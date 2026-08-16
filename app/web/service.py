# install-to: app/web
"""
Service layer between the HTTP API and the reasoning stack.

Deliberately the only place that knows about both. Nothing in
`app/reasoning` or `app/retrieval` changes to support the web interface, and
nothing here reimplements search logic - it runs the same controller the
command-line script runs and reshapes the result for a map.

CORRIDORS BECOME GEOJSON. `CorridorShape.boundary_latlon()` already returns
the corridor outline in latitude/longitude; this converts it to the
[lon, lat] ordering GeoJSON requires, which is the reverse of the ordering
used everywhere else in this project. Getting that backwards puts
Pennsylvania in the Indian Ocean, so the conversion happens in exactly one
function.

FIXTURE MODE EXISTS FOR THE DEMO. A recorded walkthrough that depends on a
live metered API can be ruined by a rate limit, an empty airport pair, or a
slow response. In fixture mode the same code path runs against payloads
captured from a real probe, so what is demonstrated is the real stack on
real data, just not fetched live.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from app.reasoning.controller import Budget, SearchResult, search
from app.reasoning.critic import Corridor
from app.reasoning.generator import CorridorGenerator
from app.reasoning.geometry import CorridorShape, overlap_fraction
from app.reasoning.graph import search_graph
from app.retrieval.schema import connect
from app.sources.aeroapi import AeroAPIClient
from app.sources.fixes import cache_stats, init_fixes

DEFAULT_DB = "data/retrieval.db"
FIXTURE_DIR = Path("data/aeroapi_probe")

#: Overlap at or above this counts as dominance. Mirrors the critic so the
#: map can shade the pair that triggered a prune.
DOMINANCE_OVERLAP = 0.80


class ServiceError(RuntimeError):
    pass


# ------------------------------------------------------------------ fixtures


class FixtureTransport:
    """Serves payloads captured from a live probe.

    Matches on path shape rather than exact URL so a saved response for one
    flight id answers for any, which is what makes a replay possible.
    """

    def __init__(self, directory: Path = FIXTURE_DIR):
        self.directory = Path(directory)
        self.calls: list[str] = []

    def _load(self, name: str) -> dict | None:
        path = self.directory / f"{name}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    @staticmethod
    def _flight_ident(path: str) -> str | None:
        """`/flights/JBU1286-178...-airline-551p/route` -> `JBU1286`."""
        parts = [p for p in path.split("/") if p]
        if len(parts) >= 2 and parts[0] == "flights":
            return parts[1].split("-")[0]
        return None

    @property
    def available(self) -> bool:
        return all((self.directory / f"{n}.json").exists() for n in
                   ("airport_pair_flights", "flight_route",
                    "flight_track", "airport_routes"))

    def __call__(self, path: str, params: dict) -> tuple[int, dict | None, str]:
        self.calls.append(path)
        if "/flights/to/" in path:
            body = self._load("airport_pair_flights")
        elif path.endswith("/route"):
            # A search fetches the filed route of one flight and, when the
            # fix cache is cold, the route of a second flight that filed the
            # alternate routing. Serving one file for both would give the
            # alternate the wrong waypoints, so a per-flight fixture wins if
            # one exists.
            ident = self._flight_ident(path)
            body = (self._load(f"flight_route_{ident}") if ident else None)
            if body is None:
                body = self._load("flight_route")
        elif path.endswith("/track"):
            body = self._load("flight_track")
        elif "/routes/" in path:
            body = self._load("airport_routes")
        elif path.endswith("/usage"):
            body = self._load("account_usage")
        else:
            return 404, None, "no fixture for this path"
        if body is None:
            return 404, None, "fixture file missing"
        return 200, body, ""


# ------------------------------------------------------------------ geojson


def _ring_to_geojson(shape: CorridorShape) -> list[list[float]]:
    """Corridor outline as GeoJSON [lon, lat] pairs.

    Everywhere else in this project a point is (lat, lon). GeoJSON reverses
    it. This is the only place that flip happens.
    """
    ring = [[lon, lat] for lat, lon in shape.boundary_latlon()]
    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def _path_to_geojson(shape: CorridorShape) -> list[list[float]]:
    return [[lon, lat] for lat, lon in shape.points]


def corridor_features(shapes: dict[str, CorridorShape],
                      meta: dict[str, dict]) -> list[dict]:
    """One polygon and one centreline per corridor, sharing properties."""
    features: list[dict] = []
    for cid, shape in shapes.items():
        props = dict(meta.get(cid, {}))
        props["id"] = cid
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon",
                         "coordinates": [_ring_to_geojson(shape)]},
            "properties": {**props, "kind": "corridor"},
        })
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString",
                         "coordinates": _path_to_geojson(shape)},
            "properties": {**props, "kind": "centerline"},
        })
    return features


# ------------------------------------------------------------------ search


@dataclass
class SearchRequest:
    origin: str = "KPIT"
    dest: str = "KBOS"
    beam_width: int = 2
    depth_limit: int = 2
    confidence_threshold: float = 0.85
    max_tool_calls: int = 8
    width_nm: float = 25.0
    use_graph: bool = False
    use_fixtures: bool = False
    departure_time: str | None = None      # "HH:MM" UTC
    departure_date: str | None = None      # "YYYY-MM-DD", recorded not queried
    include_reputation: bool = False
    #: On by default. Turning it off leaves every reading unresolved, which
    #: is honest but is not what the agent is for.
    include_turbulence: bool = True
    #: Off by default. The deterministic summary is already a complete
    #: answer; the model only rewrites it into something more readable.
    include_explanation: bool = False


def _build_client(req: SearchRequest, api_key: str | None
                  ) -> tuple[AeroAPIClient, FixtureTransport | None]:
    if req.use_fixtures:
        transport = FixtureTransport()
        if not transport.available:
            raise ServiceError(
                f"Fixture mode requested but {FIXTURE_DIR} does not hold a "
                f"captured probe. Run scripts/probe_aeroapi.py first."
            )
        return (AeroAPIClient(api_key="fixture", transport=transport,
                              spacing_seconds=0.0, sleep=lambda s: None),
                transport)
    if not api_key:
        raise ServiceError(
            "AEROAPI_KEY is not set. Start the server with a key, or request "
            "fixture mode."
        )
    return AeroAPIClient(api_key=api_key), None


def _turbulence_sources(enabled: bool):
    """Build the two turbulence sources, or nothing if they are switched off.

    Either source failing to construct leaves that half unavailable and is
    reported downstream. Neither absence ever becomes a reading.
    """
    if not enabled:
        return None, None, ["Turbulence lookup was switched off for this "
                            "search, so every reading stays unresolved."]

    notes: list[str] = []
    fetch = None
    try:
        from app.reasoning.evidence import sync_pirep_fetcher
        from app.sources.awc import fetch_pireps
        fetch = sync_pirep_fetcher(fetch_pireps)
    except Exception as e:  # noqa: BLE001
        notes.append(f"Pilot reports are unavailable ({type(e).__name__}). "
                     f"The observed side will be unknown, not clear.")

    client = None
    try:
        from app.sources.gairmet import GairmetClient
        client = GairmetClient()
    except Exception as e:  # noqa: BLE001
        notes.append(f"Turbulence forecasts are unavailable "
                     f"({type(e).__name__}). The forecast side will be "
                     f"unknown, not clear.")

    return fetch, client, notes


def _corridor_meta(result: SearchResult,
                   corridors_by_id: dict[str, Corridor],
                   shapes: dict[str, CorridorShape]) -> dict[str, dict]:
    """Everything the map needs to style and label a corridor."""
    meta: dict[str, dict] = {}
    winner = result.winner.id if result.winner else None
    survivors = {c.id for c in result.survivors}

    for level in result.levels:
        for score in level.kept + level.pruned:
            cid = score.corridor_id
            corridor = corridors_by_id.get(cid)
            shape = shapes.get(cid)
            meta[cid] = {
                "depth": level.depth,
                "score": score.total,
                "components": score.components,
                "decision": score.decision.value,
                "reason": score.reason,
                "kept": score.kept,
                "is_winner": cid == winner,
                "is_survivor": cid in survivors,
                "provenance": corridor.provenance.value if corridor else None,
                "label": corridor.label if corridor else "",
                "parent_id": corridor.parent_id if corridor else None,
                "length_nm": shape.length_nm if shape else None,
                "max_dogleg_deg": shape.max_dogleg if shape else None,
                "area_nm2": shape.area_nm2() if shape else None,
                "altitude_min_ft": shape.altitude_min_ft if shape else None,
                "altitude_max_ft": shape.altitude_max_ft if shape else None,
            }
            evidence = corridor.evidence if corridor else None
            if evidence is not None:
                meta[cid].update({
                    "reading": evidence.reading.value,
                    # Observed and forecast are kept apart all the way to the
                    # interface, so a disagreement is visible rather than
                    # blended into one number.
                    "observed_reading": evidence.observed_reading.value,
                    "observed_count": evidence.observed_count,
                    "forecast_reading": evidence.forecast_reading.value,
                    "forecast_count": evidence.forecast_count,
                    "sources_disagree": evidence.sources_disagree,
                    "mean_age_minutes": evidence.mean_age_minutes,
                })
    return meta


def _explain(payload: dict[str, Any], enabled: bool) -> dict[str, Any]:
    """Ask the model for a passenger-facing paragraph, or keep the plain one.

    The explainer is an enhancement rather than a dependency. No key, no
    SDK, a timeout, or an output that fails validation all leave the reader
    with the deterministic summary that was always going to be there.
    """
    from app.reasoning.explainer import DEFAULT_MODEL, explain

    if not enabled:
        out = explain(payload, client=None)
        return {"text": out.text, "source": out.source, "model": None,
                "rejected": [], "enabled": False}

    client = None
    note = None
    try:
        from app.reasoning.explainer import AnthropicClient
        if not os.environ.get("ANTHROPIC_API_KEY"):
            note = ("ANTHROPIC_API_KEY is not set, so the plain summary was "
                    "used instead of a written explanation.")
        else:
            client = AnthropicClient()
    except Exception as e:  # noqa: BLE001
        note = (f"The explainer could not be constructed "
                f"({type(e).__name__}), so the plain summary was used.")

    out = explain(payload, client=client, model_name=DEFAULT_MODEL)
    rejected = list(out.rejected)
    if note:
        rejected.append(note)
    return {"text": out.text, "source": out.source, "model": out.model,
            "rejected": rejected, "enabled": True}


def _turbulence_summary(result: SearchResult, generator) -> dict[str, Any]:
    """The winning corridor's two readings, kept separate.

    The combined reading is the worse of the two, which is the conservative
    direction. Showing only that would hide whether the sources agreed.
    """
    winner = result.winner
    evidence = winner.evidence if winner else None
    if evidence is None:
        return {"available": False,
                "reason": "No corridor survived, so nothing was gathered.",
                "summary": "No route could be established, so nothing is "
                           "known about the air on it."}

    # One plain sentence for a reader who will not read the notes.
    gathered = getattr(generator, "evidence", {}).get(winner.id)
    summary = getattr(gathered, "summary", None) if gathered else None

    return {
        "summary": summary,
        "available": evidence.reading.value != "unresolved",
        "reading": evidence.reading.value,
        "observed": {
            "reading": evidence.observed_reading.value,
            "count": evidence.observed_count,
            "mean_age_minutes": evidence.mean_age_minutes,
        },
        "forecast": {
            "reading": evidence.forecast_reading.value,
            "count": evidence.forecast_count,
        },
        "disagree": evidence.sources_disagree,
        "coverage_fraction": evidence.coverage_fraction,
    }


def _overlaps(shapes: dict[str, CorridorShape]) -> list[dict]:
    """Pairwise airspace overlap among depth-1 corridors.

    This is the evidence behind the dominance threshold, so it is returned
    whether or not a prune actually fired.
    """
    top = [cid for cid in shapes if "/" not in cid]
    out = []
    for a, b in combinations(sorted(top), 2):
        frac = overlap_fraction(shapes[a], shapes[b])
        out.append({"a": a, "b": b, "fraction": frac,
                    "dominance_range": frac >= DOMINANCE_OVERLAP})
    out.sort(key=lambda o: -o["fraction"])
    return out


def run_corridor_search(req: SearchRequest, api_key: str | None,
                        db_path: str = DEFAULT_DB) -> dict[str, Any]:
    """Run the search and shape it for a map. No search logic lives here."""
    client, fixtures = _build_client(req, api_key)

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(path)
    try:
        init_fixes(conn)
        cache_before = cache_stats(conn)["total"]

        fetch_pireps, gairmet_client, wx_notes = _turbulence_sources(
            req.include_turbulence)

        generator = CorridorGenerator(
            client=client, conn=conn,
            origin=req.origin.upper(), dest=req.dest.upper(),
            width_nm=req.width_nm, target_time=req.departure_time,
            fetch_pireps=fetch_pireps, gairmet_client=gairmet_client,
        )
        runner = search_graph if req.use_graph else search
        result = runner(
            generator,
            beam_width=req.beam_width,
            depth_limit=req.depth_limit,
            confidence_threshold=req.confidence_threshold,
            budget=Budget(max_tool_calls=req.max_tool_calls),
            overlap_fn=generator.overlap_fn,
            # Survivors only: see controller.search. A corridor about to be
            # pruned by dominance should not cost a weather fetch.
            enrich=generator.gather_for_survivors,
        )

        corridors_by_id: dict[str, Corridor] = {}
        for level in result.levels:
            for corridor in level.generated:
                corridors_by_id[corridor.id] = corridor

        meta = _corridor_meta(result, corridors_by_id, generator.shapes)
        cache_after = cache_stats(conn)
        aircraft = _aircraft_from(generator)
        reputation = None
        if req.include_reputation:
            reputation = _reputation_for(aircraft, db_path)

        payload = {
            "request": {
                "origin": req.origin.upper(), "dest": req.dest.upper(),
                "beam_width": req.beam_width, "depth_limit": req.depth_limit,
                "confidence_threshold": req.confidence_threshold,
                "max_tool_calls": req.max_tool_calls, "width_nm": req.width_nm,
                "controller": "langgraph" if req.use_graph else "plain",
                "source": "fixtures" if req.use_fixtures else "live",
                "departure_date": req.departure_date,
                "departure_time": req.departure_time,
                "include_reputation": req.include_reputation,
                "include_turbulence": req.include_turbulence,
                "include_explanation": req.include_explanation,
            },
            "aircraft": aircraft,
            "reputation": reputation,
            "outcome": {
                "stop": result.stop.value,
                "truncated": result.truncated,
                "contested": result.contested,
                "depth_reached": result.depth_reached,
                "nodes_generated": result.nodes_generated,
                "calls_used": result.calls_used,
                "elapsed_seconds": result.elapsed,
                "winner": result.winner.id if result.winner else None,
                "reading": result.reading.value,
                "survivors": [c.id for c in result.survivors],
                "turbulence": _turbulence_summary(result, generator),
            },
            "corridors": [
                {"id": cid, **values} for cid, values in meta.items()
            ],
            "geojson": {"type": "FeatureCollection",
                        "features": corridor_features(generator.shapes, meta)},
            "overlaps": _overlaps(generator.shapes),
            "trace": result.trace(),
            "notes": result.notes + wx_notes,
            "generator_notes": generator.notes,
            "fix_cache": {
                "before": cache_before,
                "after": cache_after["total"],
                "by_type": cache_after["by_type"],
            },
            "call_log": list(client.call_log),
        }
        # The explainer reads the finished payload, so it can only restate
        # what the search already established.
        payload["explanation"] = _explain(payload, req.include_explanation)

        # Narration is derived from the finished payload, so it can describe
        # what happened but never influence it.
        from app.web.narrate import narrate
        payload["narration"] = narrate(payload)
        return payload
    finally:
        conn.close()


# ------------------------------------------------------------------ retrieval


def _aircraft_from(generator: CorridorGenerator) -> dict[str, Any] | None:
    """Canonical aircraft type for the reference flight.

    AeroAPI reports ICAO designators; the NTSB corpus is filed against
    make/model strings. Without this bridge there is no way to ask what has
    happened to the aircraft the passenger is actually booked on.
    """
    from app.retrieval.aircraft_types import resolve_icao

    flight = generator._flight
    if flight is None:
        return None
    designator = flight.aircraft_type
    resolved = resolve_icao(designator) if designator else None
    return {
        "ident": flight.ident,
        "icao_designator": designator,
        "variant": resolved.variant if resolved else None,
        "family": resolved.family if resolved else None,
        "generation": resolved.generation if resolved else None,
        "manufacturer": resolved.manufacturer if resolved else None,
        "resolved": bool(resolved and resolved.usable),
    }


def _reputation_for(aircraft: dict | None, db_path: str) -> dict[str, Any]:
    """Safety record for the aircraft type, or an explanation of why not.

    An unresolvable type returns a stated absence. Falling back to a nearby
    type would be worse than returning nothing.
    """
    if not aircraft or not aircraft.get("resolved"):
        return {
            "available": False,
            "reason": (
                f"Aircraft type "
                f"{(aircraft or {}).get('icao_designator') or 'unknown'} could "
                f"not be resolved to a type in the NTSB corpus, so no safety "
                f"record was retrieved. This is an absence of lookup, not an "
                f"absence of events."
            ),
        }
    label = aircraft["variant"] or aircraft["family"]
    try:
        out = run_reputation_search(
            label, "safety incidents and accidents", 5, db_path)
        return {"available": True, "searched_as": label, **out}
    except Exception as e:  # noqa: BLE001
        return {
            "available": False,
            "reason": f"Safety record lookup failed: {type(e).__name__}: {e}",
        }


def run_reputation_search(aircraft_type: str, query: str, k: int,
                          db_path: str = DEFAULT_DB) -> dict[str, Any]:
    """Aircraft reputation lookup over the NTSB index.

    Imported lazily: the embedding model takes several seconds to load and
    the corridor endpoints do not need it.
    """
    from app.retrieval.embedding import SentenceTransformerEncoder
    from app.retrieval.search import search_aircraft_reputation

    path = Path(db_path)
    if not path.exists():
        raise ServiceError(f"No index at {path}. Run the ingest first.")

    conn = connect(path, load_vec=True)
    try:
        out = search_aircraft_reputation(
            conn, _encoder(), aircraft_type, query, k=k)
        resolved = out.resolved_type
        return {
            "query": {"aircraft_type": aircraft_type, "query": query, "k": k},
            "resolved_type": {
                "manufacturer": resolved.manufacturer if resolved else None,
                "family": resolved.family if resolved else None,
                "variant": resolved.variant if resolved else None,
                "generation": resolved.generation if resolved else None,
                "confidence": resolved.confidence.value if resolved else None,
            },
            "coverage": {
                "cases_variant": out.coverage.cases_variant,
                "cases_variant_with_text": out.coverage.cases_variant_with_text,
                "cases_family": out.coverage.cases_family,
                "cases_family_with_text": out.coverage.cases_family_with_text,
                "oldest_event_year": out.coverage.oldest_event_year,
                "newest_event_year": out.coverage.newest_event_year,
            },
            "hits": [
                {
                    "ntsb_num": h.ntsb_num, "event_year": h.event_year,
                    "section": h.section, "score": h.score, "tier": h.tier,
                    "variant": h.variant, "family": h.family,
                    "generation": h.generation, "raw_model": h.raw_model,
                    "operator": h.operator, "report_type": h.report_type,
                    "provisional": h.provisional, "text": h.text,
                    "source": h.source, "source_class": h.source_class,
                }
                for h in out.hits
            ],
            "notes": out.notes,
        }
    finally:
        conn.close()


_ENCODER = None


def _encoder():
    """One encoder per process. Loading it costs several seconds."""
    global _ENCODER
    if _ENCODER is None:
        from app.retrieval.embedding import SentenceTransformerEncoder
        _ENCODER = SentenceTransformerEncoder()
    return _ENCODER


def fix_cache_summary(db_path: str = DEFAULT_DB) -> dict[str, Any]:
    path = Path(db_path)
    if not path.exists():
        return {"total": 0, "by_type": {}}
    conn = connect(path)
    try:
        init_fixes(conn)
        return cache_stats(conn)
    finally:
        conn.close()
