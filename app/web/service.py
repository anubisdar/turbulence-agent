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
from datetime import datetime, timedelta, timezone
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from app.logging_setup import (
    current_request_id,
    get_logger,
    kv,
    request_context,
    trip_fields,
)
from app.runs import (
    Timings,
    from_payload,
    init_runs,
    record_run,
    resolve_origin,
)
from app.reasoning.controller import Budget, SearchResult, search
from app.reasoning.critic import Corridor
from app.reasoning.generator import CorridorGenerator
from app.reasoning.geometry import (
    CorridorShape,
    overlap_fraction,
    unwrap_longitudes,
)
from app.reasoning.graph import search_graph
from app.retrieval.schema import connect
from app.sources.aeroapi import AeroAPIClient
from app.sources.fixes import cache_stats, init_fixes

log = get_logger("service")

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


def _unwrapped(points: list[tuple[float, float]]) -> list[list[float]]:
    """(lat, lon) pairs to GeoJSON [lon, lat], continuous across the date line.

    Everywhere else in this project a point is (lat, lon). GeoJSON reverses
    it, so this is the only place that flip happens.

    Longitudes are also unwrapped. A Seattle to Tokyo corridor steps from
    179 to -179, and a map library reads that as a 358 degree move: it draws
    the line back across the whole world rather than continuing across the
    edge. Emitting 181 instead of -179 keeps the path continuous, and
    Leaflet wraps out-of-range longitudes itself when drawing.
    """
    if not points:
        return []
    lons = unwrap_longitudes([lon for _, lon in points])
    return [[lon, lat] for (lat, _), lon in zip(points, lons)]


def _ring_to_geojson(shape: CorridorShape) -> list[list[float]]:
    ring = _unwrapped(shape.boundary_latlon())
    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def _path_to_geojson(shape: CorridorShape) -> list[list[float]]:
    return _unwrapped(shape.points)


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
    max_seconds: float = 60.0
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
    #: Used to resolve a country and then discarded. Never stored.
    client_ip: str | None = None
    #: How this request got past the challenge: not_required, session,
    #: solved, or absent when nothing set it.
    challenge: str | None = None


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
            "rejected": rejected, "enabled": True,
            "tokens_in": out.tokens_in, "tokens_out": out.tokens_out}


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


#: How far ahead a G-AIRMET reaches. Beyond this no forecast describes the
#: departure, whatever the date field says.
FORECAST_HORIZON_HOURS = 6


def _target_time(date: str | None, time_of_day: str | None
                 ) -> tuple[datetime | None, str | None]:
    """When the flight departs, and a note if that is out of forecast reach.

    The forecast layer already accepts a time and already filters advisories
    to those valid at it. Nothing was supplying one, so every search asked
    what the air is like *now* and presented the answer as though it were
    about the requested departure.

    Two cases, and the second is the one that matters. Inside the forecast
    horizon the target time is used and the answer is about the departure.
    Beyond it, no forecast reaches that far and the agent falls back to the
    present - which is defensible only if it says so, because otherwise it
    is describing one thing while appearing to describe another.
    """
    if not date and not time_of_day:
        return None, None

    now = datetime.now(timezone.utc)
    try:
        if date and time_of_day:
            target = datetime.strptime(f"{date} {time_of_day}",
                                       "%Y-%m-%d %H:%M")
        elif date:
            target = datetime.strptime(date, "%Y-%m-%d")
        else:
            hour, _, minute = time_of_day.partition(":")
            target = now.replace(hour=int(hour), minute=int(minute),
                                 second=0, microsecond=0)
            # A time of day already past today means the next occurrence.
            if target < now:
                target += timedelta(days=1)
        target = target.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None, None

    ahead = (target - now).total_seconds() / 3600.0
    if ahead < 0:
        return None, (
            "The requested departure is in the past, so the turbulence "
            "reading describes conditions now rather than then.")
    if ahead > FORECAST_HORIZON_HOURS:
        return None, (
            f"This departure is about {ahead:.0f} hours away and turbulence "
            f"forecasts reach roughly {FORECAST_HORIZON_HOURS}. The reading "
            f"below describes the air on this route now, not at departure.")
    return target, None


def run_corridor_search(req: SearchRequest, api_key: str | None,
                        db_path: str = DEFAULT_DB) -> dict[str, Any]:
    """Run the search and shape it for a map. No search logic lives here.

    Everything a search touches logs under one request id, so the generator,
    the critic, the controller and the explainer can be read as one story
    rather than as interleaved fragments.
    """
    # Guarded here as well as at the API, so a script or a future caller
    # cannot reach the geometry with a route that has no length. A
    # zero-length path buffers to a polygon of no area whose containment
    # test rejects its own defining point, and the search that follows
    # observes nothing and says nothing about why.
    if req.origin.strip().upper() == req.dest.strip().upper():
        raise ServiceError(
            f"{req.origin.strip().upper()} is both the origin and the "
            f"destination, so there is no route between them to examine.")

    # Reuse the id the API layer established, so a log line written before
    # the search started correlates with the ones written during it. A new
    # context here would split one request across two ids.
    existing = current_request_id()
    if existing and existing != "-":
        return _run_corridor_search(req, api_key, db_path, existing)
    with request_context() as request_id:
        return _run_corridor_search(req, api_key, db_path, request_id)


def _run_corridor_search(req: SearchRequest, api_key: str | None,
                         db_path: str, request_id: str) -> dict[str, Any]:
    timings = Timings()
    log.info("search started " + kv(
        **trip_fields(req.origin.upper(), req.dest.upper(),
                      req.departure_time),
        beam=req.beam_width, depth=req.depth_limit,
        cap=req.max_tool_calls,
        source="fixtures" if req.use_fixtures else "live",
        turbulence=req.include_turbulence,
        explanation=req.include_explanation))

    client, fixtures = _build_client(req, api_key)

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(path)
    try:
        init_fixes(conn)
        cache_before = cache_stats(conn)["total"]

        fetch_pireps, gairmet_client, wx_notes = _turbulence_sources(
            req.include_turbulence)
        if gairmet_client is not None:
            gairmet_client.timings = timings

        forecast_when, horizon_note = _target_time(
            req.departure_date, req.departure_time)
        if horizon_note:
            # Alongside the other source notes, so it appears in the same
            # place a reader already looks for what the agent could not
            # establish.
            wx_notes.append(horizon_note)

        generator = CorridorGenerator(
            client=client, conn=conn,
            origin=req.origin.upper(), dest=req.dest.upper(),
            width_nm=req.width_nm, target_time=req.departure_time,
            when=forecast_when,
            fetch_pireps=fetch_pireps, gairmet_client=gairmet_client,
        )
        runner = search_graph if req.use_graph else search
        result = runner(
            generator,
            beam_width=req.beam_width,
            depth_limit=req.depth_limit,
            confidence_threshold=req.confidence_threshold,
            budget=Budget(max_tool_calls=req.max_tool_calls,
                          max_seconds=req.max_seconds),
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
        retrieval_started = time.perf_counter()
        if req.include_reputation:
            reputation = _reputation_for(aircraft, db_path, timings)
            timings.retrieval_seconds = round(
                time.perf_counter() - retrieval_started, 4)

        payload = {
            "request": {
                "origin": req.origin.upper(), "dest": req.dest.upper(),
                "beam_width": req.beam_width, "depth_limit": req.depth_limit,
                "confidence_threshold": req.confidence_threshold,
                "max_tool_calls": req.max_tool_calls,
                "max_seconds": req.max_seconds, "width_nm": req.width_nm,
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
                # Distinct from `truncated`. A truncated search was stopped
                # by a budget; a degraded one lost a data source and
                # explored less of the tree as a result. Both produce a
                # partial answer, for different reasons.
                "degraded": bool(generator.degraded),
                "degraded_reasons": list(generator.degraded),
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
                **generator.cache_report(),
            },
            "call_log": list(client.call_log),
            "request_id": request_id,
        }
        # The explainer reads the finished payload, so it can only restate
        # what the search already established.
        explain_started = time.perf_counter()
        payload["explanation"] = _explain(payload, req.include_explanation)
        timings.explainer_seconds = round(time.perf_counter() - explain_started, 4)
        # Everything not attributed to a named source. Called what it is:
        # an earlier version labelled this "scoring", which made the chart
        # claim the deterministic core was the slowest part of the system
        # when it was really absorbing every unmeasured call.
        timings.scoring_seconds = round(
            max(0.0, (result.elapsed or 0)
                - timings.aeroapi_seconds
                - timings.awc_seconds
                - timings.retrieval_seconds), 4)

        # One row per search, so the agent's behaviour can be asked
        # questions rather than only described. Never allowed to fail a
        # search: a missing row is a gap in a chart.
        try:
            runs_conn = sqlite3.connect(db_path)
            init_runs(runs_conn)
            record_run(runs_conn, from_payload(
                payload, request_id, timings,
                origin_info=resolve_origin(req.client_ip),
                challenge=req.challenge))
            runs_conn.close()
        except Exception as e:  # noqa: BLE001
            log.warning("could not record the run " + kv(error=type(e).__name__))

        log.info("search finished " + kv(
            request_id=request_id,
            stop=result.stop.value, truncated=result.truncated,
            nodes=result.nodes_generated, calls=result.calls_used,
            elapsed=result.elapsed, winner=result.winner.id if result.winner else None,
            reading=result.reading.value,
            contested=result.contested))
        if result.truncated:
            log.warning("search stopped on a budget rather than confidence "
                        + kv(stop=result.stop.value,
                             calls=result.calls_used,
                             cap=req.max_tool_calls))
        cache = generator.cache_report()
        log.info("fix cache " + kv(
            before=cache_before, after=cache_after["total"],
            learned=max(0, cache_after["total"] - cache_before),
            served_warm=cache["routings_served_warm"],
            calls_saved=cache["calls_saved"]))

        if generator.degraded:
            log.warning("search degraded by a failing data source "
                        + kv(failures=len(generator.degraded),
                             nodes=result.nodes_generated,
                             first=generator.degraded[0]))
        if result.reading.value == "unresolved":
            log.info("no turbulence reading established "
                     + kv(observed=(payload["outcome"]["turbulence"] or {})
                          .get("observed", {}).get("reading"),
                          forecast=(payload["outcome"]["turbulence"] or {})
                          .get("forecast", {}).get("reading")))

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


def _reputation_for(aircraft: dict | None, db_path: str,
                    timings: object | None = None) -> dict[str, Any]:
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
            label, "safety incidents and accidents", 5, db_path, timings)
        return {"available": True, "searched_as": label, **out}
    except Exception as e:  # noqa: BLE001
        return {
            "available": False,
            "reason": f"Safety record lookup failed: {type(e).__name__}: {e}",
        }


def run_reputation_search(aircraft_type: str, query: str, k: int,
                          db_path: str = DEFAULT_DB,
                          timings: object | None = None) -> dict[str, Any]:
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
            conn, _timed_encoder(timings), aircraft_type, query, k=k)
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


def _timed_encoder(timings):
    """Wrap the encoder so its CPU cost is recorded.

    The embedding model is the only real computation this machine performs.
    Its cost is measured as CPU time rather than wall clock, because wall
    clock on a shared box measures the scheduler as much as the work.

    Without a timing sink this returns the encoder untouched, so scripts and
    tests are unaffected.
    """
    encoder = _encoder()
    if timings is None:
        return encoder

    class _Timed:
        def encode(self, *args, **kwargs):
            with timings.track_embedding():
                return encoder.encode(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(encoder, name)

    return _Timed()


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
