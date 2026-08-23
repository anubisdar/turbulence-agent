# install-to: app/web
"""
HTTP API for the turbulence agent.

Thin by design: request validation, dispatch to `app.web.service`, and error
translation. No reasoning, no geometry, no scoring. If a rule about the
agent's behaviour appears in this file, it is in the wrong place.

Run it:
    export AEROAPI_KEY=...
    uvicorn app.web.api:app --host 0.0.0.0 --port 8000

Interactive docs at /docs, which is also a serviceable demo surface on its
own if the map is not ready.

Fixture mode (`use_fixtures: true`) replays payloads captured from a real
probe, so a recorded walkthrough cannot be spoiled by a rate limit or an
empty airport pair. Same code path, same stack, just not fetched live.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

from app.runs import resolve_origin
from app.logging_setup import (
    current_request_id,
    get_logger,
    kv,
    request_context,
)
from app.web.service import (
    DEFAULT_DB,
    SearchRequest,
    ServiceError,
    fix_cache_summary,
    run_corridor_search,
    run_reputation_search,
)

API_TITLE = "Turbulence-aware flight ranking agent"
STATIC_DIR = Path(__file__).resolve().parent / "static"

log = get_logger("api")

#: True unless the deployment says otherwise. Public means anonymous
#: callers, which changes what the ceilings should be and whether the
#: interactive docs should exist at all.
PUBLIC = os.environ.get("TURBULENCE_PUBLIC", "").lower() in ("1", "true", "yes")

#: Ceilings for an anonymous caller. The permissive values remain available
#: to an operator running the search scripts directly; they are not
#: something a stranger should be able to ask for.
#:
#: A single request at the old limits could cost five times a normal search
#: and hold the only worker for five minutes, which is the cheapest way to
#: turn one request into real money.
PUBLIC_MAX_TOOL_CALLS = 16
PUBLIC_MAX_SECONDS = 60.0


def _fail(exc: Exception, status: int = 500) -> HTTPException:
    """Log the cause, tell the caller a reference.

    Exception text is where a credential escapes. `AeroAPIError` embeds up
    to 200 characters of the upstream response body, and a 401 from AeroAPI
    can carry key material. Redaction covers the path to the log; nothing
    covered the path to an HTTP response, so the response no longer carries
    the detail at all.
    """
    reference = current_request_id()
    log.error("request failed " + kv(error=type(exc).__name__,
                                     detail=str(exc)[:300],
                                     reference=reference))
    return HTTPException(
        status_code=status,
        detail=(f"The search could not be completed. Reference {reference} "
                f"if you are reporting this."))

app = FastAPI(
    title=API_TITLE,
    version="0.1.0",
    # The interactive docs list every parameter and its bounds, including
    # the expensive ones. That is useful to an operator and a head start to
    # anyone looking for a way to make one request cost a lot.
    docs_url=None if PUBLIC else "/docs",
    redoc_url=None if PUBLIC else "/redoc",
    openapi_url=None if PUBLIC else "/openapi.json",
    description=(
        "Corridor hypothesis search and aircraft reputation retrieval. "
        "Turbulence evidence is not yet attached, so readings come back "
        "unresolved by construction - that is the weather layer, not a gap "
        "in the search."
    ),
)

# The page is served from this same origin in normal use, but a developer
# opening the HTML from disk needs this.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def correlate(request, call_next):
    """Give every request an id before anything can fail.

    The id used to be created inside the search, which meant a failure
    before that point logged `reference=-` and handed the caller a
    reference that matched nothing. The one case where a reference matters
    most is the one where it was missing.
    """
    with request_context():
        return await call_next(request)


@app.middleware("http")
async def security_headers(request, call_next):
    """Defence in depth for the page.

    Third-party text - AeroAPI idents, NTSB narratives, forecast fields -
    reaches the browser. The page escapes on every interpolation, which is
    the right control, but it is applied by hand in around forty places. A
    policy means one missed call fails closed rather than executing.
    """
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://unpkg.com "
        "https://cdnjs.cloudflare.com; "
        "style-src 'self' 'unsafe-inline' https://unpkg.com "
        "https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; "
        "img-src 'self' data: https://*.basemaps.cartocdn.com; "
        "connect-src 'self'; "
        "frame-ancestors 'none'; base-uri 'none'"
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


def _db_path() -> str:
    return os.environ.get("TURBULENCE_DB", DEFAULT_DB)


def _api_key() -> str | None:
    return os.environ.get("AEROAPI_KEY")


# ------------------------------------------------------------------ models


class CorridorSearchBody(BaseModel):
    # Letters and digits only. Length alone let `KP/T` through, and an
    # airport code is interpolated straight into a third-party URL path
    # (`/airports/{code}`) and into the language model's prompt. Four
    # characters bounds the damage; it does not make the input safe.
    origin: str = Field("KPIT", min_length=3, max_length=4,
                        pattern=r"^[A-Za-z0-9]{3,4}$",
                        description="ICAO or IATA code, e.g. KPIT")
    dest: str = Field("KBOS", min_length=3, max_length=4,
                      pattern=r"^[A-Za-z0-9]{3,4}$")
    beam_width: int = Field(2, ge=1, le=6)
    depth_limit: int = Field(2, ge=1, le=4)
    confidence_threshold: float = Field(0.85, ge=0.0, le=1.5)
    max_tool_calls: int = Field(8, ge=1, le=40,
                                description="Hard cap on metered API calls")
    max_seconds: float = Field(
        60.0, ge=5.0, le=300.0,
        description="Elapsed-time ceiling for the whole search. A search cut "
                    "off early is marked truncated and reported as the best "
                    "of what was explored.")
    width_nm: float = Field(25.0, ge=5.0, le=150.0,
                            description="Corridor half-width")
    use_graph: bool = Field(False, description="Route through LangGraph")
    use_fixtures: bool = Field(False,
                               description="Replay a captured probe instead "
                                           "of calling the live API")
    departure_date: str | None = Field(
        None, pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Travel date, YYYY-MM-DD")
    departure_time: str | None = Field(
        None, pattern=r"^([01]\d|2[0-3]):[0-5]\d$",
        description="Departure time of day, HH:MM UTC. Selects the reference "
                    "flight by time of day rather than by recency.")
    include_reputation: bool = Field(
        False, description="Also retrieve the NTSB safety record for the "
                           "aircraft type on this route")
    include_turbulence: bool = Field(
        True, description="Gather pilot reports and turbulence forecasts "
                          "along the surviving corridors. Switching this off "
                          "leaves every reading unresolved.")
    include_explanation: bool = Field(
        False, description="Have a language model write the passenger-facing "
                           "paragraph. Falls back to the deterministic "
                           "summary if it is unavailable or its output fails "
                           "validation.")

    @model_validator(mode="after")
    def origin_and_destination_must_differ(self) -> "CorridorSearchBody":
        """A route to the airport you are already at is not a route.

        Rejected here rather than absorbed downstream, because the geometry
        absorbs it badly: a zero-length path buffers to a polygon of no
        area whose containment test rejects its own defining point. The
        search then runs, spends metered calls, observes nothing, and
        reports the reading as unresolved - an absence with no explanation,
        which is the one outcome this agent is built to avoid.
        """
        if (self.origin or "").strip().upper() == (self.dest or "").strip().upper():
            raise ValueError(
                "origin and destination are the same airport, so there is no "
                "route between them to examine")
        return self

    def clamped(self) -> "CorridorSearchBody":
        """Apply the public ceilings, if this deployment is public.

        Clamped rather than rejected: a caller asking for 40 calls gets a
        working search at 16 rather than an error, and the response echoes
        what was actually used so the difference is visible.
        """
        if not PUBLIC:
            return self
        return self.model_copy(update={
            "max_tool_calls": min(self.max_tool_calls, PUBLIC_MAX_TOOL_CALLS),
            "max_seconds": min(self.max_seconds, PUBLIC_MAX_SECONDS),
        })

    def to_request(self) -> SearchRequest:
        return SearchRequest(
            origin=self.origin, dest=self.dest,
            beam_width=self.beam_width, depth_limit=self.depth_limit,
            confidence_threshold=self.confidence_threshold,
            max_tool_calls=self.max_tool_calls, max_seconds=self.max_seconds,
            width_nm=self.width_nm,
            use_graph=self.use_graph, use_fixtures=self.use_fixtures,
            departure_date=self.departure_date,
            departure_time=self.departure_time,
            include_reputation=self.include_reputation,
            include_turbulence=self.include_turbulence,
            include_explanation=self.include_explanation,
        )


# ------------------------------------------------------------------ routes


@app.get("/api/status", tags=["meta"])
def status(days: int = 30) -> dict:
    """Aggregate behaviour over the retention window.

    Behind whatever guards the rest of the site: it reveals usage patterns
    and API spend, which is not information to hand an anonymous caller.
    """
    import sqlite3

    from app.runs import init_runs, summary

    days = max(1, min(days, 90))
    try:
        conn = sqlite3.connect(_db_path())
        init_runs(conn)
        try:
            data = summary(conn, days=days)
        finally:
            conn.close()
    except Exception as e:  # noqa: BLE001
        raise _fail(e)

    edge = _edge_blocks()
    data["blocked"] = edge["counts"]
    data["blocked_countries"] = edge["countries"]
    data["blocked_note"] = edge["note"]
    return data


def _edge_blocks() -> dict:
    """Requests Caddy stopped before they reached this application.

    A different source from everything else on the page, because a blocked
    request never becomes a search and so never becomes a row.

    Returns a reason when it cannot read rather than an empty result. An
    earlier version caught OSError and returned nothing, so a log the app
    lacked permission to open reported as a quiet edge - which is the same
    mistake as reading missing weather data as calm air, in a different
    place.
    """
    path = os.environ.get("TURBULENCE_CADDY_LOG", "/var/log/caddy/access.log")

    if not os.path.exists(path):
        return {"readable": False, "note": f"no access log at {path}",
                "counts": [], "countries": []}
    if not os.access(path, os.R_OK):
        return {"readable": False,
                "note": ("the access log is not readable by this service. "
                         "Caddy creates it mode 600; add `mode 644` to the "
                         "log output block."),
                "counts": [], "countries": []}

    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - 400_000))
            tail = fh.read().decode("utf-8", "replace")
    except OSError as e:
        return {"readable": False, "note": f"{type(e).__name__} reading {path}",
                "counts": [], "countries": []}

    counts = {"geo filter": 0, "rate limit": 0}
    addresses: dict[str, int] = {}
    console_lines = 0

    for line in tail.splitlines():
        line = line.strip()
        if not line:
            continue
        if not line.startswith("{"):
            console_lines += 1
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue

        status = entry.get("status")
        if status == 403:
            counts["geo filter"] += 1
        elif status == 429:
            counts["rate limit"] += 1
        else:
            continue

        ip = ((entry.get("request") or {}).get("remote_ip") or "").strip()
        if ip:
            addresses[ip] = addresses.get(ip, 0) + 1

    if console_lines and not any(counts.values()):
        return {"readable": False,
                "note": ("the access log is in console format, which drops "
                         "the fields this needs. Set `format json` on the "
                         "log directive."),
                "counts": [], "countries": []}

    # Resolved per unique address rather than per line: a scanner makes
    # many requests from one place, and the lookup is the expensive part.
    # Named rather than coded. A reader should not have to know that CN is
    # China, and the database carries the name already.
    countries: dict[str, int] = {}
    for ip, n in addresses.items():
        origin = resolve_origin(ip)
        name = origin.country_name or origin.country or "unknown"
        countries[name] = countries.get(name, 0) + n

    return {
        "readable": True,
        "note": None,
        "counts": [{"reason": k, "n": v} for k, v in counts.items() if v],
        "countries": [{"country": k, "n": v} for k, v in
                      sorted(countries.items(), key=lambda x: -x[1])[:8]],
    }


@app.get("/status", include_in_schema=False)
def status_page():
    page = STATIC_DIR / "status.html"
    if not page.exists():
        raise HTTPException(status_code=404,
                            detail="status page not installed")
    return FileResponse(page)


@app.get("/api/health", tags=["meta"])
def health() -> dict:
    """What this instance can actually do right now."""
    db = Path(_db_path())
    return {
        "status": "ok",
        "public": PUBLIC,
        "database": str(db),
        "database_present": db.exists(),
        "aeroapi_key_configured": bool(_api_key()),
        "fixtures_available": Path("data/aeroapi_probe").exists(),
        "static_page_available": (STATIC_DIR / "index.html").exists(),
    }


@app.post("/api/search/corridors", tags=["corridors"])
def corridor_search(body: CorridorSearchBody, request: Request) -> dict:
    """Run the Tree-of-Thought corridor search.

    Returns the corridors as GeoJSON with scores and prune reasons attached,
    the pairwise airspace overlap behind the dominance rule, the full search
    trace, and every note the generator and controller raised.
    """
    try:
        req = body.clamped().to_request()
        # Used to resolve a country and then dropped. Behind a proxy the
        # socket address is Caddy, so the forwarded header is the only place
        # the real client appears.
        req.client_ip = (request.headers.get("x-forwarded-for")
                         or (request.client.host if request.client else None))
        return run_corridor_search(req, _api_key(), _db_path())
    except ServiceError as e:
        # ServiceError messages are written for the caller and carry no
        # upstream text, so they are safe to return.
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise _fail(e)


@app.get("/api/search/reputation", tags=["reputation"])
def reputation_search(
    # Bounded on purpose. `query` reaches the embedding model, which is the
    # only CPU-bound operation in this system, on a service that runs a
    # single worker. An unbounded string there is a denial of service that
    # costs the sender nothing.
    aircraft_type: str = Query(..., min_length=2, max_length=40,
                               description="e.g. '737 MAX 8', '737-8'"),
    query: str = Query("safety incidents and accidents",
                       min_length=1, max_length=200),
    k: int = Query(8, ge=1, le=20),
) -> dict:
    """Retrieve NTSB Part 121 material for an aircraft type.

    Type is filtered exactly before any vector search runs, and the response
    carries corpus counts so a thin result is never mistaken for a clean
    record.
    """
    try:
        return run_reputation_search(aircraft_type, query, k, _db_path())
    except ServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise _fail(e)


@app.get("/api/fixes", tags=["meta"])
def fixes() -> dict:
    """Route fix cache contents. Grows as routes are searched."""
    return fix_cache_summary(_db_path())


@app.get("/", include_in_schema=False)
def index():
    page = STATIC_DIR / "index.html"
    if not page.exists():
        raise HTTPException(
            status_code=404,
            detail="No page built yet. The API is at /docs.")
    return FileResponse(page)
