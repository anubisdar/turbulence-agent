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

import ipaddress
import json
import os
import time
import re
import subprocess
from urllib.parse import unquote
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

from app.runs import resolve_origin
from app.web import turnstile
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
        "https://cdnjs.cloudflare.com https://challenges.cloudflare.com; "
        "frame-src https://challenges.cloudflare.com; "
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
    #: Supplied by the widget on the page. Ignored entirely when Turnstile
    #: is not configured.
    turnstile_token: str | None = Field(None, max_length=2048)
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


#: Networks that reach the application without passing through Caddy:
#: loopback, and the VPC the instance sits in.
_LOCAL_NETWORKS = (
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("172.31.0.0/16"),
)


def _is_local(request: Request) -> bool:
    """Whether the request came from inside the instance.

    Parsed as an address rather than matched as a string prefix. A prefix
    test on "127." also accepts "127.0.0.1.evil.com", which cannot reach
    this field in practice - it holds a parsed socket address - but
    matching text against something that should be an IP is the kind of
    check that stops being true when the plumbing changes.

    Deliberately the socket address rather than a forwarded header, which
    a caller controls. A proxied request arrives from Caddy with the real
    client in X-Forwarded-For, and Caddy is not on this list.
    """
    host = (request.client.host if request.client else "") or ""
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return any(address in network for network in _LOCAL_NETWORKS)


def _require_human(request: Request, response: Response,
                   token: str | None) -> str:
    """Let the request through if a person is plausibly behind it.

    Order matters. A valid session skips the network call entirely, which
    is what makes a demo of several searches bearable; only the first one
    pays for a round trip to Cloudflare.

    Returns the outcome so it can be recorded. An earlier version raised a
    403 and logged nothing, which left the most useful number invisible:
    how many searches were turned away for presenting no token at all,
    which is what an automated client looks like.

    Silent when Turnstile is not configured, so development and the test
    suite are unaffected.
    """
    if not turnstile.enabled():
        return "not_required"

    # Loopback and the VPC are exempt, for the same reason the geo filter
    # exempts them: a request from inside the instance is the operator, not
    # a visitor. This is what makes the load test and the validation
    # scripts runnable at all - they are automated clients, which is
    # precisely what the challenge exists to refuse.
    #
    # Note this is the *socket* address, so it cannot be spoofed by a
    # header. A forwarded request arrives from Caddy with the real client
    # in X-Forwarded-For, and Caddy is not on this list.
    if _is_local(request):
        log.info("challenge " + kv(outcome="local"))
        return "local"

    # A header rather than a query parameter, because a query parameter
    # ends up in the access log, in browser history and in any referrer
    # sent onward.
    if turnstile.operator_token_matches(
            request.headers.get("x-operator-token")):
        log.info("challenge " + kv(outcome="operator"))
        return "operator"

    if turnstile.session_is_valid(request.cookies.get(turnstile.COOKIE_NAME)):
        log.info("challenge " + kv(outcome="session"))
        return "session"

    client = (request.headers.get("x-forwarded-for")
              or (request.client.host if request.client else None))

    if not token:
        # Distinguished from a token that was presented and rejected. A
        # request with no token never saw the page, which is the signature
        # worth counting.
        log.warning("challenge " + kv(outcome="no_token"))
        raise HTTPException(
            status_code=403,
            detail=("This search needs the challenge on the page to be "
                    "completed. Reload and try again."))

    if not turnstile.verify(token, client):
        log.warning("challenge " + kv(outcome="rejected"))
        raise HTTPException(
            status_code=403,
            detail=("The challenge could not be verified. Reload and try "
                    "again."))

    response.set_cookie(
        turnstile.COOKIE_NAME, turnstile.issue_session(),
        max_age=turnstile.SESSION_SECONDS,
        httponly=True, secure=True, samesite="lax", path="/")
    log.info("challenge " + kv(outcome="solved"))
    return "solved"


@app.get("/api/config", tags=["meta"])
def config() -> dict:
    """What the page needs to render itself. Public half only."""
    return {"turnstile_site_key": turnstile.site_key(),
            "turnstile_enabled": turnstile.enabled()}


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

    data["challenge_refusals"] = _challenge_refusals()
    data["waf"] = _waf_detections()

    edge = _edge_blocks()
    data["blocked"] = edge["counts"]
    data["blocked_countries"] = edge["countries"]
    data["blocked_note"] = edge["note"]
    return data


def _challenge_refusals(hours: int = 24) -> dict:
    """Searches the challenge turned away.

    These never become rows: a refused request stops before the search
    starts. The journal is the only record, which makes this a different
    kind of number from everything else on the page - recent rather than
    thirty days, and approximate rather than exact.
    """
    if not turnstile.enabled():
        return {"enabled": False, "no_token": 0, "rejected": 0}

    counts = {"no_token": 0, "rejected": 0}
    try:
        result = subprocess.run(
            ["journalctl", "-t", "turbulence-agent", "--since",
             f"{hours} hours ago", "-o", "cat", "--no-pager"],
            capture_output=True, text=True, timeout=8)
    except (OSError, subprocess.SubprocessError):
        return {"enabled": True, "readable": False, **counts}

    for line in result.stdout.splitlines():
        if "challenge outcome=" not in line:
            continue
        for key in counts:
            if f"outcome={key}" in line:
                counts[key] += 1
    return {"enabled": True, "readable": True, "hours": hours, **counts}


#: CRS tags an attack family on each rule. Mapped to plain words because
#: "attack-lfi" is not what a reader wants to see.
_ATTACK_NAMES = {
    "attack-reputation-scanner": "scanner detection",
    "attack-lfi": "path traversal",
    "attack-rfi": "remote file inclusion",
    "attack-rce": "remote command execution",
    "attack-sqli": "SQL injection",
    "attack-xss": "cross-site scripting",
    "attack-injection-php": "PHP injection",
    "attack-injection-generic": "code injection",
    "attack-protocol": "protocol violation",
    "attack-fixation": "session fixation",
    "attack-disclosure": "information disclosure",
    "attack-reputation-ip": "known bad address",
    "attack-generic": "generic attack",
}

#: The rule that carries the request-level anomaly score. CRS refuses a
#: request at 5 or more, so this is the number that decides whether moving
#: to SecRuleEngine On would have blocked it.
_ANOMALY_RULE = 949110
_BLOCKING_THRESHOLD = 5

_SCORE = re.compile(r"Total Score:\s*(\d+)")

#: Coraza writes its messages in the bracketed format ModSecurity uses,
#: inside the JSON `msg` field. These pull the pieces back out.
_UNIQUE_ID = re.compile(r'\[unique_id \\?"([^"\\]+)')
_URI = re.compile(r'\[uri \\?"([^"\\]*)')
_CLIENT = re.compile(r'\[client \\?"([^"\\]+)')
_TAG = re.compile(r'\[tag \\?"([^"\\]+)')


def _waf_detections(hours: int = 24) -> dict:
    """What the firewall saw, counted per request rather than per rule.

    Read from journald rather than from a Coraza audit file. The audit log
    was the obvious choice and turned out to be unusable: its JSON
    formatter emits full request headers regardless of SecAuditLogParts,
    Coraza does not implement SecSanitiseRequestHeader, and the result was
    a file on disk containing the Authorization header of every
    authenticated request. There is no configuration that removes it.

    journald carries the same detections without a second copy of the
    credentials, so it is the better source despite being the more awkward
    one to parse.

    Counted per request: one request tripping four rules is one detection.
    The four test vectors run against this deployment produced thirteen log
    lines between them, so a line count would be wrong by a factor of
    three. Requests are grouped by the unique_id Coraza assigns each one.
    """
    empty = {"enabled": False, "detections": 0, "would_block": 0,
             "sources": 0, "worst_score": 0, "attacks": [], "countries": [],
             "paths": [], "note": None}

    try:
        result = subprocess.run(
            ["journalctl", "-u", "caddy", "--since", f"{hours} hours ago",
             "-o", "cat", "--no-pager"],
            capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as e:
        return {**empty, "note": f"could not read the journal: "
                                 f"{type(e).__name__}"}

    # One entry per matched rule. Grouped by unique_id so a request that
    # trips four rules counts once.
    requests: dict[str, dict] = {}

    for line in result.stdout.splitlines():
        # Parse first, match on the field. Caddy writes compact JSON with
        # no space after the colon, but matching the raw text would break
        # against any writer that formats differently - including the test
        # fixtures, which is how this was found.
        if "http.handlers.waf" not in line:
            continue
        try:
            entry_json = json.loads(line) or {}
        except ValueError:
            continue
        if entry_json.get("logger") != "http.handlers.waf":
            continue
        message = entry_json.get("msg") or ""
        if "Coraza:" not in message:
            continue

        found = _UNIQUE_ID.search(message)
        if not found:
            continue
        entry = requests.setdefault(
            found.group(1),
            {"score": 0, "families": set(), "uri": "/", "ip": ""})

        score = _SCORE.search(message)
        if score:
            entry["score"] = max(entry["score"], int(score.group(1)))

        uri = _URI.search(message)
        if uri:
            entry["uri"] = uri.group(1)

        ip = _CLIENT.search(message)
        if ip:
            entry["ip"] = ip.group(1)

        for tag in _TAG.findall(message):
            if tag in _ATTACK_NAMES:
                entry["families"].add(_ATTACK_NAMES[tag])

    if not requests:
        return {**empty, "enabled": True, "hours": hours}

    attacks: dict[str, int] = {}
    paths: dict[str, int] = {}
    addresses: dict[str, int] = {}
    would_block = 0
    worst = 0

    for entry in requests.values():
        worst = max(worst, entry["score"])
        if entry["score"] >= _BLOCKING_THRESHOLD:
            would_block += 1
        for family in entry["families"] or {"uncategorised"}:
            attacks[family] = attacks.get(family, 0) + 1
        # Stripping the query entirely makes every injection attempt look
        # like "/", because the payload is in the query and the path is
        # bare. A scanner probing /.env is described by its path; an
        # injection is described by its query. Keep a trimmed query only
        # when the path alone says nothing.
        raw = entry["uri"]
        path, _, query = raw.partition("?")
        path = path or "/"
        if path == "/" and query:
            label = f"/?{unquote(query)[:44]}"
        else:
            label = path[:60]
        paths[label] = paths.get(label, 0) + 1
        if entry["ip"]:
            addresses[entry["ip"]] = addresses.get(entry["ip"], 0) + 1

    countries: dict[str, int] = {}
    for ip, n in addresses.items():
        origin = resolve_origin(ip)
        name = origin.country_name or origin.country or "unknown"
        countries[name] = countries.get(name, 0) + n

    def top(counts: dict, limit: int = 6) -> list[dict]:
        return [{"label": k, "n": v} for k, v in
                sorted(counts.items(), key=lambda x: -x[1])[:limit]]

    return {
        "enabled": True, "hours": hours, "note": None,
        "detections": len(requests), "would_block": would_block,
        "sources": len(addresses), "worst_score": worst,
        "attacks": top(attacks), "countries": top(countries),
        "paths": top(paths),
    }


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
def corridor_search(body: CorridorSearchBody, request: Request,
                    response: Response) -> dict:
    """Run the Tree-of-Thought corridor search.

    Returns the corridors as GeoJSON with scores and prune reasons attached,
    the pairwise airspace overlap behind the dominance rule, the full search
    trace, and every note the generator and controller raised.
    """
    challenge = _require_human(request, response, body.turnstile_token)

    try:
        req = body.clamped().to_request()
        req.challenge = challenge
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
