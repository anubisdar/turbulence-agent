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

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

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

app = FastAPI(
    title=API_TITLE,
    version="0.1.0",
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


def _db_path() -> str:
    return os.environ.get("TURBULENCE_DB", DEFAULT_DB)


def _api_key() -> str | None:
    return os.environ.get("AEROAPI_KEY")


# ------------------------------------------------------------------ models


class CorridorSearchBody(BaseModel):
    origin: str = Field("KPIT", min_length=3, max_length=4,
                        description="ICAO code, e.g. KPIT")
    dest: str = Field("KBOS", min_length=3, max_length=4)
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


@app.get("/api/health", tags=["meta"])
def health() -> dict:
    """What this instance can actually do right now."""
    db = Path(_db_path())
    return {
        "status": "ok",
        "database": str(db),
        "database_present": db.exists(),
        "aeroapi_key_configured": bool(_api_key()),
        "fixtures_available": Path("data/aeroapi_probe").exists(),
        "static_page_available": (STATIC_DIR / "index.html").exists(),
    }


@app.post("/api/search/corridors", tags=["corridors"])
def corridor_search(body: CorridorSearchBody) -> dict:
    """Run the Tree-of-Thought corridor search.

    Returns the corridors as GeoJSON with scores and prune reasons attached,
    the pairwise airspace overlap behind the dominance rule, the full search
    trace, and every note the generator and controller raised.
    """
    try:
        return run_corridor_search(body.to_request(), _api_key(), _db_path())
    except ServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001 - surface the cause, do not mask it
        raise HTTPException(status_code=500,
                            detail=f"{type(e).__name__}: {e}")


@app.get("/api/search/reputation", tags=["reputation"])
def reputation_search(
    aircraft_type: str = Query(..., description="e.g. '737 MAX 8', '737-8'"),
    query: str = Query("safety incidents and accidents"),
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
        raise HTTPException(status_code=500,
                            detail=f"{type(e).__name__}: {e}")


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
