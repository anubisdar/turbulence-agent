# Turbulence-Aware Flight Ranking Agent

## What this is
An agent that takes a trip (origin, destination, date) and returns flight options
ranked by expected turbulence, plus a separate aircraft safety reputation read.
User is an anxious but technical flier.

## Non-negotiable design rules
1. **The LLM never does math.** Route geometry, polygon intersection, and turbulence
   scoring are pure Python. The LLM decides WHAT to fetch and EXPLAINS results.
   If you find yourself prompting the model to calculate something, stop.
2. **Scoring is a pure function.** Same inputs -> same score, always. No I/O, no
   randomness, no LLM calls inside `app/scoring/`.
3. **No severity defaults.** Missing turbulence data NEVER becomes an assumed value.
   Absence of data lowers confidence and widens the search. It never becomes "light."
4. **Turbulence data is never cached beyond its natural cycle.** Every cached weather
   item carries an explicit `expires_at`. Reference data (aircraft types, ADs) may be
   cached for weeks. Weather may not.
5. **Provenance travels with data.** Every retrieved item carries its source and
   timestamp through to the output. Formal sources (NTSB, FAA, AWC) and informal
   sources (forums, news) are never blended.
6. **Conflicts are surfaced, not averaged.** If forecast says moderate and PIREPs say
   light, both survive to the output with timestamps.

## Stack
- Python 3.12 on Ubuntu
- LangGraph 1.2+ for the agent loop (StateGraph, not create_react_agent �
  we need custom conditional edges for the widen-and-retry behavior)
- FastAPI + server-sent events for the UI
- SQLite for memory
- pyproj + shapely for geometry
- xarray + cfgrib for GRIB2
- Claude Sonnet 5 (`claude-sonnet-5`) for orchestration
- Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) for high-volume text extraction

## Data sources
- AWC API (aviationweather.gov/api/data/...) � PIREPs, G-AIRMET, SIGMET.
  No API key. GeoJSON output. ~100 req/min limit. Custom User-Agent required.
- NOAA GTG Nowcast � NOMADS, GRIB2, 15-min cycle, EDPARM field = EDR values
- FlightAware AeroAPI Personal tier � schedules + aircraft type.
  10 result sets/MINUTE. Cache aggressively.

## Code conventions
- Type hints everywhere. Pydantic models for anything crossing a module boundary.
- Every external call goes through a function that can be monkeypatched in tests.
- Tests use recorded fixtures, never live API calls.
- No bare `except:`. Failed tool calls return a typed Failure, they don't raise
  into the agent loop � the loop needs to SEE the failure to react to it.

## Testing
`pytest tests/` must pass before any commit.
