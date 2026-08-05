# Recorded fixtures

Real responses captured from the live AWC API. Tests replay these; they never hit
the network (see the `_block_network` autouse fixture in `tests/conftest.py`).

| File | Request | Recorded |
|---|---|---|
| `pirep_geojson_ne_us.json` | `GET /api/data/pirep?format=geojson&age=6&bbox=38,-82,45,-68` | 2026-08-04T22:12Z |
| `pirep_geojson_severe.json` | the `SEV` features of `GET /api/data/pirep?format=geojson&age=6&bbox=24,-125,50,-66` | 2026-08-04T22:14Z |

`pirep_geojson_severe.json` is a real recording with the non-severe features removed —
severe PIREPs are rare, so a bbox small enough to keep the fixture readable rarely
contains one. Property values are untouched.

To re-record:

    curl -A "$AWC_USER_AGENT" \
      "https://aviationweather.gov/api/data/pirep?format=geojson&age=6&bbox=38,-82,45,-68"
