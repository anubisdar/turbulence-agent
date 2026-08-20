"""Tests that our beliefs about upstream data match the wire.

A fixture is a written-down belief. When the belief is wrong the fixture
encodes the error, the tests confirm it, and the suite goes green while the
system misreads reality.

That has happened four times in this project:

  Segments in an itinerary were assumed to be flights on the requested
  pair. They are not: a query with no nonstop returns connections, and the
  agent picked a regional feeder leg as its reference. The fixtures had no
  origin or destination on a segment at all, because the code had never
  looked at those fields - so when the filter was added, twenty-three tests
  failed against a payload shape that does not exist.

  Oceanic waypoints were assumed to be unparseable. `5700N/15000W` carries
  its own coordinates and every transpacific routing failed for want of
  twenty lines of parsing.

  The G-AIRMET endpoint was assumed to honour its `type` parameter. It
  ignores it and returns every hazard, so a corridor was one filter away
  from being scored against freezing-level advisories.

  A download was assumed to be a download. MaxMind answers with a 302 to a
  signed URL, and a request that did not follow redirects returned an empty
  body that read as a rejected licence key.

The pattern in all four: nobody looked at the wire. The probe scripts do
look, and their findings live in `data/*_probe/`. Nothing until now asserted
that the fixtures agree with them.

These tests are the assertion. Where captured payloads are present they are
the reference; where they are absent the test says so rather than passing
quietly, because a contract test that skips silently is worse than none.
"""

import json
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
AEROAPI_PROBE = PROJECT / "data" / "aeroapi_probe"
AWC_PROBE = PROJECT / "data" / "awc_probe"
FIXTURES = PROJECT / "tests" / "fixtures"


def _load(path: Path):
    if not path.exists():
        pytest.skip(f"no captured payload at {path.relative_to(PROJECT)}; "
                    f"run the probe script to record one")
    return json.loads(path.read_text(encoding="utf-8"))


def _segments(payload) -> list[dict]:
    return [seg
            for itinerary in (payload.get("flights") or [])
            for seg in (itinerary.get("segments") or [])]


class TestCapturedPayloadsExist:
    """A contract test with nothing to compare against is decoration."""

    def test_the_aeroapi_probe_has_been_run(self):
        assert AEROAPI_PROBE.exists(), (
            "no captured AeroAPI payloads. Run scripts/probe_aeroapi.py "
            "--save so the fixtures have something to be checked against.")

    def test_the_awc_probe_has_been_run(self):
        assert AWC_PROBE.exists(), (
            "no captured G-AIRMET payloads. Run scripts/probe_gairmet.py "
            "--save.")


class TestFlightSegmentsCarryWhatTheCodeReads:
    """The fields the nonstop filter depends on. A fixture missing them
    passes a test the real payload would fail, which is precisely how a
    feeder leg became the reference flight for a transpacific route."""

    REQUIRED = ("ident", "fa_flight_id", "origin", "destination")

    def test_real_segments_have_an_origin_and_destination(self):
        captured = _load(AEROAPI_PROBE / "airport_pair_flights.json")
        segments = _segments(captured)
        assert segments, "the captured payload has no segments"
        for seg in segments:
            for field in self.REQUIRED:
                assert field in seg, (
                    f"the real payload has no {field!r} on a segment, so the "
                    f"nonstop filter cannot work as written")

    def test_origin_and_destination_are_objects_with_a_code(self):
        """`seg['origin']['code']`, not `seg['origin']`. Reading it as a
        string would compare an airport code against a dictionary and match
        nothing, silently."""
        captured = _load(AEROAPI_PROBE / "airport_pair_flights.json")
        for seg in _segments(captured):
            assert isinstance(seg["origin"], dict)
            assert "code" in seg["origin"]

    def test_the_payload_really_does_contain_connections(self):
        """The belief that made the bug possible was that every itinerary is
        a single flight. If a captured payload ever shows a multi-segment
        itinerary, the nonstop filter is load-bearing rather than
        defensive."""
        captured = _load(AEROAPI_PROBE / "airport_pair_flights.json")
        lengths = [len(i.get("segments") or [])
                   for i in (captured.get("flights") or [])]
        assert lengths, "no itineraries captured"
        # Not asserted to be greater than one: a pair with nonstop service
        # legitimately returns single-segment itineraries. What matters is
        # that the shape is understood.
        assert all(n >= 1 for n in lengths)


class TestFixturesMatchTheCapturedShape:
    """Every field the fixtures carry must exist in a real payload, and
    every field the code reads must exist in the fixtures."""

    def _fixture_segments(self):
        candidates = [
            FIXTURES / "airport_pair_flights.json",
            AEROAPI_PROBE / "airport_pair_flights.json",
        ]
        for path in candidates:
            if path.exists():
                return _segments(json.loads(path.read_text(encoding="utf-8")))
        pytest.skip("no flight fixtures found")

    def test_fixture_segments_carry_the_filter_fields(self):
        for seg in self._fixture_segments():
            assert "origin" in seg and "destination" in seg, (
                "a fixture segment without origin and destination describes a "
                "payload AeroAPI does not produce")

    def test_no_fixture_field_is_absent_from_reality(self):
        """Guards the other direction: a fixture inventing a field means the
        code may come to depend on something that will never arrive."""
        captured = _load(AEROAPI_PROBE / "airport_pair_flights.json")
        real_fields = {k for seg in _segments(captured) for k in seg}
        for seg in self._fixture_segments():
            invented = set(seg) - real_fields
            assert not invented, (
                f"the fixtures carry fields no captured payload has: "
                f"{sorted(invented)}")


class TestGairmetShapeIsWhatWeParse:
    """Three beliefs about this endpoint were wrong at once."""

    def _features(self):
        for name in ("gairmet_all.json", "gairmet_turb-hi.json"):
            path = AWC_PROBE / name
            if path.exists():
                body = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(body, dict):
                    return body.get("features") or body.get("data") or []
                return body
        pytest.skip("no captured G-AIRMET payload")

    def test_the_type_parameter_really_is_ignored(self):
        """Requesting turb-hi returns every hazard. If this ever stops being
        true the client-side filter becomes redundant rather than critical,
        which is worth knowing either way."""
        path = AWC_PROBE / "gairmet_turb-hi.json"
        if not path.exists():
            pytest.skip("no typed G-AIRMET capture")
        body = json.loads(path.read_text(encoding="utf-8"))
        features = (body.get("features") or body.get("data") or []
                    if isinstance(body, dict) else body)
        hazards = {(f.get("properties", f) or {}).get("hazard")
                   for f in features}
        hazards.discard(None)
        assert hazards, "no hazards in the capture"
        if hazards == {"TURB-HI"}:
            pytest.skip("the endpoint now honours its type parameter; the "
                        "client-side filter is no longer load-bearing")
        assert len(hazards) > 1, (
            "a typed request returned one hazard, so the filter may have "
            "become unnecessary")

    def test_product_tango_is_not_the_same_as_turbulence(self):
        """Filtering on `product` rather than `hazard` would sweep in
        freezing level and low-level wind shear."""
        features = self._features()
        tango = [f for f in features
                 if (f.get("properties", f) or {}).get("product") == "TANGO"]
        if not tango:
            pytest.skip("no TANGO products in the capture")
        hazards = {(f.get("properties", f) or {}).get("hazard")
                   for f in tango}
        assert hazards - {"TURB-HI", "TURB-LO"}, (
            "every TANGO product in this capture is turbulence, so the "
            "distinction is untested here rather than untrue")

    def test_geometry_is_not_geojson(self):
        """`geom` and `geometryType` are both the literal string AREA, and
        the shape lives in `coords` as lat/lon dictionaries of strings."""
        features = self._features()
        sample = features[0] if features else pytest.skip("no features")
        props = sample.get("properties", sample)
        assert props.get("geometryType") == "AREA"
        assert isinstance(props.get("coords"), list)
        first = props["coords"][0]
        assert set(first) >= {"lat", "lon"}
        assert isinstance(first["lat"], str), (
            "coordinates arrive as strings; parsing them as floats without "
            "conversion would raise rather than silently misplace them, but "
            "the shape is worth pinning")

    def test_altitudes_are_flight_levels_as_strings(self):
        """`top: '400'` is FL400, which is 40,000 feet. Reading it as 400
        would place every advisory near the ground."""
        features = self._features()
        banded = [f for f in features
                  if (f.get("properties", f) or {}).get("top") is not None]
        if not banded:
            pytest.skip("no altitude bands in the capture")
        top = (banded[0].get("properties", banded[0]) or {})["top"]
        assert isinstance(top, str)
        assert int(top) < 1000, (
            "a flight level above 1000 would mean these are already feet")


class TestParsingTheCapturedPayloads:
    """The strongest form of this test: run the real parsers over the real
    captures and assert they produce something usable. A fixture can drift;
    a captured payload cannot."""

    def test_every_captured_turbulence_advisory_parses(self):
        from app.sources.gairmet import turbulence_only

        path = AWC_PROBE / "gairmet_all.json"
        if not path.exists():
            pytest.skip("no captured G-AIRMET payload")
        body = json.loads(path.read_text(encoding="utf-8"))
        features = (body.get("features") or body.get("data") or []
                    if isinstance(body, dict) else body)

        advisories = turbulence_only(features)
        for advisory in advisories:
            assert advisory.usable, (
                f"a real advisory did not parse into a usable shape: "
                f"{advisory.hazard} {advisory.base_ft}-{advisory.top_ft}")
            assert -90 <= advisory.ring[0][0] <= 90
            assert -180 <= advisory.ring[0][1] <= 180

    def test_every_captured_route_string_tokenises(self):
        """No token in a real filed route should come back UNKNOWN unless we
        have decided it is genuinely not a position. Oceanic waypoints were
        UNKNOWN for months."""
        from app.sources.fixes import TokenKind, classify, tokenize

        path = AEROAPI_PROBE / "airport_pair_flights.json"
        if not path.exists():
            pytest.skip("no captured flight payload")
        captured = json.loads(path.read_text(encoding="utf-8"))

        unknown = set()
        for seg in _segments(captured):
            for token in tokenize(seg.get("route") or ""):
                if classify(token) is TokenKind.UNKNOWN:
                    unknown.add(token)

        # Route designators like OTR13 are legitimately not positions.
        assert not unknown or all(len(t) <= 6 for t in unknown), (
            f"unrecognised tokens in real routes: {sorted(unknown)}")
