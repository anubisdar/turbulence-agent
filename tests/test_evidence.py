"""Tests for turbulence evidence gathering.

No network and no event loop. The two sources are exercised separately and
together, with particular attention to the cases where one is silent.
"""

from datetime import datetime, timedelta, timezone

import pytest

from app.reasoning.critic import Corridor, Evidence, Geometry, Provenance, Severity, score
from app.reasoning.evidence import (
    bounding_box,
    coverage_fraction,
    gather_evidence,
    gather_forecast,
    gather_observed,
    to_critic_severity,
    worst,
)
from app.reasoning.geometry import build_corridor, great_circle, intersects_ring
from app.sources.gairmet import GairmetClient, GairmetFetchError, parse_advisory

KPIT = (40.4914167, -80.2326944)
KBOS = (42.3629, -71.0064)
NOW = datetime(2026, 8, 16, 12, 30, tzinfo=timezone.utc)
PATH = great_circle(KPIT, KBOS, 24)


@pytest.fixture
def shape():
    return build_corridor(PATH, altitude_min_ft=31300, altitude_max_ft=35000)


class FakeReport:
    def __init__(self, point, altitude_ft, severity, age_minutes=20):
        self.latitude, self.longitude = point
        self.altitude_ft = altitude_ft
        self.turbulence_severity = severity
        self.observation_time = NOW - timedelta(minutes=age_minutes)


def advisory(ring_points, severity="MOD", base="300", top="400"):
    return parse_advisory({
        "hazard": "TURB-HI", "severity": severity, "base": base, "top": top,
        "validTime": "2026-08-16T12:00:00.000Z", "expireTime": 1786892400,
        "coords": [{"lat": str(lat), "lon": str(lon)}
                   for lat, lon in ring_points],
    })


def wide_ring():
    """A polygon a degree either side of the route - about 60 nm, so it
    completely contains a 25 nm corridor and has no vertex inside it."""
    return [(PATH[8][0] + 1, PATH[8][1]), (PATH[16][0] + 1, PATH[16][1]),
            (PATH[16][0] - 1, PATH[16][1]), (PATH[8][0] - 1, PATH[8][1])]


class TestSeverityTranslation:
    def test_source_strings_map_to_the_critic_enum(self):
        assert to_critic_severity("moderate") is Severity.MODERATE
        assert to_critic_severity("none") is Severity.SMOOTH

    def test_absence_is_unresolved_not_smooth(self):
        """The difference the whole project turns on."""
        assert to_critic_severity(None) is Severity.UNRESOLVED
        assert to_critic_severity("") is Severity.UNRESOLVED
        assert to_critic_severity(None) is not Severity.SMOOTH

    def test_an_unreadable_value_is_unresolved(self):
        assert to_critic_severity("wobbly") is Severity.UNRESOLVED

    def test_worst_ignores_unresolved(self):
        assert worst(Severity.UNRESOLVED, Severity.LIGHT) is Severity.LIGHT
        assert worst(Severity.LIGHT, Severity.SEVERE) is Severity.SEVERE
        assert worst(Severity.UNRESOLVED) is Severity.UNRESOLVED


class TestObservedGathering:
    def test_reports_outside_the_corridor_are_excluded(self, shape):
        reports = [FakeReport(PATH[10], 34000, "light"),
                   FakeReport((25.76, -80.19), 34000, "extreme")]  # Miami
        reading, count, _, _, _, inside, considered = gather_observed(
            shape, reports, NOW)
        assert considered == 2
        assert inside == 1
        assert reading is Severity.LIGHT

    def test_reports_above_the_altitude_band_are_excluded(self, shape):
        """Laterally inside, vertically elsewhere. Not the same air."""
        reports = [FakeReport(PATH[12], 41000, "severe")]
        reading, count, _, _, _, inside, _ = gather_observed(
            shape, reports, NOW)
        assert inside == 0
        assert reading is Severity.UNRESOLVED

    def test_the_worst_report_wins(self, shape):
        reports = [FakeReport(PATH[6], 34000, "light"),
                   FakeReport(PATH[10], 34000, "light"),
                   FakeReport(PATH[14], 34000, "severe"),
                   FakeReport(PATH[18], 34000, "light")]
        reading, count, _, _, _, _, _ = gather_observed(shape, reports, NOW)
        assert reading is Severity.SEVERE
        assert count == 4

    def test_the_count_survives_so_an_outlier_is_visible(self, shape):
        """Worst-wins without a count would hide whether it was one report
        or four in agreement."""
        reports = [FakeReport(PATH[6], 34000, "light"),
                   FakeReport(PATH[14], 34000, "severe")]
        _, count, _, _, notes, _, _ = gather_observed(shape, reports, NOW)
        assert count == 2
        assert any("disagree" in n for n in notes)
        assert any("outlier" in n for n in notes)

    def test_agreeing_reports_are_reported_as_agreeing(self, shape):
        reports = [FakeReport(PATH[6], 34000, "light"),
                   FakeReport(PATH[14], 34000, "light")]
        _, _, _, _, notes, _, _ = gather_observed(shape, reports, NOW)
        assert any("all light" in n for n in notes)

    def test_fetched_but_none_inside_is_stated_as_absence(self, shape):
        reports = [FakeReport((25.76, -80.19), 34000, "extreme")]
        reading, _, _, _, notes, _, _ = gather_observed(shape, reports, NOW)
        assert reading is Severity.UNRESOLVED
        assert any("not a report of smooth air" in n for n in notes)

    def test_no_reports_at_all(self, shape):
        reading, count, age, cov, notes, _, _ = gather_observed(shape, [], NOW)
        assert reading is Severity.UNRESOLVED
        assert count == 0
        assert cov == 0.0
        assert age is None

    def test_fetching_nothing_is_still_said_out_loud(self, shape):
        """The one place absence could pass without comment. An empty fetch
        used to produce no note at all, which is the failure this project
        exists to avoid."""
        _, _, _, _, notes, _, _ = gather_observed(shape, [], NOW)
        assert notes, "an empty fetch must still be explained"
        assert any("not a report of smooth air" in n for n in notes)

    def test_mean_age_is_computed(self, shape):
        reports = [FakeReport(PATH[6], 34000, "light", age_minutes=10),
                   FakeReport(PATH[14], 34000, "light", age_minutes=30)]
        _, _, age, _, _, _, _ = gather_observed(shape, reports, NOW)
        assert age == pytest.approx(20.0, abs=0.1)


class TestCoverage:
    def test_one_report_does_not_cover_a_long_corridor(self, shape):
        """A single report on a 431 nm route is not full observation."""
        assert coverage_fraction(shape, [PATH[12]]) <= 0.2

    def test_more_spread_means_more_coverage(self, shape):
        few = coverage_fraction(shape, [PATH[12]])
        many = coverage_fraction(shape, [PATH[i] for i in range(2, 22, 2)])
        assert many > few

    def test_no_reports_is_zero(self, shape):
        assert coverage_fraction(shape, []) == 0.0

    def test_coverage_never_exceeds_one(self, shape):
        cov = coverage_fraction(shape, [PATH[i] for i in range(24)])
        assert 0.0 <= cov <= 1.0


class TestForecastGathering:
    def test_a_polygon_containing_the_corridor_matches(self, shape):
        """The bug this test exists for: a G-AIRMET is usually far larger
        than a 25 nm corridor, so a polygon covering the whole route has
        every vertex outside it. Vertex containment would miss it."""
        reading, count, _, _, _ = gather_forecast(shape, [advisory(wide_ring())])
        assert count == 1
        assert reading is Severity.MODERATE

    def test_vertex_containment_alone_would_have_missed_it(self, shape):
        a = advisory(wide_ring())
        assert not any(shape.contains(lat, lon) for lat, lon in a.ring)
        assert intersects_ring(shape, a.ring)

    def test_a_polygon_at_a_different_altitude_does_not_match(self, shape):
        low = advisory(wide_ring(), base="SFC", top="180")
        reading, count, notes, _, _ = gather_forecast(shape, [low])
        assert count == 0
        assert reading is Severity.UNRESOLVED
        assert any("not a forecast of smooth air" in n for n in notes)

    def test_a_distant_polygon_does_not_match(self, shape):
        far = advisory([(25.0, -80.0), (26.0, -80.0), (26.0, -81.0),
                        (25.0, -81.0)])
        _, count, _, _, _ = gather_forecast(shape, [far])
        assert count == 0

    def test_the_worst_overlapping_forecast_wins(self, shape):
        mild = advisory(wide_ring(), severity="LGT")
        harsh = advisory(wide_ring(), severity="SEV")
        reading, count, _, _, _ = gather_forecast(shape, [mild, harsh])
        assert reading is Severity.SEVERE
        assert count == 2

    def test_no_forecasts_at_all(self, shape):
        reading, count, _, _, _ = gather_forecast(shape, [])
        assert reading is Severity.UNRESOLVED
        assert count == 0


class TestSourcesHeldApart:
    """Observed and forecast keep their own readings so a disagreement
    reaches the critic intact."""

    def _gather(self, shape, reports=None, advisories=None):
        client = None
        if advisories is not None:
            client = GairmetClient(
                transport=lambda p, q: (200, [
                    {"hazard": "TURB-HI", "severity": sev, "base": "300",
                     "top": "400", "validTime": "2026-08-16T12:00:00.000Z",
                     "expireTime": 1786892400,
                     "coords": [{"lat": str(lat), "lon": str(lon)}
                                for lat, lon in wide_ring()]}
                    for sev in advisories], ""))
        return gather_evidence(
            shape,
            fetch_pireps=(lambda b, h: reports) if reports is not None else None,
            gairmet_client=client, when=NOW)

    def test_both_readings_are_kept(self, shape):
        res = self._gather(shape,
                           reports=[FakeReport(PATH[10], 34000, "light")],
                           advisories=["MOD"])
        e = res.evidence
        assert e.observed_reading is Severity.LIGHT
        assert e.forecast_reading is Severity.MODERATE
        assert e.sources_disagree

    def test_the_combined_reading_is_the_worse(self, shape):
        res = self._gather(shape,
                           reports=[FakeReport(PATH[10], 34000, "light")],
                           advisories=["MOD"])
        assert res.evidence.reading is Severity.MODERATE

    def test_the_disagreement_is_stated_not_averaged(self, shape):
        res = self._gather(shape,
                           reports=[FakeReport(PATH[10], 34000, "light")],
                           advisories=["MOD"])
        assert any("averaging them would produce a number neither source "
                   "supports" in n for n in res.notes)

    def test_one_silent_source_is_a_gap_not_a_disagreement(self, shape):
        res = self._gather(shape,
                           reports=[FakeReport(PATH[10], 34000, "light")],
                           advisories=[])
        e = res.evidence
        assert e.observed_reading is Severity.LIGHT
        assert e.forecast_reading is Severity.UNRESOLVED
        assert not e.sources_disagree

    def test_agreeing_sources_are_not_contested(self, shape):
        res = self._gather(shape,
                           reports=[FakeReport(PATH[10], 34000, "moderate")],
                           advisories=["MOD"])
        assert not res.evidence.sources_disagree


class TestCriticComputesAgreement:
    """Agreement is derived from the two readings, not handed over."""

    def _corridor(self, observed, forecast):
        return Corridor("c", Provenance.ACTUAL_TRACK, Geometry(478, 431, 15),
                        Evidence(observed_reading=observed,
                                 forecast_reading=forecast,
                                 coverage_fraction=0.5))

    def test_identical_readings_score_full_agreement(self):
        c = self._corridor(Severity.MODERATE, Severity.MODERATE)
        assert score(c).components["agreement"] == 1.0

    def test_one_level_apart_scores_high_but_not_full(self):
        c = self._corridor(Severity.LIGHT, Severity.MODERATE)
        assert score(c).components["agreement"] == pytest.approx(0.75)

    def test_far_apart_readings_score_low(self):
        c = self._corridor(Severity.SMOOTH, Severity.EXTREME)
        assert score(c).components["agreement"] == 0.0

    def test_a_silent_source_scores_neutral_not_zero(self):
        """Punishing a corridor for what nobody reported on it would make
        absence behave like disagreement."""
        c = self._corridor(Severity.LIGHT, Severity.UNRESOLVED)
        assert score(c).components["agreement"] == 0.5


class TestFailureIsNotSmooth:
    def test_no_evidence_from_either_source_stays_unresolved(self, shape):
        res = gather_evidence(shape, fetch_pireps=lambda b, h: [],
                              gairmet_client=None, when=NOW)
        assert res.evidence.reading is Severity.UNRESOLVED
        assert any("Unresolved is not smooth" in n for n in res.notes)

    def test_a_pirep_fetch_failure_is_reported_not_swallowed(self, shape):
        def boom(bbox, hours):
            raise ConnectionError("network down")
        res = gather_evidence(shape, fetch_pireps=boom, when=NOW)
        assert res.evidence.observed_reading is Severity.UNRESOLVED
        assert any("unknown, not clear" in n for n in res.notes)

    def test_a_forecast_fetch_failure_is_reported_not_swallowed(self, shape):
        client = GairmetClient(transport=lambda p, q: (500, None, "boom"))
        res = gather_evidence(shape, gairmet_client=client, when=NOW)
        assert res.evidence.forecast_reading is Severity.UNRESOLVED
        assert any("unknown, not clear" in n for n in res.notes)

    def test_one_source_failing_does_not_block_the_other(self, shape):
        def boom(bbox, hours):
            raise ConnectionError("down")
        client = GairmetClient(transport=lambda p, q: (200, [{
            "hazard": "TURB-HI", "severity": "MOD", "base": "300",
            "top": "400", "validTime": "2026-08-16T12:00:00.000Z",
            "expireTime": 1786892400,
            "coords": [{"lat": str(lat), "lon": str(lon)}
                       for lat, lon in wide_ring()]}], ""))
        res = gather_evidence(shape, fetch_pireps=boom,
                              gairmet_client=client, when=NOW)
        assert res.evidence.forecast_reading is Severity.MODERATE
        assert res.evidence.observed_reading is Severity.UNRESOLVED


class TestBoundingBox:
    """AWC wants (min_lat, min_lon, max_lat, max_lon). Getting the order
    wrong is silent: a Pittsburgh longitude of -80 is a legal latitude, so
    the request succeeds and searches the South Atlantic, returning zero
    reports that look exactly like a calm afternoon."""

    def test_the_box_contains_the_route(self, shape):
        min_lat, min_lon, max_lat, max_lon = bounding_box(shape)
        for lat, lon in PATH:
            assert min_lat <= lat <= max_lat
            assert min_lon <= lon <= max_lon

    def test_latitude_comes_first(self, shape):
        """The check that would have caught the original defect."""
        min_lat, min_lon, max_lat, max_lon = bounding_box(shape)
        assert -90 <= min_lat <= 90 and -90 <= max_lat <= 90
        assert 35 < min_lat < 45, "first value should be a KPIT-KBOS latitude"
        assert -90 < min_lon < -60, "second value should be a longitude"

    def test_the_box_lands_over_the_northeast_united_states(self, shape):
        """A transposed box would land in the South Atlantic and return
        nothing, which is indistinguishable from calm air."""
        min_lat, min_lon, max_lat, max_lon = bounding_box(shape)
        assert min_lat > 0, "northern hemisphere"
        assert max_lon < 0, "western hemisphere"

    def test_the_box_is_padded(self, shape):
        min_lat, min_lon, max_lat, max_lon = bounding_box(shape, pad_deg=1.0)
        lats = [p[0] for p in PATH]
        assert min_lat < min(lats)
        assert max_lat > max(lats)


class TestPlainLanguageAbsence:
    """"Unresolved" is accurate and useless on its own. A passenger needs to
    know whether nobody looked or people looked and found nothing."""

    @pytest.fixture
    def short_route(self):
        return build_corridor(great_circle(KPIT, KBOS, 24))

    @pytest.fixture
    def long_route(self):
        return build_corridor(great_circle((38.9445, -77.4558),
                                           (33.9425, -118.4081), 24))

    def test_a_short_route_says_so(self, short_route):
        from app.reasoning.evidence import explain_absence
        text = explain_absence(short_route, 21, 0, 1, 0)
        assert "short hop" in text
        assert "431" in text or "43" in text
        assert "common on routes this length" in text

    def test_a_long_route_does_not_claim_it_is_short(self, long_route):
        from app.reasoning.evidence import explain_absence
        text = explain_absence(long_route, 40, 0, 3, 0)
        assert "short hop" not in text

    def test_nothing_fetched_is_distinguished_from_nothing_found(
            self, short_route):
        from app.reasoning.evidence import explain_absence
        nothing_fetched = explain_absence(short_route, 0, 0, 0, 0)
        nothing_found = explain_absence(short_route, 21, 0, 1, 0)
        assert nothing_fetched != nothing_found
        assert "were not available" in nothing_fetched or \
               "Neither pilot reports nor forecasts were available" in nothing_fetched

    def test_one_source_speaking_is_described_as_such(self, long_route):
        from app.reasoning.evidence import explain_absence
        text = explain_absence(long_route, 12, 0, 3, 1)
        assert "only thing to go on" in text

    def test_no_explanation_ever_implies_calm(self, short_route):
        from app.reasoning.evidence import explain_absence
        for args in [(0, 0, 0, 0), (21, 0, 1, 0), (12, 0, 3, 1)]:
            text = explain_absence(short_route, *args).lower()
            assert "smooth" not in text
            assert "calm" not in text
            assert "clear" not in text

    def test_the_summary_reaches_the_result(self, shape):
        res = gather_evidence(shape, fetch_pireps=lambda b, h: [],
                              gairmet_client=None, when=NOW)
        assert res.summary
        assert res.evidence.reading is Severity.UNRESOLVED

    def test_a_resolved_reading_explains_rather_than_excuses(self, shape):
        from app.sources.gairmet import GairmetClient
        client = GairmetClient(transport=lambda p, q: (200, [{
            "hazard": "TURB-HI", "severity": "MOD", "base": "300",
            "top": "400", "validTime": "2026-08-16T12:00:00.000Z",
            "expireTime": 1786892400,
            "coords": [{"lat": str(lat), "lon": str(lon)}
                       for lat, lon in wide_ring()]}], ""))
        res = gather_evidence(shape, gairmet_client=client, when=NOW)
        assert res.evidence.reading is not Severity.UNRESOLVED
        # A resolved reading gets an explanation of what it means and where
        # it came from, not an excuse for having nothing.
        assert res.summary
        assert "Nothing is known" not in res.summary
        assert "short hop" not in res.summary


class TestRejectionReasonsAreDistinguished:
    """Somewhere else entirely and directly overhead at 3,000 feet are
    different facts, and only one is worth worrying about."""

    def test_low_level_reports_along_the_route_are_named_as_such(self, shape):
        reports = [FakeReport(PATH[i], 3000, "light") for i in (4, 8, 12, 16)]
        _, _, _, _, notes, inside, _ = gather_observed(shape, reports, NOW)
        assert inside == 0
        text = " ".join(notes)
        assert "along this route" in text
        assert "under the cruise altitude" in text

    def test_distant_reports_are_counted_separately(self, shape):
        reports = [FakeReport(PATH[8], 3000, "light"),
                   FakeReport((25.76, -80.19), 34000, "severe")]
        _, _, _, _, notes, _, _ = gather_observed(shape, reports, NOW)
        text = " ".join(notes)
        assert "surrounding airspace" in text

    def test_the_closing_note_never_implies_smooth(self, shape):
        reports = [FakeReport(PATH[8], 3000, "light")]
        _, _, _, _, notes, _, _ = gather_observed(shape, reports, NOW)
        assert any("not a report of smooth air" in n for n in notes)


class TestReadingExplanations:
    """A resolved reading needs plain language too: what it means, where it
    came from, and how much it is worth."""

    def _e(self, **kw):
        from app.reasoning.critic import Evidence
        return Evidence(**kw)

    @pytest.fixture
    def shape_(self):
        return build_corridor(great_circle(KPIT, KBOS, 24))

    def test_every_level_has_a_sensation_description(self):
        from app.reasoning.evidence import SENSATION
        for level in (Severity.SMOOTH, Severity.LIGHT, Severity.MODERATE,
                      Severity.SEVERE, Severity.EXTREME):
            assert SENSATION.get(level), level

    def test_the_descriptions_match_the_faa_criteria(self):
        """Paraphrased from AIM Table 7-1-11 and AC 120-88A. These are the
        published effects on occupants, not invented ones."""
        from app.reasoning.evidence import SENSATION
        assert "slight strain" in SENSATION[Severity.LIGHT]
        assert "definite strain" in SENSATION[Severity.MODERATE]
        assert "positive control" in SENSATION[Severity.MODERATE]
        assert "violently" in SENSATION[Severity.SEVERE]
        assert "momentarily out of control" in SENSATION[Severity.SEVERE]
        assert "structural damage" in SENSATION[Severity.EXTREME]

    def test_no_description_promises_comfort(self):
        """Light is the level most likely to be read as reassurance. None of
        these may tell a passenger the flight will be fine."""
        from app.reasoning.evidence import SENSATION
        banned = ("you'll be fine", "nothing to worry", "perfectly safe",
                  "don't worry", "no need to worry", "harmless")
        for level, text in SENSATION.items():
            low = text.lower()
            for phrase in banned:
                assert phrase not in low, f"{level}: {phrase}"

    def test_a_forecast_only_reading_says_it_is_a_forecast(self, shape_):
        from app.reasoning.evidence import explain_reading
        text = explain_reading(self._e(
            reading=Severity.MODERATE, forecast_reading=Severity.MODERATE,
            forecast_count=1), shape_)
        assert "forecast" in text
        assert "rather than what anyone has felt" in text

    def test_a_pilot_only_reading_says_the_forecast_is_silent(self, shape_):
        from app.reasoning.evidence import explain_reading
        text = explain_reading(self._e(
            reading=Severity.LIGHT, observed_reading=Severity.LIGHT,
            observed_count=2, coverage_fraction=0.2,
            mean_age_minutes=27), shape_)
        assert "No forecast covers this route" in text
        assert "27 minutes old" in text

    def test_thin_coverage_is_stated(self, shape_):
        from app.reasoning.evidence import explain_reading
        text = explain_reading(self._e(
            reading=Severity.LIGHT, observed_reading=Severity.LIGHT,
            observed_count=2, coverage_fraction=0.2), shape_)
        assert "20%" in text
        assert "unobserved" in text

    def test_a_disagreement_is_explained_not_hidden(self, shape_):
        from app.reasoning.evidence import explain_reading
        text = explain_reading(self._e(
            reading=Severity.MODERATE, observed_reading=Severity.LIGHT,
            observed_count=3, forecast_reading=Severity.MODERATE,
            forecast_count=1), shape_)
        assert "disagree" in text
        assert "average would match neither" in text

    def test_a_lone_severe_report_is_qualified(self, shape_):
        """One aircraft hitting severe air is worth knowing and is not the
        same as a pattern."""
        from app.reasoning.evidence import explain_reading
        text = explain_reading(self._e(
            reading=Severity.SEVERE, observed_reading=Severity.SEVERE,
            observed_count=1, coverage_fraction=0.1), shape_)
        assert "single report" in text
        assert "does not mean every aircraft will" in text

    def test_counts_read_naturally(self, shape_):
        from app.reasoning.evidence import explain_reading
        one = explain_reading(self._e(
            reading=Severity.LIGHT, observed_reading=Severity.LIGHT,
            observed_count=1), shape_)
        many = explain_reading(self._e(
            reading=Severity.LIGHT, observed_reading=Severity.LIGHT,
            observed_count=3), shape_)
        assert "1 pilot report " in one and "(s)" not in one
        assert "3 pilot reports" in many

    def test_an_unresolved_reading_gets_no_sensation_text(self, shape_):
        from app.reasoning.evidence import explain_reading
        assert explain_reading(self._e(reading=Severity.UNRESOLVED),
                               shape_) == ""

    def test_a_resolved_gather_carries_the_explanation(self, shape):
        from app.sources.gairmet import GairmetClient
        client = GairmetClient(transport=lambda p, q: (200, [{
            "hazard": "TURB-HI", "severity": "MOD", "base": "300",
            "top": "400", "validTime": "2026-08-16T12:00:00.000Z",
            "expireTime": 1786892400,
            "coords": [{"lat": str(lat), "lon": str(lon)}
                       for lat, lon in wide_ring()]}], ""))
        res = gather_evidence(shape, gairmet_client=client, when=NOW)
        assert res.summary
        assert "Moderate turbulence" in res.summary


class TestBoundingBoxAcrossTheDateLine:
    """A box from -122 to +139 spans most of the planet rather than the
    narrow band a Seattle to Tokyo route occupies."""

    @pytest.fixture
    def transpacific(self):
        return build_corridor(great_circle((47.4502, -122.3088),
                                           (35.5533, 139.7811), 24))

    def test_the_box_is_legal(self, transpacific):
        min_lat, min_lon, max_lat, max_lon = bounding_box(transpacific)
        assert -90 <= min_lat <= 90 and -90 <= max_lat <= 90
        assert -180 <= min_lon <= 180 and -180 <= max_lon <= 180

    def test_latitude_still_comes_first(self, transpacific):
        min_lat, min_lon, max_lat, max_lon = bounding_box(transpacific)
        assert 30 < min_lat < 60
        assert min_lat < max_lat

    def test_a_domestic_box_is_unchanged(self, shape):
        min_lat, min_lon, max_lat, max_lon = bounding_box(shape)
        assert 38 < min_lat < 42
        assert -83 < min_lon < -79
