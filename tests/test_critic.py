"""Tests for the corridor critic.

The classes named after design rules are the ones that matter - each asserts
a commitment made in the checkpoint write-up actually holds in code.
"""

import pytest

from app.reasoning.critic import (
    DOMINANCE_OVERLAP,
    TIE_EPSILON,
    WEIGHTS,
    Corridor,
    Decision,
    Evidence,
    Geometry,
    Provenance,
    Severity,
    evaluate,
    final_reading,
    score,
    worse_reading,
)


def make(cid, prov=Provenance.FILED_ROUTE, length=520.0, gc=500.0,
         dogleg=10.0, coverage=None, age=None, agreement=None,
         reading=Severity.UNRESOLVED, endpoints=True, alt_ok=True):
    return Corridor(
        id=cid,
        provenance=prov,
        geometry=Geometry(length_nm=length, great_circle_nm=gc,
                          max_dogleg_deg=dogleg,
                          endpoints_match_airports=endpoints,
                          altitude_profile_valid=alt_ok),
        evidence=Evidence(coverage_fraction=coverage, mean_age_minutes=age,
                          agreement=agreement, reading=reading),
    )


class TestWeights:
    def test_weights_sum_to_one(self):
        assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

    def test_provenance_is_the_heaviest_criterion(self):
        assert WEIGHTS["provenance"] == max(WEIGHTS.values())

    def test_coverage_is_the_lightest(self):
        assert WEIGHTS["coverage"] == min(WEIGHTS.values())


class TestProvenanceOrdering:
    def test_better_provenance_scores_higher_all_else_equal(self):
        order = [Provenance.ACTUAL_TRACK, Provenance.FILED_ROUTE,
                 Provenance.PUBLISHED_AIRWAY, Provenance.GREAT_CIRCLE]
        totals = [score(make(p.value, prov=p)).total for p in order]
        assert totals == sorted(totals, reverse=True)

    def test_actual_track_beats_great_circle_with_better_coverage(self):
        """Provenance is heavy enough to outweigh a coverage advantage."""
        track = make("track", prov=Provenance.ACTUAL_TRACK)
        circle = make("circle", prov=Provenance.GREAT_CIRCLE,
                      coverage=1.0, age=5.0)
        assert score(track).total > score(circle).total


class TestCoverageNeverPrunes:
    """The failure mode from the write-up: pruning the true path because
    nobody filed a report on it."""

    def test_zero_coverage_is_kept(self):
        s = score(make("empty", coverage=0.0))
        assert s.decision is Decision.KEEP

    def test_absent_coverage_is_kept(self):
        s = score(make("none", coverage=None))
        assert s.decision is Decision.KEEP

    def test_no_prune_reason_ever_mentions_coverage(self):
        cands = [make("a", coverage=0.0), make("b", coverage=1.0, age=10.0),
                 make("c", coverage=None)]
        res = evaluate(cands, beam_width=1)
        for s in res.pruned:
            assert "coverage" not in s.reason.lower()

    def test_a_zero_coverage_branch_can_still_win(self):
        """Better provenance with no data beats worse provenance with data."""
        cands = [
            make("track", prov=Provenance.ACTUAL_TRACK, coverage=0.0),
            make("circle", prov=Provenance.GREAT_CIRCLE, coverage=1.0, age=5.0),
        ]
        res = evaluate(cands, beam_width=1)
        assert res.kept[0].corridor_id == "track"


class TestGeometryIsTheOnlySoloEliminator:
    @pytest.mark.parametrize("kwargs", [
        {"endpoints": False},
        {"alt_ok": False},
        {"dogleg": 120.0},
        {"length": 400.0, "gc": 500.0},     # shorter than great circle
        {"length": 900.0, "gc": 500.0},     # implausibly long
    ])
    def test_implausible_geometry_prunes(self, kwargs):
        s = score(make("bad", **kwargs))
        assert s.decision is Decision.PRUNE_IMPLAUSIBLE

    def test_plausible_geometry_survives(self):
        assert score(make("ok")).decision is Decision.KEEP

    def test_great_circle_length_scores_best_on_geometry(self):
        tight = score(make("tight", length=500.0, gc=500.0, dogleg=0.0))
        loose = score(make("loose", length=760.0, gc=500.0, dogleg=0.0))
        assert tight.components["geometry"] > loose.components["geometry"]


class TestAbsentEvidenceIsNeutral:
    def test_missing_agreement_scores_neutral_not_zero(self):
        assert score(make("x", agreement=None)).components["agreement"] == 0.5

    def test_actual_disagreement_scores_below_neutral(self):
        s = score(make("x", agreement=0.1))
        assert s.components["agreement"] < 0.5

    def test_disagreement_lowers_score_but_does_not_prune(self):
        s = score(make("x", agreement=0.0))
        assert s.decision is Decision.KEEP


class TestRecency:
    def test_fresh_observation_beats_stale(self):
        fresh = score(make("f", coverage=1.0, age=10.0))
        stale = score(make("s", coverage=1.0, age=170.0))
        assert fresh.components["coverage"] > stale.components["coverage"]

    def test_observation_past_the_window_contributes_nothing(self):
        assert score(make("old", coverage=1.0, age=300.0)).components["coverage"] == 0.0


class TestBeamWidth:
    def test_only_beam_width_survivors_are_kept(self):
        cands = [make(f"c{i}", prov=p) for i, p in enumerate(Provenance)]
        res = evaluate(cands, beam_width=2)
        assert len(res.kept) == 2

    def test_the_rest_are_pruned_by_beam_not_silently_dropped(self):
        cands = [make(f"c{i}", prov=p) for i, p in enumerate(Provenance)]
        res = evaluate(cands, beam_width=2)
        beam_pruned = [s for s in res.pruned
                       if s.decision is Decision.PRUNE_BEAM]
        assert len(beam_pruned) == 2

    def test_every_candidate_is_accounted_for(self):
        cands = [make(f"c{i}", prov=p) for i, p in enumerate(Provenance)]
        res = evaluate(cands, beam_width=2)
        assert len(res.all_scores) == len(cands)


class TestDominancePruning:
    def test_worse_provenance_over_the_same_airspace_is_pruned(self):
        a = make("circle", prov=Provenance.GREAT_CIRCLE)
        b = make("track", prov=Provenance.ACTUAL_TRACK)
        res = evaluate([a, b], beam_width=4,
                       overlap_fn=lambda x, y: 0.95)
        pruned = {s.corridor_id: s for s in res.pruned}
        assert "circle" in pruned
        assert pruned["circle"].decision is Decision.PRUNE_DOMINATED

    def test_distinct_airspace_is_not_dominated(self):
        a = make("circle", prov=Provenance.GREAT_CIRCLE)
        b = make("track", prov=Provenance.ACTUAL_TRACK)
        res = evaluate([a, b], beam_width=4, overlap_fn=lambda x, y: 0.2)
        assert not any(s.decision is Decision.PRUNE_DOMINATED
                       for s in res.pruned)

    def test_better_provenance_is_never_dominated_by_worse(self):
        a = make("track", prov=Provenance.ACTUAL_TRACK)
        b = make("circle", prov=Provenance.GREAT_CIRCLE)
        res = evaluate([a, b], beam_width=4, overlap_fn=lambda x, y: 1.0)
        assert res.kept[0].corridor_id == "track"

    def test_dominance_is_skipped_without_a_geometry_function(self):
        a = make("circle", prov=Provenance.GREAT_CIRCLE)
        b = make("track", prov=Provenance.ACTUAL_TRACK)
        res = evaluate([a, b], beam_width=4)
        assert not any(s.decision is Decision.PRUNE_DOMINATED
                       for s in res.pruned)


class TestTiePolicy:
    def _tied_pair(self, reading_a, reading_b):
        return [
            make("a", prov=Provenance.FILED_ROUTE, reading=reading_a),
            make("b", prov=Provenance.FILED_ROUTE, reading=reading_b),
        ]

    def test_agreeing_tie_is_noted_as_corroboration(self):
        res = evaluate(self._tied_pair(Severity.LIGHT, Severity.LIGHT),
                       beam_width=2)
        assert not res.contested
        assert any("corroborat" in n for n in res.notes)

    def test_disagreeing_tie_is_marked_contested(self):
        res = evaluate(self._tied_pair(Severity.LIGHT, Severity.MODERATE),
                       beam_width=2)
        assert ("a", "b") in res.contested

    def test_disagreeing_tie_keeps_both_branches(self):
        cands = self._tied_pair(Severity.LIGHT, Severity.MODERATE)
        res = evaluate(cands, beam_width=2)
        assert {s.corridor_id for s in res.kept} == {"a", "b"}

    def test_ranking_uses_the_worse_reading(self):
        cands = self._tied_pair(Severity.LIGHT, Severity.MODERATE)
        res = evaluate(cands, beam_width=2)
        assert final_reading(res, cands) is Severity.MODERATE

    def test_both_readings_are_visible_in_the_notes(self):
        res = evaluate(self._tied_pair(Severity.LIGHT, Severity.MODERATE),
                       beam_width=2)
        note = " ".join(res.notes)
        assert "light" in note and "moderate" in note

    def test_scores_further_apart_than_epsilon_are_not_a_tie(self):
        cands = [
            make("a", prov=Provenance.ACTUAL_TRACK, reading=Severity.LIGHT),
            make("b", prov=Provenance.GREAT_CIRCLE, reading=Severity.MODERATE),
        ]
        res = evaluate(cands, beam_width=2)
        assert not res.contested


class TestWorseReading:
    def test_more_severe_wins(self):
        assert worse_reading(Severity.LIGHT, Severity.SEVERE) is Severity.SEVERE

    def test_unresolved_never_softens_a_real_reading(self):
        assert worse_reading(Severity.UNRESOLVED, Severity.MODERATE) is Severity.MODERATE
        assert worse_reading(Severity.MODERATE, Severity.UNRESOLVED) is Severity.MODERATE

    def test_unresolved_is_not_treated_as_smooth(self):
        assert worse_reading(Severity.UNRESOLVED, Severity.SMOOTH) is Severity.SMOOTH
        assert worse_reading(Severity.UNRESOLVED, Severity.UNRESOLVED) is Severity.UNRESOLVED


class TestDeterminism:
    def test_identical_input_gives_identical_output(self):
        cands = [make(f"c{i}", prov=p) for i, p in enumerate(Provenance)]
        a = evaluate(cands, beam_width=2)
        b = evaluate(cands, beam_width=2)
        assert [s.corridor_id for s in a.kept] == [s.corridor_id for s in b.kept]
        assert [s.total for s in a.kept] == [s.total for s in b.kept]

    def test_input_order_does_not_change_the_result(self):
        cands = [make(f"c{i}", prov=p) for i, p in enumerate(Provenance)]
        a = evaluate(cands, beam_width=2)
        b = evaluate(list(reversed(cands)), beam_width=2)
        assert [s.corridor_id for s in a.kept] == [s.corridor_id for s in b.kept]

    def test_exact_ties_break_on_provenance_then_freshness(self):
        cands = [
            make("later", prov=Provenance.FILED_ROUTE, coverage=0.5, age=90.0),
            make("fresher", prov=Provenance.FILED_ROUTE, coverage=0.5, age=90.0),
        ]
        res = evaluate(cands, beam_width=1)
        assert res.kept[0].corridor_id == "fresher"


class TestNoSurvivors:
    def test_total_failure_is_reported_as_failure_not_smooth_air(self):
        cands = [make("bad1", endpoints=False), make("bad2", dogleg=170.0)]
        res = evaluate(cands, beam_width=2)
        assert res.kept == []
        assert any("not a finding of smooth air" in n for n in res.notes)

    def test_final_reading_with_no_survivors_is_unresolved(self):
        cands = [make("bad", endpoints=False)]
        res = evaluate(cands, beam_width=2)
        assert final_reading(res, cands) is Severity.UNRESOLVED


class TestDominanceRecordsWhatTriggeredIt:
    """The rule fired 303 times in production without recording the overlap
    that caused it. The verdict was logged and the number behind it was
    computed, used, and thrown away - so whether the 0.80 threshold had
    ever been near the line could not be asked at all.

    A decision that cannot be second-guessed later is a decision nobody can
    calibrate.
    """

    def _dominated(self, overlap):
        result = evaluate(
            [make("filed", prov=Provenance.FILED_ROUTE),
             make("track", prov=Provenance.ACTUAL_TRACK)],
            beam_width=4, overlap_fn=lambda a, b: overlap)
        return [s for s in result.pruned
                if s.decision is Decision.PRUNE_DOMINATED]

    def test_the_reason_carries_the_overlap(self):
        dominated = self._dominated(0.937)
        assert dominated, "the better-provenanced corridor should dominate"
        assert "0.937" in dominated[0].reason

    def test_the_overlap_is_readable_by_the_analyser(self):
        """The exact pattern the analysis script looks for. Asserting the
        shape here means a reword cannot silently break the measurement -
        which is how three hundred decisions went unrecorded."""
        import re as _re

        dominated = self._dominated(0.812)
        found = _re.search(r"overlap[^0-9]{0,24}([01]?\.\d+)",
                           dominated[0].reason, _re.I)
        assert found and abs(float(found.group(1)) - 0.812) < 1e-6

    def test_a_corridor_below_the_threshold_is_not_dominated(self):
        assert not self._dominated(0.5)

    def test_the_reason_still_names_the_dominating_corridor(self):
        """The overlap was added alongside the explanation, not instead of
        it. A number without the corridor it lost to is not a reason."""
        assert "track" in self._dominated(0.937)[0].reason


def test_the_root_copy_matches_the_installed_one():
    """This file imports `critic` bare, which resolves to a copy at the
    repository root rather than to app/reasoning/critic.py. The two were
    identical by luck until an edit to the real one left the tests reading
    the old logic and reporting green.

    Rather than rely on remembering, the drift is now a failing test.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    a = (root / "critic.py")
    b = (root / "app" / "reasoning" / "critic.py")
    if not a.exists():
        return                      # nothing to drift from
    assert a.read_text() == b.read_text(), (
        "critic.py at the repository root has drifted from "
        "app/reasoning/critic.py. The tests import the root copy, so they "
        "would be checking code the application does not run.")
