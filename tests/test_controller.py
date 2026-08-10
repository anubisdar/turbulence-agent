"""Tests for the beam search controller.

The controller is control flow only, so these exercise expansion, pruning,
termination, and traceability - never scoring, which belongs to the critic.
"""

import pytest

from app.reasoning.controller import (
    Budget,
    SearchResult,
    Stop,
    search,
)
from app.reasoning.critic import (
    Corridor,
    Evidence,
    Geometry,
    Provenance,
    Severity,
)


#: A threshold no corridor can reach, for tests isolating something other
#: than early stopping. The default (0.85) is met at depth 1 by a flown
#: track with good coverage, which is realistic but hides deeper behaviour.
NO_EARLY_STOP = 1.1


def corridor(cid, prov=Provenance.FILED_ROUTE, depth=1, parent=None,
             length=520.0, gc=500.0, dogleg=10.0, coverage=None, age=None,
             agreement=None, reading=Severity.UNRESOLVED, endpoints=True):
    return Corridor(
        id=cid, provenance=prov, depth=depth, parent_id=parent,
        geometry=Geometry(length_nm=length, great_circle_nm=gc,
                          max_dogleg_deg=dogleg,
                          endpoints_match_airports=endpoints),
        evidence=Evidence(coverage_fraction=coverage, mean_age_minutes=age,
                          agreement=agreement, reading=reading),
    )


class FakeClock:
    """Advances only when told to, so time-limit tests never sleep."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def fanout_generator(spend_per_call=1):
    """Depth 1 offers four corridor sources; deeper levels offer two
    refinements of each parent."""

    def gen(parent, depth, budget):
        if depth == 1:
            budget.spend(spend_per_call)
            return [
                corridor("track", Provenance.ACTUAL_TRACK, coverage=0.7, age=20.0, agreement=0.9),
                corridor("filed", Provenance.FILED_ROUTE, coverage=0.6, age=40.0, agreement=0.8),
                corridor("airway", Provenance.PUBLISHED_AIRWAY, coverage=0.4, age=60.0),
                corridor("gc", Provenance.GREAT_CIRCLE, length=500.0, dogleg=0.0),
            ]
        budget.spend(spend_per_call)
        base = parent.id if parent else "root"
        return [
            corridor(f"{base}-cruise", parent.provenance, depth=depth, parent=base,
                     coverage=0.6, age=30.0, agreement=0.85),
            corridor(f"{base}-step", parent.provenance, depth=depth, parent=base,
                     coverage=0.5, age=35.0, agreement=0.8),
        ]

    return gen


class TestExpansion:
    def test_reaches_the_depth_limit(self):
        res = search(fanout_generator(), depth_limit=3,
                     confidence_threshold=NO_EARLY_STOP,
                     budget=Budget(max_tool_calls=50))
        assert res.depth_reached == 3
        assert res.stop is Stop.DEPTH_LIMIT

    def test_beam_width_bounds_every_level(self):
        res = search(fanout_generator(), beam_width=2, depth_limit=3,
                     confidence_threshold=NO_EARLY_STOP,
                     budget=Budget(max_tool_calls=50))
        for level in res.levels:
            assert len(level.kept) <= 2

    def test_depth_one_expands_from_no_parent(self):
        res = search(fanout_generator(), depth_limit=1, budget=Budget(max_tool_calls=50))
        assert len(res.levels[0].generated) == 4

    def test_deeper_levels_expand_only_survivors(self):
        res = search(fanout_generator(), beam_width=2, depth_limit=2,
                     confidence_threshold=NO_EARLY_STOP,
                     budget=Budget(max_tool_calls=50))
        # two survivors at depth 1, two children each
        assert len(res.levels[1].generated) == 4

    def test_duplicate_ids_are_not_expanded_twice(self):
        def gen(parent, depth, budget):
            budget.spend()
            return [corridor("same"), corridor("same"), corridor("other")]
        res = search(gen, depth_limit=1, budget=Budget(max_tool_calls=50))
        assert len(res.levels[0].generated) == 2


class TestTermination:
    def test_confidence_threshold_stops_early(self):
        res = search(fanout_generator(), depth_limit=3,
                     confidence_threshold=0.50, budget=Budget(max_tool_calls=50))
        assert res.stop is Stop.CONFIDENCE_MET
        assert res.depth_reached < 3
        assert any("Stopped early" in n for n in res.notes)

    def test_a_contested_pair_blocks_early_stopping(self):
        """High score over disagreeing readings is exactly when not to stop."""
        def gen(parent, depth, budget):
            budget.spend()
            return [
                corridor(f"a{depth}", Provenance.ACTUAL_TRACK, depth=depth,
                         coverage=0.7, age=20.0, agreement=0.9,
                         reading=Severity.LIGHT),
                corridor(f"b{depth}", Provenance.ACTUAL_TRACK, depth=depth,
                         coverage=0.7, age=20.0, agreement=0.9,
                         reading=Severity.MODERATE),
            ]
        res = search(gen, depth_limit=2, confidence_threshold=0.10,
                     budget=Budget(max_tool_calls=50))
        assert res.stop is not Stop.CONFIDENCE_MET
        assert res.contested

    def test_tool_budget_stops_the_search(self):
        res = search(fanout_generator(), depth_limit=5,
                     confidence_threshold=NO_EARLY_STOP,
                     budget=Budget(max_tool_calls=2))
        assert res.stop is Stop.TOOL_BUDGET
        assert res.truncated

    def test_time_limit_stops_the_search(self):
        clock = FakeClock()
        budget = Budget(max_tool_calls=99, max_seconds=10.0, clock=clock)

        def gen(parent, depth, budget_):
            clock.advance(6.0)
            budget_.spend()
            return [corridor(f"c{depth}", depth=depth)]

        res = search(gen, depth_limit=5, budget=budget)
        assert res.stop is Stop.TIME_LIMIT
        assert res.truncated

    def test_truncation_is_flagged_in_the_notes(self):
        res = search(fanout_generator(), depth_limit=5,
                     confidence_threshold=NO_EARLY_STOP,
                     budget=Budget(max_tool_calls=2))
        assert any("not because a corridor met the confidence threshold" in n
                   for n in res.notes)

    def test_completing_normally_is_not_truncated(self):
        res = search(fanout_generator(), depth_limit=2,
                     confidence_threshold=NO_EARLY_STOP,
                     budget=Budget(max_tool_calls=50))
        assert not res.truncated


class TestFailureIsNotSmoothAir:
    def test_no_survivors_is_reported_as_a_route_failure(self):
        def gen(parent, depth, budget):
            budget.spend()
            return [corridor("bad", endpoints=False)]
        res = search(gen, depth_limit=2, budget=Budget(max_tool_calls=50))
        assert res.stop is Stop.NO_SURVIVORS
        assert any("not a finding of smooth air" in n for n in res.notes)

    def test_no_candidates_yields_no_conclusion(self):
        res = search(lambda p, d, b: [], depth_limit=3,
                     budget=Budget(max_tool_calls=50))
        assert res.stop is Stop.NO_CANDIDATES
        assert any("no turbulence conclusion follows" in n for n in res.notes)

    def test_a_winner_without_a_reading_says_so(self):
        def gen(parent, depth, budget):
            budget.spend()
            return [corridor("track", Provenance.ACTUAL_TRACK,
                             reading=Severity.UNRESOLVED)]
        res = search(gen, depth_limit=1, budget=Budget(max_tool_calls=50))
        assert res.winner is not None
        assert res.reading is Severity.UNRESOLVED
        assert any("Unresolved is not smooth" in n for n in res.notes)

    def test_no_survivors_leaves_the_reading_unresolved(self):
        def gen(parent, depth, budget):
            budget.spend()
            return [corridor("bad", endpoints=False)]
        res = search(gen, depth_limit=2, budget=Budget(max_tool_calls=50))
        assert res.reading is Severity.UNRESOLVED


class TestTraceability:
    def test_pruned_branches_are_retained(self):
        res = search(fanout_generator(), beam_width=2, depth_limit=1,
                     budget=Budget(max_tool_calls=50))
        assert len(res.levels[0].pruned) == 2

    def test_every_generated_node_is_accounted_for(self):
        res = search(fanout_generator(), beam_width=2, depth_limit=2,
                     confidence_threshold=NO_EARLY_STOP,
                     budget=Budget(max_tool_calls=50))
        for level in res.levels:
            assert len(level.kept) + len(level.pruned) == len(level.generated)

    def test_trace_mentions_pruned_nodes_and_reasons(self):
        res = search(fanout_generator(), beam_width=2, depth_limit=1,
                     budget=Budget(max_tool_calls=50))
        trace = "\n".join(res.trace())
        assert "prune" in trace
        assert "beam width" in trace

    def test_node_count_covers_all_levels(self):
        res = search(fanout_generator(), beam_width=2, depth_limit=2,
                     confidence_threshold=NO_EARLY_STOP,
                     budget=Budget(max_tool_calls=50))
        assert res.nodes_generated == 8


class TestBudgetAccounting:
    def test_calls_used_is_reported(self):
        res = search(fanout_generator(), depth_limit=2,
                     confidence_threshold=NO_EARLY_STOP,
                     budget=Budget(max_tool_calls=50))
        assert res.calls_used > 0

    def test_spending_beyond_the_cap_is_refused(self):
        b = Budget(max_tool_calls=3)
        assert b.spend(2) is True
        assert b.spend(2) is False
        assert b.calls_used == 2

    def test_calls_never_exceed_the_cap(self):
        res = search(fanout_generator(), depth_limit=10,
                     confidence_threshold=NO_EARLY_STOP,
                     budget=Budget(max_tool_calls=4))
        assert res.calls_used <= 4


class TestDeterminism:
    def test_identical_runs_give_identical_results(self):
        a = search(fanout_generator(), depth_limit=3,
                   confidence_threshold=NO_EARLY_STOP, budget=Budget(max_tool_calls=50))
        b = search(fanout_generator(), depth_limit=3,
                   confidence_threshold=NO_EARLY_STOP, budget=Budget(max_tool_calls=50))
        assert [c.id for c in a.survivors] == [c.id for c in b.survivors]
        assert a.stop is b.stop
        assert a.trace() == b.trace()

    def test_the_controller_does_no_scoring_of_its_own(self):
        """Survivor order must match the critic's scores exactly."""
        res = search(fanout_generator(), beam_width=2, depth_limit=1,
                     budget=Budget(max_tool_calls=50))
        totals = [s.total for s in res.levels[0].kept]
        assert totals == sorted(totals, reverse=True)
        assert [c.id for c in res.survivors] == \
               [s.corridor_id for s in res.levels[0].kept]
