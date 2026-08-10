"""Tests for the LangGraph search wrapper.

The parity class is the point of this file. The graph and the plain loop are
one algorithm expressed two ways, and any divergence between them is a bug
in the wiring - so every scenario the plain loop handles is asserted to come
out identical through the graph.
"""

import pytest

from app.reasoning.controller import Budget, Stop, search
from app.reasoning.critic import (
    Corridor,
    Evidence,
    Geometry,
    Provenance,
    Severity,
)
from app.reasoning.graph import build_graph, initial_state, search_graph

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
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def fanout(parent, depth, budget):
    budget.spend()
    if depth == 1:
        return [
            corridor("track", Provenance.ACTUAL_TRACK, coverage=0.7, age=20.0,
                     agreement=0.9, reading=Severity.MODERATE),
            corridor("filed", Provenance.FILED_ROUTE, coverage=0.6, age=40.0,
                     agreement=0.8, reading=Severity.MODERATE),
            corridor("airway", Provenance.PUBLISHED_AIRWAY, coverage=0.4, age=60.0),
            corridor("gc", Provenance.GREAT_CIRCLE, length=500.0, dogleg=0.0),
        ]
    p = parent.id
    return [
        corridor(f"{p}/cruise", parent.provenance, depth=depth, parent=p,
                 coverage=0.66, age=28.0, agreement=0.88, reading=Severity.MODERATE),
        corridor(f"{p}/step", parent.provenance, depth=depth, parent=p,
                 coverage=0.58, age=33.0, agreement=0.72, reading=Severity.LIGHT),
    ]


def contested(parent, depth, budget):
    budget.spend()
    return [
        corridor(f"a{depth}", Provenance.ACTUAL_TRACK, depth=depth, coverage=0.7,
                 age=20.0, agreement=0.9, reading=Severity.LIGHT),
        corridor(f"b{depth}", Provenance.ACTUAL_TRACK, depth=depth, coverage=0.7,
                 age=20.0, agreement=0.9, reading=Severity.MODERATE),
    ]


def all_implausible(parent, depth, budget):
    budget.spend()
    return [corridor("bad", endpoints=False)]


def nothing(parent, depth, budget):
    return []


class TestParity:
    """One algorithm, two expressions. They must not diverge."""

    SCENARIOS = [
        ("full depth", fanout, dict(beam_width=2, depth_limit=3,
                                    confidence_threshold=NO_EARLY_STOP), 12),
        ("early stop", fanout, dict(beam_width=2, depth_limit=3,
                                    confidence_threshold=0.50), 12),
        ("contested", contested, dict(beam_width=2, depth_limit=2,
                                      confidence_threshold=0.10), 12),
        ("no survivors", all_implausible, dict(beam_width=2, depth_limit=2,
                                               confidence_threshold=0.90), 12),
        ("no candidates", nothing, dict(beam_width=2, depth_limit=3,
                                        confidence_threshold=0.90), 12),
        ("tool budget", fanout, dict(beam_width=2, depth_limit=5,
                                     confidence_threshold=NO_EARLY_STOP), 2),
        ("width one", fanout, dict(beam_width=1, depth_limit=3,
                                   confidence_threshold=NO_EARLY_STOP), 12),
    ]

    @pytest.mark.parametrize("name,gen,kw,calls",
                             SCENARIOS, ids=[s[0] for s in SCENARIOS])
    def test_stop_reason_matches(self, name, gen, kw, calls):
        a = search(gen, budget=Budget(max_tool_calls=calls), **kw)
        b = search_graph(gen, budget=Budget(max_tool_calls=calls), **kw)
        assert a.stop is b.stop

    @pytest.mark.parametrize("name,gen,kw,calls",
                             SCENARIOS, ids=[s[0] for s in SCENARIOS])
    def test_survivors_match(self, name, gen, kw, calls):
        a = search(gen, budget=Budget(max_tool_calls=calls), **kw)
        b = search_graph(gen, budget=Budget(max_tool_calls=calls), **kw)
        assert [c.id for c in a.survivors] == [c.id for c in b.survivors]

    @pytest.mark.parametrize("name,gen,kw,calls",
                             SCENARIOS, ids=[s[0] for s in SCENARIOS])
    def test_trace_matches(self, name, gen, kw, calls):
        a = search(gen, budget=Budget(max_tool_calls=calls), **kw)
        b = search_graph(gen, budget=Budget(max_tool_calls=calls), **kw)
        assert a.trace() == b.trace()

    @pytest.mark.parametrize("name,gen,kw,calls",
                             SCENARIOS, ids=[s[0] for s in SCENARIOS])
    def test_notes_and_reading_match(self, name, gen, kw, calls):
        a = search(gen, budget=Budget(max_tool_calls=calls), **kw)
        b = search_graph(gen, budget=Budget(max_tool_calls=calls), **kw)
        assert a.notes == b.notes
        assert a.reading is b.reading

    @pytest.mark.parametrize("name,gen,kw,calls",
                             SCENARIOS, ids=[s[0] for s in SCENARIOS])
    def test_budget_accounting_matches(self, name, gen, kw, calls):
        a = search(gen, budget=Budget(max_tool_calls=calls), **kw)
        b = search_graph(gen, budget=Budget(max_tool_calls=calls), **kw)
        assert a.calls_used == b.calls_used
        assert a.depth_reached == b.depth_reached


class TestGraphMechanics:
    def test_the_loop_edge_actually_loops(self):
        res = search_graph(fanout, beam_width=2, depth_limit=3,
                           confidence_threshold=NO_EARLY_STOP,
                           budget=Budget(max_tool_calls=12))
        assert len(res.levels) == 3

    def test_the_conditional_edge_ends_on_stop(self):
        res = search_graph(fanout, beam_width=2, depth_limit=1,
                           confidence_threshold=NO_EARLY_STOP,
                           budget=Budget(max_tool_calls=12))
        assert len(res.levels) == 1
        assert res.stop is Stop.DEPTH_LIMIT

    def test_graph_compiles_and_is_invokable_directly(self):
        budget = Budget(max_tool_calls=12)
        budget.start()
        app = build_graph(fanout, budget)
        state = app.invoke(initial_state(2, 2, NO_EARLY_STOP))
        assert len(state["levels"]) == 2
        assert state["stop"] is Stop.DEPTH_LIMIT

    def test_state_retains_pruned_branches(self):
        budget = Budget(max_tool_calls=12)
        budget.start()
        app = build_graph(fanout, budget)
        state = app.invoke(initial_state(2, 1, NO_EARLY_STOP))
        assert len(state["levels"][0].pruned) == 2

    def test_deep_search_does_not_hit_the_recursion_limit(self):
        res = search_graph(fanout, beam_width=2, depth_limit=8,
                           confidence_threshold=NO_EARLY_STOP,
                           budget=Budget(max_tool_calls=99))
        assert res.stop is Stop.DEPTH_LIMIT
        assert len(res.levels) == 8


class TestGuardrailsInTheGraph:
    def test_time_limit_stops_the_graph(self):
        clock = FakeClock()
        budget = Budget(max_tool_calls=99, max_seconds=10.0, clock=clock)

        def slow(parent, depth, b):
            clock.advance(6.0)
            b.spend()
            return [corridor(f"c{depth}", depth=depth)]

        res = search_graph(slow, depth_limit=5, confidence_threshold=NO_EARLY_STOP,
                           budget=budget)
        assert res.stop is Stop.TIME_LIMIT
        assert res.truncated

    def test_tool_budget_stops_the_graph(self):
        res = search_graph(fanout, depth_limit=5,
                           confidence_threshold=NO_EARLY_STOP,
                           budget=Budget(max_tool_calls=2))
        assert res.stop is Stop.TOOL_BUDGET
        assert res.truncated

    def test_truncation_note_is_present(self):
        res = search_graph(fanout, depth_limit=5,
                           confidence_threshold=NO_EARLY_STOP,
                           budget=Budget(max_tool_calls=2))
        assert any("best of what was explored" in n for n in res.notes)

    def test_contested_survivors_block_early_stop_in_the_graph(self):
        res = search_graph(contested, depth_limit=2, confidence_threshold=0.10,
                           budget=Budget(max_tool_calls=12))
        assert res.stop is not Stop.CONFIDENCE_MET
        assert res.contested
