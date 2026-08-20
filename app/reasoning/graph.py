"""
The beam search as a LangGraph StateGraph.

This is the controller role from the Tree-of-Thought split, expressed as
explicit control flow rather than as a prompt. Two nodes and one conditional
edge: `expand` generates a level, `assess` scores and prunes it, and the edge
decides whether to loop or stop.

There is deliberately no search logic here. Scoring comes from the critic,
and the confidence test, termination annotations, and result shape come from
`controller`. This module only wires them into a graph. `test_graph.py`
asserts parity - the graph and the plain loop must produce identical
survivors, identical stop reasons, and identical traces for the same input.
Two implementations that could disagree would be worse than one.

The generator, budget, and overlap function are bound when the graph is
built rather than carried in state. State holds data; those are behaviour.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence, TypedDict

from langgraph.graph import END, START, StateGraph

from app.reasoning.controller import (
    DEFAULT_BEAM_WIDTH,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_DEPTH_LIMIT,
    Budget,
    Enricher,
    Generator,
    Level,
    SearchResult,
    Stop,
    annotate,
    is_confident,
)
from app.reasoning.critic import Corridor, Severity, evaluate, final_reading


class SearchState(TypedDict, total=False):
    """The reasoning tree as it stands mid-search.

    This is the state manager role: every node generated is retained in
    `levels`, including the ones pruned at depth 1, so the search can be
    replayed and explained after the fact.
    """
    depth: int
    frontier: list[Corridor]
    candidates: list[Corridor]
    levels: list[Level]
    survivors: list[Corridor]
    reading: Severity
    stop: Stop | None
    notes: list[str]
    beam_width: int
    depth_limit: int
    confidence_threshold: float


def initial_state(
    beam_width: int = DEFAULT_BEAM_WIDTH,
    depth_limit: int = DEFAULT_DEPTH_LIMIT,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> SearchState:
    return SearchState(
        depth=0,
        frontier=[],
        candidates=[],
        levels=[],
        survivors=[],
        reading=Severity.UNRESOLVED,
        stop=None,
        notes=[],
        beam_width=beam_width,
        depth_limit=depth_limit,
        confidence_threshold=confidence_threshold,
    )


def build_graph(
    generate: Generator,
    budget: Budget,
    overlap_fn: Callable[[Corridor, Corridor], float] | None = None,
    enrich: Enricher | None = None,
):
    """Compile the search graph.

    `budget` is mutated as the search runs, so a fresh one is required per
    request. That is the same contract the plain loop has.
    """

    def expand(state: SearchState) -> dict[str, Any]:
        """Generate the next level of the tree."""
        depth = state["depth"] + 1

        # Guardrails are checked before spending anything on this level.
        if budget.out_of_time():
            return {"depth": depth, "candidates": [], "stop": Stop.TIME_LIMIT}
        if budget.out_of_calls():
            return {"depth": depth, "candidates": [], "stop": Stop.TOOL_BUDGET}

        parents: list[Corridor | None] = state["frontier"] or [None]
        candidates: list[Corridor] = []
        seen: set[str] = set()
        for parent in parents:
            for c in generate(parent, depth, budget):
                if c.id not in seen:
                    seen.add(c.id)
                    candidates.append(c)
            if budget.out_of_time() or budget.out_of_calls():
                break

        return {"depth": depth, "candidates": candidates}

    def assess(state: SearchState) -> dict[str, Any]:
        """Score the level, keep the beam, record what was discarded."""
        depth = state["depth"]
        candidates = state["candidates"]

        # A guardrail that fired before generation means no level was
        # attempted, so none is recorded. Only a generator that ran and
        # returned nothing produces an empty level.
        if not candidates and state.get("stop") is not None:
            return {"stop": state["stop"]}

        level = Level(depth=depth, generated=list(candidates))

        if not candidates:
            if budget.out_of_time():
                stop = Stop.TIME_LIMIT
            elif budget.out_of_calls():
                stop = Stop.TOOL_BUDGET
            elif depth == 1:
                stop = Stop.NO_CANDIDATES
            else:
                stop = Stop.DEPTH_LIMIT
            return {"levels": state["levels"] + [level], "stop": stop}

        beam = evaluate(candidates,
                        beam_width=state["beam_width"],
                        overlap_fn=overlap_fn)
        level.kept = list(beam.kept)
        level.pruned = list(beam.pruned)
        level.notes = list(beam.notes)
        level.contested = list(beam.contested)

        by_id = {c.id: c for c in candidates}
        frontier = [by_id[s.corridor_id] for s in beam.kept]

        # Survivors only, matching the plain loop. See controller.search.
        if enrich and frontier:
            frontier = list(enrich(frontier, budget))
            for c in frontier:
                by_id[c.id] = c

        stop: Stop | None = None
        if not frontier:
            stop = Stop.NO_SURVIVORS
        elif is_confident(beam, state["confidence_threshold"]):
            stop = Stop.CONFIDENCE_MET
        elif depth >= state["depth_limit"]:
            stop = Stop.DEPTH_LIMIT

        return {
            "levels": state["levels"] + [level],
            "frontier": frontier,
            "survivors": list(frontier),
            "reading": final_reading(beam, list(by_id.values())),
            "notes": state["notes"] + list(beam.notes),
            "stop": stop,
        }

    def route(state: SearchState) -> str:
        """The pruning decision, as a conditional edge."""
        return END if state.get("stop") is not None else "expand"

    graph = StateGraph(SearchState)
    graph.add_node("expand", expand)
    graph.add_node("assess", assess)
    graph.add_edge(START, "expand")
    graph.add_edge("expand", "assess")
    graph.add_conditional_edges("assess", route, {"expand": "expand", END: END})
    return graph.compile()


def to_result(state: SearchState, budget: Budget) -> SearchResult:
    """Convert graph state into the same SearchResult the plain loop returns,
    so callers cannot tell which ran."""
    result = SearchResult(
        levels=list(state.get("levels", [])),
        survivors=list(state.get("survivors", [])),
        reading=state.get("reading", Severity.UNRESOLVED),
        stop=state.get("stop") or Stop.DEPTH_LIMIT,
        depth_reached=len([lv for lv in state.get("levels", [])]),
        calls_used=budget.calls_used,
        elapsed=round(budget.elapsed, 4),
        notes=list(state.get("notes", [])),
    )
    annotate(result, state.get("confidence_threshold",
                               DEFAULT_CONFIDENCE_THRESHOLD))
    return result


def search_graph(
    generate: Generator,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    depth_limit: int = DEFAULT_DEPTH_LIMIT,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    budget: Budget | None = None,
    overlap_fn: Callable[[Corridor, Corridor], float] | None = None,
    enrich: Enricher | None = None,
) -> SearchResult:
    """Run the search through the graph. Signature matches `controller.search`."""
    budget = budget or Budget()
    budget.start()
    app = build_graph(generate, budget, overlap_fn=overlap_fn, enrich=enrich)
    state = app.invoke(
        initial_state(beam_width, depth_limit, confidence_threshold),
        config={"recursion_limit": max(50, depth_limit * 4 + 10)},
    )
    return to_result(state, budget)
