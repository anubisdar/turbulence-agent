"""
Beam search controller for corridor hypothesis search.

This is the decision maker in the Tree-of-Thought role split: it expands a
level, hands the candidates to the critic, keeps the survivors, and repeats
until a termination condition fires. It contains no scoring logic of its own
and makes no model calls - it is control flow, and nothing else.

Deliberately framework-free. Proving the search logic here means that when it
is wrapped in a LangGraph StateGraph, any bug is in the graph wiring rather
than in the search.

TERMINATION. The three guardrails were defined in Checkpoint 2.1 and are
unchanged here: a confidence threshold that stops early, a cap on external
tool calls because AeroAPI is metered, and an elapsed-time limit on the whole
request. The result always records which one fired. If time ran out rather
than confidence being met, the caller must be able to say so.

TRACEABILITY. Every node generated is retained with its score and the reason
it was kept or pruned, including nodes eliminated at depth 1. A ranking the
user cannot interrogate is not much better than a guess, and the explanation
step needs the discarded branches as much as the surviving ones.

Early stopping on confidence requires that the survivors agree. A high score
on a contested pair means the search is confident about a corridor whose
turbulence readings disagree, which is exactly the case where stopping early
is wrong.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Protocol, Sequence

from app.logging_setup import get_logger, kv
from app.reasoning.critic import (
    BeamResult,
    Corridor,
    Decision,
    Score,
    Severity,
    evaluate,
    final_reading,
)

log = get_logger("controller")

DEFAULT_BEAM_WIDTH = 2

#: Two, because the generator implements two levels: corridor source at
#: depth 1 and cruise altitude band at depth 2. This constant read 3 for
#: a while, which cost nothing in API calls - the third pass produced no
#: candidates and the search stopped - but the interface offered a depth
#: that could not do anything, and a caller who chose it reasonably
#: assumed they had searched deeper.
#:
#: A third level is designed but not built. It would split a corridor
#: longitudinally, and only when the evidence says the route is not
#: uniform: partial coverage, reports disagreeing inside one corridor, or
#: a forecast overlapping only part of it. Raise this when that exists,
#: not before.
MAX_IMPLEMENTED_DEPTH = 2
DEFAULT_DEPTH_LIMIT = 2
DEFAULT_CONFIDENCE_THRESHOLD = 0.85
DEFAULT_MAX_TOOL_CALLS = 12
#: Elapsed-time ceiling for a whole search. Set at 25 seconds when a warm
#: domestic search took about a second, which turned out to be the wrong
#: measurement to design against: real AeroAPI latency runs 12 to 25 seconds
#: for the same search, so the limit was firing on ordinary runs rather than
#: on stuck ones. A search cut off at depth 1 returns a different corridor
#: from the same query, which reads as non-determinism even though it is
#: correctly marked truncated.
#:
#: 60 seconds sits well clear of the observed worst case while still
#: catching a search that has genuinely hung.
DEFAULT_MAX_SECONDS = 60.0


class Stop(str, Enum):
    """Why the search ended. Recorded on every result."""
    CONFIDENCE_MET = "confidence_met"
    DEPTH_LIMIT = "depth_limit"
    TOOL_BUDGET = "tool_budget_exhausted"
    TIME_LIMIT = "time_limit"
    NO_SURVIVORS = "no_survivors"
    NO_CANDIDATES = "generator_returned_nothing"


#: Stops that mean the search was cut short rather than finishing its work.
TRUNCATED = {Stop.TOOL_BUDGET, Stop.TIME_LIMIT}


@dataclass
class Budget:
    """External call and wall-clock allowance for one request.

    The clock is injectable so tests do not sleep.
    """
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS
    max_seconds: float = DEFAULT_MAX_SECONDS
    clock: Callable[[], float] = time.monotonic
    calls_used: int = 0
    _started: float | None = None

    def start(self) -> None:
        self._started = self.clock()

    @property
    def elapsed(self) -> float:
        return 0.0 if self._started is None else self.clock() - self._started

    @property
    def calls_remaining(self) -> int:
        return max(0, self.max_tool_calls - self.calls_used)

    def spend(self, n: int = 1) -> bool:
        """Consume budget. Returns False if the request cannot afford it."""
        if self.calls_used + n > self.max_tool_calls:
            return False
        self.calls_used += n
        return True

    def out_of_calls(self) -> bool:
        return self.calls_used >= self.max_tool_calls

    def out_of_time(self) -> bool:
        return self.elapsed >= self.max_seconds


class Generator(Protocol):
    """Produces candidate corridors for one level of the tree.

    `parent` is None at depth 1. Implementations should spend from `budget`
    for each external call they make and return early if it is exhausted.
    """

    def __call__(self, parent: Corridor | None, depth: int,
                 budget: Budget) -> Sequence[Corridor]: ...


#: Optional hook: given the corridors that survived a level, return them
#: with turbulence evidence attached. Run after pruning rather than before,
#: so a corridor about to be discarded by dominance does not cost a fetch.
Enricher = Callable[[Sequence[Corridor], Budget], Sequence[Corridor]]


@dataclass
class Level:
    """One depth of the tree, retained for traceability."""
    depth: int
    generated: list[Corridor] = field(default_factory=list)
    kept: list[Score] = field(default_factory=list)
    pruned: list[Score] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    contested: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class SearchResult:
    levels: list[Level] = field(default_factory=list)
    survivors: list[Corridor] = field(default_factory=list)
    reading: Severity = Severity.UNRESOLVED
    stop: Stop = Stop.DEPTH_LIMIT
    depth_reached: int = 0
    calls_used: int = 0
    elapsed: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def winner(self) -> Corridor | None:
        return self.survivors[0] if self.survivors else None

    @property
    def truncated(self) -> bool:
        """True when a budget ran out rather than the search completing."""
        return self.stop in TRUNCATED

    @property
    def contested(self) -> bool:
        return bool(self.levels and self.levels[-1].contested)

    @property
    def nodes_generated(self) -> int:
        return sum(len(lv.generated) for lv in self.levels)

    def trace(self) -> list[str]:
        """Flat, human-readable account of the search, discarded branches
        included. This is what the explanation step reads from."""
        out: list[str] = []
        for lv in self.levels:
            out.append(f"depth {lv.depth}: {len(lv.generated)} generated, "
                       f"{len(lv.kept)} kept, {len(lv.pruned)} pruned")
            for s in lv.kept:
                out.append(f"  keep  {s.corridor_id:<16} {s.total:.4f}")
            for s in lv.pruned:
                out.append(f"  prune {s.corridor_id:<16} {s.total:.4f}  "
                           f"{s.decision.value}: {s.reason}")
        return out


def is_confident(result: BeamResult, threshold: float) -> bool:
    """Confident enough to stop early.

    Requires both a high top score and agreement among survivors. A confident
    score over a contested pair is the case where stopping early is worst.
    """
    if not result.kept:
        return False
    if result.contested:
        return False
    return result.kept[0].total >= threshold


def search(
    generate: Generator,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    depth_limit: int = DEFAULT_DEPTH_LIMIT,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    budget: Budget | None = None,
    overlap_fn: Callable[[Corridor, Corridor], float] | None = None,
    enrich: Enricher | None = None,
) -> SearchResult:
    """Run the beam search and return the tree, the survivors, and why it stopped."""
    budget = budget or Budget()
    budget.start()

    result = SearchResult()
    frontier: list[Corridor] = []

    for depth in range(1, depth_limit + 1):
        # Check the guardrails before spending anything on this level.
        if budget.out_of_time():
            result.stop = Stop.TIME_LIMIT
            break
        if budget.out_of_calls():
            result.stop = Stop.TOOL_BUDGET
            break

        level = Level(depth=depth)
        parents: list[Corridor | None] = frontier if frontier else [None]

        candidates: list[Corridor] = []
        seen: set[str] = set()
        for parent in parents:
            for c in generate(parent, depth, budget):
                if c.id not in seen:
                    seen.add(c.id)
                    candidates.append(c)
            if budget.out_of_time() or budget.out_of_calls():
                break

        level.generated = list(candidates)

        if not candidates:
            result.levels.append(level)
            result.depth_reached = depth
            # A level that produced nothing after a budget ran out is a
            # truncation, not an exhausted search space.
            if budget.out_of_time():
                result.stop = Stop.TIME_LIMIT
            elif budget.out_of_calls():
                result.stop = Stop.TOOL_BUDGET
            elif depth == 1:
                result.stop = Stop.NO_CANDIDATES
            else:
                result.stop = Stop.DEPTH_LIMIT
            break

        beam = evaluate(candidates, beam_width=beam_width, overlap_fn=overlap_fn)

        # Logged here rather than in the critic, which is a pure function
        # with no I/O and stays that way. One line per decision makes
        # questions like "how often does dominance actually fire" a grep
        # rather than a guess: the answer has been assumed from a single
        # observed search since the threshold was first calibrated.
        for scored in beam.all_scores:
            log.info("critic decision " + kv(
                depth=depth,
                corridor=scored.corridor_id,
                decision=scored.decision.value,
                score=round(scored.total, 4),
                reason=scored.reason or None))
        level.kept = list(beam.kept)
        level.pruned = list(beam.pruned)
        level.notes = list(beam.notes)
        level.contested = list(beam.contested)

        result.levels.append(level)
        result.depth_reached = depth
        result.notes.extend(beam.notes)

        by_id = {c.id: c for c in candidates}
        frontier = [by_id[s.corridor_id] for s in beam.kept]

        # Evidence is gathered only for survivors. Fetching turbulence for a
        # corridor that dominance is about to discard spends a call on an
        # answer nobody reads.
        if enrich and frontier:
            frontier = list(enrich(frontier, budget))
            for c in frontier:
                by_id[c.id] = c

        result.survivors = list(frontier)
        result.reading = final_reading(beam, list(by_id.values()))

        if not frontier:
            result.stop = Stop.NO_SURVIVORS
            break

        if is_confident(beam, confidence_threshold):
            result.stop = Stop.CONFIDENCE_MET
            break

        if depth == depth_limit:
            result.stop = Stop.DEPTH_LIMIT

    result.calls_used = budget.calls_used
    result.elapsed = round(budget.elapsed, 4)

    annotate(result, confidence_threshold)
    return result


def annotate(result: SearchResult, threshold: float) -> None:
    """Attach the caveats the caller must not be able to overlook."""
    if result.stop is Stop.TIME_LIMIT:
        result.notes.append(
            f"Search stopped on the elapsed-time limit after {result.elapsed:.1f}s, "
            f"not because a corridor met the confidence threshold. The result is "
            f"the best of what was explored, not the best available."
        )
    elif result.stop is Stop.TOOL_BUDGET:
        result.notes.append(
            f"Search stopped after exhausting the {result.calls_used}-call tool "
            f"budget, not because a corridor met the confidence threshold. The "
            f"result is the best of what was explored."
        )
    elif result.stop is Stop.NO_SURVIVORS:
        result.notes.append(
            "No corridor survived evaluation. The agent could not establish a "
            "route for this flight. This is a failure to determine the path, "
            "not a finding of smooth air."
        )
    elif result.stop is Stop.NO_CANDIDATES:
        result.notes.append(
            "No corridor hypotheses could be generated for this trip. Nothing "
            "was evaluated, and no turbulence conclusion follows."
        )
    elif result.stop is Stop.CONFIDENCE_MET:
        result.notes.append(
            f"Stopped early at depth {result.depth_reached}: the leading corridor "
            f"scored at or above {threshold} and the survivors agreed."
        )

    if result.contested:
        pairs = ", ".join(f"{a}/{b}" for a, b in result.levels[-1].contested)
        result.notes.append(
            f"Surviving corridors disagree ({pairs}). Both are reported and the "
            f"ranking uses the worse reading."
        )

    if result.survivors and result.reading is Severity.UNRESOLVED:
        result.notes.append(
            "A corridor was selected but no turbulence reading could be "
            "established for it. Unresolved is not smooth."
        )
