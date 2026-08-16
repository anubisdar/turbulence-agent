"""
Deterministic critic for corridor hypothesis search.

The generator proposes corridors and the LLM explains the outcome. Nothing in
this module is generative: given the same candidates it returns the same
scores, the same keep/prune decisions, and the same ordering, every run. That
is what lets Tree-of-Thought sit upstream of a scoring function that was
promised to be deterministic.

Four weighted criteria:

    provenance      0.40   how the corridor was derived
    geometry        0.25   does it behave like a flight
    agreement       0.20   do PIREPs and forecast point the same way
    coverage        0.15   how much of it has fresh observation

Provenance carries the most weight deliberately. It is the only criterion
that measures whether the corridor is real rather than inferred; the other
three measure how well a possibly-fictional corridor can be described. For a
nervous passenger, "this is the path the aircraft actually flew" is worth
more than a well-observed guess.

Two separate mechanisms, and the distinction matters:

    ELIMINATION  a branch is removed only for being wrong (implausible
                 geometry) or redundant (dominated by a better-provenanced
                 corridor over the same airspace).
    RANKING      everything else competes on weighted score.

Coverage affects ranking only and can never eliminate a branch. A corridor
nobody has flown recently has no pilot reports, and sparse data is not smooth
air. If coverage could prune, the search would discard the true path
precisely because nobody reported on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Sequence

# --------------------------------------------------------------- vocabulary


class Provenance(str, Enum):
    """How a corridor hypothesis was derived. Ordered best to worst."""
    ACTUAL_TRACK = "actual_track"        # flown track for this flight number
    FILED_ROUTE = "filed_route"          # route from the flight plan
    PUBLISHED_AIRWAY = "published_airway"  # standard airway routing
    GREAT_CIRCLE = "great_circle"        # geometric fallback


PROVENANCE_SCORE: dict[Provenance, float] = {
    Provenance.ACTUAL_TRACK: 1.00,
    Provenance.FILED_ROUTE: 0.75,
    Provenance.PUBLISHED_AIRWAY: 0.50,
    Provenance.GREAT_CIRCLE: 0.25,
}

PROVENANCE_RANK: dict[Provenance, int] = {
    Provenance.ACTUAL_TRACK: 0,
    Provenance.FILED_ROUTE: 1,
    Provenance.PUBLISHED_AIRWAY: 2,
    Provenance.GREAT_CIRCLE: 3,
}


class Severity(str, Enum):
    """Turbulence reading. UNRESOLVED is not a severity - it is the absence
    of one, and must never be treated as SMOOTH."""
    UNRESOLVED = "unresolved"
    SMOOTH = "smooth"
    LIGHT = "light"
    MODERATE = "moderate"
    SEVERE = "severe"
    EXTREME = "extreme"


#: Ordering for "rank on the worse reading". UNRESOLVED sits outside the
#: scale - it is not comparable and never wins a worse-of comparison.
SEVERITY_ORDER: dict[Severity, int] = {
    Severity.SMOOTH: 0,
    Severity.LIGHT: 1,
    Severity.MODERATE: 2,
    Severity.SEVERE: 3,
    Severity.EXTREME: 4,
}


class Decision(str, Enum):
    KEEP = "keep"
    PRUNE_IMPLAUSIBLE = "prune_implausible"
    PRUNE_DOMINATED = "prune_dominated"
    PRUNE_BEAM = "prune_beam"


WEIGHTS = {
    "provenance": 0.40,
    "geometry": 0.25,
    "agreement": 0.20,
    "coverage": 0.15,
}

#: Two scores closer than this are treated as tied.
TIE_EPSILON = 0.02

#: Airspace overlap above which a worse-provenanced corridor is redundant.
DOMINANCE_OVERLAP = 0.80

#: Geometry sanity bounds.
MAX_LENGTH_RATIO = 1.60      # vs great-circle distance
MAX_DOGLEG_DEG = 90.0        # sharpest turn a transport aircraft would fly


# --------------------------------------------------------------- inputs


@dataclass(frozen=True)
class Geometry:
    """Shape facts about a candidate corridor."""
    length_nm: float
    great_circle_nm: float
    max_dogleg_deg: float = 0.0
    endpoints_match_airports: bool = True
    altitude_profile_valid: bool = True


@dataclass(frozen=True)
class Evidence:
    """Turbulence evidence gathered along a corridor.

    `coverage_fraction` and `agreement` are None when nothing has been
    gathered yet - distinct from 0.0, which means gathered and empty.

    OBSERVED AND FORECAST ARE HELD APART. A pilot report is one aircraft at
    one moment saying what the ride was actually like. A forecast polygon
    covers hours and a wide band of sky saying what is expected. They can
    disagree, and conflicts surface rather than average, so each keeps its
    own reading and its own count all the way to the critic. `agreement` is
    then something the critic computes from the two rather than a blended
    number handed to it.

    `reading` is the corridor's single severity, taken as the worse of the
    two. That is the conservative direction and the one a nervous passenger
    cares about: a lone severe report is the thing to surface, not the thing
    to outvote.
    """
    coverage_fraction: float | None = None   # 0..1 of corridor with any obs
    mean_age_minutes: float | None = None
    agreement: float | None = None           # 0..1, computed from the two
    reading: Severity = Severity.UNRESOLVED
    observation_count: int = 0

    # --- observed: pilot reports
    observed_reading: Severity = Severity.UNRESOLVED
    observed_count: int = 0
    observed_worst_at: str | None = None      # where the worst report came from

    # --- forecast: G-AIRMET polygons
    forecast_reading: Severity = Severity.UNRESOLVED
    forecast_count: int = 0

    @property
    def has_observed(self) -> bool:
        return self.observed_reading is not Severity.UNRESOLVED

    @property
    def has_forecast(self) -> bool:
        return self.forecast_reading is not Severity.UNRESOLVED

    @property
    def sources_disagree(self) -> bool:
        """True only when both sources spoke and said different things.

        One source being silent is not a disagreement, it is a gap.
        """
        return (self.has_observed and self.has_forecast
                and self.observed_reading is not self.forecast_reading)


@dataclass(frozen=True)
class Corridor:
    """One node in the reasoning tree."""
    id: str
    provenance: Provenance
    geometry: Geometry
    evidence: Evidence = field(default_factory=Evidence)
    depth: int = 1
    parent_id: str | None = None
    label: str = ""


# --------------------------------------------------------------- scoring


@dataclass(frozen=True)
class Score:
    corridor_id: str
    total: float
    components: dict[str, float]
    decision: Decision
    reason: str

    @property
    def kept(self) -> bool:
        return self.decision is Decision.KEEP


def score_provenance(c: Corridor) -> float:
    return PROVENANCE_SCORE[c.provenance]


def score_geometry(c: Corridor) -> float:
    """Zero means implausible, which is the only score that eliminates on
    its own. A corridor that does not behave like a flight is wrong however
    much data sits along it."""
    g = c.geometry
    if not g.endpoints_match_airports:
        return 0.0
    if not g.altitude_profile_valid:
        return 0.0
    if g.great_circle_nm <= 0:
        return 0.0
    if g.max_dogleg_deg > MAX_DOGLEG_DEG:
        return 0.0

    ratio = g.length_nm / g.great_circle_nm
    if ratio < 0.995:                     # shorter than great circle
        return 0.0
    if ratio > MAX_LENGTH_RATIO:
        return 0.0

    # 1.0 at great-circle length, tapering to 0 at the ratio ceiling.
    excess = (ratio - 1.0) / (MAX_LENGTH_RATIO - 1.0)
    length_term = max(0.0, 1.0 - excess)
    dogleg_term = max(0.0, 1.0 - (g.max_dogleg_deg / MAX_DOGLEG_DEG))
    return round(0.7 * length_term + 0.3 * dogleg_term, 4)


#: How far apart two severity levels can be before agreement scores zero.
#: Light against moderate is a mild disagreement; light against severe is a
#: real one.
_MAX_SEVERITY_GAP = 4


def score_agreement(c: Corridor) -> float:
    """Do the observed and forecast readings point the same way?

    Computed here rather than supplied, so the critic sees the two sources
    as separate opinions instead of a pre-blended number.

    Absent evidence scores neutral, not zero. One silent source is a gap,
    and scoring a gap as disagreement would punish a corridor for what
    nobody reported on it.
    """
    e = c.evidence
    if not (e.has_observed and e.has_forecast):
        return 0.5 if e.agreement is None else max(0.0, min(1.0, e.agreement))

    gap = abs(SEVERITY_ORDER.get(e.observed_reading, 0)
              - SEVERITY_ORDER.get(e.forecast_reading, 0))
    return round(max(0.0, 1.0 - gap / _MAX_SEVERITY_GAP), 4)


def score_coverage(c: Corridor) -> float:
    """Fresh observation over more of the corridor scores higher.

    Contributes to ranking only. See the module docstring: this value can
    never cause a branch to be eliminated.
    """
    frac = c.evidence.coverage_fraction
    if frac is None:
        return 0.0
    frac = max(0.0, min(1.0, frac))
    age = c.evidence.mean_age_minutes
    if age is None:
        return round(frac * 0.5, 4)      # present but undated
    # Full credit under 30 minutes, decaying to nothing at 3 hours.
    recency = max(0.0, min(1.0, (180.0 - age) / 150.0))
    return round(frac * recency, 4)


def score(c: Corridor) -> Score:
    components = {
        "provenance": score_provenance(c),
        "geometry": score_geometry(c),
        "agreement": score_agreement(c),
        "coverage": score_coverage(c),
    }
    total = round(sum(WEIGHTS[k] * v for k, v in components.items()), 4)

    if components["geometry"] == 0.0:
        return Score(c.id, total, components, Decision.PRUNE_IMPLAUSIBLE,
                     "corridor geometry is not physically plausible")

    return Score(c.id, total, components, Decision.KEEP, "")


# --------------------------------------------------------------- beam


@dataclass
class BeamResult:
    kept: list[Score] = field(default_factory=list)
    pruned: list[Score] = field(default_factory=list)
    contested: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def all_scores(self) -> list[Score]:
        return self.kept + self.pruned


OverlapFn = Callable[[Corridor, Corridor], float]


def _tie_break_key(c: Corridor, s: Score):
    """Deterministic ordering for exact ties: provenance, then freshness,
    then id. Never random, never input order."""
    age = c.evidence.mean_age_minutes
    return (
        PROVENANCE_RANK[c.provenance],
        age if age is not None else float("inf"),
        c.id,
    )


def worse_reading(a: Severity, b: Severity) -> Severity:
    """The more severe of two readings.

    UNRESOLVED is not on the scale. When one side is unresolved the other
    stands - an absent reading must never be allowed to soften a real one,
    and must never masquerade as SMOOTH.
    """
    if a is Severity.UNRESOLVED:
        return b
    if b is Severity.UNRESOLVED:
        return a
    return a if SEVERITY_ORDER[a] >= SEVERITY_ORDER[b] else b


def evaluate(
    candidates: Sequence[Corridor],
    beam_width: int = 2,
    overlap_fn: OverlapFn | None = None,
) -> BeamResult:
    """Score every candidate, eliminate the wrong and the redundant, then
    keep the top `beam_width` survivors.

    `overlap_fn` returns 0..1 airspace overlap between two corridors. When
    omitted, dominance pruning is skipped - the geometry layer that computes
    real overlap is not required for the critic to be testable.
    """
    result = BeamResult()
    by_id = {c.id: c for c in candidates}
    scores = {c.id: score(c) for c in candidates}

    for s in scores.values():
        if s.decision is Decision.PRUNE_IMPLAUSIBLE:
            result.pruned.append(s)

    alive = [c for c in candidates
             if scores[c.id].decision is Decision.KEEP]

    # Dominance: a worse-provenanced corridor covering substantially the same
    # airspace adds nothing. Note this is provenance-based, never coverage.
    if overlap_fn is not None:
        dominated: set[str] = set()
        for a in alive:
            for b in alive:
                if a.id == b.id or a.id in dominated:
                    continue
                better = PROVENANCE_RANK[b.provenance] < PROVENANCE_RANK[a.provenance]
                if better and overlap_fn(a, b) >= DOMINANCE_OVERLAP:
                    dominated.add(a.id)
                    old = scores[a.id]
                    scores[a.id] = Score(
                        old.corridor_id, old.total, old.components,
                        Decision.PRUNE_DOMINATED,
                        f"covers the same airspace as {b.id}, "
                        f"which has better provenance",
                    )
                    result.pruned.append(scores[a.id])
                    break
        alive = [c for c in alive if c.id not in dominated]

    ranked = sorted(
        alive,
        key=lambda c: (-scores[c.id].total, _tie_break_key(c, scores[c.id])),
    )

    survivors = ranked[:beam_width]
    for c in ranked[beam_width:]:
        old = scores[c.id]
        scores[c.id] = Score(old.corridor_id, old.total, old.components,
                             Decision.PRUNE_BEAM,
                             f"outside beam width {beam_width}")
        result.pruned.append(scores[c.id])

    result.kept = [scores[c.id] for c in survivors]

    # Ties among survivors. Same reading is agreement worth noting; different
    # readings are a conflict that must reach the user intact.
    for i, a in enumerate(survivors):
        for b in survivors[i + 1:]:
            if abs(scores[a.id].total - scores[b.id].total) > TIE_EPSILON:
                continue
            ra, rb = a.evidence.reading, b.evidence.reading
            if ra is rb and ra is Severity.UNRESOLVED:
                # Two corridors knowing nothing is the same silence twice,
                # not two sources agreeing. Calling it corroboration would
                # dress an absence up as a finding.
                result.notes.append(
                    f"{a.id} and {b.id} score within {TIE_EPSILON} and "
                    f"neither has a turbulence reading. That is the same "
                    f"absence twice, not corroboration."
                )
            elif ra is rb:
                result.notes.append(
                    f"{a.id} and {b.id} score within {TIE_EPSILON} and agree "
                    f"on {ra.value}; corroborating rather than conflicting."
                )
            else:
                result.contested.append((a.id, b.id))
                result.notes.append(
                    f"{a.id} and {b.id} score within {TIE_EPSILON} but "
                    f"disagree: {ra.value} vs {rb.value}. Both are carried "
                    f"forward; ranking uses the worse reading "
                    f"({worse_reading(ra, rb).value})."
                )

    if not result.kept:
        result.notes.append(
            "No corridor survived evaluation. This is a failure to establish "
            "a route, not a finding of smooth air."
        )

    return result


def final_reading(result: BeamResult, candidates: Sequence[Corridor]) -> Severity:
    """The severity the ranking should use.

    Where survivors disagree, the worse reading is used and both are shown.
    A nervous passenger is better served by "this could be moderate, sources
    disagree" than by an average that matches neither source.
    """
    by_id = {c.id: c for c in candidates}
    readings = [by_id[s.corridor_id].evidence.reading for s in result.kept]
    if not readings:
        return Severity.UNRESOLVED
    out = readings[0]
    for r in readings[1:]:
        out = worse_reading(out, r)
    return out
