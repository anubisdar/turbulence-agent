# install-to: app/reasoning
"""
Shape checks on the facts that reach the model.

WHY SHAPE AND NOT MEANING. Generic prompt injection detection looks for
phrases - "ignore previous instructions" and its relatives - which is fuzzy,
easy to evade and comes with a false positive class this project has already
paid for twice in the explainer's own validator.

None of that is needed here, because the prompt has no free text. Every one
of the twelve facts has a known shape: six enum values for a reading, a
float between zero and one for coverage, integers for counts, and strings
built from airport codes that were validated against a character class at
the API boundary. A field outside its shape is either an upstream data
problem or an injection attempt, and both are worth knowing about.

WHERE THE EXPOSURE ACTUALLY IS. Not user input. Eleven of the twelve fields
are computed by this system from validated inputs and cannot carry a
payload. The two that can are `aircraft`, which is a variant string passed
through from the flight data provider with no check on it, and
`plain_summary`, which is written by this system but from readings that
came from external weather APIs. The surface is the data providers, not the
caller.

WHAT THIS IS NOT. It is not the defence. The defence is that the model
cannot change a reading: severity comes from the deterministic layer and
the model only writes prose about it, and output validation discards a
paragraph that says otherwise. This is detection, so that a provider
sending something strange is visible rather than silent.
"""

from __future__ import annotations

import re
from typing import Any

from app.logging_setup import get_logger, kv

log = get_logger("facts")

#: The six words a reading may be. Anything else did not come from the
#: scoring layer.
_READINGS = {"unresolved", "smooth", "light", "moderate", "severe",
             "extreme"}

#: An aircraft variant as the provider sends it: "737-900", "A321neo",
#: "E175", "CRJ-900". Letters, digits, spaces, hyphens and slashes.
_AIRCRAFT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 \-/().]{0,39}$")

#: "KPIT to KBOS", built from codes already matched against
#: ^[A-Za-z0-9]{3,4}$ at the API.
_ROUTE = re.compile(r"^[A-Za-z0-9]{3,4} to [A-Za-z0-9]{3,4}$")

#: "FL320 to FL340", formatted from integers.
_CRUISE_BAND = re.compile(r"^FL\d{3} to FL\d{3}$")

#: The deterministic summary. Long, but bounded, and prose rather than
#: markup or instructions.
_MAX_SUMMARY_CHARS = 900

#: Characters that have no business in any of these fields. Not an
#: injection signature - a signal that something is not the kind of value
#: it claims to be.
_STRUCTURAL = re.compile(r"[<>{}\[\]\\\x00-\x08\x0b\x0c\x0e-\x1f]|```|"
                         r"\bsystem:|\bassistant:|\buser:", re.I)


def _check_text(name: str, value: Any, pattern: re.Pattern | None = None,
                max_chars: int | None = None) -> list[str]:
    problems = []
    if not isinstance(value, str):
        return [f"{name} is {type(value).__name__}, expected a string"]
    if max_chars is not None and len(value) > max_chars:
        problems.append(f"{name} is {len(value)} characters, over "
                        f"{max_chars}")
    if pattern is not None and not pattern.match(value):
        problems.append(f"{name} does not match its expected shape")
    found = _STRUCTURAL.search(value)
    if found:
        problems.append(f"{name} contains {found.group(0)!r}, which is "
                        f"markup or a role marker rather than data")
    return problems


def check_facts(facts: dict[str, Any]) -> list[str]:
    """Every way the fact dictionary can be wrong, as plain sentences.

    Returns an empty list for a normal search. Does not raise and does not
    modify the facts: a strange aircraft variant should still produce an
    answer, because refusing to explain a search because a provider sent an
    odd string would be a worse failure than explaining it.
    """
    problems: list[str] = []

    unexpected = set(facts) - {
        "route", "reading", "pilot_reports", "forecast", "sources_disagree",
        "route_coverage_fraction", "corridors_considered", "corridors_kept",
        "search_was_truncated", "plain_summary", "aircraft", "cruise_band"}
    if unexpected:
        problems.append(f"unexpected field(s) reaching the model: "
                        f"{', '.join(sorted(unexpected))}")

    problems += _check_text("route", facts.get("route"), _ROUTE)

    reading = facts.get("reading")
    if reading not in _READINGS:
        problems.append(f"reading is {reading!r}, which is not one of the "
                        f"six severities")

    for side in ("pilot_reports", "forecast"):
        block = facts.get(side)
        if not isinstance(block, dict):
            problems.append(f"{side} is {type(block).__name__}, expected an "
                            f"object")
            continue
        if block.get("reading") not in _READINGS:
            problems.append(f"{side}.reading is {block.get('reading')!r}, "
                            f"which is not one of the six severities")
        count = block.get("count")
        if not isinstance(count, int) or isinstance(count, bool) \
                or not 0 <= count <= 10_000:
            problems.append(f"{side}.count is {count!r}, expected a small "
                            f"non-negative integer")

    fraction = facts.get("route_coverage_fraction")
    if fraction is not None and (
            not isinstance(fraction, (int, float))
            or isinstance(fraction, bool)
            or not 0.0 <= float(fraction) <= 1.0):
        problems.append(f"route_coverage_fraction is {fraction!r}, expected "
                        f"a number between zero and one")

    for name in ("corridors_considered", "corridors_kept"):
        value = facts.get(name)
        if not isinstance(value, int) or isinstance(value, bool) \
                or not 0 <= value <= 1_000:
            problems.append(f"{name} is {value!r}, expected a small "
                            f"non-negative integer")

    for name in ("sources_disagree", "search_was_truncated"):
        if not isinstance(facts.get(name), bool):
            problems.append(f"{name} is {facts.get(name)!r}, expected true "
                            f"or false")

    if "plain_summary" in facts:
        problems += _check_text("plain_summary", facts["plain_summary"],
                                max_chars=_MAX_SUMMARY_CHARS)

    # The two fields carrying text this system did not compute.
    if "aircraft" in facts:
        problems += _check_text("aircraft", facts["aircraft"], _AIRCRAFT)
    if "cruise_band" in facts:
        problems += _check_text("cruise_band", facts["cruise_band"],
                                _CRUISE_BAND)

    return problems


def report(facts: dict[str, Any]) -> list[str]:
    """Check the facts and log anything wrong. Returns the problems.

    Logged at WARNING with the field named, because a fact outside its
    shape means either a provider changed its output or something is trying
    to reach the model through one. Both want a person looking.
    """
    problems = check_facts(facts)
    for problem in problems:
        log.warning("fact shape " + kv(problem=problem))
    return problems
