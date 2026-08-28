# install-to: app/web
"""
Human-readable narration of what the agent did.

The strongest properties of this system are invisible in its output. Nobody
looking at a ranked list can see that the critic is deterministic, that
pruning happened on airspace overlap rather than on data coverage, or that
the language model was nowhere near any number on the page. This turns those
into something a person can read.

A pure function of the response payload. It emits no events, threads no
logger through the reasoning code, and cannot change what the agent did -
which also means it cannot flatter it. Every beat below is derived from a
value the search actually produced.

Each beat carries the Tree-of-Thought role that acted and the concept it
illustrates, so the narration reads as an implementation of the pattern
rather than as generic logging.

The unflattering beats matter more than the flattering ones. A narration
that only reports success is marketing.
"""

from __future__ import annotations

from typing import Any

GENERATOR = "Thought generator"
CRITIC = "Critic"
CONTROLLER = "Controller"
STATE = "State manager"

#: Milliseconds to hold before the next beat when streaming. Longer pauses
#: sit before a conclusion so it does not scroll past unread.
BEAT = 420
BEAT_LONG = 900
BEAT_SHORT = 240

SOURCE_BLURB = {
    "actual_track": "the path an aircraft actually flew on this route",
    "filed_route": "the route this flight told air traffic control it would fly",
    "published_airway": "a routing other traffic on this pair commonly files",
    "great_circle": "the shortest line between the airports, always available",
}


def _beat(role: str, concept: str, text: str, detail: str | None = None,
          kind: str = "info", pause: int = BEAT) -> dict[str, Any]:
    return {"role": role, "concept": concept, "text": text,
            "detail": detail, "kind": kind, "pause_ms": pause}


def _fl(ft: int | None) -> str:
    return f"FL{round(ft / 100):03d}" if ft else "-"


def narrate(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn a corridor search response into an ordered narration."""
    req = payload.get("request", {})
    out = payload.get("outcome", {})
    corridors = payload.get("corridors", [])
    overlaps = payload.get("overlaps", [])
    beats: list[dict[str, Any]] = []

    depth1 = [c for c in corridors if c.get("depth") == 1]
    depth2 = [c for c in corridors if c.get("depth", 1) > 1]

    # ---------------------------------------------------------- setup
    beats.append(_beat(
        CONTROLLER, "Search strategy",
        f"Planning {req.get('origin')} to {req.get('dest')}. Rather than "
        f"picking one flight path and reasoning forward from it, the agent "
        f"generates several competing paths, scores them against each other, "
        f"and discards the weak ones.",
        "beam search · beam width "
        f"{req.get('beam_width', '?')} · depth limit {req.get('depth_limit', '?')}",
        pause=BEAT_LONG))

    if req.get("departure_time"):
        beats.append(_beat(
            GENERATOR, "Problem framing",
            f"Departure time {req['departure_time']} UTC is used to choose a "
            f"reference flight departing near that hour. A morning and an "
            f"evening flight on the same route fly through different air.",
            pause=BEAT_SHORT))

    # ---------------------------------------------------------- branching
    if depth1:
        listed = ", ".join(c["id"] for c in depth1)
        beats.append(_beat(
            GENERATOR, "Branching",
            f"Generated {len(depth1)} candidate corridors. Each comes from a "
            f"different kind of source, so they fail in different ways - the "
            f"point of branching is that one bad assumption cannot take the "
            f"whole answer with it.",
            listed, pause=BEAT_LONG))

        for c in depth1:
            blurb = SOURCE_BLURB.get(c.get("provenance"), "")
            length = f"{c['length_nm']:.0f} nm" if c.get("length_nm") else "-"
            alt = ""
            if c.get("altitude_min_ft"):
                alt = f" · {_fl(c['altitude_min_ft'])}-{_fl(c['altitude_max_ft'])}"
            beats.append(_beat(
                GENERATOR, "Branching",
                f"{c['id']}: {blurb}.",
                f"{length}{alt}", pause=BEAT_SHORT))

    fixes = payload.get("fix_cache", {})
    gained = (fixes.get("after", 0) or 0) - (fixes.get("before", 0) or 0)
    if gained > 0:
        beats.append(_beat(
            STATE, "Waypoint cache",
            f"Resolved {gained} new waypoint names to coordinates and stored "
            f"them. A route arrives as names with no positions, and those "
            f"positions do not change, so the next route across this "
            f"airspace does not pay for the lookup again.",
            f"{fixes.get('before', 0)} to {fixes.get('after', 0)} cached fixes",
            pause=BEAT))
    elif fixes.get("after"):
        beats.append(_beat(
            STATE, "Waypoint cache",
            f"Every waypoint on these routes was already resolved by an "
            f"earlier search, so this one paid for no lookups.",
            f"{fixes['after']} cached fixes", pause=BEAT_SHORT))

    # ---------------------------------------------------------- evaluation
    beats.append(_beat(
        CRITIC, "Evaluation",
        "Scoring each corridor on four weighted criteria: where it came from "
        "(40%), whether it behaves like a real flight path (25%), whether the "
        "weather sources agree (20%), and how much fresh observation covers it "
        "(15%). This is ordinary code. No language model produced any number "
        "on this page, so the same inputs always give the same result.",
        pause=BEAT_LONG))

    for c in sorted(depth1, key=lambda x: -x.get("score", 0)):
        comp = c.get("components", {})
        beats.append(_beat(
            CRITIC, "Evaluation",
            f"{c['id']} scored {c.get('score', 0):.4f}.",
            "  ".join(f"{k[:4]} {v:.2f}" for k, v in comp.items()),
            pause=BEAT_SHORT))

    # ---------------------------------------------------------- pruning
    dominated = [c for c in corridors if c.get("decision") == "prune_dominated"]
    if dominated:
        pairs = {o["a"]: o for o in overlaps if o.get("dominance_range")}
        pairs.update({o["b"]: o for o in overlaps if o.get("dominance_range")})
        for c in dominated:
            hit = pairs.get(c["id"])
            frac = f"{hit['fraction']:.0%} shared airspace" if hit else ""
            beats.append(_beat(
                CONTROLLER, "Pruning",
                f"Discarded {c['id']}: it covers nearly the same airspace as a "
                f"corridor with better evidence behind it, so keeping both "
                f"would add nothing.",
                frac, kind="caution", pause=BEAT))

    beamed = [c for c in corridors if c.get("decision") == "prune_beam"]
    if beamed:
        beats.append(_beat(
            CONTROLLER, "Pruning",
            f"Dropped {len(beamed)} lower-scoring branch(es) to stay within "
            f"the beam. This is the cost control: every branch kept alive "
            f"costs metered API calls at the next level.",
            ", ".join(c["id"] for c in beamed), kind="caution", pause=BEAT))

    implausible = [c for c in corridors
                   if c.get("decision") == "prune_implausible"]
    for c in implausible:
        beats.append(_beat(
            CONTROLLER, "Pruning",
            f"Rejected {c['id']} outright: the path does not behave like a "
            f"flight. This is the only check that can eliminate a corridor on "
            f"its own.",
            c.get("reason"), kind="caution", pause=BEAT))

    # ---------------------------------------------------------- depth 2
    if depth2:
        beats.append(_beat(
            GENERATOR, "Branching",
            f"Took the surviving corridor and branched it {len(depth2)} ways "
            f"on cruise altitude, using the altitudes real traffic files on "
            f"this route rather than invented ones. Turbulence at one flight "
            f"level is not turbulence at another.",
            " · ".join(
                f"{c['id']} {_fl(c.get('altitude_min_ft'))}-{_fl(c.get('altitude_max_ft'))}"
                for c in depth2), pause=BEAT_LONG))

    # ---------------------------------------------------------- evidence
    wx = out.get("turbulence") or {}
    observed = wx.get("observed") or {}
    forecast = wx.get("forecast") or {}

    turbulence_off = req.get("include_turbulence") is False
    looked = bool(wx) and not turbulence_off

    if observed.get("count") or forecast.get("count"):
        beats.append(_beat(
            STATE, "Evidence",
            f"Gathered turbulence evidence along the surviving corridors, and "
            f"only those: fetching for a corridor about to be discarded would "
            f"spend a call on an answer nobody reads.",
            f"{observed.get('count', 0)} pilot report(s) · "
            f"{forecast.get('count', 0)} forecast(s)", pause=BEAT))
    elif looked:
        # Saying nothing about a search that happened is the same gap as
        # saying nothing about a result. A reader should be able to tell
        # "we looked and found nothing" from "we never looked."
        beats.append(_beat(
            STATE, "Evidence",
            "Looked for pilot reports and turbulence forecasts along the "
            "surviving corridors and found neither. Both sources were "
            "queried; neither had anything to say about this route.",
            wx.get("summary"), kind="caution", pause=BEAT_LONG))
    elif turbulence_off:
        beats.append(_beat(
            STATE, "Evidence",
            "Turbulence lookup was switched off for this search, so neither "
            "pilot reports nor forecasts were queried. Nothing is known "
            "about the air on this route.",
            kind="caution", pause=BEAT))

    if observed.get("count"):
        age = observed.get("mean_age_minutes")
        beats.append(_beat(
            STATE, "Evidence",
            f"Pilots flying inside this corridor reported "
            f"{observed.get('reading')}. Where reports disagree the worst one "
            f"is used and the count is kept, so one alarming report is "
            f"distinguishable from several calm ones.",
            f"{observed['count']} report(s)"
            + (f", average {age:.0f} minutes old" if age else ""),
            pause=BEAT))

    if forecast.get("count"):
        beats.append(_beat(
            STATE, "Evidence",
            f"The forecast covering this corridor calls for "
            f"{forecast.get('reading')}. A forecast is a wide shape over "
            f"several hours, so it is treated as a separate opinion rather "
            f"than merged with what pilots actually felt.",
            f"{forecast['count']} advisory(s)", pause=BEAT))

    if wx.get("disagree"):
        beats.append(_beat(
            CRITIC, "Guardrail",
            f"The two sources disagree: pilots say {observed.get('reading')}, "
            f"the forecast says {forecast.get('reading')}. Both are reported "
            f"and the worse one is used. Averaging them would produce a number "
            f"neither source supports.",
            kind="caution", pause=BEAT_LONG))

    # ---------------------------------------------------------- guardrail
    no_coverage = [c for c in depth1
                   if (c.get("components", {}) or {}).get("coverage", 0) == 0]
    if len(no_coverage) == len(depth1) and depth1:
        beats.append(_beat(
            CRITIC, "Guardrail",
            "Every corridor scored zero on data coverage, because no "
            "turbulence observations were found. That lowered the scores but "
            "eliminated nothing: a corridor nobody has reported on is "
            "unobserved, not smooth. Coverage is never allowed to prune.",
            kind="caution", pause=BEAT_LONG))

    # ---------------------------------------------------------- termination
    stop = out.get("stop")
    reasons = {
        "confidence_met": "one corridor scored high enough that exploring "
                          "further would not change the answer",
        "depth_limit": "the tree reached its depth limit",
        "tool_budget_exhausted": "the cap on metered API calls was reached",
        "time_limit": "the elapsed-time limit expired",
        "no_survivors": "no corridor survived evaluation",
        "generator_returned_nothing": "no corridor could be generated at all",
    }
    beats.append(_beat(
        CONTROLLER, "Termination",
        f"Stopped because {reasons.get(stop, stop)}.",
        f"{out.get('nodes_generated', 0)} nodes explored · "
        f"{out.get('calls_used', 0)} of {req.get('max_tool_calls', '?')} API calls · "
        f"{out.get('elapsed_seconds', 0)}s",
        kind="caution" if out.get("truncated") else "info", pause=BEAT))

    if out.get("truncated"):
        beats.append(_beat(
            CONTROLLER, "Guardrail",
            "A budget ran out rather than the search finishing. What follows "
            "is the best of what was explored, not the best available, and "
            "the agent says so rather than presenting it as final.",
            kind="caution", pause=BEAT))

    if out.get("degraded"):
        reasons = out.get("degraded_reasons") or []
        beats.append(_beat(
            CONTROLLER, "Guardrail",
            "A data source failed during this search, so fewer corridors "
            "were built than usual. The answer below rests on what could be "
            "reached, not on everything the agent would normally consider. "
            "This is a different thing from a budget running out.",
            reasons[0] if reasons else None,
            kind="caution", pause=BEAT_LONG))

    # ---------------------------------------------------------- verdict
    if out.get("winner"):
        beats.append(_beat(
            CONTROLLER, "Decision",
            f"Selected {out['winner']} as the corridor this flight is most "
            f"likely to fly.", pause=BEAT))

    reading = out.get("reading")
    if reading == "unresolved" and not out.get("winner"):
        # No corridor at all is a different failure from a corridor with no
        # weather. Saying "the agent found the corridor" on a run where it
        # found none contradicts the termination beat two lines above.
        beats.append(_beat(
            CRITIC, "Guardrail",
            "Turbulence reading: unresolved, because no route could be "
            "established in the first place. With no corridor there is "
            "nothing to gather evidence along, and the agent reports that "
            "rather than guessing at a path.",
            kind="caution", pause=BEAT_LONG))
    elif reading == "unresolved":
        beats.append(_beat(
            CRITIC, "Guardrail",
            "Turbulence reading: unresolved. The agent found the corridor but "
            "has nothing to say about the air in it, and it will not fill that "
            "silence with a guess. Unresolved is not smooth.",
            # The plain explanation already appeared on the Evidence beat
            # above; repeating it here makes the panel read like a loop.
            wx.get("summary") if not looked else None,
            kind="caution", pause=BEAT_LONG))
    elif reading:
        detail = None
        if observed.get("count") or forecast.get("count"):
            detail = (f"pilots {observed.get('reading', 'unknown')} · "
                      f"forecast {forecast.get('reading', 'unknown')}")
        beats.append(_beat(
            CRITIC, "Decision",
            f"Turbulence reading along the selected corridor: {reading}. "
            f"Where the two sources differ this is the worse of them, which "
            f"is the direction a nervous passenger cares about.",
            detail, pause=BEAT_LONG))

    if out.get("contested"):
        beats.append(_beat(
            CRITIC, "Guardrail",
            "Two corridors scored within a hair of each other but disagree "
            "about the ride. Both are reported and the worse reading is used. "
            "Averaging them would produce a number neither source supports.",
            kind="caution", pause=BEAT))

    # ---------------------------------------------------------- reputation
    rep = payload.get("reputation")
    craft = payload.get("aircraft") or {}
    if rep and rep.get("available"):
        cov = rep.get("coverage", {})
        beats.append(_beat(
            STATE, "Retrieval",
            f"Separately, looked up the safety record for the "
            f"{craft.get('variant') or 'aircraft'}. The type was matched "
            f"exactly before any search ran, so a MAX query cannot return a "
            f"Next Generation aircraft by resemblance.",
            f"{craft.get('icao_designator')} to {rep.get('searched_as')} · "
            f"{cov.get('cases_variant_with_text', 0)} of "
            f"{cov.get('cases_variant', 0)} cases have a written report",
            pause=BEAT))
    elif rep:
        beats.append(_beat(
            STATE, "Guardrail",
            "The safety record could not be retrieved, and the agent reports "
            "that rather than leaving the panel silently empty.",
            rep.get("reason"), kind="caution", pause=BEAT))

    # ---------------------------------------------------------- explainer
    ex = payload.get("explanation") or {}
    if ex.get("enabled"):
        if ex.get("source") == "model":
            beats.append(_beat(
                STATE, "Explanation",
                "A language model wrote the paragraph you are reading. It was "
                "given the finished numbers and allowed only to restate them, "
                "then checked: an explanation that invented a severity, "
                "reassured, or hid a caveat would have been discarded.",
                f"accepted · {ex.get('model')}", pause=BEAT))
        elif ex.get("rejected"):
            beats.append(_beat(
                CRITIC, "Guardrail",
                "The model's explanation was discarded and the plain summary "
                "used instead. The written version is an improvement on the "
                "wording, never a substitute for the evidence, so a failed "
                "check costs prose rather than accuracy.",
                "; ".join(str(r) for r in ex["rejected"][:2]),
                kind="caution", pause=BEAT_LONG))

    beats.append(_beat(
        CONTROLLER, "Summary",
        f"Done. {len(corridors)} corridors considered, "
        f"{len([c for c in corridors if c.get('kept')])} kept, "
        f"{out.get('calls_used', 0)} API calls spent. Every discarded branch "
        f"is still on screen with the reason it was discarded.",
        pause=BEAT_LONG))

    return beats
