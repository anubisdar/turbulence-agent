"""Tests that the prose this system writes agrees with the data behind it.

This agent explains itself. The Agent Processing View narrates each step,
the evidence layer writes a plain summary of what was found, and the status
page captions its own charts. All of that text is generated from data, and
all of it can contradict the data it was generated from.

It has. The status page carried the sentence "Scoring is 14.49s of 14.2s.
The part that decides the answer is the part that costs nothing" directly
above a chart showing scoring as the largest bar. The claim was true when it
was written and false by the time it shipped, because the number underneath
it changed and the sentence did not.

There is an asymmetry worth naming. The explainer, which is a language
model, is held to four rules and its output is rejected when it breaks them:
no invented severity, no reassurance, no hidden caveat, no smoothing an
absence. The prose this system writes itself is held to none of them, even
though it makes the same claims to the same reader and is trusted more
precisely because a person wrote the template.

These tests apply the explainer's rules to everything else that speaks.
"""

import re

import pytest

from app.reasoning.critic import Evidence, Severity
from app.reasoning.evidence import (
    SENSATION,
    explain_absence,
    explain_reading,
)
from app.reasoning.geometry import build_corridor, great_circle
from app.web.narrate import narrate

#: The phrases the explainer's validator rejects. Applied here to text this
#: system writes for itself.
REASSURANCE = (
    "should be fine", "will be fine", "nothing to worry", "no need to worry",
    "don't worry", "do not worry", "perfectly safe", "quite safe",
    "rest easy", "sit back and relax", "smooth sailing", "no cause for concern",
)

SEVERITY_WORDS = ("smooth", "light", "moderate", "severe", "extreme")


def _corridors(count, winner):
    """A corridor list of the right size. The narrator counts corridors from
    this list and nodes from the outcome, and a payload where the two
    disagree is not one the search can produce."""
    ids = ["track", "filed", "alternate", "gc",
           "track/high", "track/low", "filed/high", "filed/low"]
    out = []
    for i in range(count):
        cid = ids[i % len(ids)]
        out.append({
            "id": cid, "depth": 1 if "/" not in cid else 2,
            "kept": i < 2, "score": 0.7 - i * 0.05,
            "components": {"provenance": 1.0, "geometry": 0.9,
                           "agreement": 0.5, "coverage": 0.2},
            "decision": "keep" if i < 2 else "prune_beam",
            "reason": "" if i < 2 else "outside beam width 2",
            "provenance": "actual_track",
            "is_winner": cid == winner,
        })
    return out


def payload(reading="moderate", observed="unresolved", obs_count=0,
            forecast="moderate", fc_count=1, disagree=False, coverage=0.2,
            nodes=8, calls=8, winner="track/high", truncated=False,
            degraded=False, stop="depth_limit", corridors=None):
    """A finished search, shaped as the narrator receives it."""
    return {
        "request": {"origin": "KPIT", "dest": "KBOS", "beam_width": 2,
                    "depth_limit": 2, "max_tool_calls": 16},
        "outcome": {
            "stop": stop, "nodes_generated": nodes, "calls_used": calls,
            "depth_reached": 2, "winner": winner, "reading": reading,
            "elapsed_seconds": 14.2, "truncated": truncated,
            "degraded": degraded, "degraded_reasons": [],
            "turbulence": {
                "available": reading != "unresolved", "reading": reading,
                "observed": {"reading": observed, "count": obs_count,
                             "mean_age_minutes": 25 if obs_count else None},
                "forecast": {"reading": forecast, "count": fc_count},
                "disagree": disagree, "coverage_fraction": coverage,
            },
        },
        "corridors": (corridors if corridors is not None
                      else _corridors(nodes, winner)),
        "overlaps": [],
        "fix_cache": {"before": 27, "after": 27, "by_type": {}},
    }


def narration_text(data) -> str:
    return " ".join(b.get("text", "") for b in narrate(data))


class TestNarrationNeverReassures:
    """The rule the explainer is held to, applied to the narrator."""

    @pytest.mark.parametrize("reading", [
        "unresolved", "smooth", "light", "moderate", "severe", "extreme"])
    def test_no_beat_offers_comfort(self, reading):
        text = narration_text(payload(reading=reading)).lower()
        for phrase in REASSURANCE:
            assert phrase not in text, (
                f"the narration reassures on a {reading} reading: {phrase!r}")

    def test_no_beat_offers_comfort_when_a_source_failed(self):
        text = narration_text(payload(reading="unresolved", degraded=True,
                                      forecast="unresolved", fc_count=0)).lower()
        for phrase in REASSURANCE:
            assert phrase not in text


class TestNarrationNeverInventsASeverity:
    """A beat naming a severity the evidence does not hold is the same
    defect the explainer's validator exists to catch."""

    @pytest.mark.parametrize("reading,observed,forecast", [
        ("moderate", "light", "moderate"),
        ("light", "light", "unresolved"),
        ("unresolved", "unresolved", "unresolved"),
    ])
    def test_only_severities_in_evidence_appear(self, reading, observed,
                                                forecast):
        data = payload(reading=reading, observed=observed, forecast=forecast,
                       obs_count=2 if observed != "unresolved" else 0,
                       fc_count=1 if forecast != "unresolved" else 0)
        text = narration_text(data).lower()
        held = {reading, observed, forecast} - {"unresolved"}
        for word in SEVERITY_WORDS:
            if word in held:
                continue
            # "not smooth" is this project's own phrasing for an absence and
            # is not a claim that the air is smooth.
            occurrences = [m for m in re.finditer(rf"\b{word}\b", text)
                           if not text[max(0, m.start() - 4):m.start()]
                           .endswith("not ")]
            assert not occurrences, (
                f"the narration says {word!r} when the evidence holds "
                f"{sorted(held) or 'nothing'}")


class TestUnresolvedIsNeverDressedUp:
    """The rule this whole project turns on, applied to its own prose."""

    def test_an_unresolved_reading_says_so(self):
        text = narration_text(payload(reading="unresolved",
                                      forecast="unresolved", fc_count=0))
        assert "unresolved" in text.lower()

    def test_an_unresolved_reading_says_it_is_not_smooth(self):
        text = narration_text(payload(reading="unresolved",
                                      forecast="unresolved", fc_count=0))
        assert "not smooth" in text.lower() or "not the same as" in text.lower()

    def test_no_corridor_reads_differently_from_no_weather(self):
        """Two absences with different causes must not share a sentence.
        A beat claiming the agent found a corridor, on a run where none was
        generated, contradicts the termination beat two lines above it."""
        no_corridor = narration_text(payload(
            reading="unresolved", winner=None, nodes=0, calls=2,
            stop="generator_returned_nothing", forecast="unresolved",
            fc_count=0))
        no_weather = narration_text(payload(
            reading="unresolved", winner="track/high", nodes=8,
            forecast="unresolved", fc_count=0))
        assert "found the corridor" not in no_corridor
        assert no_corridor != no_weather


class TestNarrationMatchesItsNumbers:
    """A beat that states a count must state the right one. The status page
    caption that failed did exactly this: it named a figure that its own
    chart contradicted."""

    def test_the_corridor_count_matches_the_nodes_explored(self):
        """The narrator counts corridors from the corridor list and nodes
        from the outcome. Nothing else checks that those two agree, and a
        payload where they diverge produces a summary claiming zero
        corridors were considered on a search that explored six."""
        for nodes in (2, 6, 8):
            text = narration_text(payload(nodes=nodes))
            assert str(nodes) in text, (
                f"the narration never mentions the {nodes} corridors")

    def test_a_mismatched_payload_is_visible_rather_than_silent(self):
        """If the two ever do diverge, the summary should not quietly
        report the wrong one. This documents current behaviour: it reports
        the corridor list and ignores the node count."""
        data = payload(nodes=6, corridors=[])
        text = narration_text(data)
        assert "0 corridors considered" in text, (
            "the summary counts the corridor list; if this changes, the "
            "narration and the outcome have been reconciled")

    def test_the_call_count_is_the_payload_count(self):
        for calls in (3, 8, 14):
            text = narration_text(payload(calls=calls))
            assert str(calls) in text

    def test_no_beat_names_a_count_that_is_not_in_the_payload(self):
        """Guards the inverse: a hardcoded number in a template survives
        every change to the data underneath it."""
        data = payload(nodes=7, calls=11, obs_count=3, fc_count=2)
        allowed = {"7", "11", "3", "2", "2026",
                   str(data["request"]["beam_width"]),
                   str(data["request"]["depth_limit"]),
                   str(data["request"]["max_tool_calls"]),
                   str(data["outcome"]["depth_reached"]),
                   "14", "0", "1", "4", "25", "80", "40", "20", "15", "60",
                   "27", "100", "5"}
        for beat in narrate(data):
            # Decimals first: a score of 0.7000 would otherwise register as
            # the integer 7000.
            text = re.sub(r"\d+\.\d+", " ", beat.get("text", ""))
            for number in re.findall(r"\b\d+\b", text):
                assert number in allowed, (
                    f"a beat names {number!r}, which is not in the payload: "
                    f"{beat['text'][:90]}")

    def test_the_winner_named_is_the_winner_chosen(self):
        for winner in ("track/high", "filed", "gc"):
            text = narration_text(payload(winner=winner))
            assert winner in text


class TestTruncatedAndDegradedAreDistinct:
    """Different causes, different sentences. A budget stopping a search and
    a source failing during one both give a partial answer, and a reader who
    cannot tell them apart cannot judge the result."""

    def test_a_truncated_search_says_a_budget_ran_out(self):
        text = narration_text(payload(truncated=True, stop="tool_budget"))
        assert "budget" in text.lower()

    def test_a_degraded_search_says_a_source_failed(self):
        data = payload(degraded=True)
        data["outcome"]["degraded_reasons"] = [
            "Could not list flights on this pair: rate limited twice"]
        text = narration_text(data)
        assert "data source failed" in text.lower()

    def test_they_do_not_share_wording(self):
        truncated = narration_text(payload(truncated=True, stop="tool_budget"))
        data = payload(degraded=True)
        data["outcome"]["degraded_reasons"] = ["Could not fetch the track"]
        degraded = narration_text(data)
        assert truncated != degraded


class TestTheSensationDescriptionsAreHeldToTheSameRules:
    """These are the most consequential sentences the system writes: they
    tell a nervous passenger what a severity level feels like."""

    def test_none_of_them_reassures(self):
        for level, text in SENSATION.items():
            low = text.lower()
            for phrase in REASSURANCE:
                assert phrase not in low, f"{level}: {phrase!r}"

    def test_each_one_describes_only_its_own_level(self):
        """The description of light turbulence must not mention severe."""
        for level, text in SENSATION.items():
            low = text.lower()
            others = {w for w in SEVERITY_WORDS
                      if w != level.value and w in low}
            assert not others, (
                f"the {level.value} description mentions {sorted(others)}")

    def test_they_describe_effects_rather_than_predictions(self):
        """"Occupants feel", not "you will feel". The distinction is that
        one describes a category and the other promises an experience."""
        for level, text in SENSATION.items():
            assert " you will " not in text.lower(), level
            assert " you'll " not in text.lower(), level


class TestAbsenceExplanationsNeverImplyCalm:
    """Written to be friendlier than "unresolved", which is exactly the
    place reassurance creeps in."""

    @pytest.fixture
    def short(self):
        return build_corridor(great_circle((40.49, -80.23), (42.36, -71.01), 16))

    @pytest.fixture
    def long(self):
        return build_corridor(great_circle((38.94, -77.46), (33.94, -118.41), 16))

    @pytest.mark.parametrize("args", [
        (0, 0, 0, 0), (21, 0, 1, 0), (12, 0, 3, 1), (5, 0, 0, 0),
    ])
    def test_no_variant_mentions_calm_air(self, short, args):
        text = explain_absence(short, *args).lower()
        for word in ("smooth", "calm", "clear", "fine", "gentle"):
            assert word not in text, f"{word!r} in: {text}"

    def test_a_short_route_is_described_as_short(self, short):
        assert "short" in explain_absence(short, 21, 0, 1, 0).lower()

    def test_a_long_route_is_not(self, long):
        assert "short hop" not in explain_absence(long, 21, 0, 1, 0).lower()

    def test_nothing_fetched_reads_differently_from_nothing_found(self, short):
        assert explain_absence(short, 0, 0, 0, 0) != \
            explain_absence(short, 21, 0, 1, 0)


class TestReadingExplanationsMatchTheirEvidence:
    """The deterministic counterpart to the explainer's validator."""

    @pytest.fixture
    def shape(self):
        return build_corridor(great_circle((40.49, -80.23), (42.36, -71.01), 16))

    def test_a_stated_count_matches_the_evidence(self, shape):
        for count in (1, 3, 7):
            evidence = Evidence(reading=Severity.LIGHT,
                                observed_reading=Severity.LIGHT,
                                observed_count=count)
            assert str(count) in explain_reading(evidence, shape)

    def test_a_stated_coverage_matches_the_evidence(self, shape):
        evidence = Evidence(reading=Severity.LIGHT,
                            observed_reading=Severity.LIGHT,
                            observed_count=2, coverage_fraction=0.2)
        assert "20%" in explain_reading(evidence, shape)

    def test_no_severity_appears_that_the_evidence_lacks(self, shape):
        evidence = Evidence(reading=Severity.LIGHT,
                            observed_reading=Severity.LIGHT,
                            observed_count=2, coverage_fraction=0.5)
        text = explain_reading(evidence, shape).lower()
        for word in ("severe", "extreme", "moderate"):
            assert not re.search(rf"\b{word}\b", text), (
                f"{word!r} appears in an explanation of a light reading")

    def test_a_disagreement_is_never_silently_resolved(self, shape):
        evidence = Evidence(reading=Severity.MODERATE,
                            observed_reading=Severity.LIGHT, observed_count=3,
                            forecast_reading=Severity.MODERATE,
                            forecast_count=1, coverage_fraction=0.5)
        text = explain_reading(evidence, shape).lower()
        assert "disagree" in text
        assert "average" in text

    def test_an_unresolved_reading_gets_no_sensation_text(self, shape):
        assert explain_reading(Evidence(reading=Severity.UNRESOLVED),
                               shape) == ""
