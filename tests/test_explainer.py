"""Tests for the explainer.

This is the only place a language model speaks to the passenger, so the
rules are enforced here rather than trusted to a prompt. No network: the
model client is injected.
"""

import pytest

from app.reasoning.explainer import (
    SOFTENING,
    SYSTEM_PROMPT,
    Explanation,
    build_facts,
    explain,
    validate,
)


def payload(reading="moderate", observed="unresolved", obs_count=0,
            forecast="moderate", fc_count=1, disagree=False, coverage=0.0,
            summary="Deterministic summary of the assessment.",
            truncated=False):
    return {
        "request": {"origin": "KIAD", "dest": "KLAX"},
        "outcome": {
            "reading": reading, "truncated": truncated,
            "turbulence": {
                "reading": reading,
                "observed": {"reading": observed, "count": obs_count,
                             "mean_age_minutes": 25 if obs_count else None},
                "forecast": {"reading": forecast, "count": fc_count},
                "disagree": disagree, "coverage_fraction": coverage,
                "summary": summary,
            },
        },
        "corridors": [{"id": "track", "kept": True, "is_winner": True,
                       "altitude_min_ft": 32000, "altitude_max_ft": 34000}],
        "aircraft": {"variant": "A321neo"},
    }


class FakeClient:
    """Returns whatever it is told to, so the rules can be exercised."""

    def __init__(self, reply="", raises=None):
        self.reply = reply
        self.raises = raises
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        if self.raises:
            raise self.raises
        return self.reply


GOOD_MODERATE = (
    "A turbulence forecast covers the route you are flying, and it calls for "
    "moderate conditions at cruise altitude. No pilot flying this corridor "
    "has reported what the air was actually like, so the forecast is the "
    "only source here. A forecast describes a broad area over several hours "
    "rather than what any one aircraft felt. You are seeing what is "
    "expected, not a measurement.")

GOOD_UNRESOLVED = (
    "Nothing is known about the air on this route right now. No turbulence "
    "forecast covers it, and no pilot flying it at cruise has filed a report "
    "in the last few hours. An absence of information is not the same as "
    "calm air, so this is genuinely unknown rather than quiet.")


class TestFacts:
    def test_only_structured_facts_are_sent(self):
        facts = build_facts(payload())
        assert facts["reading"] == "moderate"
        assert facts["forecast"]["count"] == 1
        assert facts["pilot_reports"]["reading"] == "unresolved"

    def test_the_deterministic_summary_travels_with_the_facts(self):
        """It is the fallback, so it has to be there before the call."""
        assert build_facts(payload())["plain_summary"]

    def test_the_cruise_band_is_included_when_known(self):
        assert build_facts(payload())["cruise_band"] == "FL320 to FL340"


class TestItMayNotInventASeverity:
    def test_a_severity_the_evidence_lacks_is_rejected(self):
        text = GOOD_MODERATE.replace(
            "moderate conditions",
            "moderate conditions, with severe patches near the mountains")
        v = validate(text, build_facts(payload()))
        assert not v.ok
        assert any("severity" in r for r in v.reasons)

    def test_the_readings_that_are_present_are_allowed(self):
        facts = build_facts(payload(observed="light", obs_count=3,
                                    disagree=True, coverage=0.5))
        text = ("Pilots reported light conditions while the forecast calls "
                "for moderate, so the two sources disagree. The worse of the "
                "two is used here rather than an average. Their reports cover "
                "about half the route.")
        assert validate(text, facts).ok

    def test_the_projects_own_phrase_is_not_a_severity_claim(self):
        """"Unresolved is not smooth" must not read as claiming smooth."""
        facts = build_facts(payload(reading="unresolved", forecast="unresolved",
                                    fc_count=0))
        text = ("Nothing is known about the air on this route. No forecast "
                "covers it and no pilot has reported along it recently. That "
                "is not smooth air, it is an absence of information.")
        assert validate(text, facts).ok


class TestItMayNotSoften:
    @pytest.mark.parametrize("phrase", [
        "you should be fine", "nothing to worry about",
        "it will be perfectly safe", "sit back and relax",
        "expect a comfortable ride",
    ])
    def test_reassurance_is_rejected(self, phrase):
        text = GOOD_MODERATE + " Overall, " + phrase + "."
        v = validate(text, build_facts(payload()))
        assert not v.ok
        assert any("reassurance" in r for r in v.reasons)

    def test_the_prompt_forbids_it_too(self):
        """Belt and braces: the rule is stated and then checked."""
        assert "Never reassure" in SYSTEM_PROMPT

    def test_the_banned_list_is_not_empty(self):
        assert len(SOFTENING) > 10


class TestItMustCarryTheUncertainty:
    def test_an_unresolved_reading_must_say_nothing_is_known(self):
        facts = build_facts(payload(reading="unresolved", forecast="unresolved",
                                    fc_count=0))
        evasive = ("The system considered four possible corridors and chose "
                   "the one this aircraft most likely flies. It looked at "
                   "pilot reports and forecasts along that path. The "
                   "assessment is complete for your route.")
        v = validate(evasive, facts)
        assert not v.ok
        assert any("nothing is known" in r for r in v.reasons)

    def test_a_good_unresolved_paragraph_passes(self):
        facts = build_facts(payload(reading="unresolved", forecast="unresolved",
                                    fc_count=0))
        assert validate(GOOD_UNRESOLVED, facts).ok

    def test_a_disagreement_must_be_mentioned(self):
        facts = build_facts(payload(observed="light", obs_count=3,
                                    disagree=True, coverage=0.5))
        hiding = ("The assessment came back as moderate at cruise altitude. "
                  "Pilots have filed reports and a forecast covers the same "
                  "airspace. Both were taken into account when producing "
                  "this reading for your route.")
        v = validate(hiding, facts)
        assert not v.ok
        assert any("disagree" in r for r in v.reasons)

    def test_thin_coverage_must_be_mentioned(self):
        facts = build_facts(payload(observed="light", obs_count=2,
                                    forecast="light", coverage=0.2))
        silent = ("Pilots flying this route reported light conditions and the "
                  "forecast agrees. Both sources point the same way at cruise "
                  "altitude for this aircraft. That is what the assessment "
                  "found for your flight today.")
        v = validate(silent, facts)
        assert not v.ok
        assert any("how little of the route" in r for r in v.reasons)


class TestItFailsToTheDeterministicSummary:
    def test_no_client_means_the_plain_summary(self):
        out = explain(payload(), client=None)
        assert out.source == "deterministic"
        assert out.text == "Deterministic summary of the assessment."

    def test_a_model_outage_degrades_prose_not_truth(self):
        out = explain(payload(),
                      client=FakeClient(raises=ConnectionError("down")))
        assert out.source == "deterministic"
        assert out.text == "Deterministic summary of the assessment."
        assert any("model call failed" in r for r in out.rejected)

    def test_a_rejected_explanation_falls_back_with_reasons(self):
        bad = GOOD_MODERATE + " You should be fine."
        out = explain(payload(), client=FakeClient(reply=bad))
        assert out.source == "deterministic"
        assert out.rejected
        assert any("reassurance" in r for r in out.rejected)

    def test_an_accepted_explanation_is_used(self):
        out = explain(payload(), client=FakeClient(reply=GOOD_MODERATE))
        assert out.source == "model"
        assert out.text == GOOD_MODERATE

    def test_the_fallback_is_never_empty(self):
        empty = {"request": {}, "outcome": {}, "corridors": []}
        out = explain(empty, client=FakeClient(raises=RuntimeError("x")))
        assert out.text


class TestTheModelSeesOnlyFacts:
    def test_the_prompt_carries_the_structured_facts(self):
        client = FakeClient(reply=GOOD_MODERATE)
        explain(payload(), client=client)
        system, user = client.calls[0]
        assert "moderate" in user
        assert "KIAD to KLAX" in user

    def test_the_system_prompt_states_all_four_rules(self):
        for rule in ("State only the severity level given to you",
                     "Never reassure",
                     "Always carry the uncertainty",
                     "not the same as the air being calm"):
            assert rule in SYSTEM_PROMPT


class TestRejectedOutputIsRecoverable:
    """The reason names which rule fired. The text says what was nearly
    shown to a passenger, and that is the part worth reviewing."""

    def test_the_discarded_text_is_kept(self):
        bad = GOOD_MODERATE + " Some segments could see severe conditions."
        out = explain(payload(), client=FakeClient(reply=bad))
        assert out.source == "deterministic"
        assert out.discarded_text == bad

    def test_an_accepted_explanation_discards_nothing(self):
        out = explain(payload(), client=FakeClient(reply=GOOD_MODERATE))
        assert out.discarded_text is None

    def test_a_failed_call_has_nothing_to_discard(self):
        out = explain(payload(),
                      client=FakeClient(raises=ConnectionError("down")))
        assert out.discarded_text is None

    def test_the_rejection_is_logged_with_its_reason(self):
        import io
        from app.logging_setup import configure
        buf = io.StringIO()
        configure(level="DEBUG", use_syslog=False, stream=buf, force=True)
        bad = GOOD_MODERATE + " You should be fine."
        explain(payload(), client=FakeClient(reply=bad))
        logged = buf.getvalue()
        assert "explainer output rejected" in logged
        assert "reassurance" in logged

    def test_the_discarded_text_reaches_the_log(self):
        import io
        from app.logging_setup import configure
        buf = io.StringIO()
        configure(level="DEBUG", use_syslog=False, stream=buf, force=True)
        bad = GOOD_MODERATE + " Some segments could see severe conditions."
        explain(payload(), client=FakeClient(reply=bad))
        assert "Some segments could see severe" in buf.getvalue()
