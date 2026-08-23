# install-to: app/reasoning
"""
The explainer: the one place a language model talks to the passenger.

Everything upstream of this module is deterministic. Corridors are scored by
arithmetic, pruning is a rule, and severity comes from pilot reports and
forecast polygons. This module takes that finished result and writes the
paragraph a nervous flier actually reads.

It is the last step on purpose. A wrong sentence here degrades an
explanation; a wrong number anywhere upstream corrupts a score. That is the
whole reason the model sits at the edge rather than inside the loop.

FOUR RULES, EACH ENFORCED RATHER THAN REQUESTED

  IT MAY NOT INVENT A SEVERITY. The reading comes from the critic. The
  explainer states it. Output naming a severity the evidence does not hold
  is rejected, not repaired.

  IT MAY NOT SOFTEN. No "probably fine", no "should be smooth", no
  reassurance. Thin evidence is the case where comfort language is most
  tempting and least earned, and it is the case this agent exists to
  handle.

  IT MUST CARRY THE UNCERTAINTY. If coverage is a fifth of the route, the
  paragraph says so. If two corridors disagree, it says both. Fluent prose
  is exactly how a caveat gets smoothed away, so the caveats are checked
  for rather than hoped for.

  IT FAILS TO THE DETERMINISTIC SUMMARY. If the model is unavailable, slow,
  or produces something that fails validation, the reader gets the existing
  `evidence.summary` instead. This makes the explainer an enhancement, not
  a dependency: an outage degrades the prose, never the answer.

The client is injectable, so the whole module tests offline against a fake.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from app.logging_setup import get_logger, kv
from app.reasoning.critic import Severity

log = get_logger("explainer")

DEFAULT_MODEL = os.environ.get("TURBULENCE_EXPLAINER_MODEL",
                               "claude-sonnet-5")
DEFAULT_MAX_TOKENS = 400
DEFAULT_TIMEOUT_SECONDS = 20.0

#: Words that promise comfort. The model is told not to use them and the
#: output is checked afterwards, because a rule only in a prompt is a
#: request rather than a constraint.
SOFTENING = (
    "probably fine", "should be fine", "should be smooth", "nothing to worry",
    "no need to worry", "don't worry", "do not worry", "rest easy",
    "you'll be fine", "you will be fine", "perfectly safe", "quite safe",
    "very safe", "no cause for concern", "not a concern", "sit back and relax",
    "smooth sailing", "clear skies", "expect a comfortable", "should be okay",
    "should be ok", "reassur",
)

#: Severity words the model must not use unless the evidence holds them.
SEVERITY_WORDS = {
    "smooth": Severity.SMOOTH,
    "light": Severity.LIGHT,
    "moderate": Severity.MODERATE,
    "severe": Severity.SEVERE,
    "extreme": Severity.EXTREME,
}

SYSTEM_PROMPT = """\
You write one short paragraph for an anxious air passenger, explaining what \
a turbulence assessment found.

You are given structured facts. You may only restate them. You may not add \
weather knowledge, aviation knowledge, or any judgement of your own.

Hard rules:
1. State only the severity level given to you. Never name a different level, \
and never say a level is likely, expected, or possible unless the facts say so.
2. Never reassure. Do not write that the flight will be fine, smooth, safe, \
comfortable, or nothing to worry about. You are describing evidence, not \
predicting an experience.
3. Always carry the uncertainty. If coverage is thin, say so. If sources \
disagree, say both. If nothing is known, say that plainly and do not fill \
the gap.
4. Write 3 to 5 sentences of plain prose. No lists, no headings, no bold. \
Address the reader as "you". Do not open with a greeting.

If the facts say the reading is unresolved, your paragraph must make clear \
that nothing is known about the air on this route, and that this is not the \
same as the air being calm."""


class ModelClient(Protocol):
    """Anything that turns a prompt into text."""

    def complete(self, system: str, user: str) -> str: ...


@dataclass
class AnthropicClient:
    """Real client. Imported lazily so this module loads without the SDK."""
    api_key: str | None = None
    model: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    timeout: float = DEFAULT_TIMEOUT_SECONDS
    _client: Any = field(default=None, repr=False)

    def _load(self):
        if self._client is None:
            import anthropic
            key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set")
            self._client = anthropic.Anthropic(api_key=key,
                                               timeout=self.timeout)
        return self._client

    #: Token counts from the last call. The API returns these on every
    #: response and they were being discarded with the rest of the object,
    #: which left the only measure of what this agent costs unavailable.
    last_usage: dict = field(default_factory=dict)

    def complete(self, system: str, user: str) -> str:
        message = self._load().messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        usage = getattr(message, "usage", None)
        self.last_usage = {
            "tokens_in": getattr(usage, "input_tokens", None),
            "tokens_out": getattr(usage, "output_tokens", None),
        }
        return "".join(block.text for block in message.content
                       if getattr(block, "type", None) == "text").strip()


# ------------------------------------------------------------------ facts


def build_facts(payload: dict[str, Any]) -> dict[str, Any]:
    """The structured facts the model is allowed to restate.

    Deliberately narrow. Anything not in here cannot appear in the output
    without failing validation, which is easier to enforce than trusting a
    prompt to constrain scope.
    """
    outcome = payload.get("outcome") or {}
    wx = outcome.get("turbulence") or {}
    observed = wx.get("observed") or {}
    forecast = wx.get("forecast") or {}
    request = payload.get("request") or {}
    aircraft = payload.get("aircraft") or {}

    kept = [c for c in payload.get("corridors") or [] if c.get("kept")]
    winner = next((c for c in kept if c.get("is_winner")), None)

    facts: dict[str, Any] = {
        "route": f"{request.get('origin')} to {request.get('dest')}",
        "reading": wx.get("reading") or outcome.get("reading"),
        "pilot_reports": {
            "reading": observed.get("reading"),
            "count": observed.get("count", 0),
            "average_age_minutes": observed.get("mean_age_minutes"),
        },
        "forecast": {
            "reading": forecast.get("reading"),
            "count": forecast.get("count", 0),
        },
        "sources_disagree": bool(wx.get("disagree")),
        "route_coverage_fraction": wx.get("coverage_fraction"),
        "corridors_considered": len(payload.get("corridors") or []),
        "corridors_kept": len(kept),
        "search_was_truncated": bool(outcome.get("truncated")),
        "plain_summary": wx.get("summary"),
    }
    if aircraft.get("variant"):
        facts["aircraft"] = aircraft["variant"]
    if winner and winner.get("altitude_min_ft"):
        facts["cruise_band"] = (
            f"FL{winner['altitude_min_ft'] // 100:03d} to "
            f"FL{(winner.get('altitude_max_ft') or 0) // 100:03d}")
    return facts


# ------------------------------------------------------------------ checks


@dataclass
class Verdict:
    """Whether a candidate explanation may be shown."""
    ok: bool
    reasons: list[str] = field(default_factory=list)


def validate(text: str, facts: dict[str, Any]) -> Verdict:
    """Check the output against the four rules.

    Rejection is the outcome, not repair. Editing a model's paragraph to
    remove a reassurance would leave the sentence around it built on the
    same assumption.
    """
    reasons: list[str] = []
    if not text or len(text.split()) < 15:
        return Verdict(False, ["the explanation is empty or too short"])
    if len(text.split()) > 220:
        reasons.append("the explanation is far longer than asked for")

    low = text.lower()

    for phrase in SOFTENING:
        if phrase in low:
            reasons.append(f"contains reassurance: {phrase!r}")

    allowed = {str(facts.get("reading") or "").lower(),
               str((facts.get("pilot_reports") or {}).get("reading") or "").lower(),
               str((facts.get("forecast") or {}).get("reading") or "").lower()}
    allowed.discard("")
    allowed.discard("unresolved")
    for word in SEVERITY_WORDS:
        if re.search(rf"\b{word}\b", low) and word not in allowed:
            # "not smooth" is the project's own phrasing and is not a claim
            # that the air is smooth.
            if word == "smooth" and re.search(r"not smooth|isn.t smooth", low):
                continue

            # A comparison between two named readings is not a claim about
            # a third. Observed in production: a paragraph that correctly
            # reported moderate, explained the disagreement and gave the
            # coverage caveat was discarded for the phrase "the more severe
            # of the two, moderate, is the one used".
            #
            # Narrow deliberately. The comparative has to be followed by
            # "of" or "than", so it is comparing things already named.
            # "more severe turbulence is expected" predicts rather than
            # compares and still fails.
            if re.search(rf"\b(?:more|less|most)\s+{word}\s+(?:of|than)\b",
                         low):
                continue
            reasons.append(f"names a severity the evidence does not hold: "
                           f"{word!r}")

    if str(facts.get("reading")).lower() == "unresolved":
        if not re.search(r"not known|nothing is known|no .*(report|forecast)"
                         r"|unknown|not the same as", low):
            reasons.append("does not make clear that nothing is known")

    if facts.get("sources_disagree") and "disagree" not in low:
        reasons.append("does not mention that the sources disagree")

    coverage = facts.get("route_coverage_fraction")
    if isinstance(coverage, (int, float)) and 0 < coverage < 0.34:
        if not re.search(r"cover|only part|much of the route|most of the route",
                         low):
            reasons.append("does not mention how little of the route is covered")

    return Verdict(not reasons, reasons)


# ------------------------------------------------------------------ agent


@dataclass
class Explanation:
    text: str
    source: str                 # "model" or "deterministic"
    model: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    rejected: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)
    #: What the model actually wrote when its output was rejected. Kept so
    #: a discarded explanation can be read afterwards. The reason alone -
    #: "names a severity the evidence does not hold" - says which rule
    #: fired but not what was nearly shown to a passenger, and that is the
    #: part worth reviewing.
    discarded_text: str | None = None


def _log_exchange(facts: dict, text: str | None) -> None:
    """Record what went to the model and what came back.

    Off unless TURBULENCE_LOG_EXPLAINER_IO is set, for the same reason trip
    content is off: the facts carry the route, and origin plus destination
    plus a time is an itinerary.

    A rejected response is already logged in full, because studying the
    failure mode is the point of keeping it. This adds the two halves that
    were missing: the prompt, and the text of a response that was accepted.
    """
    if os.environ.get("TURBULENCE_LOG_EXPLAINER_IO", "").lower() not in (
            "1", "true", "yes"):
        return
    log.info("explainer prompt " + kv(facts=json.dumps(facts, default=str)))
    if text is not None:
        log.info("explainer response " + kv(text=text.strip()))


def explain(payload: dict[str, Any], client: ModelClient | None = None,
            model_name: str | None = None) -> Explanation:
    """Write the passenger-facing paragraph, or fall back to the plain one.

    The fallback is not an error path. A search whose model call failed
    still has a complete, honest answer, because the deterministic summary
    was always going to be there.
    """
    facts = build_facts(payload)
    fallback = (facts.get("plain_summary")
                or "No turbulence assessment is available for this route.")

    if client is None:
        return Explanation(text=fallback, source="deterministic", facts=facts)

    user = ("Facts about this turbulence assessment:\n\n"
            + json.dumps(facts, indent=2)
            + "\n\nWrite the paragraph.")

    try:
        text = client.complete(SYSTEM_PROMPT, user)
    except Exception as e:  # noqa: BLE001 - an outage degrades prose, not truth
        log.warning("explainer call failed, using the plain summary "
                    + kv(error=type(e).__name__,
                         reading=facts.get("reading")))
        return Explanation(text=fallback, source="deterministic", facts=facts,
                           rejected=[f"model call failed: {type(e).__name__}"])

    verdict = validate(text, facts)
    if not verdict.ok:
        # An explanation that failed the checks is the most interesting
        # thing this agent produces. Log the reasons at warning and the
        # text itself at info, so a reviewer can see what was nearly shown
        # rather than only which rule caught it.
        _log_exchange(facts, None)
        log.warning("explainer output rejected "
                    + kv(reasons="; ".join(verdict.reasons),
                         reading=facts.get("reading"),
                         model=model_name or DEFAULT_MODEL))
        log.info("explainer discarded text " + kv(text=text.strip()))
        return Explanation(text=fallback, source="deterministic", facts=facts,
                           rejected=verdict.reasons, discarded_text=text.strip())

    _log_exchange(facts, text)
    usage = getattr(client, "last_usage", None) or {}
    log.info("explainer output accepted "
             + kv(reading=facts.get("reading"), words=len(text.split()),
                  tokens_in=usage.get("tokens_in"),
                  tokens_out=usage.get("tokens_out"),
                  model=model_name or DEFAULT_MODEL))
    return Explanation(text=text.strip(), source="model",
                       model=model_name or DEFAULT_MODEL, facts=facts,
                       tokens_in=usage.get("tokens_in"),
                       tokens_out=usage.get("tokens_out"))
