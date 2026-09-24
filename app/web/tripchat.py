# install-to: app/web
"""
Conversational intake: turns a back-and-forth into the trip fields the
corridor search already required.

Nothing downstream changes. `CorridorSearchBody` still takes `origin`,
`dest`, `departure_date`, `departure_time` as plain strings - the same
four values the form's text boxes used to hold. This module's only job is
to arrive at those values through a chat instead of four inputs, and to
know when it hasn't yet.

THE MODEL NEVER INVENTS AN AIRPORT CODE THAT REACHES THE SEARCH. It may
propose one - "Pittsburgh" becomes a guess at "PIT" - but that guess is
checked against `app.retrieval.airports.resolve_airport`, the same
function the form's submission already ran through server-side. A
proposal that doesn't resolve to a plausible code is not silently
dropped or passed on hoping the search will explain the gap; it becomes
a question, here, before a call is ever spent on it.

THE MODEL NEVER MARKS A TRIP COMPLETE ON ITS OWN SAY-SO. `complete` is
decided in `respond`, by checking that both ends resolved and differ -
not by trusting a boolean the model returns. A model that calls a trip
finished when it isn't would reproduce, one layer up, the exact failure
this whole agent exists to avoid: a confident answer standing in for a
missing one.

FAILS TO A PLAIN QUESTION. No key, no SDK, a timeout, or output that
doesn't parse all fall back to asking for origin and destination in the
plainest possible terms. The chat degrades to something less pleasant to
use; it never blocks the trip on the model being available.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.logging_setup import get_logger, kv, trip_fields
from app.retrieval.airports import Airport, resolve_pair

log = get_logger("tripchat")

DEFAULT_MODEL = os.environ.get("TURBULENCE_CHAT_MODEL", "claude-sonnet-5")
MAX_TOKENS = 500
TIMEOUT_SECONDS = 20.0

#: A runaway conversation replays its whole history to the model on every
#: turn. Bounded rather than trimmed, because trimming a trip conversation
#: silently could drop the one message that named the destination.
MAX_MESSAGES = 40

SYSTEM_PROMPT = """\
You are the intake step in front of a turbulence-aware flight search. \
Your only job is to work out four things from the conversation: an origin \
airport, a destination airport, a departure date, and a departure time in \
UTC. Nothing else about the search - not the aircraft, not turbulence, not \
what the report should look like - is yours to ask about.

For origin and destination, give your best reading of what the airport is \
- a city name, an airport name, or a code - as a short string. You do not \
decide whether it's a valid airport; a system downstream of you resolves \
it and will tell you if it couldn't. If a city has more than one major \
airport and the user hasn't said which, ask rather than guess.

Departure date and time are useful but not required to proceed - if the \
user never gives them, leave both null rather than asking again. Do not \
invent a date or time yourself.

Always respond by calling record_trip, exactly once, with everything you \
currently know. Leave a field null if you don't have it. Set `question` \
to the single next thing you'd ask the user, or null if you have enough \
to move on. Ask about at most one thing per turn."""

TOOL_NAME = "record_trip"

TOOL = {
    "name": TOOL_NAME,
    "description": "Record what is known so far about the requested trip.",
    "input_schema": {
        "type": "object",
        "properties": {
            "origin": {
                "type": ["string", "null"],
                "description": "Origin as typed or inferred - a city, "
                              "airport name, IATA or ICAO code. Not yet "
                              "resolved to a real airport.",
            },
            "dest": {
                "type": ["string", "null"],
                "description": "Destination, same rules as origin.",
            },
            "departure_date": {
                "type": ["string", "null"],
                "description": "YYYY-MM-DD, or null if not given.",
            },
            "departure_time": {
                "type": ["string", "null"],
                "description": "HH:MM in UTC, or null if not given.",
            },
            "question": {
                "type": ["string", "null"],
                "description": "The single next question for the user, "
                              "or null if nothing more is needed.",
            },
        },
        "required": ["origin", "dest", "departure_date",
                     "departure_time", "question"],
    },
}

#: Loose enough to catch "9/23", "2026-09-23", "Sept 23". Strict enough
#: that anything failing this is not silently forwarded to the search,
#: which requires exactly YYYY-MM-DD and would otherwise turn a good
#: extraction into a 422 the user never asked for.
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_HHMM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


class ChatClient(Protocol):
    """Anything that turns a system prompt and a message history into the
    tool's structured arguments."""

    def extract(self, system: str, messages: list[dict[str, str]]
               ) -> dict[str, Any]: ...


@dataclass
class AnthropicChatClient:
    """Real client. Imported lazily so this module loads without the SDK."""

    api_key: str | None = None
    model: str = DEFAULT_MODEL
    max_tokens: int = MAX_TOKENS
    timeout: float = TIMEOUT_SECONDS
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

    def extract(self, system: str, messages: list[dict[str, str]]
               ) -> dict[str, Any]:
        message = self._load().messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            tools=[TOOL],
            tool_choice={"type": "tool", "name": TOOL_NAME},
        )
        for block in message.content:
            if getattr(block, "type", None) == "tool_use" \
                    and block.name == TOOL_NAME:
                return dict(block.input)
        raise ValueError("model did not call record_trip")


# ------------------------------------------------------------------ result


@dataclass
class ChatResult:
    #: What to show the user next: a clarifying question, or an
    #: acknowledgement once the trip is complete. None only in the
    #: fallback path's terse mode.
    reply: str
    complete: bool
    #: Present only when complete: the exact fields CorridorSearchBody
    #: takes, ready to submit unchanged.
    trip: dict[str, str] | None
    origin_resolution: Airport | None
    dest_resolution: Airport | None
    notes: list[str]
    source: str  # "model" or "fallback"


_FALLBACK_QUESTIONS = [
    "Where are you flying from? A city or an airport code works.",
    "And where are you headed?",
]


def _fallback(history_len: int) -> ChatResult:
    """No model available. Ask for origin, then destination, in order.

    Deliberately dumb: two fixed questions rather than an attempt to parse
    free text with rules, because a rule-based parser that gets a city
    name wrong fails the same way a bad model guess would, without any of
    the "ask again" behaviour that makes the model's version safe.
    """
    idx = min(history_len, len(_FALLBACK_QUESTIONS) - 1)
    return ChatResult(
        reply=_FALLBACK_QUESTIONS[idx],
        complete=False, trip=None,
        origin_resolution=None, dest_resolution=None,
        notes=["The trip assistant isn't available right now, so this is "
              "asking for origin and destination directly."],
        source="fallback",
    )


def _resolution_note(field_name: str, airport: Airport | None,
                     raw: str | None) -> str | None:
    if not raw:
        return None
    if airport is None:
        return (f"“{raw}” doesn't look like an airport - try a "
                f"city name, or a three- or four-letter code.")
    return None


def respond(history: list[dict[str, str]], client: ChatClient | None = None,
           model_name: str = DEFAULT_MODEL) -> ChatResult:
    """Advance the trip conversation by one turn.

    `history` is the full conversation so far, oldest first, each entry
    `{"role": "user" | "assistant", "content": str}`, ending with the
    user's latest message. Stateless: nothing is kept between calls, so
    the caller (the page) owns replaying it.
    """
    history = history[-MAX_MESSAGES:]

    if client is None:
        try:
            client = AnthropicChatClient(model=model_name)
            if not (client.api_key or os.environ.get("ANTHROPIC_API_KEY")):
                log.info("trip chat " + kv(outcome="no_api_key"))
                return _fallback(len(history))
        except Exception as e:  # noqa: BLE001
            log.warning("trip chat client unavailable "
                        + kv(error=type(e).__name__))
            return _fallback(len(history))

    try:
        args = client.extract(SYSTEM_PROMPT, history)
    except Exception as e:  # noqa: BLE001
        # Anything from here down is a provider or parsing failure, not a
        # statement about the trip - never shown as though the model had
        # an opinion about the airports.
        log.warning("trip chat extraction failed "
                    + kv(error=type(e).__name__, detail=str(e)[:200]))
        return _fallback(len(history))

    origin_raw = (args.get("origin") or "").strip() or None
    dest_raw = (args.get("dest") or "").strip() or None
    date = (args.get("departure_date") or "").strip() or None
    time_ = (args.get("departure_time") or "").strip() or None
    question = (args.get("question") or "").strip() or None

    if date and not _ISO_DATE.match(date):
        log.info("trip chat dropped an unparseable date "
                + kv(outcome="bad_date_shape"))
        date = None
    if time_ and not _HHMM.match(time_):
        log.info("trip chat dropped an unparseable time "
                + kv(outcome="bad_time_shape"))
        time_ = None

    origin_air, dest_air = resolve_pair(origin_raw or "", dest_raw or "")

    notes = [n for n in (
        _resolution_note("origin", origin_air, origin_raw),
        _resolution_note("dest", dest_air, dest_raw),
    ) if n]

    same_airport = bool(
        origin_air and dest_air and origin_air.code == dest_air.code)
    if same_airport:
        notes.append("That's the same airport on both ends - where are you "
                    "actually headed?")

    complete = bool(origin_air and dest_air and not same_airport
                    and not question)

    if complete:
        log.info("trip chat complete " + kv(
            **trip_fields(origin_air.code, dest_air.code, time_),
            date_given=bool(date), time_given=bool(time_)))
        trip = {"origin": origin_raw, "dest": dest_raw}
        if date:
            trip["departure_date"] = date
        if time_:
            trip["departure_time"] = time_
        reply = (f"Got it — {origin_air.code} to {dest_air.code}. "
                "Searching now.")
        return ChatResult(reply=reply, complete=True, trip=trip,
                          origin_resolution=origin_air,
                          dest_resolution=dest_air,
                          notes=notes, source="model")

    # Not complete. Prefer a resolution problem over the model's own
    # question, because a question about a field that's already unusable
    # ("what time?") while the origin doesn't resolve talks past the
    # actual blocker.
    reply = notes[0] if notes else (
        question or "Where are you flying from, and to?")
    return ChatResult(reply=reply, complete=False, trip=None,
                      origin_resolution=origin_air, dest_resolution=dest_air,
                      notes=notes, source="model")
