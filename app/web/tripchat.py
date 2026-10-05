# install-to: app/web
"""
Conversational intake: turns a back-and-forth into the trip fields the
corridor search already required.

Nothing downstream changes. `CorridorSearchBody` still takes `origin`,
`dest`, `departure_date`, `departure_time` as plain strings - the same
four values the form's text boxes used to hold - plus an optional
`flight_number` this module also arrives at through the chat. This
module's only job is to arrive at those values through a chat instead of
inputs, and to know when it hasn't yet.

FLIGHT NUMBER IS OPTIONAL AND NEVER BLOCKS COMPLETION. A trip is complete
once origin and destination resolve, whether or not a flight number was
ever mentioned - the field only sharpens which flight the search treats
as the reference, so `question` is never generated to chase it. Loosely
validated shape (letters then digits, e.g. "UA1234") rather than checked
against a real schedule here; a number that doesn't match anything real
is simply a number the search later fails to pin against, same as any
other lookup miss.

A FLIGHT NUMBER SPENDS A METERED CALL, IN THREE DIFFERENT SITUATIONS. A
number tells you nothing about origin or destination without asking
somewhere real - the model is deliberately never asked to guess a route
from a flight number the way it converts a city to a code, because a
route is a live schedule fact that gets reassigned every season and an
airport code is not. So when a flight number arrives and neither airport
is otherwise known, this looks the flight up against AeroAPI to find its
route. It never overrides an airport the user (or the model, from what
they said) already gave; partial information is trusted over a guess at
which of an ident's possibly-several legs is meant. A missing key, a
lookup failure, or an unknown ident all fall through to asking for the
airports directly, exactly as if no flight number had been given.

THE SECOND CASE IS THE REVERSE ONE: a flight number given alongside
airports that were already known isn't looked up for its route - it's
pinned on trust - so nothing previously caught "AS305" being pinned to a
Boston-Detroit trip Alaska doesn't fly. This looks the flight up too, in
that case to check its real route against the trip rather than to find
one, and only warns on a mismatch rather than unpinning it - the
generator's own matcher already falls back safely at search time if the
pin doesn't actually match, so the worst case was already handled; this
just says so sooner. Skipped whenever the exact flight number and route
pairing was already checked as of the previous turn, so an idle,
unchanged trip doesn't spend a fresh call every time the user says
anything at all.

THE THIRD CASE HAS NOTHING TO DO WITH A FLIGHT NUMBER AT ALL: it fires
the moment a trip completes with none given. A rider who doesn't know
their flight number is the common case, not the edge one, and asking
them to go find it defeats the point of pinning a reference flight in
the first place - so this looks up real flights on the resolved route
and offers a short list instead. Offered once per resolved route, same
discipline as the mismatch check: skipped whenever a flight number is
already pinned, and skipped again on a later turn for the same route
with still no number pinned. Never required, never trusted as the pin
on its own - picking one just reaches `flight_number` the same way
typing one does, and the search still matches it against real segments
either way. See `_upcoming_flights`.

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
from datetime import datetime, timezone
from typing import Any, Protocol

from app.logging_setup import get_logger, kv, trip_fields
from app.retrieval.airports import Airport, local_to_utc, resolve_pair, \
    timezone_for

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
airport, a destination airport, a departure date, and a departure time. \
Nothing else about the search - not the aircraft, not turbulence, not \
what the report should look like - is yours to ask about.

Record the departure time exactly as the traveler stated it, on its own \
clock - a boarding pass says "4:15 PM," not a UTC offset, and that is the \
form you should record too. Do not convert it to UTC yourself; a system \
downstream of you does that conversion using the origin airport's real \
timezone, and a conversion you attempt here would just be a second, \
uncontrolled guess sitting in front of the one that's actually checked. \
The only judgment call that's yours: if the traveler explicitly says the \
time is UTC or "Zulu" - not the common case, but it happens - record it \
as given and say so via `departure_time_is_utc`; otherwise leave that \
false and let the conversion downstream treat it as local to wherever \
they're flying from.

For origin and destination, the user may name a city, an airport, or a \
code - but what you record must always be a three-letter IATA code (BOS, \
PIT) or a four-letter ICAO code (KBOS, KPIT). Translate anything else \
yourself using what you know: "Boston" becomes BOS, "Pittsburgh" becomes \
PIT. Never record a bare city or airport name - a system downstream of \
you checks the code against a fixed table and cannot look up a name, so \
a name that reaches it always fails, even when the city itself is real \
and well known. If you don't know the code, or a city has more than one \
major airport and the user hasn't said which (Chicago, London, the New \
York area), ask rather than guess - do not record a name and hope it \
resolves.

If the user gives you a flight number but no city or airport at all, do \
not ask for the airports yourself - leave origin, dest, and question all \
null. A flight number is looked up for its real route downstream of you, \
and that lookup asks for the airports itself if it comes up empty; \
asking for them yourself first is a question the user shouldn't have to \
answer twice.

Departure date and time are useful but not required to proceed. Ask \
about them at most once, total, in the whole conversation - if the user's \
answer doesn't resolve to a real date and time even with today's date in \
hand, leave both null and move on rather than asking again in any form. \
Do not invent a date or time yourself, and do not ask the user what \
today's date or the current time is - you're told that below, and \
resolving "tonight", "tomorrow", or "6pm" against it is your job, not \
theirs.

If the user volunteers a specific flight number - "I'm on UA1234", "it's \
flight DL 45" - record it as given, letters then digits, e.g. "UA1234". \
Never ask for one yourself; it is a bonus, not a requirement, and asking \
for it would hold up a trip that's otherwise ready to search.

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
                "description": "Origin as a three-letter IATA code or "
                              "four-letter ICAO code - e.g. BOS, KBOS. "
                              "Translate any city or airport name to its "
                              "code yourself before recording it. Never a "
                              "bare name. Not yet resolved to a real "
                              "airport.",
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
                "description": "HH:MM, exactly as the traveler stated it "
                              "on their own clock - do not convert this "
                              "to UTC yourself. Null if not given.",
            },
            "departure_time_is_utc": {
                "type": "boolean",
                "description": "True only if the traveler explicitly said "
                              "the time is UTC or Zulu. False (the "
                              "default) means treat departure_time as "
                              "local to wherever they're flying from.",
            },
            "flight_number": {
                "type": ["string", "null"],
                "description": "Flight number as given, letters then "
                              "digits (e.g. UA1234), only if the user "
                              "volunteered one. Never ask for it; leave "
                              "null rather than prompt.",
            },
            "question": {
                "type": ["string", "null"],
                "description": "The single next question for the user, "
                              "or null if nothing more is needed.",
            },
        },
        "required": ["origin", "dest", "departure_date",
                     "departure_time", "departure_time_is_utc",
                     "flight_number", "question"],
    },
}

#: Loose enough to catch "9/23", "2026-09-23", "Sept 23". Strict enough
#: that anything failing this is not silently forwarded to the search,
#: which requires exactly YYYY-MM-DD and would otherwise turn a good
#: extraction into a 422 the user never asked for.
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_HHMM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")

#: One to three letters (an airline's IATA or ICAO prefix), then one to
#: four digits, with an optional trailing letter for a codeshare suffix -
#: "UA1234", "DAL45", "BA249A". Loose on purpose: this only screens out
#: garbage before it reaches a note, a log line, or an AeroAPI path -
#: matching it against a real flight is what the lookup and the search's
#: own reference-flight pin are for.
_FLIGHT_NUMBER = re.compile(r"^[A-Za-z]{1,3}\s?-?\d{1,4}[A-Za-z]?$")


def _grounded_system_prompt(now: datetime | None = None) -> str:
    """SYSTEM_PROMPT plus the one fact it cannot know on its own.

    The model has no clock. Without today's date in the prompt, "tonight"
    or "today" are unresolvable, and the observed failure mode wasn't a
    graceful null - it was the model asking the user what today's date
    is, in a loop, once it had already been told once not to invent one.
    Computed fresh on every call rather than baked into the module-level
    template, so a long-lived worker process doesn't keep answering with
    the date it started up on.

    `now` is test-only; production always uses the real clock.
    """
    now = now or datetime.now(timezone.utc)
    return SYSTEM_PROMPT + (
        f"\n\nRight now it is {now.strftime('%Y-%m-%d')} "
        f"({now.strftime('%A')}), {now.strftime('%H:%M')} UTC. Resolve "
        f"\"tonight\", \"today\", \"tomorrow\", \"in a few hours\", and "
        f"similar against this yourself.")


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


class FlightLookupClient(Protocol):
    """Anything that can turn a flight number into the flight it flies, or
    an airport pair into the flights that fly it."""

    def flight_by_ident(self, ident: str): ...
    def flights_between(self, origin: str, dest: str, timeout: int = 60): ...


def _real_aero_client() -> FlightLookupClient | None:
    """A live AeroAPI client, or None if no key is configured.

    Imported lazily - the common path through this module (two cities,
    no flight number) never needs AeroAPI at all, and importing it should
    not require the key any more than importing anthropic requires one.
    """
    key = os.environ.get("AEROAPI_KEY")
    if not key:
        return None
    from app.sources.aeroapi import AeroAPIClient
    return AeroAPIClient(api_key=key)


def _lookup_route(flight_number: str, aero_client: FlightLookupClient | None
                  ) -> tuple[str | None, str | None, str | None]:
    """Resolve a bare flight number to the airports it actually flies.

    Only called when neither origin nor destination is otherwise known -
    trusting an airport the user (or the model, reading what they said)
    already gave outranks a guess at which of an ident's possibly-several
    legs is meant, so this never overwrites one. Returns
    (origin, dest, note); note is set only on a miss, and is written to
    become the reply directly - the one place this module still asks a
    question of its own, now that the model has been told not to.
    """
    from app.sources.aeroapi import AeroAPIError

    client = aero_client if aero_client is not None else _real_aero_client()
    if client is None:
        # The system itself can't attempt this, same category as the
        # model being offline in _fallback - not a search that came back
        # empty, a capability that isn't there right now.
        log.info("trip chat flight lookup skipped "
                + kv(outcome="no_api_key"))
        return None, None, (
            f"Unable to comply: I can't look up flight {flight_number} "
            f"right now - what are the origin and destination airports?")

    try:
        flight = client.flight_by_ident(flight_number)
    except AeroAPIError as e:
        log.warning("trip chat flight lookup failed "
                    + kv(error=type(e).__name__, detail=str(e)[:200]))
        return None, None, (
            f"Unable to comply: I couldn't look up flight {flight_number} "
            f"- what are the origin and destination airports?")

    if flight is None or not flight.origin or not flight.destination:
        # A real attempt that came back empty, distinct from the two
        # cases above where the attempt itself couldn't be made.
        log.info("trip chat flight lookup found nothing "
                + kv(outcome="no_match"))
        return None, None, (
            f"Does not compute: I couldn't find flight {flight_number} - "
            f"what are the origin and destination airports?")

    return flight.origin, flight.destination, None


def _verify_flight_route(flight_number: str, origin_air: Airport,
                         dest_air: Airport,
                         aero_client: FlightLookupClient | None
                         ) -> str | None:
    """Sanity-check a flight number given alongside airports that are
    already known, rather than pinning it on trust alone.

    `_lookup_route` only ever runs when neither airport is known yet, so
    a flight number added to a trip that already has both ends never gets
    checked against anything - "AS305" pins cleanly onto a Boston-Detroit
    trip even though Alaska doesn't fly that route. This spends one more
    metered AeroAPI call to catch exactly that: a plausible-looking flight
    number that doesn't actually fly the route it's being pinned to.

    Only ever warns. A mismatch doesn't unpin the number, drop it, or
    block completion - the generator's own matcher (`_match_flight_number`)
    already ignores a non-matching pin at search time and falls back to
    the nearest reference flight, so the worst case was already handled
    downstream. This just says so sooner, while the user can still fix it
    themselves instead of finding out from a search result they didn't
    expect.
    """
    from app.sources.aeroapi import AeroAPIError

    client = aero_client if aero_client is not None else _real_aero_client()
    if client is None:
        # Can't check right now - say nothing rather than accuse a real
        # flight number of not matching a route no one actually verified.
        return None

    try:
        flight = client.flight_by_ident(flight_number)
    except AeroAPIError:
        return None

    if flight is None or not flight.origin or not flight.destination:
        return None

    flown_origin = (flight.origin or "").strip().upper()
    flown_dest = (flight.destination or "").strip().upper()
    if flown_origin != origin_air.code or flown_dest != dest_air.code:
        # Logged at warning, not just returned as a note, because a
        # mismatch here rests entirely on what AeroAPI handed back for
        # this exact ident string - if that's wrong (an IATA/ICAO ident
        # AeroAPI didn't resolve the way expected, a stale or off-season
        # schedule instance, the wrong leg of a same-day pair) the note
        # itself is a false positive, and there is otherwise no record of
        # what was actually compared to go find out.
        pinned = trip_fields(origin_air.code, dest_air.code)
        flown = ({"flown_origin": flown_origin, "flown_dest": flown_dest}
                if "route" not in pinned else {})
        log.warning("trip chat flight route mismatch " + kv(
            flight_number=flight_number, ident_returned=flight.ident,
            **pinned, **flown))
        return (f"Heads up — {flight_number} doesn't look like it flies "
                f"{origin_air.code} to {dest_air.code}. I'll still search "
                f"with it pinned, but the reference flight may not match "
                f"this route - tell me if that's not the flight you meant.")
    return None


#: Shown at most this many flights per suggestion, soonest-still-scheduled
#: first. Enough to be useful without turning a chat bubble into a
#: timetable.
UPCOMING_FLIGHTS_LIMIT = 5

#: A short leash on this one call, well under any reverse-proxy or worker
#: read timeout blueadept (or any deployment) is likely to run. Every
#: other AeroAPI call in this module is something the user explicitly
#: asked for and is worth a long wait; this one fires silently the moment
#: a trip completes, on every such trip rather than only when a flight
#: number was typed, so a slow upstream response must never turn into a
#: multi-second (or worse, request-timeout) hang on an otherwise-finished
#: reply. Missing the suggestion is a shrug; hanging the whole turn on it
#: is not.
UPCOMING_FLIGHTS_TIMEOUT_SECONDS = 8


def _upcoming_flights(origin_code: str, dest_code: str,
                      aero_client: FlightLookupClient | None
                      ) -> list[dict]:
    """Real flights on this pair, for a rider who doesn't know their
    flight number rather than one who already does.

    Only ever offered - never required, never blocking completion, and
    never itself trusted as the pin. Picking one from this list reaches
    the same `flight_number` field as typing one; a later search still
    matches it against real segments the same way either way. Any
    failure here (no key, no service on this pair, a network blip)
    returns empty rather than a note - this is enrichment on an already-
    complete reply, not a question the user asked, so a miss is quiet
    rather than an "Unable to comply" over a bonus.

    One entry per flight number, keeping whichever instance is soonest
    still scheduled, or most recently flown if nothing is still ahead -
    a recurring number listed five times over with five different dates
    is a worse picker than the same number listed once with its real,
    nearest time.
    """
    from app.sources.aeroapi import AeroAPIError

    client = aero_client if aero_client is not None else _real_aero_client()
    if client is None:
        return []

    try:
        segments = client.flights_between(
            origin_code, dest_code, timeout=UPCOMING_FLIGHTS_TIMEOUT_SECONDS)
    except AeroAPIError as e:
        log.info("trip chat upcoming-flights lookup failed "
                + kv(error=type(e).__name__, detail=str(e)[:200]))
        return []

    best: dict[str, object] = {}
    for seg in segments or []:
        if not seg.ident:
            continue
        current = best.get(seg.ident)
        if current is None or _prefer(seg, current):
            best[seg.ident] = seg

    upcoming = sorted((s for s in best.values() if not s.has_flown),
                      key=lambda s: s.scheduled_out or "")
    past = sorted((s for s in best.values() if s.has_flown),
                 key=lambda s: s.actual_off or s.scheduled_out or "",
                 reverse=True)

    chosen = (upcoming + past)[:UPCOMING_FLIGHTS_LIMIT]
    return [{"flight_number": s.ident, "scheduled_out": s.scheduled_out,
             "aircraft_type": s.aircraft_type, "has_flown": s.has_flown}
            for s in chosen]


def _prefer(candidate, current) -> bool:
    """Whether `candidate` should replace `current` as the one instance
    shown for a repeated flight number: not-yet-flown outranks flown, and
    within the same category the one nearer to right now wins - soonest
    scheduled if upcoming, most recently departed if not."""
    if candidate.has_flown != current.has_flown:
        return not candidate.has_flown
    cand_key = candidate.scheduled_out or candidate.actual_off or ""
    cur_key = current.scheduled_out or current.actual_off or ""
    if candidate.has_flown:
        return cand_key > cur_key
    return bool(cand_key) and (not cur_key or cand_key < cur_key)


def _upcoming_flights_note(flights: list[dict]) -> str | None:
    """Plain-language fallback alongside the structured list - the page
    may render these as clickable chips, but the note works even where it
    doesn't, and it's what a caller with no UI at all still gets."""
    if not flights:
        return None
    parts = []
    for f in flights[:UPCOMING_FLIGHTS_LIMIT]:
        when = (f["scheduled_out"] or "").replace("T", " ").replace("Z", " UTC")
        bits = [f["flight_number"]]
        if when.strip():
            bits.append(when.strip())
        if f["aircraft_type"]:
            bits.append(f["aircraft_type"])
        parts.append(" · ".join(bits))
    return (f"Flights on this route: {'; '.join(parts)}. Tell me one (or "
            f"click it) to use it as the reference flight.")


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
    #: Set as soon as the user volunteers one, independent of `complete` -
    #: the page can show "flight UA1234 noted" while still asking about the
    #: destination. Never required to reach `complete`.
    flight_number: str | None = None
    #: Real flights on the resolved route, offered once, only when the
    #: trip just became complete without one already pinned. Never
    #: required, never itself trusted as the pin - see _upcoming_flights.
    upcoming_flights: list[dict] | None = None


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

    A real outage, not a bad guess - the one place in this module where
    the assistant itself, not just an answer, is unavailable. "Unable to
    comply" over the more obvious HAL line: HAL is refusing a request it
    understood, and that is the wrong story for a plain, boring timeout.
    """
    idx = min(history_len, len(_FALLBACK_QUESTIONS) - 1)
    return ChatResult(
        reply=f"Unable to comply: {_FALLBACK_QUESTIONS[idx]}",
        complete=False, trip=None,
        origin_resolution=None, dest_resolution=None,
        notes=["The trip assistant isn't available right now, so this is "
              "asking for origin and destination directly."],
        source="fallback",
    )


def _parse_got_it(text: str | None) -> dict[str, str | None] | None:
    """Recover the trip state a previous `respond()` call announced.

    `respond` never keeps state between calls - the caller replays the
    whole history instead - so the only record of what the assistant
    already told the user is the text of its last reply. This parses
    exactly the shape the complete branch below generates, nothing more
    general; if that shape ever changes, this changes with it. Returns
    None for anything that isn't one of those replies (a question, a
    resolution note, or no prior assistant turn at all) - never guesses.
    """
    if not text or not text.startswith("Got it — ") \
            or not text.endswith(". Click Search corridors when you're "
                                 "ready."):
        return None
    middle = text[len("Got it — "):-len(". Click Search corridors when "
                                        "you're ready.")]
    parts = middle.split(", ")
    heads = parts[0].split(" to ")
    if len(heads) != 2:
        return None
    state: dict[str, str | None] = {
        "origin": heads[0], "dest": heads[1],
        "date": None, "time": None, "flight_number": None,
    }
    for part in parts[1:]:
        if part.startswith("flight "):
            state["flight_number"] = part[len("flight "):]
        elif _ISO_DATE.match(part.split(" ")[0]):
            bits = part.split(" ")
            state["date"] = bits[0]
            if len(bits) == 3 and bits[2] == "UTC" and _HHMM.match(bits[1]):
                state["time"] = bits[1]
        elif part.endswith(" UTC") and _HHMM.match(part[:-len(" UTC")]):
            state["time"] = part[:-len(" UTC")]
        else:
            return None
    return state


def _nothing_new_reply(date: str | None, time_: str | None,
                       flight_number: str | None) -> str:
    """An honest non-repeat when this turn's message changed nothing.

    Used only once a trip is already complete and the model's fresh
    extraction landed on the exact same origin, destination, date, time,
    and flight number as the last turn already announced - most often
    because the user asked a genuine question ("what is that flight
    number?") that this module has no way to actually answer, rather
    than giving one, or updating a field, it says plainly that nothing
    changed instead of silently repeating the identical reply.
    """
    missing = []
    if not date:
        missing.append("date")
    if not time_:
        missing.append("time")
    if not flight_number:
        missing.append("flight number")
    if not missing:
        return ("Nothing new to update — this trip already has everything "
                "I can gather. Click Search corridors, or tell me what to "
                "change.")
    what = missing[0] if len(missing) == 1 else " or ".join(missing)
    verb = "was" if len(missing) == 1 else "were"
    return (f"Nothing new to update — no {what} {verb} given for this "
            f"trip. Click Search corridors, or tell me what to change.")


def _resolve_local_time(date: str | None, time_: str | None,
                        time_is_utc: bool, origin_air: Airport | None
                        ) -> tuple[str | None, str | None, str | None]:
    """Turn a stated departure time into the UTC value the search wants.

    Only converts when both a date and a time were given: a bare clock
    time with no date ("6pm", nothing else) has no unambiguous local-to-UTC
    offset without guessing which day is meant, and `_target_time` in
    `service.py` already does something reasonable with a bare UTC time by
    rolling it to its next occurrence - a job this function shouldn't
    duplicate or second-guess. That case, and a time already stated as
    UTC, both pass `date`/`time_` through unchanged.

    Returns `(date, time_, note)`: the same two fields respond() already
    tracks, updated in place when a conversion happened, plus a note to
    show the reader exactly what was resolved - the same transparency
    `Airport.note()` gives a guessed airport code, extended to a guessed
    (or here, looked-up) timezone. `note` is None only when there was
    nothing to convert or nothing new to say.
    """
    if not time_ or not date or time_is_utc:
        return date, time_, None
    if origin_air is None:
        # No resolved origin yet to look a timezone up against - leave it
        # for a later turn once the origin resolves, rather than guess.
        return date, time_, None
    tz_name = timezone_for(origin_air.code)
    if not tz_name:
        return date, time_, (
            f"I don't have timezone data for {origin_air.code}, so I'm "
            f"treating {time_} as UTC.")
    utc_date, utc_time = local_to_utc(date, time_, tz_name)
    note = (f"{time_} local time at {origin_air.code} on {date} is "
           f"{utc_time} UTC" +
           (f" on {utc_date}" if utc_date != date else "") + ".")
    return utc_date, utc_time, note


def _resolution_note(field_name: str, airport: Airport | None,
                     raw: str | None) -> str | None:
    if not raw:
        return None
    if airport is None:
        # The one line this whole feature was requested for: the system
        # genuinely doesn't recognise what it was given, which is exactly
        # HAL's refusal, minus the malice.
        return (f"I'm sorry, Dave. I'm afraid I can't do that — "
                f"“{raw}” doesn't look like an airport. Try a city name, "
                f"or a three- or four-letter code.")
    return None


def respond(history: list[dict[str, str]], client: ChatClient | None = None,
           model_name: str = DEFAULT_MODEL,
           aero_client: FlightLookupClient | None = None) -> ChatResult:
    """Advance the trip conversation by one turn.

    `history` is the full conversation so far, oldest first, each entry
    `{"role": "user" | "assistant", "content": str}`, ending with the
    user's latest message. Stateless: nothing is kept between calls, so
    the caller (the page) owns replaying it.

    `aero_client` is test-only; production always resolves the real
    client lazily inside `_lookup_route`, and only when a flight number
    turn actually needs one.
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
        args = client.extract(_grounded_system_prompt(), history)
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
    time_is_utc = bool(args.get("departure_time_is_utc"))
    flight_number = (args.get("flight_number") or "").strip() or None
    question = (args.get("question") or "").strip() or None

    if date and not _ISO_DATE.match(date):
        log.info("trip chat dropped an unparseable date "
                + kv(outcome="bad_date_shape"))
        date = None
    if time_ and not _HHMM.match(time_):
        log.info("trip chat dropped an unparseable time "
                + kv(outcome="bad_time_shape"))
        time_ = None
    if flight_number and not _FLIGHT_NUMBER.match(flight_number):
        log.info("trip chat dropped an unparseable flight number "
                + kv(outcome="bad_flight_number_shape"))
        flight_number = None
    if flight_number:
        # Normalize once here rather than at every downstream matcher:
        # "ua 1234" and "UA-1234" and "UA1234" should all reach the search
        # as the same string.
        flight_number = re.sub(r"[\s-]", "", flight_number).upper()

    notes: list[str] = []

    # What the assistant's own last reply already told the user, if
    # anything - recovered from that reply's text since nothing is kept
    # between calls. Used twice below: to skip re-verifying a flight
    # number/route pairing that hasn't actually changed, and to give an
    # honest non-repeat when this whole turn changed nothing.
    prev_message = history[-2] if len(history) >= 2 else None
    prev_state = (
        _parse_got_it(prev_message.get("content"))
        if prev_message is not None and prev_message.get("role") == "assistant"
        else None)

    # A flight number is the only thing that arrived - neither airport is
    # otherwise known FOR THIS TURN. "Otherwise known" has to include what
    # the last turn already established, not just this message on its
    # own: clicking a suggested-flight chip sends nothing but the bare
    # flight number (see index.html's addChatFlights), so a trip that was
    # already confirmed last turn would otherwise look, right here, like
    # it had never had airports at all - and get silently replaced by
    # whatever route that flight number itself happens to fly, with no
    # visible warning in the chat. Recover last turn's airports first, so
    # that case instead falls through to the mismatch check below, which
    # keeps the confirmed trip and only flags a disagreement.
    prev_origin = prev_state.get("origin") if prev_state else None
    prev_dest = prev_state.get("dest") if prev_state else None
    if not origin_raw and not dest_raw and prev_origin and prev_dest:
        origin_raw, dest_raw = prev_origin, prev_dest

    # A flight number is the only thing that arrived - neither airport is
    # otherwise known, even counting last turn's confirmed trip above.
    # Look the flight up for its real route rather than asking the user
    # something a lookup can answer, or the model something it was told
    # not to guess. Skipped entirely whenever either airport is already
    # in hand: partial information from the user, or a trip already on
    # the books, outranks a guess at which of an ident's legs is meant.
    looked_up_route_this_turn = False
    if flight_number and not origin_raw and not dest_raw:
        looked_origin, looked_dest, lookup_note = _lookup_route(
            flight_number, aero_client)
        if looked_origin and looked_dest:
            origin_raw, dest_raw = looked_origin, looked_dest
            looked_up_route_this_turn = True
        elif lookup_note:
            notes.append(lookup_note)

    origin_air, dest_air = resolve_pair(origin_raw or "", dest_raw or "")

    notes += [n for n in (
        _resolution_note("origin", origin_air, origin_raw),
        _resolution_note("dest", dest_air, dest_raw),
    ) if n]

    # The traveler's own clock, converted to the UTC the search actually
    # runs on - never trusted from the model, which was told to record the
    # stated time as-is rather than convert it. See _resolve_local_time.
    date, time_, time_note = _resolve_local_time(
        date, time_, time_is_utc, origin_air)
    if time_note:
        notes.append(time_note)

    # A flight number given alongside airports that were already known
    # (not the case just above, where the flight number was the only
    # thing given and its own lookup produced these exact airports - a
    # route can't mismatch itself) is pinned on trust today; check it
    # actually flies the route it's being pinned to. Skipped when this
    # exact flight number/route pairing was already verified as of the
    # last turn - re-checking something that hasn't changed would spend
    # another metered call for no new information, every single turn a
    # complete trip sits idle.
    if flight_number and origin_air and dest_air \
            and not looked_up_route_this_turn:
        already_checked = bool(
            prev_state
            and prev_state.get("flight_number") == flight_number
            and prev_state.get("origin") == origin_air.code
            and prev_state.get("dest") == dest_air.code)
        if not already_checked:
            mismatch_note = _verify_flight_route(
                flight_number, origin_air, dest_air, aero_client)
            if mismatch_note:
                notes.append(mismatch_note)

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
        if flight_number:
            trip["flight_number"] = flight_number

        new_state = {
            "origin": origin_air.code, "dest": dest_air.code,
            "date": date, "time": time_, "flight_number": flight_number,
        }

        if prev_state is not None and prev_state == new_state:
            # The trip was already complete with these exact values as of
            # the last turn, and this turn's extraction changed nothing -
            # most likely a side question ("what is that flight number?")
            # this module has no way to actually answer. Say so honestly
            # rather than repeating the identical "Got it" line, which
            # reads as though the message was ignored.
            reply = _nothing_new_reply(date, time_, flight_number)
        else:
            # Every resolved field, not just the airports - otherwise a
            # real update (adding a time, say) is invisible: the reply
            # looks identical to the turn before it, and looks like the
            # message was ignored even when it wasn't.
            when = " ".join(p for p in (
                date, f"{time_} UTC" if time_ else None) if p)
            reply = f"Got it — {origin_air.code} to {dest_air.code}"
            if when:
                reply += f", {when}"
            if flight_number:
                reply += f", flight {flight_number}"
            reply += ". Click Search corridors when you're ready."

        # Offered once per resolved route: skip when a flight number is
        # already pinned (nothing to suggest an alternative to), and skip
        # re-fetching when this exact route already got a list as of the
        # last turn (a date/time-only edit shouldn't spend a second call
        # for the same suggestions).
        upcoming_flights: list[dict] | None = None
        already_suggested = bool(
            prev_state and prev_state.get("origin") == origin_air.code
            and prev_state.get("dest") == dest_air.code
            and not prev_state.get("flight_number"))
        if not flight_number and not already_suggested:
            upcoming_flights = _upcoming_flights(
                origin_air.code, dest_air.code, aero_client)
            suggestion_note = _upcoming_flights_note(upcoming_flights)
            if suggestion_note:
                notes.append(suggestion_note)

        return ChatResult(reply=reply, complete=True, trip=trip,
                          origin_resolution=origin_air,
                          dest_resolution=dest_air,
                          notes=notes, source="model",
                          flight_number=flight_number,
                          upcoming_flights=upcoming_flights)

    # Not complete. Prefer a resolution problem over the model's own
    # question, because a question about a field that's already unusable
    # ("what time?") while the origin doesn't resolve talks past the
    # actual blocker.
    reply = notes[0] if notes else (
        question or "Where are you flying from, and to?")
    return ChatResult(reply=reply, complete=False, trip=None,
                      origin_resolution=origin_air, dest_resolution=dest_air,
                      notes=notes, source="model",
                      flight_number=flight_number)
