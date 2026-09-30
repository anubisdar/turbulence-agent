"""Tests for the conversational trip intake.

Runs entirely against a fake model client - no network, no API key. The
fake returns canned tool-call arguments, so these tests are about the
contract between the model and `respond`, not about a real model's
behaviour.
"""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.sources.aeroapi import AeroAPIError, NetworkError
from app.web.tripchat import (ChatClient, UPCOMING_FLIGHTS_TIMEOUT_SECONDS,
                              _grounded_system_prompt, respond)


class FakeClient:
    """Returns one canned response per call, in order."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []
        self.systems: list[str] = []

    def extract(self, system, messages):
        self.calls.append(messages)
        self.systems.append(system)
        if not self.responses:
            raise AssertionError("FakeClient asked for more than it was given")
        return self.responses.pop(0)


class FakeAeroClient:
    """Returns one canned flight (or list of flights), or raises one
    canned error, per lookup."""

    def __init__(self, flight=None, error=None, flights=None):
        self.flight = flight
        self.error = error
        self.flights = flights if flights is not None else []
        self.idents_asked: list[str] = []
        self.pairs_asked: list[tuple[str, str]] = []
        self.timeouts_asked: list[int] = []

    def flight_by_ident(self, ident):
        self.idents_asked.append(ident)
        if self.error:
            raise self.error
        return self.flight

    def flights_between(self, origin, dest, timeout=60):
        self.pairs_asked.append((origin, dest))
        self.timeouts_asked.append(timeout)
        if self.error:
            raise self.error
        return self.flights


def msg(role, content):
    return {"role": role, "content": content}


class TestHappyPath:
    def test_both_airports_known_completes_the_trip(self):
        client = FakeClient({
            "origin": "PIT", "dest": "BOS",
            "departure_date": None, "departure_time": None,
            "question": None,
        })
        result = respond([msg("user", "PIT to BOS")], client=client)
        assert result.complete
        assert result.trip == {"origin": "PIT", "dest": "BOS"}
        assert result.source == "model"

    def test_date_and_time_carry_through_when_given(self):
        client = FakeClient({
            "origin": "PIT", "dest": "BOS",
            "departure_date": "2026-09-25", "departure_time": "14:00",
            "departure_time_is_utc": True,
            "question": None,
        })
        result = respond([msg("user", "PIT to BOS tomorrow at 2pm UTC")],
                         client=client)
        assert result.complete
        assert result.trip == {
            "origin": "PIT", "dest": "BOS",
            "departure_date": "2026-09-25", "departure_time": "14:00",
        }

    def test_the_reply_shows_resolved_date_time_and_flight_number(self):
        """A real update has to be visible in the reply text itself - a
        reply that never changes looks exactly like an ignored message,
        even when the underlying trip did update."""
        client = FakeClient({
            "origin": "MIA", "dest": "SEA",
            "departure_date": "2026-09-24", "departure_time": "16:00",
            "departure_time_is_utc": True,
            "flight_number": "AA100", "question": None,
        })
        result = respond([msg("user", "MIA to SEA, AA100, 4pm UTC today")],
                         client=client)
        assert "2026-09-24" in result.reply
        assert "16:00" in result.reply
        assert "AA100" in result.reply

    def test_the_reply_omits_fields_that_were_never_given(self):
        client = FakeClient({
            "origin": "MIA", "dest": "SEA",
            "departure_date": None, "departure_time": None,
            "question": None,
        })
        result = respond([msg("user", "MIA to SEA")], client=client)
        assert result.reply == ("Got it — KMIA to KSEA. Click Search "
                                "corridors when you're ready.")

    def test_a_city_name_resolves_through_the_same_table_the_form_used(self):
        """The model may propose a city; resolution still runs the airport
        table, not the model's own judgement of validity."""
        client = FakeClient({
            "origin": "Pittsburgh", "dest": "BOS",
            "departure_date": None, "departure_time": None,
            "question": None,
        })
        result = respond([msg("user", "from Pittsburgh to Boston")],
                         client=client)
        # "Pittsburgh" is not a code resolve_airport recognises, so this
        # cannot complete - the model's confidence is not the authority.
        assert not result.complete
        assert result.origin_resolution is None


class TestClarification:
    def test_a_pending_question_is_not_complete(self):
        client = FakeClient({
            "origin": "ORD", "dest": None,
            "departure_date": None, "departure_time": None,
            "question": "Where are you headed?",
        })
        result = respond([msg("user", "flying out of Chicago")],
                         client=client)
        assert not result.complete
        assert result.trip is None
        assert result.reply == "Where are you headed?"

    def test_an_unresolvable_origin_is_surfaced_even_if_the_model_moved_on(self):
        """The model asked about something else, but the origin it already
        settled on doesn't resolve. The resolution problem takes priority
        over a question that talks past it."""
        client = FakeClient({
            "origin": "not a place", "dest": "BOS",
            "departure_date": None, "departure_time": None,
            "question": "What time do you want to fly?",
        })
        result = respond([msg("user", "somewhere to Boston, morning-ish")],
                         client=client)
        assert not result.complete
        assert "look" in result.reply.lower() or "code" in result.reply.lower()

    def test_an_unresolved_airport_gets_the_hal_9000_line(self):
        client = FakeClient({
            "origin": "not a place", "dest": "BOS",
            "departure_date": None, "departure_time": None,
            "question": None,
        })
        result = respond([msg("user", "somewhere to Boston")], client=client)
        assert "I'm sorry, Dave" in result.reply

    def test_same_origin_and_destination_is_rejected(self):
        client = FakeClient({
            "origin": "BOS", "dest": "BOS",
            "departure_date": None, "departure_time": None,
            "question": None,
        })
        result = respond([msg("user", "BOS to BOS")], client=client)
        assert not result.complete
        assert "same airport" in result.reply.lower()


class TestMalformedFields:
    def test_a_badly_shaped_date_is_dropped_not_forwarded(self):
        """A date that would fail CorridorSearchBody's pattern must never
        reach it - dropped here means the search runs without one rather
        than 422ing on something the user never typed that way."""
        client = FakeClient({
            "origin": "PIT", "dest": "BOS",
            "departure_date": "next tuesday", "departure_time": None,
            "question": None,
        })
        result = respond([msg("user", "PIT to BOS next tuesday")],
                         client=client)
        assert result.complete
        assert "departure_date" not in result.trip

    def test_a_badly_shaped_time_is_dropped_not_forwarded(self):
        client = FakeClient({
            "origin": "PIT", "dest": "BOS",
            "departure_date": None, "departure_time": "2pm",
            "question": None,
        })
        result = respond([msg("user", "PIT to BOS at 2pm")], client=client)
        assert result.complete
        assert "departure_time" not in result.trip


class TestFlightNumber:
    def test_a_volunteered_flight_number_carries_through_when_complete(self):
        client = FakeClient({
            "origin": "PIT", "dest": "BOS",
            "departure_date": None, "departure_time": None,
            "flight_number": "UA1234", "question": None,
        })
        result = respond([msg("user", "PIT to BOS, I'm on UA1234")],
                         client=client)
        assert result.complete
        assert result.trip["flight_number"] == "UA1234"
        assert result.flight_number == "UA1234"

    def test_a_lowercase_spaced_flight_number_is_normalized(self):
        client = FakeClient({
            "origin": "PIT", "dest": "BOS",
            "departure_date": None, "departure_time": None,
            "flight_number": "ua 1234", "question": None,
        })
        result = respond([msg("user", "PIT to BOS on ua 1234")],
                         client=client)
        assert result.trip["flight_number"] == "UA1234"

    def test_absent_flight_number_never_blocks_completion(self):
        client = FakeClient({
            "origin": "PIT", "dest": "BOS",
            "departure_date": None, "departure_time": None,
            "flight_number": None, "question": None,
        })
        result = respond([msg("user", "PIT to BOS")], client=client)
        assert result.complete
        assert "flight_number" not in result.trip
        assert result.flight_number is None

    def test_a_garbled_flight_number_is_dropped_not_forwarded(self):
        client = FakeClient({
            "origin": "PIT", "dest": "BOS",
            "departure_date": None, "departure_time": None,
            "flight_number": "my usual flight", "question": None,
        })
        result = respond([msg("user", "PIT to BOS, my usual flight")],
                         client=client)
        assert result.complete
        assert "flight_number" not in result.trip
        assert result.flight_number is None

    def test_a_flight_number_is_visible_even_before_the_trip_completes(self):
        """The page can show 'flight UA1234 noted' while still chasing the
        destination - the field is never gated on completeness."""
        client = FakeClient({
            "origin": "PIT", "dest": None,
            "departure_date": None, "departure_time": None,
            "flight_number": "UA1234",
            "question": "Where are you headed?",
        })
        result = respond([msg("user", "flying PIT on UA1234")], client=client)
        assert not result.complete
        assert result.flight_number == "UA1234"


class TestFlightNumberLookup:
    """A bare flight number - no city, no airport - is looked up for its
    real route rather than turned into a question."""

    def test_a_bare_flight_number_resolves_the_trip_in_one_turn(self):
        chat_client = FakeClient({
            "origin": None, "dest": None,
            "departure_date": None, "departure_time": None,
            "flight_number": "DL5731", "question": None,
        })
        aero_client = FakeAeroClient(
            flight=SimpleNamespace(origin="KATL", destination="KJFK"))
        result = respond([msg("user", "I'm on flight DL5731")],
                         client=chat_client, aero_client=aero_client)
        assert result.complete
        assert result.trip == {"origin": "KATL", "dest": "KJFK",
                               "flight_number": "DL5731"}
        assert aero_client.idents_asked == ["DL5731"]

    def test_an_unmatched_ident_asks_for_the_airports_directly(self):
        chat_client = FakeClient({
            "origin": None, "dest": None,
            "departure_date": None, "departure_time": None,
            "flight_number": "ZZ0000", "question": None,
        })
        aero_client = FakeAeroClient(flight=None)
        result = respond([msg("user", "I'm on flight ZZ0000")],
                         client=chat_client, aero_client=aero_client)
        assert not result.complete
        assert "ZZ0000" in result.reply
        assert "airport" in result.reply.lower()
        assert result.reply.startswith("Does not compute")

    def test_a_lookup_failure_asks_for_the_airports_directly(self):
        chat_client = FakeClient({
            "origin": None, "dest": None,
            "departure_date": None, "departure_time": None,
            "flight_number": "DL5731", "question": None,
        })
        aero_client = FakeAeroClient(error=AeroAPIError("rate limited"))
        result = respond([msg("user", "I'm on flight DL5731")],
                         client=chat_client, aero_client=aero_client)
        assert not result.complete
        assert "DL5731" in result.reply
        assert "airport" in result.reply.lower()
        assert result.reply.startswith("Unable to comply")

    def test_a_partially_known_trip_never_triggers_a_lookup(self):
        """The user already said Pittsburgh - trusting that outranks a
        guess at which of the ident's legs is meant."""
        chat_client = FakeClient({
            "origin": "PIT", "dest": None,
            "departure_date": None, "departure_time": None,
            "flight_number": "DL5731", "question": None,
        })
        aero_client = FakeAeroClient(
            flight=SimpleNamespace(origin="KATL", destination="KJFK"))
        respond([msg("user", "from Pittsburgh, flight DL5731")],
               client=chat_client, aero_client=aero_client)
        assert aero_client.idents_asked == []

    def test_no_api_key_falls_back_to_asking_for_the_airports(self, monkeypatch):
        monkeypatch.delenv("AEROAPI_KEY", raising=False)
        chat_client = FakeClient({
            "origin": None, "dest": None,
            "departure_date": None, "departure_time": None,
            "flight_number": "DL5731", "question": None,
        })
        result = respond([msg("user", "I'm on flight DL5731")],
                         client=chat_client)
        assert not result.complete
        assert "airport" in result.reply.lower()
        assert result.reply.startswith("Unable to comply")


class TestFallback:
    def test_no_client_and_no_key_asks_plainly(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        result = respond([msg("user", "I need a flight")])
        assert not result.complete
        assert result.trip is None
        assert result.source == "fallback"
        assert result.reply
        assert result.reply.startswith("Unable to comply")

    def test_a_client_that_raises_falls_back_rather_than_erroring(self):
        class Boom:
            def extract(self, system, messages):
                raise RuntimeError("provider is down")

        result = respond([msg("user", "PIT to BOS")], client=Boom())
        assert result.source == "fallback"
        assert not result.complete

    def test_fallback_never_marks_a_trip_complete(self):
        """However far the conversation has gone, the fallback path must
        never fabricate a completed trip - it has no way to know one."""
        history = [msg("user", "PIT to BOS"), msg("assistant", "when?"),
                  msg("user", "tomorrow at 2pm")]
        result = respond(history)
        assert not result.complete
        assert result.trip is None


class TestHistoryHandling:
    def test_a_very_long_conversation_is_bounded_before_reaching_the_model(self):
        client = FakeClient({
            "origin": "PIT", "dest": "BOS",
            "departure_date": None, "departure_time": None,
            "question": None,
        })
        history = [msg("user", f"message {i}") for i in range(100)]
        respond(history, client=client)
        assert len(client.calls[0]) <= 40


class TestDateGrounding:
    """The model has no clock of its own - without today's date in the
    prompt, "tonight" and "today" are unresolvable, and the observed
    failure was the model asking the user what today's date is."""

    def test_todays_date_is_in_the_prompt(self):
        now = datetime(2026, 9, 24, 18, 30, tzinfo=timezone.utc)
        prompt = _grounded_system_prompt(now)
        assert "2026-09-24" in prompt
        assert "18:30" in prompt
        assert "Thursday" in prompt

    def test_a_fresh_date_is_used_on_every_call_by_default(self):
        before = datetime.now(timezone.utc)
        prompt = _grounded_system_prompt()
        assert before.strftime("%Y-%m-%d") in prompt

    def test_the_model_actually_receives_the_grounded_prompt(self):
        client = FakeClient({
            "origin": "PIT", "dest": "BOS",
            "departure_date": None, "departure_time": None,
            "question": None,
        })
        respond([msg("user", "PIT to BOS")], client=client)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert today in client.systems[0]
        assert "Resolve" in client.systems[0]


class TestNothingNewToReport:
    """Once a trip is already complete, a follow-up message that changes
    none of the trip fields - typically a genuine question the model has
    no way to answer ("what is that flight number?") - should say
    honestly that nothing changed, not repeat the identical "Got it"
    line as though the message had been acted on."""

    def test_a_side_question_after_completion_gets_an_honest_non_repeat(self):
        client = FakeClient({
            "origin": "MIA", "dest": "SEA",
            "departure_date": None, "departure_time": None,
            "flight_number": None, "question": None,
        })
        history = [
            msg("user", "MIA to SEA"),
            msg("assistant", "Got it — KMIA to KSEA. Click Search "
                             "corridors when you're ready."),
            msg("user", "what is that flight number?"),
        ]
        result = respond(history, client=client)
        assert result.complete
        assert result.reply.startswith("Nothing new to update")
        assert "flight number" in result.reply
        assert "Got it" not in result.reply

    def test_a_real_change_still_updates_the_reply_normally(self):
        """The nothing-new check must not swallow an actual update - this
        is the scenario the previous fix (showing date/time/flight in the
        reply) was for, and it has to keep working alongside this one."""
        client = FakeClient({
            "origin": "MIA", "dest": "SEA",
            "departure_date": "2026-09-24", "departure_time": "16:00",
            "departure_time_is_utc": True,
            "flight_number": None, "question": None,
        })
        history = [
            msg("user", "MIA to SEA"),
            msg("assistant", "Got it — KMIA to KSEA. Click Search "
                             "corridors when you're ready."),
            msg("user", "try 4pm UTC today"),
        ]
        result = respond(history, client=client)
        assert result.complete
        assert "16:00" in result.reply
        assert "Nothing new" not in result.reply

    def test_no_prior_assistant_turn_is_never_treated_as_nothing_new(self):
        client = FakeClient({
            "origin": "MIA", "dest": "SEA",
            "departure_date": None, "departure_time": None,
            "flight_number": None, "question": None,
        })
        result = respond([msg("user", "MIA to SEA")], client=client)
        assert result.reply.startswith("Got it")

    def test_a_prior_question_turn_is_never_treated_as_nothing_new(self):
        """The previous assistant turn wasn't a completed trip at all -
        parsing it must fail closed, not accidentally match."""
        client = FakeClient({
            "origin": "MIA", "dest": "SEA",
            "departure_date": None, "departure_time": None,
            "flight_number": None, "question": None,
        })
        history = [
            msg("user", "MIA to somewhere"),
            msg("assistant", "And where are you headed?"),
            msg("user", "MIA to SEA"),
        ]
        result = respond(history, client=client)
        assert result.reply.startswith("Got it")

    def test_the_nothing_new_reply_names_every_still_missing_field(self):
        client = FakeClient({
            "origin": "MIA", "dest": "SEA",
            "departure_date": None, "departure_time": None,
            "flight_number": None, "question": None,
        })
        history = [
            msg("user", "MIA to SEA"),
            msg("assistant", "Got it — KMIA to KSEA. Click Search "
                             "corridors when you're ready."),
            msg("user", "what else do you know?"),
        ]
        result = respond(history, client=client)
        assert "date" in result.reply
        assert "time" in result.reply
        assert "flight number" in result.reply

    def test_the_nothing_new_reply_when_every_field_is_already_known(self):
        client = FakeClient({
            "origin": "MIA", "dest": "SEA",
            "departure_date": "2026-09-24", "departure_time": "16:00",
            "departure_time_is_utc": True,
            "flight_number": "AA100", "question": None,
        })
        history = [
            msg("user", "MIA to SEA, 4pm UTC today, AA100"),
            msg("assistant", "Got it — KMIA to KSEA, 2026-09-24 16:00 "
                             "UTC, flight AA100. Click Search corridors "
                             "when you're ready."),
            msg("user", "what is that flight number?"),
        ]
        result = respond(history, client=client)
        assert result.reply == (
            "Nothing new to update — this trip already has everything I "
            "can gather. Click Search corridors, or tell me what to "
            "change.")


class TestFlightNumberRouteVerification:
    """A flight number given alongside airports that are already known
    is pinned without ever being checked against AeroAPI, unlike the
    bare-flight-number case above - so a plausible-looking but wrong
    number (an Alaska flight pinned to a Boston-Detroit trip) went
    through silently. This checks it and warns, without unpinning it."""

    def test_a_mismatched_flight_number_gets_a_warning_note(self):
        chat_client = FakeClient({
            "origin": "BOS", "dest": "DTW",
            "departure_date": None, "departure_time": None,
            "flight_number": "AS305", "question": None,
        })
        aero_client = FakeAeroClient(
            # AS305 actually flies somewhere else entirely.
            flight=SimpleNamespace(
                ident="ASA305", origin="KSEA", destination="KANC"))
        result = respond([msg("user", "BOS to DTW, flight AS305")],
                         client=chat_client, aero_client=aero_client)
        assert result.complete
        assert result.trip["flight_number"] == "AS305"
        assert any("AS305" in n and "doesn't look like it flies" in n
                  for n in result.notes)
        assert aero_client.idents_asked == ["AS305"]

    def test_a_matching_flight_number_gets_no_warning(self):
        chat_client = FakeClient({
            "origin": "BOS", "dest": "DTW",
            "departure_date": None, "departure_time": None,
            "flight_number": "DL123", "question": None,
        })
        aero_client = FakeAeroClient(
            flight=SimpleNamespace(origin="KBOS", destination="KDTW"))
        result = respond([msg("user", "BOS to DTW, flight DL123")],
                         client=chat_client, aero_client=aero_client)
        assert result.complete
        assert result.notes == []

    def test_a_bare_number_lookup_that_produced_the_route_is_not_reverified(
            self):
        """When the flight number is the only thing given, its own lookup
        already produced this exact route - checking it against itself
        would be pointless and would double the call."""
        chat_client = FakeClient({
            "origin": None, "dest": None,
            "departure_date": None, "departure_time": None,
            "flight_number": "DL5731", "question": None,
        })
        aero_client = FakeAeroClient(
            flight=SimpleNamespace(origin="KATL", destination="KJFK"))
        result = respond([msg("user", "I'm on flight DL5731")],
                         client=chat_client, aero_client=aero_client)
        assert result.complete
        assert result.notes == []
        assert aero_client.idents_asked == ["DL5731"]

    def test_an_unresolvable_lookup_during_verification_says_nothing(self):
        """The check can't confirm a mismatch, so it stays quiet rather
        than accusing a real flight number of not matching a route no
        one actually verified."""
        chat_client = FakeClient({
            "origin": "BOS", "dest": "DTW",
            "departure_date": None, "departure_time": None,
            "flight_number": "AS305", "question": None,
        })
        aero_client = FakeAeroClient(error=AeroAPIError("rate limited"))
        result = respond([msg("user", "BOS to DTW, flight AS305")],
                         client=chat_client, aero_client=aero_client)
        assert result.complete
        assert result.notes == []

    def test_no_api_key_during_verification_says_nothing(self, monkeypatch):
        monkeypatch.delenv("AEROAPI_KEY", raising=False)
        chat_client = FakeClient({
            "origin": "BOS", "dest": "DTW",
            "departure_date": None, "departure_time": None,
            "flight_number": "AS305", "question": None,
        })
        result = respond([msg("user", "BOS to DTW, flight AS305")],
                         client=chat_client)
        assert result.complete
        assert result.notes == []

    def test_an_already_verified_pairing_is_not_rechecked_next_turn(self):
        """Once a flight number/route pairing has been checked, an idle
        trip sitting through more side questions shouldn't spend another
        call each time for the same answer."""
        chat_client = FakeClient({
            "origin": "BOS", "dest": "DTW",
            "departure_date": None, "departure_time": None,
            "flight_number": "AS305", "question": None,
        })
        aero_client = FakeAeroClient(
            flight=SimpleNamespace(
                ident="ASA305", origin="KSEA", destination="KANC"))
        history = [
            msg("user", "BOS to DTW, flight AS305"),
            msg("assistant", "Got it — KBOS to KDTW, flight AS305. Click "
                             "Search corridors when you're ready."),
            msg("user", "are you sure about that flight?"),
        ]
        result = respond(history, client=chat_client, aero_client=aero_client)
        assert aero_client.idents_asked == []

    def test_changing_the_destination_re_triggers_verification(self):
        """The route changed even though the flight number didn't - the
        old check no longer says anything about the new pairing."""
        chat_client = FakeClient({
            "origin": "BOS", "dest": "ORD",
            "departure_date": None, "departure_time": None,
            "flight_number": "AS305", "question": None,
        })
        aero_client = FakeAeroClient(
            flight=SimpleNamespace(
                ident="ASA305", origin="KSEA", destination="KANC"))
        history = [
            msg("user", "BOS to DTW, flight AS305"),
            msg("assistant", "Got it — KBOS to KDTW, flight AS305. Click "
                             "Search corridors when you're ready."),
            msg("user", "actually make it Chicago"),
        ]
        result = respond(history, client=chat_client, aero_client=aero_client)
        assert aero_client.idents_asked == ["AS305"]


class TestNetworkFailureDuringAeroAPILookups:
    """A plain network blip (timeout, DNS failure, connection refused)
    during either AeroAPI path reaches this module as a NetworkError - an
    AeroAPIError subclass - not a raw socket/urllib exception, so both
    lookup paths degrade to the same honest question a lookup miss
    already produces, rather than blowing up the whole turn."""

    def test_a_network_failure_during_the_bare_number_lookup_degrades_gracefully(
            self):
        chat_client = FakeClient({
            "origin": None, "dest": None,
            "departure_date": None, "departure_time": None,
            "flight_number": "DL5731", "question": None,
        })
        aero_client = FakeAeroClient(
            error=NetworkError("Name or service not known"))
        result = respond([msg("user", "I'm on flight DL5731")],
                         client=chat_client, aero_client=aero_client)
        assert not result.complete
        assert result.reply.startswith("Unable to comply")

    def test_a_network_failure_during_route_verification_degrades_gracefully(
            self):
        chat_client = FakeClient({
            "origin": "BOS", "dest": "DTW",
            "departure_date": None, "departure_time": None,
            "flight_number": "AS305", "question": None,
        })
        aero_client = FakeAeroClient(
            error=NetworkError("Name or service not known"))
        result = respond([msg("user", "BOS to DTW, flight AS305")],
                         client=chat_client, aero_client=aero_client)
        # The trip still completes - verification only ever warns, never
        # blocks - and a check that couldn't run says nothing at all
        # rather than accusing a real flight number of a mismatch no one
        # actually confirmed.
        assert result.complete
        assert result.notes == []


class TestFlightRouteMismatchIsLogged:
    """A mismatch warning is only as trustworthy as what AeroAPI actually
    returned - there was no record of that anywhere, so a false positive
    (a real flight flagged as not matching a route it does fly) was
    undiagnosable from outside a debugger. This locks in that the
    comparison itself, including the ident AeroAPI resolved the query
    to, reaches the logs."""

    def test_the_actual_returned_route_and_ident_are_logged(self, caplog):
        import logging
        chat_client = FakeClient({
            "origin": "BOS", "dest": "DTW",
            "departure_date": None, "departure_time": None,
            "flight_number": "AS305", "question": None,
        })
        aero_client = FakeAeroClient(
            flight=SimpleNamespace(
                ident="ASA305", origin="KSEA", destination="KANC"))
        with caplog.at_level(logging.WARNING):
            respond([msg("user", "BOS to DTW, flight AS305")],
                   client=chat_client, aero_client=aero_client)
        mismatch_logs = [r for r in caplog.records
                         if "flight route mismatch" in r.message]
        assert len(mismatch_logs) == 1
        assert "AS305" in mismatch_logs[0].message
        assert "ASA305" in mismatch_logs[0].message


class TestUpcomingFlights:
    """Once a trip completes with no flight number, real flights on that
    route are offered so a rider who doesn't know their number can just
    pick one instead of being left to go find it."""

    def test_a_completed_trip_with_no_flight_number_gets_suggestions(self):
        chat_client = FakeClient({
            "origin": "PIT", "dest": "BOS",
            "departure_date": None, "departure_time": None,
            "flight_number": None, "question": None,
        })
        aero_client = FakeAeroClient(flights=[
            SimpleNamespace(ident="UA1234", scheduled_out="2026-09-25T14:00:00Z",
                            actual_off=None, aircraft_type="737-800",
                            has_flown=False),
            SimpleNamespace(ident="DL5678", scheduled_out="2026-09-25T18:30:00Z",
                            actual_off=None, aircraft_type="A320",
                            has_flown=False),
        ])
        result = respond([msg("user", "PIT to BOS")],
                         client=chat_client, aero_client=aero_client)
        assert result.complete
        assert result.upcoming_flights is not None
        idents = [f["flight_number"] for f in result.upcoming_flights]
        assert idents == ["UA1234", "DL5678"]
        assert any("UA1234" in n and "DL5678" in n for n in result.notes)
        assert aero_client.pairs_asked == [("KPIT", "KBOS")]

    def test_a_flight_number_already_given_gets_no_suggestions(self):
        chat_client = FakeClient({
            "origin": "PIT", "dest": "BOS",
            "departure_date": None, "departure_time": None,
            "flight_number": "UA1234", "question": None,
        })
        aero_client = FakeAeroClient(flights=[
            SimpleNamespace(ident="DL5678", scheduled_out="2026-09-25T18:30:00Z",
                            actual_off=None, aircraft_type="A320",
                            has_flown=False),
        ])
        result = respond([msg("user", "PIT to BOS, flight UA1234")],
                         client=chat_client, aero_client=aero_client)
        assert result.complete
        assert result.upcoming_flights is None
        assert aero_client.pairs_asked == []

    def test_no_service_on_the_route_is_quiet(self):
        chat_client = FakeClient({
            "origin": "PIT", "dest": "BOS",
            "departure_date": None, "departure_time": None,
            "flight_number": None, "question": None,
        })
        aero_client = FakeAeroClient(flights=[])
        result = respond([msg("user", "PIT to BOS")],
                         client=chat_client, aero_client=aero_client)
        assert result.complete
        assert result.upcoming_flights == []
        assert not any("route" in n.lower() and "flight" in n.lower()
                      for n in result.notes)

    def test_a_lookup_failure_is_quiet_not_an_error(self):
        chat_client = FakeClient({
            "origin": "PIT", "dest": "BOS",
            "departure_date": None, "departure_time": None,
            "flight_number": None, "question": None,
        })
        aero_client = FakeAeroClient(error=AeroAPIError("rate limited"))
        result = respond([msg("user", "PIT to BOS")],
                         client=chat_client, aero_client=aero_client)
        assert result.complete
        assert result.upcoming_flights == []

    def test_no_api_key_is_quiet(self, monkeypatch):
        monkeypatch.delenv("AEROAPI_KEY", raising=False)
        chat_client = FakeClient({
            "origin": "PIT", "dest": "BOS",
            "departure_date": None, "departure_time": None,
            "flight_number": None, "question": None,
        })
        result = respond([msg("user", "PIT to BOS")], client=chat_client)
        assert result.complete
        assert result.upcoming_flights == []

    def test_a_repeated_flight_number_shows_once_with_its_soonest_time(self):
        chat_client = FakeClient({
            "origin": "PIT", "dest": "BOS",
            "departure_date": None, "departure_time": None,
            "flight_number": None, "question": None,
        })
        aero_client = FakeAeroClient(flights=[
            SimpleNamespace(ident="UA1234", scheduled_out="2026-09-26T14:00:00Z",
                            actual_off=None, aircraft_type="737-800",
                            has_flown=False),
            SimpleNamespace(ident="UA1234", scheduled_out="2026-09-25T14:00:00Z",
                            actual_off=None, aircraft_type="737-800",
                            has_flown=False),
        ])
        result = respond([msg("user", "PIT to BOS")],
                         client=chat_client, aero_client=aero_client)
        assert len(result.upcoming_flights) == 1
        assert result.upcoming_flights[0]["scheduled_out"] == \
            "2026-09-25T14:00:00Z"

    def test_flown_instances_are_preferred_over_scheduled_for_the_pick(self):
        """Not-yet-flown outranks flown for the SAME ident, so the list
        stays forward-looking when a real upcoming instance exists."""
        chat_client = FakeClient({
            "origin": "PIT", "dest": "BOS",
            "departure_date": None, "departure_time": None,
            "flight_number": None, "question": None,
        })
        aero_client = FakeAeroClient(flights=[
            SimpleNamespace(ident="UA1234", scheduled_out="2026-09-20T14:00:00Z",
                            actual_off="2026-09-20T14:05:00Z",
                            aircraft_type="737-800", has_flown=True),
            SimpleNamespace(ident="UA1234", scheduled_out="2026-09-25T14:00:00Z",
                            actual_off=None, aircraft_type="737-800",
                            has_flown=False),
        ])
        result = respond([msg("user", "PIT to BOS")],
                         client=chat_client, aero_client=aero_client)
        assert len(result.upcoming_flights) == 1
        assert result.upcoming_flights[0]["has_flown"] is False

    def test_a_second_turn_on_the_same_route_does_not_ask_again(self):
        chat_client = FakeClient({
            "origin": "PIT", "dest": "BOS",
            "departure_date": "2026-09-25", "departure_time": "16:00",
            "flight_number": None, "question": None,
        })
        aero_client = FakeAeroClient(flights=[
            SimpleNamespace(ident="UA1234", scheduled_out="2026-09-25T14:00:00Z",
                            actual_off=None, aircraft_type="737-800",
                            has_flown=False),
        ])
        history = [
            msg("user", "PIT to BOS"),
            msg("assistant", "Got it — KPIT to KBOS. Click Search "
                             "corridors when you're ready."),
            msg("user", "try 4pm tomorrow"),
        ]
        result = respond(history, client=chat_client, aero_client=aero_client)
        assert result.upcoming_flights is None
        assert aero_client.pairs_asked == []

    def test_a_different_destination_asks_again(self):
        chat_client = FakeClient({
            "origin": "PIT", "dest": "ORD",
            "departure_date": None, "departure_time": None,
            "flight_number": None, "question": None,
        })
        aero_client = FakeAeroClient(flights=[
            SimpleNamespace(ident="UA1234", scheduled_out="2026-09-25T14:00:00Z",
                            actual_off=None, aircraft_type="737-800",
                            has_flown=False),
        ])
        history = [
            msg("user", "PIT to BOS"),
            msg("assistant", "Got it — KPIT to KBOS. Click Search "
                             "corridors when you're ready."),
            msg("user", "actually make it Chicago"),
        ]
        result = respond(history, client=chat_client, aero_client=aero_client)
        assert aero_client.pairs_asked == [("KPIT", "KORD")]


class TestUpcomingFlightsTimeout:
    """The suggestion lookup fires on every completed trip with no flight
    number - far more often than the flight-number lookups it sits
    alongside - so it carries its own short leash rather than the general
    60s default: a slow AeroAPI response must degrade to an empty
    suggestion list, never hang the whole chat turn."""

    def test_the_suggestion_lookup_uses_a_short_timeout(self):
        chat_client = FakeClient({
            "origin": "PIT", "dest": "BOS",
            "departure_date": None, "departure_time": None,
            "flight_number": None, "question": None,
        })
        aero_client = FakeAeroClient(flights=[])
        respond([msg("user", "PIT to BOS")],
               client=chat_client, aero_client=aero_client)
        assert aero_client.timeouts_asked == [UPCOMING_FLIGHTS_TIMEOUT_SECONDS]
        assert UPCOMING_FLIGHTS_TIMEOUT_SECONDS < 60


class TestLocalDepartureTimes:
    """A departure time is local to the origin airport unless the model
    says the traveler stated it as UTC. The model is told never to do this
    conversion itself - see SYSTEM_PROMPT - so these tests exercise
    `respond`'s own conversion, not anything the fake client computes."""

    def test_a_stated_local_time_is_converted_to_utc(self):
        """PIT is America/New_York. 2pm EDT in September is 18:00 UTC."""
        client = FakeClient({
            "origin": "PIT", "dest": "BOS",
            "departure_date": "2026-09-25", "departure_time": "14:00",
            "departure_time_is_utc": False,
            "question": None,
        })
        result = respond([msg("user", "PIT to BOS tomorrow at 2pm")],
                         client=client, aero_client=FakeAeroClient())
        assert result.complete
        assert result.trip == {
            "origin": "PIT", "dest": "BOS",
            "departure_date": "2026-09-25", "departure_time": "18:00",
        }

    def test_the_resolution_is_explained_in_a_note(self):
        client = FakeClient({
            "origin": "PIT", "dest": "BOS",
            "departure_date": "2026-09-25", "departure_time": "14:00",
            "departure_time_is_utc": False,
            "question": None,
        })
        result = respond([msg("user", "PIT to BOS tomorrow at 2pm")],
                         client=client, aero_client=FakeAeroClient())
        assert any("14:00" in n and "18:00 UTC" in n for n in result.notes)

    def test_a_time_explicitly_stated_as_utc_is_not_converted_again(self):
        client = FakeClient({
            "origin": "PIT", "dest": "BOS",
            "departure_date": "2026-09-25", "departure_time": "18:00",
            "departure_time_is_utc": True,
            "question": None,
        })
        result = respond([msg("user", "PIT to BOS tomorrow at 18:00 UTC")],
                         client=client, aero_client=FakeAeroClient())
        assert result.trip["departure_time"] == "18:00"
        # Not "no note mentions UTC" - the upcoming-flights suggestion
        # note (suppressed here via an empty FakeAeroClient) legitimately
        # says "UTC" too, from a flight's own scheduled time. What this
        # guards against is specifically a second, unwanted conversion
        # note - the resolution note this same class's
        # test_the_resolution_is_explained_in_a_note checks for.
        assert not any("local time at" in n for n in result.notes)

    def test_a_late_local_time_can_roll_the_date_forward(self):
        """A conversion that only fixed the clock and left the date alone
        would silently misplace the departure by a day - the exact class
        of quiet wrong answer this feature exists to avoid. LAX is
        UTC-7 in September, so a late evening departure is already past
        midnight in UTC."""
        client = FakeClient({
            "origin": "LAX", "dest": "HND",
            "departure_date": "2026-09-25", "departure_time": "23:30",
            "departure_time_is_utc": False,
            "question": None,
        })
        result = respond([msg("user", "LA to Tokyo tomorrow at 11:30pm")],
                         client=client, aero_client=FakeAeroClient())
        assert result.trip["departure_date"] == "2026-09-26"
        assert result.trip["departure_time"] == "06:30"

    def test_an_early_local_time_can_roll_the_date_backward(self):
        """HND is Asia/Tokyo, UTC+9 - a 2 AM departure is still the
        evening before in UTC."""
        client = FakeClient({
            "origin": "HND", "dest": "LAX",
            "departure_date": "2026-09-25", "departure_time": "02:00",
            "departure_time_is_utc": False,
            "question": None,
        })
        result = respond([msg("user", "Tokyo to LA tomorrow at 2am")],
                         client=client, aero_client=FakeAeroClient())
        assert result.trip["departure_date"] == "2026-09-24"
        assert result.trip["departure_time"] == "17:00"

    def test_arizona_gets_no_dst_shift(self):
        client = FakeClient({
            "origin": "PHX", "dest": "LAX",
            "departure_date": "2026-01-15", "departure_time": "09:00",
            "departure_time_is_utc": False,
            "question": None,
        })
        result = respond([msg("user", "Phoenix to LA at 9am in January")],
                         client=client, aero_client=FakeAeroClient())
        assert result.trip["departure_time"] == "16:00"

    def test_a_bare_time_with_no_date_is_left_for_target_time_to_handle(self):
        """No date means no unambiguous local offset to convert with -
        this stays exactly as `_target_time` in service.py already
        expects: a plain UTC clock time it rolls to its next occurrence."""
        client = FakeClient({
            "origin": "PIT", "dest": "BOS",
            "departure_date": None, "departure_time": "14:00",
            "departure_time_is_utc": False,
            "question": None,
        })
        result = respond([msg("user", "PIT to BOS around 2pm")],
                         client=client, aero_client=FakeAeroClient())
        assert result.trip["departure_time"] == "14:00"
        assert "departure_date" not in result.trip

    def test_an_airport_with_no_timezone_data_falls_back_to_utc_honestly(self):
        """An ASSUMED (K-rule) guess never gets a timezone entry - see
        `_TIMEZONE`'s docstring - so this is the ordinary, expected path
        for a smaller airport, not a bug to paper over."""
        client = FakeClient({
            "origin": "ZZZ", "dest": "BOS",
            "departure_date": "2026-09-25", "departure_time": "14:00",
            "departure_time_is_utc": False,
            "question": None,
        })
        result = respond([msg("user", "ZZZ to BOS tomorrow at 2pm")],
                         client=client, aero_client=FakeAeroClient())
        assert result.trip["departure_time"] == "14:00"
        assert any("timezone data" in n for n in result.notes)

    def test_conversion_waits_for_the_origin_to_resolve(self):
        """A time given before the origin is known can't be converted yet
        - it's carried through unconverted rather than guessed at, and
        picked up once the origin actually resolves."""
        client = FakeClient({
            "origin": None, "dest": "BOS",
            "departure_date": "2026-09-25", "departure_time": "14:00",
            "departure_time_is_utc": False,
            "question": "Where are you flying from?",
        })
        result = respond([msg("user", "to BOS tomorrow at 2pm")],
                         client=client)
        assert not result.complete
        assert not any("UTC" in n for n in result.notes)
