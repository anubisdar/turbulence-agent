"""Tests for the conversational trip intake.

Runs entirely against a fake model client - no network, no API key. The
fake returns canned tool-call arguments, so these tests are about the
contract between the model and `respond`, not about a real model's
behaviour.
"""

import pytest

from app.web.tripchat import ChatClient, respond


class FakeClient:
    """Returns one canned response per call, in order."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    def extract(self, system, messages):
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("FakeClient asked for more than it was given")
        return self.responses.pop(0)


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
            "question": None,
        })
        result = respond([msg("user", "PIT to BOS tomorrow at 2pm UTC")],
                         client=client)
        assert result.complete
        assert result.trip == {
            "origin": "PIT", "dest": "BOS",
            "departure_date": "2026-09-25", "departure_time": "14:00",
        }

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


class TestFallback:
    def test_no_client_and_no_key_asks_plainly(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        result = respond([msg("user", "I need a flight")])
        assert not result.complete
        assert result.trip is None
        assert result.source == "fallback"
        assert result.reply

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
