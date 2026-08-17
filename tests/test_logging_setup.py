"""Tests for logging.

Two things matter more than the rest: a credential must never reach a log
line, and a trip itinerary must not be logged by default. Both are checked
against the shapes they would actually arrive in.
"""

import io
import logging

import pytest

from app.logging_setup import (
    APP_NAME,
    RedactingFilter,
    configure,
    current_request_id,
    get_logger,
    kv,
    new_request_id,
    redact,
    request_context,
    trip_fields,
)


@pytest.fixture
def captured():
    """A logger writing to a buffer, so records can be read back."""
    buf = io.StringIO()
    configure(level="DEBUG", use_syslog=False, stream=buf, force=True)
    return buf, get_logger("test")


class TestRedaction:
    @pytest.mark.parametrize("secret,text", [
        ("sk-ant-api03-XyZ9876543210abcdef",
         "auth failed with sk-ant-api03-XyZ9876543210abcdef"),
        ("abc123def456ghi789",
         'AeroAPIError: 401 - {"x-apikey":"abc123def456ghi789"}'),
        ("SECRETVALUE123456789",
         "GET /flights?api_key=SECRETVALUE123456789&start=2026-08-01"),
        ("eyJhbGciOiJIUzI1NiJ9.body.sig",
         "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.body.sig"),
    ])
    def test_credentials_are_stripped(self, secret, text):
        assert secret not in redact(text)
        assert "REDACTED" in redact(text)

    def test_the_aeroapi_error_shape_is_covered(self):
        """`AeroAPIError` carries response bodies into its message, which is
        exactly how a key reaches a log line."""
        message = ('AeroAPIError: 401 on /flights/x - key rejected: '
                   '{"title":"Invalid API key","x-apikey":"live_9f8e7d6c5b"}')
        assert "live_9f8e7d6c5b" not in redact(message)

    def test_ordinary_text_survives(self):
        message = "search finished stop=depth_limit calls=8 winner=track/high"
        assert redact(message) == message

    def test_redaction_happens_in_the_filter_not_the_formatter(self, captured):
        """A filter cannot be bypassed by adding another handler."""
        buf, log = captured
        log.info("connecting with sk-ant-api03-SHOULDNOTAPPEAR123456")
        assert "SHOULDNOTAPPEAR" not in buf.getvalue()

    def test_a_secret_in_an_argument_is_also_stripped(self):
        """Keys arrive interpolated and as arguments; both must be caught."""
        buf = io.StringIO()
        configure(level="DEBUG", use_syslog=False, stream=buf, force=True)
        get_logger("t").info("key is %s", "sk-ant-api03-ARGSECRET9876")
        assert "ARGSECRET" not in buf.getvalue()

    def test_an_exception_traceback_is_stripped(self):
        buf = io.StringIO()
        configure(level="DEBUG", use_syslog=False, stream=buf, force=True)
        log = get_logger("t")
        try:
            raise RuntimeError("failed with sk-ant-api03-TRACEBACKSECRET1")
        except RuntimeError:
            log.exception("call failed")
        assert "TRACEBACKSECRET" not in buf.getvalue()


class TestRequestCorrelation:
    def test_lines_in_one_context_share_an_id(self, captured):
        buf, log = captured
        with request_context("fixedid00001"):
            log.info("first")
            log.info("second")
        lines = [l for l in buf.getvalue().splitlines() if l.strip()]
        assert all("req=fixedid00001" in l for l in lines)

    def test_separate_contexts_get_separate_ids(self, captured):
        buf, log = captured
        with request_context():
            log.info("one")
        with request_context():
            log.info("two")
        ids = {l.split("req=")[1].split()[0]
               for l in buf.getvalue().splitlines() if "req=" in l}
        assert len(ids) == 2

    def test_the_previous_id_is_restored(self):
        with request_context("outer0000001"):
            with request_context("inner0000001"):
                assert current_request_id() == "inner0000001"
            assert current_request_id() == "outer0000001"

    def test_ids_are_short_enough_to_read(self):
        assert len(new_request_id()) <= 16


class TestTripContentPolicy:
    """A trip is an itinerary. Where someone is going and when is a
    different kind of record from how the search behaved."""

    def test_trip_content_is_withheld_by_default(self, monkeypatch):
        monkeypatch.delenv("TURBULENCE_LOG_TRIP_CONTENT", raising=False)
        fields = trip_fields("KPIT", "KBOS", "13:00")
        assert fields == {"route": "redacted"}
        assert "KPIT" not in str(fields)

    def test_it_can_be_enabled_deliberately(self):
        fields = trip_fields("KPIT", "KBOS", "13:00", include=True)
        assert fields["origin"] == "KPIT"
        assert fields["dest"] == "KBOS"

    def test_the_environment_switch_works(self, monkeypatch):
        monkeypatch.setenv("TURBULENCE_LOG_TRIP_CONTENT", "true")
        assert trip_fields("KPIT", "KBOS")["origin"] == "KPIT"

    def test_operational_shape_is_never_withheld(self):
        """Scores, counts and stop reasons carry no itinerary."""
        line = kv(stop="depth_limit", calls=8, nodes=6, reading="moderate")
        assert "depth_limit" in line and "moderate" in line


class TestFormatting:
    def test_values_with_spaces_are_quoted(self):
        assert 'reason="outside beam width"' in kv(reason="outside beam width")

    def test_none_is_dropped_rather_than_printed(self):
        """Logging None reads as a value rather than as an absence."""
        line = kv(winner=None, calls=8)
        assert "winner" not in line
        assert "calls=8" in line

    def test_simple_values_are_unquoted(self):
        assert kv(calls=8) == "calls=8"


class TestFallback:
    def test_a_missing_syslog_socket_falls_back_to_a_stream(self, monkeypatch):
        """No log socket must never take down a search.

        The socket has to be made absent rather than assumed absent: on a
        machine that has one, asking for syslog gets syslog, and the line
        correctly goes to journald instead of the buffer.
        """
        import app.logging_setup as mod
        monkeypatch.setattr(mod.os.path, "exists", lambda p: False)
        buf = io.StringIO()
        logger = configure(level="INFO", use_syslog=True, stream=buf,
                           force=True)
        logger.info("still works")
        assert "still works" in buf.getvalue()

    def test_syslog_is_used_when_the_socket_is_there(self, monkeypatch):
        """The other half: a present socket must actually be used."""
        import logging.handlers
        import app.logging_setup as mod
        monkeypatch.setattr(mod.os.path, "exists",
                            lambda p: p == mod.SYSLOG_SOCKET)

        attached = {}

        class FakeSysLog(logging.Handler):
            def __init__(self, address=None):
                super().__init__()
                attached["address"] = address

            def emit(self, record):
                pass

        monkeypatch.setattr(logging.handlers, "SysLogHandler", FakeSysLog)
        configure(level="INFO", use_syslog=True, force=True)
        assert attached.get("address") == mod.SYSLOG_SOCKET

    def test_configure_is_idempotent(self):
        first = configure(use_syslog=False, stream=io.StringIO(), force=True)
        second = configure(use_syslog=False)
        assert first is second
        assert len(second.handlers) == 1

    def test_child_loggers_share_the_configuration(self, captured):
        buf, _ = captured
        get_logger("child.one").info("from a child")
        assert "from a child" in buf.getvalue()
        assert f"{APP_NAME}.child.one" in buf.getvalue()
