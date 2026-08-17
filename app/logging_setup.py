# install-to: app
"""
Logging for the turbulence agent.

One search touches the generator, three external APIs, the critic, the
controller and possibly a language model. Without correlation those lines
interleave with every other request and none of them can be followed. So
every log line carries a request id, and the id is generated once per
search and threaded through everything it touches.

WHY SYSLOG. The box already runs journald, which handles rotation,
persistence and querying. Writing to `/dev/log` means `journalctl -t
turbulence-agent -f` works with no configuration, and a deployment can ship
lines to a collector by changing one address. Where there is no syslog
socket - a container, a test runner, a laptop - the handler falls back to
stderr rather than failing, because a missing log socket should never take
down a search.

REDACTION IS NOT OPTIONAL. Two credentials pass through this system, and
one of them is metered. `AeroAPIError` carries response bodies into its
message, which is exactly the shape that puts a key in a log line. Every
record is filtered before it is emitted, and the filter runs on the message
after formatting rather than on the arguments, because a key can arrive by
either route.

WHAT IS DELIBERATELY NOT LOGGED. A trip is an itinerary. Origin,
destination and travel date together say where a person is going and when,
which is different from logging how the search behaved. Operational shape -
scores, call counts, stop reasons, durations - is logged freely. Trip
content is logged only when `log_trip_content` is enabled, and it is off by
default.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import re
import socket
import sys
import uuid
from contextvars import ContextVar
from typing import Any

APP_NAME = "turbulence-agent"
SYSLOG_SOCKET = "/dev/log"

#: The current search's id. A ContextVar rather than a parameter so the
#: reasoning layer does not have to carry a logger through every function
#: signature to stay correlated.
_request_id: ContextVar[str] = ContextVar("request_id", default="-")

#: Anything matching these is replaced before a record leaves the process.
#: Patterns are deliberately broad: a false positive costs a redacted string
#: in a log line, a false negative costs a credential.
_SECRET_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"), "sk-ant-REDACTED"),
    (re.compile(r"\bsk-[A-Za-z0-9]{20,}"), "sk-REDACTED"),
    # AeroAPI keys are opaque; catch them by their header and query names.
    (re.compile(r"(x-apikey['\"?:=\s]+)[^\s'\"&,}]+", re.I), r"\1REDACTED"),
    (re.compile(r"(api[_-]?key['\"?:=\s]+)[^\s'\"&,}]+", re.I), r"\1REDACTED"),
    # Bearer first: an "Authorization: Bearer <token>" header matches both,
    # and the broader pattern would otherwise eat the scheme and leave the
    # token behind.
    (re.compile(r"(bearer\s+)[A-Za-z0-9._\-]+", re.I), r"\1REDACTED"),
    (re.compile(r"(authorization['\"?:=\s]+)(?!REDACTED)[^\s'\"&,}]+", re.I),
     r"\1REDACTED"),
]


def redact(text: str) -> str:
    """Strip anything that looks like a credential.

    Runs on the formatted message, so a key reaches this function whether it
    was interpolated in, passed as an argument, or embedded in an exception
    message by a library that never considered logging.
    """
    if not text:
        return text
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class RedactingFilter(logging.Filter):
    """Applies redaction and attaches the current request id.

    A filter rather than a formatter, so redaction happens once regardless
    of how many handlers are attached and cannot be bypassed by adding one.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - a broken format string is not fatal
            message = str(record.msg)
        cleaned = redact(message)
        if cleaned != message:
            record.msg = cleaned
            record.args = ()
        # A traceback is the likeliest place for a credential to appear,
        # since an exception message often carries a response body. But
        # `exc_text` is only populated during formatting, which runs after
        # filters, so the traceback has to be rendered here to be redacted.
        if record.exc_info:
            import traceback
            record.exc_text = redact(
                "".join(traceback.format_exception(*record.exc_info)).rstrip())
            record.exc_info = None
        elif record.exc_text:
            record.exc_text = redact(record.exc_text)
        record.request_id = _request_id.get()
        return True


# ------------------------------------------------------------------ setup

_configured = False


def configure(level: str | int = "INFO", *, use_syslog: bool = True,
              stream: Any = None, force: bool = False) -> logging.Logger:
    """Set up the application logger. Safe to call more than once.

    Falls back to stderr where there is no syslog socket, which covers
    containers, test runners and development machines.
    """
    global _configured
    logger = logging.getLogger(APP_NAME)
    if _configured and not force:
        return logger

    logger.handlers.clear()
    logger.setLevel(level if isinstance(level, int)
                    else getattr(logging, str(level).upper(), logging.INFO))
    logger.propagate = False

    handler: logging.Handler | None = None
    if use_syslog and os.path.exists(SYSLOG_SOCKET):
        try:
            handler = logging.handlers.SysLogHandler(address=SYSLOG_SOCKET)
            handler.setFormatter(logging.Formatter(
                f"{APP_NAME}[%(process)d]: %(levelname)s "
                f"req=%(request_id)s %(name)s %(message)s"))
        except (OSError, socket.error):
            handler = None

    if handler is None:
        handler = logging.StreamHandler(stream or sys.stderr)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s req=%(request_id)s "
            "%(name)s %(message)s", datefmt="%H:%M:%S"))

    handler.addFilter(RedactingFilter())
    logger.addHandler(handler)
    _configured = True
    return logger


def get_logger(name: str = "") -> logging.Logger:
    """A child logger. Configures the parent on first use."""
    if not _configured:
        configure(level=os.environ.get("TURBULENCE_LOG_LEVEL", "INFO"))
    return logging.getLogger(f"{APP_NAME}.{name}" if name else APP_NAME)


# ------------------------------------------------------------------ context


def new_request_id() -> str:
    """Start a new correlated request. Short: these are read by humans."""
    rid = uuid.uuid4().hex[:12]
    _request_id.set(rid)
    return rid


def set_request_id(rid: str) -> None:
    _request_id.set(rid)


def current_request_id() -> str:
    return _request_id.get()


class request_context:
    """Scopes a request id to a block, restoring the previous one after.

    A context manager rather than a bare setter so a nested search cannot
    leave its id behind for whatever runs next in the same worker.
    """

    def __init__(self, rid: str | None = None):
        self.rid = rid or uuid.uuid4().hex[:12]
        self._token = None

    def __enter__(self) -> str:
        self._token = _request_id.set(self.rid)
        return self.rid

    def __exit__(self, *exc) -> None:
        if self._token is not None:
            _request_id.reset(self._token)


# ------------------------------------------------------------------ policy


def trip_fields(origin: str, dest: str, when: str | None = None,
                include: bool | None = None) -> dict[str, Any]:
    """Trip details for a log line, subject to the content policy.

    Origin, destination and date together are an itinerary: they say where
    a person is going and when. That is a different kind of record from how
    the search behaved, and it carries a retention question the operational
    fields do not. Off unless explicitly enabled.
    """
    if include is None:
        include = os.environ.get("TURBULENCE_LOG_TRIP_CONTENT",
                                 "").lower() in ("1", "true", "yes")
    if not include:
        return {"route": "redacted"}
    return {"origin": origin, "dest": dest,
            **({"when": when} if when else {})}


def kv(**fields: Any) -> str:
    """Render fields as `key=value`, which journald and grep both handle.

    Values containing spaces are quoted; None is dropped rather than logged
    as the string "None", which reads as a value rather than an absence.
    """
    parts = []
    for key, value in fields.items():
        if value is None:
            continue
        text = str(value)
        parts.append(f'{key}="{text}"' if " " in text else f"{key}={text}")
    return " ".join(parts)
