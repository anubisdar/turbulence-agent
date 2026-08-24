# install-to: app/web
"""
Cloudflare Turnstile: a challenge in front of the searches that cost money.

WHY THIS RATHER THAN A LOGIN. There is one user and no accounts, so a login
page would authenticate nobody against nothing. What the endpoint actually
needs is a reason to believe a human asked, because every search spends
against a metered flight API and, with an explanation switched on, against a
model API with no fixed ceiling.

WHY TURNSTILE RATHER THAN reCAPTCHA. It is free, it usually resolves without
the visitor doing anything, and it does not ship visitor data to an
advertising company - which matters here given trip content is deliberately
withheld from the logs.

WHAT IT DOES NOT DO. It raises the cost of starting, not the cost of
continuing. Anyone who solves it once and then scripts against the resulting
session is unaffected. The rate limit at the edge is still the control that
bounds spend, and the console spend cap is still the one that survives every
other failure.

THREE DECISIONS, MADE DELIBERATELY.

  A session follows the first solve. Turnstile tokens are single-use and
  expire in about five minutes, and a demo runs several searches. Without a
  session the visitor would be challenged on each one.

  Verification fails closed. If Cloudflare cannot be reached the search is
  refused rather than allowed, because the alternative is that an outage
  silently removes the control.

  It is off unless configured. With no secret set the module allows
  everything, so development and the test suite are unaffected.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

from app.logging_setup import get_logger, kv

log = get_logger("turnstile")

VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"

#: How long a solve is honoured for. Long enough to run a demo without a
#: second challenge, short enough that a shared machine does not stay open.
SESSION_SECONDS = 2 * 60 * 60

COOKIE_NAME = "turbulence_session"

#: Cloudflare's own timeout is generous; ours is not. A verification that
#: takes longer than this has already cost more than the search would.
VERIFY_TIMEOUT = 6.0


def site_key() -> str | None:
    """The public half, rendered into the page."""
    return os.environ.get("TURNSTILE_SITE_KEY") or None


def _secret() -> str | None:
    return os.environ.get("TURNSTILE_SECRET_KEY") or None


def enabled() -> bool:
    return bool(_secret() and site_key())


#: Minimum length for the operator token. Short enough to type, long enough
#: that guessing it is not a strategy.
MIN_OPERATOR_TOKEN = 24


def operator_token_matches(presented: str | None) -> bool:
    """Whether a caller presented the operator token.

    For the scripts that drive this system from outside the instance: the
    load test and the validation runs are automated clients, which is
    exactly what the challenge exists to refuse, and they cannot solve it.

    THE FAILURE MODE THIS GUARDS. With no token configured, a naive
    comparison of two empty values succeeds and every request bypasses the
    challenge - a total failure that looks like nothing at all. So an
    unset or short token disables the mechanism rather than opening it.
    """
    configured = os.environ.get("TURBULENCE_OPERATOR_TOKEN") or ""
    if len(configured) < MIN_OPERATOR_TOKEN:
        return False
    if not presented:
        return False
    return hmac.compare_digest(configured, presented)


def _signing_key() -> bytes:
    """The key that signs session cookies.

    Separate from the Turnstile secret by derivation rather than by
    configuration, so there is one fewer thing to set and the two are still
    not the same value. An explicit TURBULENCE_SESSION_SECRET overrides it.
    """
    explicit = os.environ.get("TURBULENCE_SESSION_SECRET")
    if explicit:
        return explicit.encode()
    return hashlib.sha256(
        b"turbulence-session|" + (_secret() or "").encode()).digest()


def issue_session(now: float | None = None) -> str:
    """A signed expiry. No identity, because there is none to carry."""
    expires = int((now or time.time()) + SESSION_SECONDS)
    payload = str(expires).encode()
    signature = hmac.new(_signing_key(), payload, hashlib.sha256).hexdigest()
    return f"{expires}.{signature}"


def session_is_valid(cookie: str | None, now: float | None = None) -> bool:
    """Whether a cookie was issued by us and has not expired."""
    if not cookie or "." not in cookie:
        return False
    expires_raw, _, signature = cookie.partition(".")
    try:
        expires = int(expires_raw)
    except ValueError:
        return False

    expected = hmac.new(_signing_key(), expires_raw.encode(),
                        hashlib.sha256).hexdigest()
    # Constant time: a comparison that returns early leaks how much of a
    # forged signature was correct.
    if not hmac.compare_digest(expected, signature):
        return False
    return (now or time.time()) < expires


def verify(token: str | None, remote_ip: str | None = None) -> bool:
    """Ask Cloudflare whether this token is good.

    Returns False on anything that is not an explicit success, including a
    network failure. Failing closed is the point: an outage that quietly
    disabled the challenge would be worse than one that refused searches.
    """
    secret = _secret()
    if not secret:
        return True
    if not token:
        return False

    fields = {"secret": secret, "response": token}
    if remote_ip:
        fields["remoteip"] = remote_ip.split(",")[0].strip()

    request = urllib.request.Request(
        VERIFY_URL, data=urllib.parse.urlencode(fields).encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"})

    try:
        with urllib.request.urlopen(request, timeout=VERIFY_TIMEOUT) as resp:
            body = json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError, ValueError, TimeoutError) as e:
        log.warning("turnstile verification could not be completed "
                    + kv(error=type(e).__name__))
        return False

    if body.get("success"):
        return True

    # Cloudflare names the reason, and the names are worth keeping: a
    # steady stream of `timeout-or-duplicate` means the page is reusing a
    # token, which is a bug rather than an attack.
    log.warning("turnstile rejected a token "
                + kv(errors=",".join(body.get("error-codes") or ["unknown"])))
    return False
