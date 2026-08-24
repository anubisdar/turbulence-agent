"""Turnstile: the challenge in front of the searches that spend money.

There is one user and no accounts, so a login would authenticate nobody
against nothing. What the endpoint needs is a reason to believe a person
asked, because every search spends against a metered flight API and, with
an explanation on, against a model API with no fixed ceiling.
"""

import time

import pytest
from fastapi.testclient import TestClient

from app.web import turnstile


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("TURNSTILE_SITE_KEY", "0x4AAAAAAAtest_site")
    monkeypatch.setenv("TURNSTILE_SECRET_KEY", "0x4AAAAAAAtest_secret")
    return True


@pytest.fixture
def client():
    from app.web.api import app
    return TestClient(app)


class TestItIsOffUnlessConfigured:
    """Development and the test suite must be unaffected by a control that
    needs a key nobody has set."""

    def test_disabled_without_keys(self, monkeypatch):
        monkeypatch.delenv("TURNSTILE_SITE_KEY", raising=False)
        monkeypatch.delenv("TURNSTILE_SECRET_KEY", raising=False)
        assert turnstile.enabled() is False

    def test_disabled_with_only_one_key(self, monkeypatch):
        monkeypatch.setenv("TURNSTILE_SITE_KEY", "site")
        monkeypatch.delenv("TURNSTILE_SECRET_KEY", raising=False)
        assert turnstile.enabled() is False

    def test_verification_passes_when_unconfigured(self, monkeypatch):
        monkeypatch.delenv("TURNSTILE_SECRET_KEY", raising=False)
        assert turnstile.verify(None) is True

    def test_a_search_is_not_blocked_when_unconfigured(self, client,
                                                       monkeypatch):
        monkeypatch.delenv("TURNSTILE_SECRET_KEY", raising=False)
        response = client.post(
            "/api/search/corridors",
            json={"origin": "KPIT", "dest": "KBOS", "use_fixtures": True})
        assert response.status_code != 403


class TestSessions:
    """A token is single use and expires in about five minutes. A demo runs
    several searches, so the first solve issues a session."""

    def test_a_fresh_session_is_valid(self, configured):
        assert turnstile.session_is_valid(turnstile.issue_session())

    def test_an_expired_session_is_not(self, configured):
        old = turnstile.issue_session(now=time.time()
                                      - turnstile.SESSION_SECONDS - 10)
        assert turnstile.session_is_valid(old) is False

    def test_a_forged_signature_is_rejected(self, configured):
        expires = int(time.time() + 3600)
        assert turnstile.session_is_valid(f"{expires}.deadbeef") is False

    def test_an_extended_expiry_is_rejected(self, configured):
        """The obvious attack: keep the signature, move the deadline."""
        cookie = turnstile.issue_session()
        _, _, signature = cookie.partition(".")
        forged = f"{int(time.time() + 999999)}.{signature}"
        assert turnstile.session_is_valid(forged) is False

    @pytest.mark.parametrize("cookie", [
        None, "", "nonsense", "no-dot", ".", "abc.def", "12345"])
    def test_malformed_cookies_are_rejected(self, configured, cookie):
        assert turnstile.session_is_valid(cookie) is False

    def test_a_session_from_a_different_secret_is_rejected(self, monkeypatch):
        monkeypatch.setenv("TURNSTILE_SECRET_KEY", "first")
        cookie = turnstile.issue_session()
        monkeypatch.setenv("TURNSTILE_SECRET_KEY", "second")
        assert turnstile.session_is_valid(cookie) is False


class TestVerificationFailsClosed:
    """An outage that quietly disabled the challenge would be worse than one
    that refused searches."""

    def test_a_missing_token_fails(self, configured):
        assert turnstile.verify(None) is False
        assert turnstile.verify("") is False

    def test_a_network_failure_fails(self, configured, monkeypatch):
        def unreachable(*a, **kw):
            raise OSError("connection refused")
        monkeypatch.setattr("urllib.request.urlopen", unreachable)
        assert turnstile.verify("some-token") is False

    def test_a_timeout_fails(self, configured, monkeypatch):
        def slow(*a, **kw):
            raise TimeoutError("took too long")
        monkeypatch.setattr("urllib.request.urlopen", slow)
        assert turnstile.verify("some-token") is False

    def test_unparseable_output_fails(self, configured, monkeypatch):
        class Body:
            def read(self): return b"not json"
            def __enter__(self): return self
            def __exit__(self, *a): return False
        monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: Body())
        assert turnstile.verify("some-token") is False

    def test_an_explicit_rejection_fails(self, configured, monkeypatch):
        import json

        class Body:
            def read(self):
                return json.dumps({"success": False,
                                   "error-codes": ["invalid-input-response"]}
                                  ).encode()
            def __enter__(self): return self
            def __exit__(self, *a): return False
        monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: Body())
        assert turnstile.verify("some-token") is False

    def test_an_explicit_success_passes(self, configured, monkeypatch):
        import json

        class Body:
            def read(self): return json.dumps({"success": True}).encode()
            def __enter__(self): return self
            def __exit__(self, *a): return False
        monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: Body())
        assert turnstile.verify("some-token") is True

    def test_the_secret_is_never_returned_or_logged(self, configured,
                                                    monkeypatch):
        import io

        from app.logging_setup import configure

        buf = io.StringIO()
        configure(level="DEBUG", use_syslog=False, stream=buf, force=True)
        monkeypatch.setattr("urllib.request.urlopen",
                            lambda *a, **kw: (_ for _ in ()).throw(
                                OSError("boom")))
        turnstile.verify("some-token")
        assert "test_secret" not in buf.getvalue()


class TestTheEndpointGate:
    def test_the_site_key_is_published_but_not_the_secret(self, client,
                                                          configured):
        body = client.get("/api/config").json()
        assert body["turnstile_site_key"] == "0x4AAAAAAAtest_site"
        assert "secret" not in str(body).lower()

    def test_a_search_without_a_token_is_refused(self, client, configured):
        response = client.post(
            "/api/search/corridors",
            json={"origin": "KPIT", "dest": "KBOS", "use_fixtures": True})
        assert response.status_code == 403
        assert "challenge" in response.text.lower()

    def test_a_valid_session_skips_verification(self, client, configured,
                                                monkeypatch):
        """The point of the session: only the first search waits on a round
        trip to Cloudflare."""
        def should_not_be_called(*a, **kw):
            raise AssertionError("verification ran despite a valid session")
        monkeypatch.setattr(turnstile, "verify", should_not_be_called)

        client.cookies.set(turnstile.COOKIE_NAME, turnstile.issue_session())
        response = client.post(
            "/api/search/corridors",
            json={"origin": "KPIT", "dest": "KBOS", "use_fixtures": True})
        assert response.status_code != 403

    def test_a_forged_session_does_not(self, client, configured, monkeypatch):
        monkeypatch.setattr(turnstile, "verify", lambda *a, **kw: False)
        client.cookies.set(turnstile.COOKIE_NAME,
                           f"{int(time.time() + 9999)}.forged")
        response = client.post(
            "/api/search/corridors",
            json={"origin": "KPIT", "dest": "KBOS", "use_fixtures": True})
        assert response.status_code == 403

    def test_the_token_field_is_length_bounded(self, client, configured):
        response = client.post(
            "/api/search/corridors",
            json={"origin": "KPIT", "dest": "KBOS", "use_fixtures": True,
                  "turnstile_token": "x" * 5000})
        assert response.status_code == 422

    def test_the_policy_permits_the_widget(self, client):
        csp = client.get("/api/health").headers["Content-Security-Policy"]
        assert "https://challenges.cloudflare.com" in csp
        assert "frame-src" in csp


class TestChallengeTelemetry:
    """An earlier version raised a 403 and logged nothing, which left the
    most useful number invisible: how many searches presented no token at
    all, which is what an automated client looks like."""

    def _logged(self, client, configured, **kw):
        import io

        from app.logging_setup import configure

        buf = io.StringIO()
        configure(level="DEBUG", use_syslog=False, stream=buf, force=True)
        body = {"origin": "KPIT", "dest": "KBOS", "use_fixtures": True}
        body.update(kw)
        client.post("/api/search/corridors", json=body)
        return buf.getvalue()

    def test_a_missing_token_is_recorded_distinctly(self, client, configured):
        assert "challenge outcome=no_token" in self._logged(client, configured)

    def test_a_rejected_token_is_recorded_distinctly(self, client, configured,
                                                     monkeypatch):
        """Different from no token at all: one saw the page and failed, the
        other never saw it."""
        monkeypatch.setattr(turnstile, "verify", lambda *a, **kw: False)
        logged = self._logged(client, configured, turnstile_token="abc")
        assert "challenge outcome=rejected" in logged
        assert "outcome=no_token" not in logged

    def test_a_session_is_recorded(self, client, configured):
        client.cookies.set(turnstile.COOKIE_NAME, turnstile.issue_session())
        assert "challenge outcome=session" in self._logged(client, configured)

    def test_a_fresh_solve_is_recorded(self, client, configured, monkeypatch):
        monkeypatch.setattr(turnstile, "verify", lambda *a, **kw: True)
        assert "challenge outcome=solved" in self._logged(
            client, configured, turnstile_token="abc")

    def test_the_outcome_reaches_the_run_record(self):
        from app.runs import RunRecord, from_payload
        record = from_payload({"request": {}, "outcome": {}, "corridors": []},
                              "req", challenge="solved")
        assert isinstance(record, RunRecord)
        assert record.challenge == "solved"

    def test_the_summary_groups_by_outcome(self):
        import sqlite3

        from app.runs import RunRecord, init_runs, record_run, summary

        conn = sqlite3.connect(":memory:")
        init_runs(conn)
        for outcome in ("solved", "session", "session"):
            record_run(conn, RunRecord(request_id="x", challenge=outcome))
        counts = {r["outcome"]: r["n"] for r in summary(conn)["challenges"]}
        assert counts == {"session": 2, "solved": 1}

    def test_refusals_are_counted_separately_from_the_database(self):
        """A refused request never becomes a row, so it cannot be counted
        from search_runs. The journal is the only record."""
        from app.web.api import _challenge_refusals
        result = _challenge_refusals()
        assert "no_token" in result and "rejected" in result

    def test_refusals_report_disabled_rather_than_zero(self, monkeypatch):
        """Zero refusals and a disabled challenge are different facts."""
        monkeypatch.delenv("TURNSTILE_SECRET_KEY", raising=False)
        from app.web.api import _challenge_refusals
        assert _challenge_refusals()["enabled"] is False


class TestLocalRequestsAreExempt:
    """The operator scripts are automated clients, which is exactly what the
    challenge exists to refuse. Loopback and the VPC are exempt for the same
    reason the geo filter exempts them: a request from inside the instance
    is the operator, not a visitor."""

    @pytest.mark.parametrize("host", ["127.0.0.1", "::1", "172.31.29.123"])
    def test_local_addresses_are_exempt(self, host):
        from app.web.api import _is_local

        class Request:
            client = type("C", (), {"host": host})()

        assert _is_local(Request()) is True

    @pytest.mark.parametrize("host", [
        "71.116.60.24", "45.155.205.7", "", "127.0.0.1.evil.com",
        "1172.31.0.1"])
    def test_public_addresses_are_not(self, host):
        from app.web.api import _is_local

        class Request:
            client = type("C", (), {"host": host})()

        assert _is_local(Request()) is False

    def test_a_forwarded_header_cannot_claim_to_be_local(self, client,
                                                         configured):
        """The exemption reads the socket address, which a caller does not
        control. A forwarded request arrives from Caddy with the real
        client in a header, and Caddy is not on the exempt list."""
        response = client.post(
            "/api/search/corridors",
            json={"origin": "KPIT", "dest": "KBOS", "use_fixtures": True},
            headers={"X-Forwarded-For": "127.0.0.1"})
        # TestClient presents itself as testclient, not a local address.
        assert response.status_code == 403

    def test_the_exemption_is_recorded(self, client, configured,
                                       monkeypatch):
        """Otherwise a search that skipped the challenge is
        indistinguishable from one that solved it."""
        import io

        import app.web.api as api
        from app.logging_setup import configure

        monkeypatch.setattr(api, "_is_local", lambda request: True)
        buf = io.StringIO()
        configure(level="DEBUG", use_syslog=False, stream=buf, force=True)
        client.post("/api/search/corridors",
                    json={"origin": "KPIT", "dest": "KBOS",
                          "use_fixtures": True})
        assert "challenge outcome=local" in buf.getvalue()


class TestOperatorToken:
    """The load test and the validation scripts are automated clients,
    which is exactly what the challenge exists to refuse. The token is how
    an operator says so rather than pretending to be a browser."""

    GOOD = "k" * 40

    @pytest.fixture
    def token(self, monkeypatch):
        monkeypatch.setenv("TURBULENCE_OPERATOR_TOKEN", self.GOOD)
        return self.GOOD

    def test_the_correct_token_matches(self, token):
        assert turnstile.operator_token_matches(token) is True

    def test_a_wrong_token_does_not(self, token):
        assert turnstile.operator_token_matches("x" * 40) is False

    @pytest.mark.parametrize("presented", [None, "", "   "])
    def test_an_absent_token_does_not(self, token, presented):
        assert turnstile.operator_token_matches(presented) is False

    @pytest.mark.parametrize("presented", [None, "", "anything at all"])
    def test_an_unconfigured_token_never_matches(self, monkeypatch,
                                                 presented):
        """The failure mode worth guarding: comparing two empty values
        succeeds, and every request bypasses the challenge. A total failure
        that looks like nothing at all."""
        monkeypatch.delenv("TURBULENCE_OPERATOR_TOKEN", raising=False)
        assert turnstile.operator_token_matches(presented) is False

    @pytest.mark.parametrize("configured", ["", "short", "k" * 23])
    def test_a_short_token_disables_rather_than_opens(self, monkeypatch,
                                                      configured):
        """Presenting exactly the configured value still fails, so a weak
        token cannot be used at all rather than being used weakly."""
        monkeypatch.setenv("TURBULENCE_OPERATOR_TOKEN", configured)
        assert turnstile.operator_token_matches(configured) is False

    def test_it_lets_a_search_through(self, client, configured, token):
        response = client.post(
            "/api/search/corridors",
            json={"origin": "KPIT", "dest": "KBOS", "use_fixtures": True},
            headers={"X-Operator-Token": token})
        assert response.status_code != 403

    def test_a_wrong_one_does_not(self, client, configured, token):
        response = client.post(
            "/api/search/corridors",
            json={"origin": "KPIT", "dest": "KBOS", "use_fixtures": True},
            headers={"X-Operator-Token": "x" * 40})
        assert response.status_code == 403

    def test_its_use_is_recorded(self, client, configured, token):
        """Otherwise a search that skipped the challenge is
        indistinguishable from one that solved it."""
        import io

        from app.logging_setup import configure

        buf = io.StringIO()
        configure(level="DEBUG", use_syslog=False, stream=buf, force=True)
        client.post("/api/search/corridors",
                    json={"origin": "KPIT", "dest": "KBOS",
                          "use_fixtures": True},
                    headers={"X-Operator-Token": token})
        assert "challenge outcome=operator" in buf.getvalue()

    def test_the_token_is_never_logged(self, client, configured, token):
        import io

        from app.logging_setup import configure

        buf = io.StringIO()
        configure(level="DEBUG", use_syslog=False, stream=buf, force=True)
        client.post("/api/search/corridors",
                    json={"origin": "KPIT", "dest": "KBOS",
                          "use_fixtures": True},
                    headers={"X-Operator-Token": token})
        assert token not in buf.getvalue()
