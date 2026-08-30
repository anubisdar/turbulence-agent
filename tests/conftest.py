"""Shared test fixtures.

Two things are kept away from the suite here.

`_block_network` is autouse, so any test that reaches a real socket through
httpx fails loudly instead of silently depending on aviationweather.gov
being up.

`_isolate_the_database` is autouse for the same reason applied to storage.
`_db_path()` falls back to data/retrieval.db when TURBULENCE_DB is unset,
and thirty tests reach the corridor search endpoint, so every deploy - which
runs the full suite first - left about eighty rows in the live run record.
They were not obviously fake: a reading, a corridor, a degraded reason, a
timestamp. What gave them away was the shape, 0.9 seconds where a real
search takes fourteen, and no resolved country, because a fixture run never
resolves an origin. The status page counted them as behaviour three separate
times before anyone noticed.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import httpx
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict[str, Any]:
    """Load a recorded API response by filename."""
    return json.loads((FIXTURE_DIR / name).read_text())


@pytest.fixture
def anyio_backend() -> str:
    """anyio's pytest plugin runs `@pytest.mark.anyio` tests on this backend."""
    return "asyncio"


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that tries to open a real connection."""
    def _refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(
            "test attempted a live network call - use a recorded fixture instead"
        )

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", _refuse)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _refuse)


@pytest.fixture
def ne_us_geojson() -> dict[str, Any]:
    """55 real PIREPs/AIREPs over the northeastern US, 6 hours to 2026-08-04T22:12Z."""
    return load_fixture("pirep_geojson_ne_us.json")


@pytest.fixture
def severe_geojson() -> dict[str, Any]:
    """Three real severe-turbulence reports (two of them urgent PIREPs)."""
    return load_fixture("pirep_geojson_severe.json")


# --------------------------------------------------------------- database

#: Resolved against this file rather than the working directory. pytest
#: does not guarantee cwd during fixture setup, and a relative path
#: silently missed - the copy below was skipped and the retrieval tables
#: went absent, which surfaced as "no such table: case_aircraft".
REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DB = str(REPO_ROOT / "data" / "retrieval.db")


@pytest.fixture(scope="session", autouse=True)
def _isolate_the_database(tmp_path_factory) -> Path:
    """Point the suite at a copy of the database for the whole session.

    A copy rather than an empty file, because the retrieval index and the
    run record share one file: starting empty takes the NTSB corpus away
    along with search_runs, and the reputation tests then fail looking for
    a table that is only missing because of this fixture.

    Session-scoped because the variable is read inside request handlers,
    and a function-scoped fixture would leave a window where a background
    thread could still resolve the old path.
    """
    scratch = tmp_path_factory.mktemp("db") / "test-retrieval.db"

    source = Path(PRODUCTION_DB)
    if source.exists():
        # sqlite3's backup API rather than a file copy: it takes a
        # consistent snapshot including anything still in a write-ahead
        # log, which a file copy would leave behind in the sidecar.
        src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
        dst = sqlite3.connect(scratch)
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()

    saved = {
        "TURBULENCE_DB": os.environ.get("TURBULENCE_DB"),
        # Trip content stays out of test logs for the same reason it stays
        # out of production ones.
        "TURBULENCE_LOG_TRIP_CONTENT": os.environ.get(
            "TURBULENCE_LOG_TRIP_CONTENT"),
    }
    os.environ["TURBULENCE_DB"] = str(scratch)
    os.environ["TURBULENCE_LOG_TRIP_CONTENT"] = "0"

    yield scratch

    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture(scope="session", autouse=True)
def _refuse_to_write_to_production(_isolate_the_database):
    """Fail loudly if anything still reaches the real database.

    The redirect above is the fix; this is the alarm. A test that hard-codes
    the path, or a code path that ignores the environment variable, would
    otherwise write to the live record silently and be found weeks later in
    a metric that looked wrong for no visible reason.

    Reads stay allowed - two tests deliberately check the shipped retrieval
    index has its tables, and a read cannot pollute anything.
    """
    real = Path(PRODUCTION_DB).resolve()
    original_connect = sqlite3.connect

    class _ReadOnlyToProduction(sqlite3.Connection):
        """Connection.execute is read-only on the C type, so it cannot be
        patched on an instance; a factory subclass is the supported way."""

        _WRITES = ("INSERT", "UPDATE", "DELETE", "REPLACE",
                   "CREATE", "DROP", "ALTER", "TRUNCATE")

        def _check(self, sql: Any) -> None:
            first = str(sql).lstrip().split(None, 1)[:1]
            if first and first[0].upper() in self._WRITES:
                raise AssertionError(
                    f"a test tried to write to the production database at "
                    f"{real}. Use the TURBULENCE_DB scratch path from "
                    f"conftest, or pass db_path explicitly.")

        def execute(self, sql, *a, **k):
            self._check(sql)
            return super().execute(sql, *a, **k)

        def executemany(self, sql, *a, **k):
            self._check(sql)
            return super().executemany(sql, *a, **k)

        def executescript(self, sql, *a, **k):
            raise AssertionError(
                f"a test tried to run a script against the production "
                f"database at {real}.")

    def guarded(database, *args, **kwargs):
        try:
            target = Path(str(database)).resolve()
        except (OSError, ValueError, TypeError):
            return original_connect(database, *args, **kwargs)
        if target == real and "factory" not in kwargs:
            kwargs["factory"] = _ReadOnlyToProduction
        return original_connect(database, *args, **kwargs)

    sqlite3.connect = guarded
    yield
    sqlite3.connect = original_connect
