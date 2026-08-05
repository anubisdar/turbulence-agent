"""Shared test fixtures.

Everything here exists to keep the test suite offline: `_block_network` is autouse,
so any test that reaches a real socket through httpx fails loudly instead of
silently depending on aviationweather.gov being up.
"""

from __future__ import annotations

import json
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
