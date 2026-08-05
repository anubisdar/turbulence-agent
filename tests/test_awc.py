"""Tests for app.sources.awc.

All AWC responses come from recorded fixtures (see tests/fixtures/README.md).
The autouse `_block_network` fixture in conftest.py fails the test if anything
reaches for a real socket.
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from app.config import Settings
from app.sources import awc
from app.sources.awc import (
    AwcFetchError,
    PilotReport,
    TurbulenceSeverity,
    fetch_pireps,
    parse_turbulence_severity,
)

pytestmark = pytest.mark.anyio

NE_BBOX = (38.0, -82.0, 45.0, -68.0)

TEST_SETTINGS = Settings(
    awc_user_agent="turbulence-agent-test/0.1 (test@example.com)",
    awc_base_url="https://aviationweather.example/api/data",
)


def _client(handler: Any) -> httpx.AsyncClient:
    """An AsyncClient wired to a MockTransport - never touches the network."""
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _json_responder(payload: Any, status_code: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=payload)

    return handler


def _recording_handler(payload: Any) -> tuple[list[httpx.Request], Any]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=payload)

    return seen, handler


async def _fetch(payload: Any, **kwargs: Any) -> list[PilotReport]:
    bbox = kwargs.pop("bbox", NE_BBOX)
    hours_back = kwargs.pop("hours_back", 6)
    async with _client(_json_responder(payload)) as client:
        return await fetch_pireps(
            bbox, hours_back, client=client, settings=TEST_SETTINGS, **kwargs
        )


# --------------------------------------------------------------------------
# request construction
# --------------------------------------------------------------------------


async def test_request_targets_pirep_endpoint_with_geojson_output(ne_us_geojson):
    seen, handler = _recording_handler(ne_us_geojson)

    async with _client(handler) as client:
        await fetch_pireps(NE_BBOX, 6, client=client, settings=TEST_SETTINGS)

    (request,) = seen
    assert request.url.path == "/api/data/pirep"
    assert request.url.params["format"] == "geojson"
    assert request.url.params["age"] == "6"
    # AWC bbox order is (min_lat, min_lon, max_lat, max_lon)
    assert request.url.params["bbox"] == "38.0,-82.0,45.0,-68.0"


async def test_user_agent_comes_from_config(ne_us_geojson):
    seen, handler = _recording_handler(ne_us_geojson)

    async with _client(handler) as client:
        await fetch_pireps(NE_BBOX, 6, client=client, settings=TEST_SETTINGS)

    assert seen[0].headers["user-agent"] == TEST_SETTINGS.awc_user_agent


async def test_fractional_hours_back_is_passed_through(ne_us_geojson):
    seen, handler = _recording_handler(ne_us_geojson)

    async with _client(handler) as client:
        await fetch_pireps(NE_BBOX, 1.5, client=client, settings=TEST_SETTINGS)

    assert seen[0].url.params["age"] == "1.5"


async def test_the_offline_guard_is_armed():
    """Meta-test: without a mock transport, a fetch must not reach the internet."""
    with pytest.raises(AssertionError, match="live network call"):
        await fetch_pireps(NE_BBOX, 6, settings=TEST_SETTINGS)


async def test_network_call_goes_through_a_monkeypatchable_seam(
    monkeypatch, ne_us_geojson
):
    """CLAUDE.md: every external call must be patchable without a live client."""
    calls: list[tuple[Any, Any]] = []

    async def fake(bbox, hours_back, *, settings, client=None):
        calls.append((bbox, hours_back))
        return ne_us_geojson

    monkeypatch.setattr(awc, "fetch_pirep_geojson", fake)

    reports = await fetch_pireps(NE_BBOX, 6, settings=TEST_SETTINGS)

    assert calls == [(NE_BBOX, 6)]
    assert len(reports) == 55


# --------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bbox",
    [
        (45.0, -82.0, 38.0, -68.0),  # lat inverted
        (38.0, -68.0, 45.0, -82.0),  # lon inverted
        (38.0, -82.0, 95.0, -68.0),  # lat out of range
        (38.0, -200.0, 45.0, -68.0),  # lon out of range
        (38.0, -82.0, 45.0),  # too short
    ],
)
async def test_invalid_bbox_is_rejected(bbox, ne_us_geojson):
    with pytest.raises(ValueError):
        await _fetch(ne_us_geojson, bbox=bbox)


@pytest.mark.parametrize("hours_back", [0, -1, float("nan"), float("inf")])
async def test_invalid_hours_back_is_rejected(hours_back, ne_us_geojson):
    with pytest.raises(ValueError):
        await _fetch(ne_us_geojson, hours_back=hours_back)


# --------------------------------------------------------------------------
# parsing the recorded fixture
# --------------------------------------------------------------------------


async def test_parses_every_feature_in_the_recording(ne_us_geojson):
    reports = await _fetch(ne_us_geojson)

    assert len(reports) == len(ne_us_geojson["features"]) == 55
    assert all(isinstance(r, PilotReport) for r in reports)


async def test_first_report_fields(ne_us_geojson):
    """The 2157Z CRJ9 out of BOS, field by field."""
    reports = await _fetch(ne_us_geojson)
    report = next(r for r in reports if r.raw_text.startswith("BOS UA"))

    assert report.latitude == pytest.approx(42.3575)
    assert report.longitude == pytest.approx(-70.9895)
    assert report.altitude_ft == 28_000
    assert report.turbulence_severity is TurbulenceSeverity.LIGHT
    assert report.aircraft_type == "CRJ9"
    assert report.observation_time == datetime(2026, 8, 4, 21, 57, tzinfo=timezone.utc)
    assert report.raw_text == (
        "BOS UA /OV BOS/TM 2157/FL280/TP CRJ9/TB CONS LGT MOD CHOP 280 NEG 260"
        "/RM PARCH4 JFK"
    )
    assert report.source == "AWC"


async def test_observation_times_are_timezone_aware_utc(ne_us_geojson):
    reports = await _fetch(ne_us_geojson)

    assert reports, "fixture should not be empty"
    for report in reports:
        assert report.observation_time.tzinfo is not None
        assert report.observation_time.utcoffset() == timedelta(0)


async def test_flight_level_becomes_feet(ne_us_geojson):
    reports = await _fetch(ne_us_geojson)
    by_raw = {r.raw_text: r for r in reports}

    assert (
        by_raw["ROC UA /OV ROC/TM 2130/FL350/TP E75S/TB CONS MOD CHOP"].altitude_ft
        == 35_000
    )
    assert by_raw["SBY UA /OV KSBY/TM 2125/FL020/TP E145"].altitude_ft == 2_000


async def test_missing_flight_level_is_none_not_zero(ne_us_geojson):
    """A during-climb report carries no flight level; don't invent one."""
    reports = await _fetch(ne_us_geojson)
    durc = next(r for r in reports if r.raw_text.startswith("AGC UA /OV AGC/TM 1947"))

    assert durc.altitude_ft is None


async def test_reports_are_sorted_newest_first(ne_us_geojson):
    reports = await _fetch(ne_us_geojson)
    times = [r.observation_time for r in reports]

    assert times == sorted(times, reverse=True)


async def test_fetched_at_is_utc_and_shared_across_the_batch(ne_us_geojson):
    before = datetime.now(timezone.utc)
    reports = await _fetch(ne_us_geojson)
    after = datetime.now(timezone.utc)

    stamps = {r.fetched_at for r in reports}
    assert len(stamps) == 1, "one fetch = one provenance timestamp"
    (stamp,) = stamps
    assert stamp.tzinfo is not None and stamp.utcoffset() == timedelta(0)
    assert before <= stamp <= after


# --------------------------------------------------------------------------
# turbulence severity - CLAUDE.md rule 3: no severity defaults
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("NEG", TurbulenceSeverity.NONE),
        ("NEGATIVE", TurbulenceSeverity.NONE),
        ("SMTH", TurbulenceSeverity.NONE),
        ("SMOOTH", TurbulenceSeverity.NONE),
        ("NIL", TurbulenceSeverity.NONE),
        ("NONE", TurbulenceSeverity.NONE),
        ("LGT", TurbulenceSeverity.LIGHT),
        ("LIGHT", TurbulenceSeverity.LIGHT),
        ("LT", TurbulenceSeverity.LIGHT),
        ("MOD", TurbulenceSeverity.MODERATE),
        ("MODERATE", TurbulenceSeverity.MODERATE),
        ("MDT", TurbulenceSeverity.MODERATE),
        ("SEV", TurbulenceSeverity.SEVERE),
        ("SEVERE", TurbulenceSeverity.SEVERE),
        ("EXTM", TurbulenceSeverity.EXTREME),
        ("EXTRM", TurbulenceSeverity.EXTREME),
        ("EXTREME", TurbulenceSeverity.EXTREME),
        ("  mod  ", TurbulenceSeverity.MODERATE),
    ],
)
def test_severity_codes_map_to_the_enum(code, expected):
    assert parse_turbulence_severity(code) is expected


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("LGT-MOD", TurbulenceSeverity.MODERATE),
        ("MOD-SEV", TurbulenceSeverity.SEVERE),
        ("SEV-EXTM", TurbulenceSeverity.EXTREME),
        ("NEG-LGT", TurbulenceSeverity.LIGHT),
    ],
)
def test_intensity_ranges_report_the_worst_end(code, expected):
    """A LGT-MOD report can be moderate; ranking for an anxious flier rounds up."""
    assert parse_turbulence_severity(code) is expected


@pytest.mark.parametrize("code", [None, "", "   ", "WAT", "???", "-", "///"])
def test_unrecognised_or_absent_codes_yield_none(code):
    """Rule 3: absence of data never becomes an assumed severity."""
    assert parse_turbulence_severity(code) is None


def test_partly_recognised_range_keeps_only_what_was_coded():
    assert parse_turbulence_severity("LGT-WAT") is TurbulenceSeverity.LIGHT


async def test_report_with_no_coded_turbulence_has_severity_none(ne_us_geojson):
    """Rule 3, end to end: no tbInt field must not become LIGHT."""
    reports = await _fetch(ne_us_geojson)
    by_raw = {r.raw_text: r for r in reports}

    # no /TB group at all
    assert by_raw["SBY UA /OV KSBY/TM 2125/FL020/TP E145"].turbulence_severity is None
    # a /TB group with a type but no intensity
    assert (
        by_raw["POU UA /OV IGN/TM 1930/FL300/TP CRJ7/TB CONS CHOP"].turbulence_severity
        is None
    )
    # intensity mentioned only in a free-text remark, never coded by AWC
    assert (
        by_raw["AGC UA /OV AGC/TM 1755/FL020/TP C172/RM LIGHT TURB"].turbulence_severity
        is None
    )


async def test_negative_report_is_none_enum_not_missing(ne_us_geojson):
    """'Smooth ride' is data. It is distinct from 'we don't know'."""
    reports = await _fetch(ne_us_geojson)
    neg = next(r for r in reports if r.raw_text.startswith("ALB UA /OV ALB140020"))

    assert neg.turbulence_severity is TurbulenceSeverity.NONE
    assert neg.turbulence_severity is not None


async def test_severe_reports_from_recording(severe_geojson):
    reports = await _fetch(severe_geojson)

    assert len(reports) == 3
    assert {r.turbulence_severity for r in reports} == {TurbulenceSeverity.SEVERE}
    urgent = next(r for r in reports if r.raw_text.startswith("DRO UUA"))
    assert urgent.altitude_ft == 10_500
    assert urgent.aircraft_type == "BE95"


async def test_second_turbulence_group_can_raise_the_severity(ne_us_geojson):
    """AWC splits layered turbulence across tbInt1/tbInt2; keep the worst."""
    payload = copy.deepcopy(ne_us_geojson)
    payload["features"] = [payload["features"][0]]
    payload["features"][0]["properties"]["tbInt1"] = "LGT"
    payload["features"][0]["properties"]["tbInt2"] = "SEV"

    (report,) = await _fetch(payload)

    assert report.turbulence_severity is TurbulenceSeverity.SEVERE


# --------------------------------------------------------------------------
# malformed input
# --------------------------------------------------------------------------


async def test_empty_feature_collection_returns_empty_list():
    assert await _fetch({"type": "FeatureCollection", "features": []}) == []


async def test_missing_features_key_returns_empty_list():
    assert await _fetch({"type": "FeatureCollection"}) == []


async def test_unusable_features_are_skipped_not_fatal(ne_us_geojson):
    payload = copy.deepcopy(ne_us_geojson)
    good = payload["features"][0]
    payload["features"] = [
        good,
        {"type": "Feature", "properties": {"rawOb": "x"}, "geometry": None},
        {
            "type": "Feature",
            "properties": {"rawOb": "x", "obsTime": "2026-08-04T21:00:00.000Z"},
            "geometry": {"type": "Point", "coordinates": []},
        },
        {
            "type": "Feature",
            "properties": {"rawOb": "x"},
            "geometry": {"type": "Point", "coordinates": [-70.0, 42.0]},
        },  # no obsTime
        {
            "type": "Feature",
            "properties": {"rawOb": "x", "obsTime": "not a timestamp"},
            "geometry": {"type": "Point", "coordinates": [-70.0, 42.0]},
        },
        "not a feature at all",
    ]

    reports = await _fetch(payload)

    assert len(reports) == 1
    assert reports[0].raw_text == good["properties"]["rawOb"]


async def test_epoch_observation_times_are_accepted(ne_us_geojson):
    """Some AWC endpoints return obsTime as epoch seconds rather than ISO."""
    payload = copy.deepcopy(ne_us_geojson)
    payload["features"] = [payload["features"][0]]
    payload["features"][0]["properties"]["obsTime"] = 1_785_880_620

    (report,) = await _fetch(payload)

    assert report.observation_time == datetime(2026, 8, 4, 21, 57, tzinfo=timezone.utc)


# --------------------------------------------------------------------------
# failure modes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("status", [400, 429, 500, 502, 503])
async def test_http_error_status_raises_awc_fetch_error(status):
    async with _client(_json_responder({"error": "nope"}, status_code=status)) as client:
        with pytest.raises(AwcFetchError) as excinfo:
            await fetch_pireps(NE_BBOX, 6, client=client, settings=TEST_SETTINGS)

    assert str(status) in str(excinfo.value)


async def test_non_json_body_raises_awc_fetch_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>maintenance</html>")

    async with _client(handler) as client:
        with pytest.raises(AwcFetchError):
            await fetch_pireps(NE_BBOX, 6, client=client, settings=TEST_SETTINGS)


async def test_json_that_is_not_an_object_raises_awc_fetch_error():
    with pytest.raises(AwcFetchError):
        await _fetch(["not", "a", "feature", "collection"])


async def test_transport_error_raises_awc_fetch_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out", request=request)

    async with _client(handler) as client:
        with pytest.raises(AwcFetchError):
            await fetch_pireps(NE_BBOX, 6, client=client, settings=TEST_SETTINGS)


# --------------------------------------------------------------------------
# the model itself
# --------------------------------------------------------------------------


def _model_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "latitude": 42.0,
        "longitude": -70.0,
        "altitude_ft": 30_000,
        "turbulence_severity": TurbulenceSeverity.LIGHT,
        "aircraft_type": "B738",
        "observation_time": datetime(2026, 8, 4, 21, 57, tzinfo=timezone.utc),
        "raw_text": "x",
        "fetched_at": datetime(2026, 8, 4, 22, 12, tzinfo=timezone.utc),
    }
    kwargs.update(overrides)
    return kwargs


def test_pilot_report_rejects_naive_observation_time():
    with pytest.raises(ValidationError):
        PilotReport(**_model_kwargs(observation_time=datetime(2026, 8, 4, 21, 57)))


def test_pilot_report_normalises_offset_times_to_utc():
    report = PilotReport(
        **_model_kwargs(
            observation_time=datetime(
                2026, 8, 4, 16, 57, tzinfo=timezone(timedelta(hours=-5))
            )
        )
    )

    assert report.observation_time == datetime(2026, 8, 4, 21, 57, tzinfo=timezone.utc)
    assert report.observation_time.utcoffset() == timedelta(0)


def test_pilot_report_rejects_out_of_range_coordinates():
    with pytest.raises(ValidationError):
        PilotReport(**_model_kwargs(latitude=91.0))


def test_pilot_report_allows_everything_optional_to_be_missing():
    report = PilotReport(
        **_model_kwargs(altitude_ft=None, turbulence_severity=None, aircraft_type=None)
    )

    assert report.turbulence_severity is None


def test_severity_ranks_worst_last():
    ordered = sorted(TurbulenceSeverity, key=lambda s: s.rank)

    assert ordered == [
        TurbulenceSeverity.NONE,
        TurbulenceSeverity.LIGHT,
        TurbulenceSeverity.MODERATE,
        TurbulenceSeverity.SEVERE,
        TurbulenceSeverity.EXTREME,
    ]
