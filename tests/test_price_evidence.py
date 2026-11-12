"""Synthetic, offline checks for price-policy and correction-source evidence."""

from copy import deepcopy
import json
from pathlib import Path
import socket

import numpy as np
import pandas as pd
import pytest

from robo_advisor.data.collectors import CollectionResult
from robo_advisor.data.importers import TEAM12_ASSETS
from robo_advisor.data.price_evidence import audit_price_evidence
from robo_advisor.data.snapshot import Snapshot


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Price evidence tests must not access external APIs")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


@pytest.fixture
def sources():
    dates = pd.bdate_range("2020-03-09", periods=3, name="Date")
    parents, selected = {}, {}
    for asset in TEAM12_ASSETS:
        domestic = asset in TEAM12_ASSETS[-2:]
        frame = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0],
                "High": [102.0, 103.0, 104.0],
                "Low": [98.0, 99.0, 100.0],
                "Close": [100.0, 101.0, 102.0],
                "Adj Close": [95.0, 95.95, 96.9],
                "Volume": [0.0, 10.0, 20.0],
            },
            index=dates,
        )
        if asset == "KTB10Y":
            frame.loc[:, ["Open", "High", "Low", "Close"]] = [
                127475.0,
                127500.0,
                127400.0,
                127500.0,
            ]
            frame.loc[dates[1], "Low"] = 127500.0
        status = "failed" if asset == "KTB10Y" else "unverified"
        metadata = {
            "dataset_id": asset,
            "symbol": {"KODEX_200TR": "278530.KS", "KTB10Y": "148070.KS"}.get(asset, asset),
            "source": "yahoo",
            "role": "etf",
            "market": "KRX" if domestic else "US",
            "currency": "KRW" if domestic else "USD",
            "timezone": "Asia/Seoul" if domestic else "America/New_York",
            "price_basis": "Synthetic archived Close and Adj Close; adjustment unverified",
            "auto_adjust": None,
            "availability_status": "unverified",
            "available_at": None,
            "requested_start": "2020-03-09",
            "requested_end_exclusive": "2020-03-12",
            "status": status,
        }
        validation = {"status": status, "issues": [{"code": "synthetic_source_status"}]}
        parents[asset] = CollectionResult(frame, frame.copy(deep=True), metadata, validation)
        adopted = frame.copy(deep=True)
        if asset == "KTB10Y":
            adopted.loc[dates[1], "Low"] = 126795.0
        selected[asset] = CollectionResult(
            adopted,
            frame.copy(deep=True),
            {
                **metadata,
                "source": (
                    "yahoo_krx_low_correction" if asset == "KTB10Y" else "yahoo_archived_close"
                ),
                "status": "unverified",
            },
            {
                "status": "unverified",
                "training_approved": False,
                "parent_validation": deepcopy(validation),
            },
        )
    candidate = CollectionResult(
        selected["KTB10Y"].data.loc[[dates[1]], ["Open", "High", "Low", "Close", "Volume"]].copy(),
        None,
        {
            "dataset_id": "KTB10Y_KRX_CORRECTION",
            "source": "krx",
            "symbol": "148070",
            "role": "etf",
            "market": "KRX",
            "currency": "KRW",
            "timezone": "Asia/Seoul",
            "status": "unverified",
        },
        {"status": "unverified", "training_approved": False},
    )
    snapshots = [
        Snapshot(Path("."), {"snapshot_id": name}, datasets, digit * 64)
        for name, datasets, digit in (
            ("parent", parents, "1"),
            ("selected", selected, "2"),
            ("candidate", {"KTB10Y_KRX_CORRECTION": candidate}, "3"),
        )
    ]
    return (*snapshots, {"US": dates, "KRX": dates})


def _run(sources):
    parent, selected, candidate, calendars = sources
    return audit_price_evidence(parent, selected, candidate, calendars=calendars)


def _codes(report):
    return {issue["code"] for issue in report["issues"]}


def test_reviewed_low_delta_is_traced_and_inputs_are_unchanged(sources):
    original = deepcopy(sources)
    report = _run(sources)
    comparison = report["metrics"]["ktb_comparison"]
    assert report["status"] == "failed"  # The original failed source is preserved.
    assert report["training_approved"] is False
    assert comparison["status"] == "passed"
    assert comparison["metrics"]["parent_status"] == "failed"
    assert comparison["metrics"]["parent_raw_preserved"] is True
    assert comparison["metrics"]["parent_validation_preserved"] is True
    assert comparison["metrics"]["changed_cells"] == [
        {"date": "2020-03-10", "column": "Low", "old": 127500.0, "new": 126795.0}
    ]
    json.dumps(report, allow_nan=False)
    for before, after in zip(original[:3], sources[:3]):
        for name in before.datasets:
            pd.testing.assert_frame_equal(before.datasets[name].data, after.datasets[name].data)
            assert before.datasets[name].metadata == after.datasets[name].metadata
            assert before.datasets[name].validation == after.datasets[name].validation


@pytest.mark.parametrize("equal", [True, False])
def test_close_adjusted_relationship_never_proves_dividend_or_split_policy(sources, equal):
    for snapshot in sources[:2]:
        result = snapshot.datasets["GLD"]
        result.data["Adj Close"] = result.data["Close"] * (1.0 if equal else 0.95)
    report = _run(sources)
    source = report["metrics"]["assets"]["GLD"]["selected"]
    ratio = source["metrics"]["close_adjusted_ratio"]
    assert source["status"] == "unverified"
    assert ratio["equal_close_rows"] == (3 if equal else 0)
    assert ratio["different_close_rows"] == (0 if equal else 3)
    assert source["metrics"]["dividend_check"]["status"] == "unverified"
    assert "DIVIDEND_NOT_REFLECTED" not in _codes(source["metrics"]["dividend_check"])


@pytest.mark.parametrize("column", ["Close", "Volume"])
def test_unreviewed_selected_cell_change_fails(sources, column):
    sources[1].datasets["KTB10Y"].data.loc["2020-03-11", column] += 1
    report = _run(sources)
    assert "KTB_UNEXPECTED_SOURCE_DELTA" in _codes(report)
    assert report["metrics"]["ktb_comparison"]["status"] == "failed"


@pytest.mark.parametrize("column", ["Open", "High", "Low", "Close"])
def test_correction_candidate_must_match_all_selected_ohlc(sources, column):
    candidate = sources[2].datasets["KTB10Y_KRX_CORRECTION"]
    candidate.data.loc["2020-03-10", column] += 1
    report = _run(sources)
    assert "KTB_CORRECTION_CANDIDATE_MISMATCH" in _codes(report)


@pytest.mark.parametrize("part", ["raw_data", "parent_validation"])
def test_original_failed_evidence_must_remain_preserved(sources, part):
    selected = sources[1].datasets["KTB10Y"]
    if part == "raw_data":
        selected.raw_data.iloc[0, 0] += 1
    else:
        selected.validation["parent_validation"] = {"status": "passed"}
    report = _run(sources)
    expected = (
        "KTB_PARENT_RAW_NOT_PRESERVED"
        if part == "raw_data"
        else "KTB_PARENT_VALIDATION_NOT_PRESERVED"
    )
    assert expected in _codes(report)


@pytest.mark.parametrize(("field", "value"), [("source", "fred"), ("currency", "KRW")])
def test_wrong_source_identity_or_currency_fails(sources, field, value):
    sources[1].datasets["SPY"].metadata[field] = value
    report = _run(sources)
    asset = report["metrics"]["assets"]["SPY"]["selected"]
    assert asset["status"] == "failed"
    assert "PRICE_SOURCE_IDENTITY_MISMATCH" in _codes(asset)


def test_missing_reference_session_and_observed_nan_are_distinct(sources):
    for snapshot in sources[:2]:
        result = snapshot.datasets["SPY"]
        result.data = result.data.drop(pd.Timestamp("2020-03-10"))
        result.data.loc["2020-03-11", "Close"] = np.nan
    report = _run(sources)
    source = report["metrics"]["assets"]["SPY"]["selected"]
    assert source["metrics"]["absent_days"] == ["2020-03-10"]
    assert source["metrics"]["observed_nan"]["Close"] == 1
    assert source["metrics"]["reference_session_count"] == 3
    assert source["metrics"]["calendar_authority"] == "reference_only"
    assert "CALENDAR_AUTHORITY_UNVERIFIED" in _codes(source)
    assert source["status"] == "failed"


def test_matching_nulls_are_not_fictitious_source_deltas(sources):
    for snapshot in sources[:2]:
        snapshot.datasets["KTB10Y"].data.loc["2020-03-09", "Adj Close"] = np.nan
    sources[1].datasets["KTB10Y"].raw_data = sources[0].datasets["KTB10Y"].data.copy(deep=True)
    report = _run(sources)
    comparison = report["metrics"]["ktb_comparison"]
    assert len(comparison["metrics"]["changed_cells"]) == 1
    assert comparison["status"] == "passed"
    assert (
        report["metrics"]["assets"]["KTB10Y"]["selected"]["metrics"]["observed_nan"]["Adj Close"]
        == 1
    )
    json.dumps(report, allow_nan=False)


def test_ratio_overflow_is_a_recordable_failure_not_nonfinite_json(sources):
    for snapshot in sources[:2]:
        data = snapshot.datasets["GLD"].data
        data.loc["2020-03-09", ["Open", "High", "Low", "Close"]] = 1e-300
        data.loc["2020-03-09", "Adj Close"] = 1e300
    report = _run(sources)
    assert report["status"] == "failed"
    assert "PRICE_EVIDENCE_INPUT_INVALID" in _codes(report)
    assert "ratio arithmetic" in report["issues"][-1]["message"]
    json.dumps(report, allow_nan=False)
