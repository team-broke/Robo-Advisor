"""Synthetic offline tests for preserved-source comparisons and failure reporting."""

from copy import deepcopy
from hashlib import sha256
import json
import socket
from types import SimpleNamespace

import pandas as pd
import pytest

from robo_advisor.data import source_evidence as evidence
from robo_advisor.data.collectors import CollectionResult


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Source evidence tests must not access external APIs")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


@pytest.fixture
def yahoo_result():
    dates = pd.date_range("2025-01-01", periods=4)
    ticks = list((dates.tz_localize("Europe/London").asi8 // 10**9).astype(int))
    ticks = [int(value) for value in ticks]
    quote = {
        "open": [None, 1300.0, 1300.0, None],
        "high": [None, 1310.0, 1310.0, None],
        "low": [None, 1290.0, 1290.0, None],
        "close": [None, 1305.0, 1325.0, None],
        "volume": [None, 0, 0, None],
    }
    payload = {
        "chart": {
            "error": None,
            "result": [
                {
                    "meta": {
                        "symbol": "KRW=X",
                        "currency": "KRW",
                        "longName": "USD/KRW",
                        "exchangeTimezoneName": "Europe/London",
                    },
                    "timestamp": ticks,
                    "indicators": {"quote": [quote]},
                }
            ],
        }
    }
    original = pd.DataFrame({key.title(): value for key, value in quote.items()}, index=dates)
    raw = original.copy(deep=True)
    raw["ProviderTimestampUnix"] = ticks
    raw["ProviderBarDate"] = dates
    return CollectionResult(
        data=original.iloc[:3].copy(deep=True),
        raw_data=raw,
        raw_text=json.dumps(payload, allow_nan=False),
        metadata={
            "source": "yahoo",
            "symbol": "KRW=X",
            "currency": "KRW",
            "timezone": "Asia/Seoul",
            "requested_start": "2025-01-01",
            "requested_end_exclusive": "2025-01-04",
            "status": "failed",
            "availability_status": "unverified",
            "available_at": None,
        },
        validation={"issues": []},
    )


def _assert_report(report, status):
    assert report["status"] == status
    assert report["training_approved"] is False
    assert all(issue["code"] and issue["message"] for issue in report["issues"])
    json.dumps(report, allow_nan=False)


def test_original_chain_preserves_failed_source_and_distinguishes_rows_from_cells(yahoo_result):
    original = deepcopy(yahoo_result)
    report = evidence.audit_yahoo_fx_origin(yahoo_result)
    _assert_report(report, "failed")
    metrics = report["metrics"]
    assert metrics["original_chain_status"] == "passed"
    assert metrics["source_status"] == "failed"
    assert (metrics["raw_rows"], metrics["data_rows"]) == (4, 3)
    assert metrics["ohlc_defect_rows"] == 1
    assert metrics["ohlc_defects"][0]["date"] == "2025-01-03"
    assert metrics["missing_dates"] == ["2025-01-01"]
    assert metrics["missing_price_cells"] == 4
    assert metrics["missing_volume_cells"] == 1
    assert metrics["raw_missing_rows"] == 2  # Includes the preserved out-of-scope row.
    assert metrics["provider_timezone"] == "Europe/London"
    assert metrics["label_timezone"] == "Asia/Seoul"
    pd.testing.assert_frame_equal(yahoo_result.data, original.data)
    pd.testing.assert_frame_equal(yahoo_result.raw_data, original.raw_data)
    assert yahoo_result.raw_text == original.raw_text
    assert yahoo_result.metadata == original.metadata
    assert yahoo_result.validation == original.validation


@pytest.mark.parametrize("layer", ["raw", "data", "json"])
def test_quote_tampering_cannot_match_preserved_source(yahoo_result, layer):
    if layer == "json":
        payload = json.loads(yahoo_result.raw_text)
        payload["chart"]["result"][0]["indicators"]["quote"][0]["close"][1] = 9999.0
        yahoo_result.raw_text = json.dumps(payload)
    else:
        frame = yahoo_result.raw_data if layer == "raw" else yahoo_result.data
        frame.loc[frame.index[1], "Close"] = 9999.0
    report = evidence.audit_yahoo_fx_origin(yahoo_result)
    _assert_report(report, "failed")
    assert report["issues"][0]["code"] == "YAHOO_EVIDENCE_INVALID"


@pytest.mark.parametrize("field, value", [("symbol", "JPY=X"), ("exchangeTimezoneName", "No/Zone")])
def test_original_identity_and_timezone_must_be_usable(yahoo_result, field, value):
    payload = json.loads(yahoo_result.raw_text)
    payload["chart"]["result"][0]["meta"][field] = value
    yahoo_result.raw_text = json.dumps(payload)
    report = evidence.audit_yahoo_fx_origin(yahoo_result)
    _assert_report(report, "failed")
    assert report["issues"][0]["code"] == "YAHOO_EVIDENCE_INVALID"


def test_truncated_quote_array_is_recorded_as_invalid(yahoo_result):
    payload = json.loads(yahoo_result.raw_text)
    payload["chart"]["result"][0]["indicators"]["quote"][0]["close"].pop()
    yahoo_result.raw_text = json.dumps(payload)
    report = evidence.audit_yahoo_fx_origin(yahoo_result)
    _assert_report(report, "failed")
    assert "lengths disagree" in report["issues"][0]["message"]


@pytest.mark.parametrize("source_status", ["failed", "unverified"])
def test_missing_original_json_does_not_upgrade_declared_source(yahoo_result, source_status):
    yahoo_result.raw_text = None
    yahoo_result.metadata["status"] = source_status
    report = evidence.audit_yahoo_fx_origin(yahoo_result)
    _assert_report(report, source_status)
    assert report["issues"][0]["code"] == "YAHOO_ORIGINAL_UNAVAILABLE"
    assert report["metrics"]["original_chain_status"] == "unverified"
    assert report["metrics"]["source_status"] == source_status
    assert yahoo_result.status == source_status


@pytest.fixture
def ktb_inputs(tmp_path):
    day = pd.DatetimeIndex(["2020-03-10"])
    data = pd.DataFrame(
        {"Open": [127475], "High": [127500], "Low": [126795], "Close": [127500], "Volume": [28091]},
        index=day,
    )
    original = pd.DataFrame(
        {
            "일자": ["2020/03/10"],
            "시가": ["127475"],
            "고가": ["127500"],
            "저가": ["126795"],
            "종가": ["127500"],
            "거래량": ["28091"],
        }
    )
    path = tmp_path / "ktb.csv"
    path.write_bytes(original.to_csv(index=False).encode("cp949"))
    descriptor = {"path": path.name, "sha256": sha256(path.read_bytes()).hexdigest()}
    result = CollectionResult(
        data=data,
        raw_data=original.copy(deep=True),
        metadata={"symbol": "148070", "currency": "KRW", "status": "unverified"},
        validation={"issues": []},
    )
    return tmp_path, descriptor, SimpleNamespace(datasets={"KTB10Y_KRX_CORRECTION": result})


def test_matching_cp949_values_do_not_create_a_correction_receipt(ktb_inputs):
    root, descriptor, candidate = ktb_inputs
    before = deepcopy(candidate.datasets["KTB10Y_KRX_CORRECTION"])
    original = (root / descriptor["path"]).read_bytes()
    report = evidence.audit_ktb_original(root, descriptor, candidate)
    _assert_report(report, "unverified")
    assert report["metrics"]["original_values_status"] == "passed"
    assert report["metrics"]["low"] == 126795
    assert report["metrics"]["collection_receipt_status"] == "not_supplied"
    assert report["metrics"]["training_adoption"] == "blocked"
    actual = candidate.datasets["KTB10Y_KRX_CORRECTION"]
    pd.testing.assert_frame_equal(actual.data, before.data)
    pd.testing.assert_frame_equal(actual.raw_data, before.raw_data)
    assert actual.metadata == before.metadata
    assert actual.validation == before.validation
    assert (root / descriptor["path"]).read_bytes() == original


def test_changed_ktb_bytes_fail_the_external_anchor(ktb_inputs):
    root, descriptor, candidate = ktb_inputs
    path = root / descriptor["path"]
    path.write_bytes(path.read_bytes() + b"\n")
    report = evidence.audit_ktb_original(root, descriptor, candidate)
    _assert_report(report, "failed")
    assert "SHA-256 mismatch" in report["issues"][0]["message"]


def test_candidate_cannot_disagree_with_ktb_original(ktb_inputs):
    root, descriptor, candidate = ktb_inputs
    candidate.datasets["KTB10Y_KRX_CORRECTION"].data.iloc[0, 2] = 126800
    report = evidence.audit_ktb_original(root, descriptor, candidate)
    _assert_report(report, "failed")
    assert "differs" in report["issues"][0]["message"]


def test_multirow_csv_is_not_the_single_date_correction(ktb_inputs):
    root, descriptor, candidate = ktb_inputs
    path = root / descriptor["path"]
    raw = pd.read_csv(path, encoding="cp949", dtype=str)
    raw = pd.concat([raw, raw], ignore_index=True)
    path.write_bytes(raw.to_csv(index=False).encode("cp949"))
    descriptor["sha256"] = sha256(path.read_bytes()).hexdigest()
    report = evidence.audit_ktb_original(root, descriptor, candidate)
    _assert_report(report, "failed")
    assert "single-date" in report["issues"][0]["message"]


def test_invalid_source_specification_is_a_finite_failure_report(tmp_path):
    report = evidence.audit_source_evidence({}, root=tmp_path, base_audit={})
    _assert_report(report, "failed")
    assert report["issues"][0]["code"] == "SOURCE_EVIDENCE_INPUT_INVALID"


def test_cli_bad_configuration_records_failure(tmp_path, capsys):
    config = tmp_path / "source.json"
    config.write_text("{}", encoding="utf-8")
    code = evidence.main(["--config", str(config), "--root", str(tmp_path)])
    assert code == 1
    report = json.loads(capsys.readouterr().out)
    _assert_report(report, "failed")


def test_cli_never_replaces_existing_report(tmp_path, monkeypatch, capsys):
    config = tmp_path / "source.json"
    config.write_text(json.dumps({"base_audit_config": "base.json"}), encoding="utf-8")
    (tmp_path / "base.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "existing.json"
    original = b'{"preserved":true}\n'
    output.write_bytes(original)
    monkeypatch.setattr(
        evidence,
        "audit_source_evidence",
        lambda *args, **kwargs: {"status": "unverified", "training_approved": False},
    )
    code = evidence.main(
        ["--config", str(config), "--root", str(tmp_path), "--output", str(output)]
    )
    assert code == 1
    assert output.read_bytes() == original
    _assert_report(json.loads(capsys.readouterr().out), "failed")
