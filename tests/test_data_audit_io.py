"""Offline failure reporting, immutable CLI outputs and full source-pointer checks."""

from copy import deepcopy
from hashlib import sha256
import json
import socket
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from robo_advisor.data import audit
from robo_advisor.data.collectors import CollectionResult
from robo_advisor.features.dataset import TEAM12_ASSETS


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Data audit IO tests must not access external APIs")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def _config(calendar):
    return {
        "schema_version": 1,
        "calendars": {"KRX": deepcopy(calendar), "US": deepcopy(calendar)},
        "snapshots": {
            name: {"path": f"snapshots/{name}", "sha256": "0" * 64}
            for name in ("parent", "selected")
        },
        "scenarios": {"smbs": {}, "alfred": {}},
    }


def _assert_input_failed(report):
    assert report["status"] == "failed"
    assert report["training_approved"] is False
    assert "AUDIT_INPUT_INVALID" in {issue["code"] for issue in report["issues"]}
    assert all(issue["message"] for issue in report["issues"])
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("config", [None, {}, {"schema_version": True}, {"schema_version": 2}])
def test_invalid_or_unknown_specification_never_passes(tmp_path, config):
    _assert_input_failed(audit.audit_research_package(config, root=tmp_path))


def test_empty_required_sections_are_not_a_successful_empty_audit(tmp_path):
    config = {"schema_version": 1, "snapshots": {}, "scenarios": {}, "calendars": {}}
    report = audit.audit_research_package(config, root=tmp_path)
    _assert_input_failed(report)
    assert "requires exactly" in report["issues"][0]["message"]


def test_missing_evidence_root_records_failure(tmp_path):
    config = _config({"path": "sessions.csv", "sha256": "0" * 64})
    root = tmp_path / "missing-package"
    report = audit.audit_research_package(config, root=root)
    _assert_input_failed(report)
    assert not root.exists()


@pytest.mark.parametrize("anchor", [None, "", "not-a-digest", "F" * 64])
def test_missing_or_malformed_external_anchor_is_explicit(tmp_path, anchor):
    (tmp_path / "sessions.csv").write_text("session_date\n2024-01-02\n", encoding="utf-8")
    config = _config({"path": "sessions.csv", "sha256": anchor})
    report = audit.audit_research_package(config, root=tmp_path)
    _assert_input_failed(report)
    assert "SHA-256 anchor" in report["issues"][0]["message"]


def test_changed_calendar_bytes_are_not_accepted_under_old_hash(tmp_path):
    content = b"session_date\n2024-01-02\n"
    calendar = tmp_path / "sessions.csv"
    calendar.write_bytes(content)
    config = _config({"path": "sessions.csv", "sha256": sha256(content).hexdigest()})
    calendar.write_bytes(content + b"2024-01-03\n")
    report = audit.audit_research_package(config, root=tmp_path)
    _assert_input_failed(report)
    assert "SHA-256 mismatch" in report["issues"][0]["message"]


def test_anchored_file_outside_package_is_still_rejected(tmp_path):
    content = b"session_date\n2024-01-02\n"
    (tmp_path / "outside.csv").write_bytes(content)
    root = tmp_path / "package"
    root.mkdir()
    config = _config({"path": "../outside.csv", "sha256": sha256(content).hexdigest()})
    report = audit.audit_research_package(config, root=root)
    _assert_input_failed(report)
    assert "within" in report["issues"][0]["message"]


def test_cli_writes_recordable_failure_and_does_not_overwrite(tmp_path, capsys):
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    output = tmp_path / "audit.json"
    args = ["--config", str(config), "--root", str(tmp_path), "--output", str(output)]
    assert audit.main(args) == 1
    original = output.read_bytes()
    report = json.loads(original)
    _assert_input_failed(report)
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
    assert audit.main(args) == 1
    assert output.read_bytes() == original
    assert json.loads(capsys.readouterr().out)["status"] == "failed"


@pytest.fixture
def provenance_inputs():
    us = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"])
    kr = us.delete(1)  # Synthetic KR closure while the US still traded on Jan 3.
    dates = kr[1:]
    evaluation = pd.Series(
        (dates + pd.Timedelta(hours=15, minutes=30)).tz_localize("Asia/Seoul").tz_convert("UTC"),
        index=dates,
        name="evaluation_at",
    )
    reference = dates.tz_localize("Asia/Seoul").tz_convert("UTC")
    selection = pd.DataFrame(
        {
            "rate": 1300.0,  # Equal rates deliberately cannot identify their source rows.
            "observation_date": dates,
            "reference_at": reference,
            "actual_available_at": reference + pd.Timedelta(hours=1),
            "assumed_usable_at": reference + pd.Timedelta(hours=2),
            "source_row_position": np.arange(len(dates)),
            "observation_age_hours": 15.5,
            "selection_status": "observed_asof_quote",
        },
        index=dates,
    )
    rows = selection.add_prefix("FX__")
    rows["evaluation_at"] = evaluation
    previous_us = us[us.get_indexer(dates) - 1]
    for asset in TEAM12_ASSETS:
        rows[f"{asset}__source_dates"] = previous_us if asset in TEAM12_ASSETS[:10] else dates
    return rows, selection, evaluation, {"KRX": kr, "US": us}


def test_matching_full_pointers_preserve_input_frames(provenance_inputs):
    original = deepcopy(provenance_inputs)
    assert audit._check_row_provenance(*provenance_inputs) is None
    for actual, expected in zip(provenance_inputs[:3], original[:3]):
        if isinstance(actual, pd.DataFrame):
            pd.testing.assert_frame_equal(actual, expected)
        else:
            pd.testing.assert_series_equal(actual, expected)


@pytest.mark.leakage
@pytest.mark.parametrize(
    "column",
    [
        "source_row_position",
        "reference_at",
        "actual_available_at",
        "assumed_usable_at",
        "observation_age_hours",
        "selection_status",
    ],
)
def test_equal_fx_rates_cannot_hide_wrong_source_pointer(provenance_inputs, column):
    rows, selection, evaluation, calendars = provenance_inputs
    date = rows.index[0]
    target = f"FX__{column}"
    if column.endswith("_at"):
        rows.loc[date, target] += pd.Timedelta(hours=1)
    elif column == "selection_status":
        rows.loc[date, target] = "no_released_observation"
    else:
        rows.loc[date, target] += 1
    assert rows.loc[date, "FX__rate"] == selection.loc[date, "rate"] == 1300.0
    with pytest.raises(ValueError, match="FX lineage"):
        audit._check_row_provenance(rows, selection, evaluation, calendars)


@pytest.mark.leakage
def test_row_evaluation_cannot_differ_from_anchored_date_audit(provenance_inputs):
    rows, selection, evaluation, calendars = provenance_inputs
    rows.loc[rows.index[0], "evaluation_at"] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="evaluation timestamps"):
        audit._check_row_provenance(rows, selection, evaluation, calendars)


@pytest.mark.leakage
def test_us_lag_must_precede_korean_holiday_intersection(provenance_inputs):
    rows, selection, evaluation, calendars = provenance_inputs
    # Correct Jan 4 US input is Jan 3, even though Jan 3 is absent from KR sessions.
    assert rows.loc["2024-01-04", "SPY__source_dates"] == pd.Timestamp("2024-01-03")
    rows.loc["2024-01-04", "SPY__source_dates"] = pd.Timestamp("2024-01-02")
    with pytest.raises(ValueError, match="full-market-session lag"):
        audit._check_row_provenance(rows, selection, evaluation, calendars)


def _snapshot_source(*, status, invalid_ohlc=False):
    dates = pd.DatetimeIndex(["2020-03-09", "2020-03-10", "2020-03-11"])
    data = pd.DataFrame(
        {"Open": 128000.0, "High": 128500.0, "Low": 127500.0, "Close": 128000.0, "Volume": 0},
        index=dates,
    )
    if invalid_ohlc:
        data.loc[dates[1], ["Open", "High", "Low", "Close"]] = [127475, 127500, 127500, 127500]
    result = CollectionResult(
        data=data,
        raw_data=data.copy(deep=True),
        metadata={
            "status": status,
            "source": "synthetic",
            "symbol": "148070.KS",
            "market": "KRX",
            "currency": "KRW",
            "availability_status": "verified",
            "requested_start": "2020-03-09",
            "requested_end_exclusive": "2020-03-12",
            "actual_start": "2020-03-09",
            "actual_end": "2020-03-11",
        },
        validation={"issues": []},
    )
    snapshot = SimpleNamespace(datasets={"KTB10Y": result}, manifest_sha256="a" * 64)
    return snapshot, {"KRX": dates}


def test_ktb_ohlc_defect_retains_date_position_and_original_values():
    snapshot, calendars = _snapshot_source(status="failed", invalid_ohlc=True)
    original = snapshot.datasets["KTB10Y"].data.copy(deep=True)
    report = audit._source_report(snapshot, calendars)
    source = report["metrics"]["sources"]["KTB10Y"]
    assert source["status"] == "failed"
    defect = next(issue for issue in source["issues"] if issue["code"] == "INVALID_OHLC_RANGE")
    assert defect["date"] == "2020-03-10"
    assert defect["position"] == 1
    assert source["training_approved"] is False
    pd.testing.assert_frame_equal(snapshot.datasets["KTB10Y"].data, original)
    assert snapshot.datasets["KTB10Y"].status == "failed"


def test_numeric_success_does_not_upgrade_preexisting_failed_source():
    snapshot, calendars = _snapshot_source(status="failed")
    source = audit._source_report(snapshot, calendars)["metrics"]["sources"]["KTB10Y"]
    assert source["metrics"]["numeric_status"] == "passed"
    assert source["metrics"]["declared_status"] == "failed"
    assert source["status"] == "failed"
    assert snapshot.datasets["KTB10Y"].status == "failed"


def test_zero_volume_is_not_checked_as_nonpositive_price():
    snapshot, calendars = _snapshot_source(status="passed")
    source = audit._source_report(snapshot, calendars)["metrics"]["sources"]["KTB10Y"]
    assert source["metrics"]["numeric_status"] == "passed"
    assert source["status"] == "passed"
    assert not any(issue.get("column") == "Volume" for issue in source["issues"])
    assert source["training_approved"] is False


@pytest.mark.parametrize("descriptor", [None, [], {"sha256": "0" * 64}])
def test_malformed_file_descriptor_records_input_failure(tmp_path, descriptor):
    report = audit.audit_research_package(_config(descriptor), root=tmp_path)
    _assert_input_failed(report)


@pytest.mark.parametrize("descriptor", [None, [], {}])
def test_malformed_snapshot_descriptor_records_component_failure(tmp_path, monkeypatch, descriptor):
    content = b"session_date\n2024-01-02\n"
    (tmp_path / "sessions.csv").write_bytes(content)
    config = _config({"path": "sessions.csv", "sha256": sha256(content).hexdigest()})
    config["snapshots"] = {name: deepcopy(descriptor) for name in ("parent", "selected")}

    def forbidden_reader(*args, **kwargs):
        pytest.fail("Malformed snapshot descriptors must be rejected before reading payloads")

    monkeypatch.setattr(audit, "read_snapshot", forbidden_reader)
    report = audit.audit_research_package(config, root=tmp_path)
    assert report["status"] == "failed"
    assert report["training_approved"] is False
    for name in ("parent", "selected"):
        component = report["metrics"]["components"][name]
        assert component["status"] == "failed"
        assert component["metrics"]["integrity"] == "failed"
        assert component["issues"][0]["code"] == "SNAPSHOT_INVALID"
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "anchor, omit", [(None, False), ("not-a-digest", False), ("F" * 64, False), (None, True)]
)
def test_snapshot_requires_external_anchor_before_reader(tmp_path, monkeypatch, anchor, omit):
    content = b"session_date\n2024-01-02\n"
    (tmp_path / "sessions.csv").write_bytes(content)
    config = _config({"path": "sessions.csv", "sha256": sha256(content).hexdigest()})
    for descriptor in config["snapshots"].values():
        folder = tmp_path / descriptor["path"]
        folder.mkdir(parents=True)
        (folder / "manifest.json").write_text("{}", encoding="utf-8")
        if omit:
            descriptor.pop("sha256")
        else:
            descriptor["sha256"] = anchor

    def forbidden_reader(*args, **kwargs):
        pytest.fail("Missing or malformed snapshot anchors must fail before reading payloads")

    monkeypatch.setattr(audit, "read_snapshot", forbidden_reader)
    report = audit.audit_research_package(config, root=tmp_path)
    assert report["status"] == "failed"
    assert report["training_approved"] is False
    for name in ("parent", "selected"):
        component = report["metrics"]["components"][name]
        assert component["status"] == "failed"
        assert component["metrics"]["integrity"] == "failed"
        assert component["issues"][0]["code"] == "SNAPSHOT_INVALID"
        assert "SHA-256 anchor" in component["issues"][0]["message"]
    json.dumps(report, allow_nan=False)
