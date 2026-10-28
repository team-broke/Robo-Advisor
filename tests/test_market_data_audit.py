"""Synthetic, offline checks for auditable point-in-time FX selection."""

from copy import deepcopy
import json
import socket

import numpy as np
import pandas as pd
import pytest

from robo_advisor.data.audit import audit_fx_data


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("FX audit tests must not access external APIs")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


@pytest.fixture
def fx_inputs():
    dates = pd.date_range("2024-01-02", periods=3, name="date")
    reference = (dates + pd.Timedelta(hours=9)).tz_localize("UTC")
    observations = pd.DataFrame(
        {
            "rate": [1300.0, 1310.0, 1320.0],
            "reference_at": reference,
            "actual_available_at": reference + pd.Timedelta(hours=1),
            "assumed_usable_at": reference + pd.Timedelta(minutes=30),
        },
        index=dates,
    )
    evaluation = pd.Series(
        (dates + pd.Timedelta(hours=12)).tz_localize("UTC"),
        index=dates,
        name="evaluation_at",
    )
    return observations, evaluation


def _audit(observations, evaluation, **overrides):
    options = {
        "source": "smbs",
        "direction": "USD/KRW",
        "unit": "KRW_per_USD",
        "max_age_days": 7,
    }
    options.update(overrides)
    return audit_fx_data(observations, evaluation, **options)


def _selected(observations, evaluation):
    selected = observations.reindex(evaluation.index).copy(deep=True)
    selected["observation_date"] = selected.index
    selected["source_row_position"] = np.arange(len(selected))
    selected["observation_age_hours"] = (
        evaluation - selected["reference_at"]
    ).dt.total_seconds() / 3600
    selected["selection_status"] = "observed_asof_quote"
    return selected.loc[
        :,
        [
            "rate",
            "observation_date",
            "reference_at",
            "actual_available_at",
            "assumed_usable_at",
            "source_row_position",
            "observation_age_hours",
            "selection_status",
        ],
    ]


def _issue_codes(report):
    return {issue["code"] for issue in report["issues"]}


@pytest.mark.parametrize("source", ["smbs", "alfred"])
def test_known_release_selection_preserves_quotes_and_evidence(fx_inputs, source):
    observations, evaluation = fx_inputs
    supplied = _selected(observations, evaluation)
    report = _audit(observations, evaluation, source=source, selected=supplied)
    assert report["status"] == "unverified"
    assert report["metrics"]["selection_check"] == "passed"
    assert report["training_approved"] is False
    assert report["metrics"]["raw_rows"] == 3
    assert report["metrics"]["missing_rate_count"] == 0
    assert report["metrics"]["actual_available_count"] == 3
    assert report["metrics"]["excluded_counts"] == {}
    assert report["metrics"]["max_used_age_hours"] == pytest.approx(3.0)
    assert [row["rate"] for row in report["metrics"]["selected"]] == [1300.0, 1310.0, 1320.0]
    assert [row["source_row_position"] for row in report["metrics"]["selected"]] == [0, 1, 2]
    json.dumps(report, allow_nan=False)


@pytest.mark.leakage
def test_future_quote_changes_do_not_change_past_selection(fx_inputs):
    observations, evaluation = fx_inputs
    evaluation = evaluation.iloc[:2]
    before = _audit(observations, evaluation)
    changed = observations.copy(deep=True)
    changed.loc[changed.index[-1], "rate"] = 1e12
    after = _audit(changed, evaluation)
    assert before["status"] == after["status"] == "unverified"
    assert before["metrics"]["selection_check"] == after["metrics"]["selection_check"] == "passed"
    for key in ("selected", "excluded_counts", "excluded_dates", "max_used_age_hours"):
        assert before["metrics"][key] == after["metrics"][key]
    assert [row["rate"] for row in after["metrics"]["selected"]] == [1300.0, 1310.0]


@pytest.mark.leakage
def test_known_future_release_overrides_earlier_research_assumption(fx_inputs):
    observations, evaluation = fx_inputs
    observations = observations.iloc[:1].copy()
    evaluation = evaluation.iloc[:1]
    observations["actual_available_at"] += pd.Timedelta(days=1)
    report = _audit(observations, evaluation)
    assert report["status"] != "passed"
    assert report["metrics"]["excluded_counts"] == {"no_released_observation": 1}
    selection = report["metrics"]["selected"][0]
    assert selection["rate"] is None
    assert selection["selection_status"] == "no_released_observation"


@pytest.mark.leakage
def test_supplied_future_source_row_is_rejected(fx_inputs):
    observations, evaluation = fx_inputs
    selected = _selected(observations, evaluation)
    selected.iloc[0] = selected.iloc[1]
    report = _audit(observations, evaluation, selected=selected)
    assert report["status"] == "failed"
    assert report["issues"]
    assert report["training_approved"] is False


@pytest.mark.leakage
def test_changed_supplied_rate_cannot_override_released_quote(fx_inputs):
    observations, evaluation = fx_inputs
    selected = _selected(observations, evaluation)
    selected.loc[selected.index[1], "rate"] = 99999.0
    report = _audit(observations, evaluation, selected=selected)
    assert report["status"] == "failed"
    assert report["issues"]


@pytest.mark.parametrize(
    "offset, excluded", [(pd.Timedelta(0), False), (pd.Timedelta(1, "ns"), True)]
)
def test_maximum_quote_age_includes_exact_boundary(fx_inputs, offset, excluded):
    observations, _ = fx_inputs
    observations = observations.iloc[:1].copy()
    observations["actual_available_at"] = observations["reference_at"]
    reference = observations["reference_at"].iloc[0]
    evaluation = pd.Series(
        [reference + pd.Timedelta(days=7) + offset],
        index=pd.DatetimeIndex(["2024-01-09"], name="date"),
    )
    report = _audit(observations, evaluation)
    if excluded:
        assert report["metrics"]["excluded_counts"] == {"observation_too_old": 1}
        assert report["metrics"]["selected"][0]["rate"] is None
        assert report["status"] != "passed"
    else:
        assert report["metrics"]["excluded_counts"] == {}
        assert report["metrics"]["max_used_age_hours"] == pytest.approx(168.0)
        assert report["metrics"]["selected"][0]["rate"] == 1300.0


def test_null_quote_remains_null_and_uses_last_released_nonnull_row(fx_inputs):
    observations, evaluation = fx_inputs
    observations.loc[observations.index[1], "rate"] = np.nan
    original = observations.copy(deep=True)
    report = _audit(observations, evaluation)
    assert report["metrics"]["missing_rate_count"] == 1
    assert report["metrics"]["raw_rows"] == 3
    assert report["metrics"]["selected"][1]["rate"] == 1300.0
    assert report["metrics"]["selected"][1]["source_row_position"] == 0
    pd.testing.assert_frame_equal(observations, original)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("missing", ["actual_available_at", "both"])
def test_research_assumption_does_not_certify_actual_release(fx_inputs, missing):
    observations, evaluation = fx_inputs
    observations["actual_available_at"] = pd.NaT
    if missing == "both":
        observations["assumed_usable_at"] = pd.NaT
    report = _audit(observations, evaluation)
    assert report["status"] == "unverified"
    assert report["training_approved"] is False
    assert report["metrics"]["actual_available_count"] == 0
    if missing == "both":
        assert report["metrics"]["excluded_counts"] == {"no_released_observation": 3}
    else:
        assert [row["rate"] for row in report["metrics"]["selected"]] == [1300.0, 1310.0, 1320.0]


@pytest.mark.parametrize("rate", [0.0, -1.0, np.inf, -np.inf, True, "1300"])
def test_invalid_rates_leave_explicit_failure(fx_inputs, rate):
    observations, evaluation = fx_inputs
    observations["rate"] = rate
    report = _audit(observations, evaluation)
    assert report["status"] == "failed"
    assert "INPUT_INVALID" in _issue_codes(report)


@pytest.mark.parametrize("max_age", [0, -1, 15, 1.5, True])
def test_maximum_age_requires_bounded_positive_integer(fx_inputs, max_age):
    report = _audit(*fx_inputs, max_age_days=max_age)
    assert report["status"] == "failed"
    assert "INPUT_INVALID" in _issue_codes(report)


@pytest.mark.parametrize(
    "direction, unit",
    [("USD/KRW", "USD_per_KRW"), ("KRW/USD", "KRW_per_USD"), ("USD_KRW", "KRW_per_USD")],
)
def test_direction_and_unit_must_agree_explicitly(fx_inputs, direction, unit):
    report = _audit(*fx_inputs, direction=direction, unit=unit)
    assert report["status"] == "failed"
    assert "INPUT_INVALID" in _issue_codes(report)


def test_explicit_reciprocal_quote_is_normalized_with_recorded_output_unit(fx_inputs):
    observations, evaluation = fx_inputs
    observations["rate"] = 1.0 / observations["rate"]
    report = _audit(observations, evaluation, direction="KRW/USD", unit="USD_per_KRW")
    assert report["status"] == "unverified"
    assert report["metrics"]["selection_check"] == "passed"
    assert report["metrics"]["output_unit"] == "KRW_per_USD"
    assert report["metrics"]["selected"][0]["rate"] == pytest.approx(1300.0)


@pytest.mark.parametrize("defect", ["timezone", "intraday", "duplicate", "reversed", "nat"])
def test_observation_dates_are_not_silently_normalized(fx_inputs, defect):
    observations, evaluation = fx_inputs
    index = observations.index
    observations.index = {
        "timezone": index.tz_localize("Asia/Seoul"),
        "intraday": index + pd.Timedelta(hours=1),
        "duplicate": pd.DatetimeIndex([index[0], index[0], index[2]]),
        "reversed": index[::-1],
        "nat": pd.DatetimeIndex([index[0], pd.NaT, index[2]]),
    }[defect]
    report = _audit(observations, evaluation)
    assert report["status"] == "failed"
    assert "INPUT_INVALID" in _issue_codes(report)


@pytest.mark.parametrize("column", ["reference_at", "actual_available_at", "assumed_usable_at"])
def test_supplied_release_timestamps_must_be_timezone_aware(fx_inputs, column):
    observations, evaluation = fx_inputs
    observations[column] = observations[column].dt.tz_localize(None)
    report = _audit(observations, evaluation)
    assert report["status"] == "failed"
    assert "INPUT_INVALID" in _issue_codes(report)


def test_evaluation_timestamp_requires_explicit_timezone(fx_inputs):
    observations, evaluation = fx_inputs
    report = _audit(observations, evaluation.dt.tz_localize(None))
    assert report["status"] == "failed"
    assert "INPUT_INVALID" in _issue_codes(report)


@pytest.mark.parametrize("defect", ["index", "missing_column", "source_position", "age"])
def test_supplied_selection_metadata_cannot_disagree(fx_inputs, defect):
    observations, evaluation = fx_inputs
    selected = _selected(observations, evaluation)
    if defect == "index":
        selected.index += pd.Timedelta(days=1)
    elif defect == "missing_column":
        selected = selected.drop(columns="actual_available_at")
    elif defect == "source_position":
        selected.loc[selected.index[0], "source_row_position"] = 2
    else:
        selected.loc[selected.index[0], "observation_age_hours"] = 0.0
    report = _audit(observations, evaluation, selected=selected)
    assert report["status"] == "failed"
    assert report["issues"]


def test_inputs_are_unchanged_after_success_or_failure(fx_inputs):
    observations, evaluation = fx_inputs
    selected = _selected(observations, evaluation)
    originals = deepcopy((observations, evaluation, selected))
    _audit(observations, evaluation, selected=selected)
    pd.testing.assert_frame_equal(observations, originals[0])
    pd.testing.assert_series_equal(evaluation, originals[1])
    pd.testing.assert_frame_equal(selected, originals[2])
    selected.loc[selected.index[0], "rate"] = -1.0
    failed_selection = selected.copy(deep=True)
    assert _audit(observations, evaluation, selected=selected)["status"] == "failed"
    pd.testing.assert_frame_equal(observations, originals[0])
    pd.testing.assert_series_equal(evaluation, originals[1])
    pd.testing.assert_frame_equal(selected, failed_selection)
