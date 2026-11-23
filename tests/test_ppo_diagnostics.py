"""합성 데이터만 사용한다. 실제 PPO 투자 성과를 입증하는 테스트가 아니다."""

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from robo_advisor.diagnostics import inputs
from robo_advisor.diagnostics.ppo_diagnostics import (
    DiagnosticPeriod,
    ReferenceReturns,
    allocation_statistics,
    compare_experiment_settings,
    diagnose_ppo,
    save_diagnostics,
    summarize_training_episodes,
    trace_policy,
    validate_weights,
    _drawdown_crossings,
)
from robo_advisor.rl.evaluation import evaluate_policy
from robo_advisor.rl.safeguard import MDDSafeguard
from robo_advisor.rl.training import EnvironmentConfig, TrainingDataset


class ConstantPolicy:
    seed = 42
    action_space = SimpleNamespace(shape=(13,))
    observation_space = SimpleNamespace(shape=(85,))

    def __init__(self, logits=None):
        self.logits = np.zeros(13) if logits is None else np.asarray(logits, dtype=float)

    def predict(self, observation, deterministic=False):
        assert deterministic
        return self.logits.copy(), None


class ObservationPolicy:
    """Changing past inputs changes actions, making leakage observable."""

    def __init__(self):
        self.observations = []

    def predict(self, observation, deterministic=False):
        assert deterministic
        self.observations.append(observation.copy())
        risky = np.clip(observation[:36].reshape(3, 12).sum(axis=0) * 100, -20, 20)
        return np.r_[risky, 0.0], None


@pytest.fixture
def case():
    dates = pd.bdate_range("2024-01-01", periods=32, tz="UTC")
    returns = pd.DataFrame(
        np.sin(np.arange(32 * 12).reshape(32, 12)) * 0.005,
        index=dates,
        columns=[f"ETF{i}" for i in range(12)],
    )
    data = TrainingDataset(
        returns, returns * 0 + 50, returns * 0, (), {"returns.parquet": "a" * 64}
    )
    config = EnvironmentConfig(lookback=3)
    periods = [
        DiagnosticPeriod("train", dates[0], dates[12]),
        DiagnosticPeriod("validation", dates[12], dates[20]),
        DiagnosticPeriod("test", dates[20], dates[-1] + pd.Timedelta(days=1)),
    ]
    manifest = {
        "status": "completed",
        "run_id": "synthetic-run",
        "seed": 42,
        "config": {
            "env": config.model_dump(mode="json"),
            "seed": 42,
            "data": {"train_start": dates[0].isoformat(), "train_end": dates[19].isoformat()},
            "validation": {"start": dates[12].isoformat()},
        },
        "data_cutoff": {
            "assets": list(returns.columns),
            "actual_first_index": dates[0].isoformat(),
            "actual_last_index": dates[11].isoformat(),
        },
        "data_file_sha256": data.file_sha256.copy(),
    }
    return data, config, periods, manifest


def diagnose(case, **kwargs):
    data, config, periods, manifest = case
    defaults = {
        "provenance": {"kind": "synthetic", "run_id": manifest["run_id"]},
        "policy_version": "synthetic-policy-v1",
        "currency": "KRW",
    }
    defaults.update(kwargs)
    model = defaults.pop("model", ConstantPolicy())
    return diagnose_ppo(model, data, manifest, config, periods, **defaults)


def test_trace_reuses_existing_evaluation_nav_and_costs(case):
    data, config, _, _ = case
    result = trace_policy(ConstantPolicy(), data, config)
    reference = evaluate_policy(ConstantPolicy(), data, config)
    np.testing.assert_allclose(result["return"], reference.returns, rtol=0, atol=0)
    np.testing.assert_allclose(result.nav, reference.nav.iloc[1:], rtol=0, atol=0)
    assert result.cost.iloc[0] > 0
    assert result.index.equals(data.returns.index[config.lookback :])


@pytest.mark.leakage
def test_periods_half_open_and_validation_test_have_only_past_context(case):
    result = diagnose(case)
    history = result.allocation_history.query("mode == 'configured'")
    for period in case[2]:
        part = history[history.period == period.name]
        assert (part.date >= period.start).all() and (part.date < period.end).all()
        assert (part.decision_at < part.date).all()
    first_test = history[history.period == "test"].iloc[0]
    assert first_test.date == case[0].returns.index[20]
    assert first_test.decision_at == case[0].returns.index[19]
    assert result.period_metrics.query("period == 'train'").complete.eq(False).all()
    assert result.period_metrics.query("period != 'train'").complete.eq(True).all()
    assert result.summary["periods"][0]["missing_context_dates"] == [
        d.isoformat() for d in case[0].returns.index[:3]
    ]


@pytest.mark.leakage
def test_future_returns_and_indicators_do_not_change_past_trace(case):
    before = diagnose(case, model=ObservationPolicy())
    changed = copy.deepcopy(case)
    changed[0].returns.iloc[20:] = 0.8
    changed[0].rsi.iloc[20:] = 100
    changed[0].macd.iloc[20:] = 999
    after = diagnose(changed, model=ObservationPolicy())
    assert_frame_equal(
        before.allocation_history.query("period != 'test'").reset_index(drop=True),
        after.allocation_history.query("period != 'test'").reset_index(drop=True),
    )


@pytest.mark.parametrize(
    "defect",
    ["overlap", "selection_cutoff", "training_cutoff", "validation", "train_start", "coverage"],
)
@pytest.mark.leakage
def test_manifest_boundaries_fail_closed(case, defect):
    data, config, periods, manifest = copy.deepcopy(case)
    if defect == "overlap":
        periods[2] = DiagnosticPeriod("test", periods[1].start, periods[2].end)
    elif defect == "selection_cutoff":
        manifest["config"]["data"]["train_end"] = periods[2].start.isoformat()
    elif defect == "training_cutoff":
        manifest["data_cutoff"]["actual_last_index"] = periods[1].start.isoformat()
    elif defect == "validation":
        manifest["config"]["validation"] = None
    elif defect == "train_start":
        manifest["config"]["data"]["train_start"] = data.returns.index[1].isoformat()
    else:
        manifest["data_cutoff"]["actual_last_index"] = data.returns.index[10].isoformat()
    with pytest.raises(ValueError):
        diagnose((data, config, periods, manifest))


@pytest.mark.leakage
def test_validation_cannot_omit_recorded_selection_rows(case):
    data, config, periods, manifest = case
    shortened = DiagnosticPeriod("validation", periods[1].start, data.returns.index[19])
    with pytest.raises(ValueError, match="validation period omits"):
        diagnose((data, config, [periods[0], shortened, periods[2]], manifest))


def test_unused_indicator_warmup_matches_native_environment(case):
    data, config, periods, manifest = case
    clean = diagnose(case)
    data.rsi.iloc[: config.lookback - 1] = np.nan
    data.macd.iloc[: config.lookback - 1] = np.nan
    result = diagnose((data, config, periods, manifest))
    assert_frame_equal(result.allocation_history, clean.allocation_history)
    data.rsi.iloc[config.lookback - 1] = np.nan
    with pytest.raises(ValueError, match="rsi.*finite"):
        diagnose((data, config, periods, manifest))


@pytest.mark.parametrize(
    "defect", ["data_hash", "asset_order", "env", "run_id", "running", "naive"]
)
def test_versions_and_invalid_inputs_rejected(case, defect):
    data, config, periods, manifest = copy.deepcopy(case)
    kwargs = {}
    if defect == "data_hash":
        manifest["data_file_sha256"] = {"returns.parquet": "b" * 64}
    elif defect == "asset_order":
        manifest["data_cutoff"]["assets"].reverse()
    elif defect == "env":
        config = config.model_copy(update={"commission_rate": 0.01})
    elif defect == "run_id":
        kwargs["provenance"] = {"kind": "synthetic", "run_id": "different-checkpoint"}
    elif defect == "running":
        manifest["status"] = "running"
    else:
        periods[0] = DiagnosticPeriod("train", "2024-01-01", periods[0].end)
    with pytest.raises(ValueError):
        diagnose((data, config, periods, manifest), **kwargs)


@pytest.mark.parametrize(
    "weights", [[-0.1, 1.1], [np.nan, 1], [np.inf, 0], [0.4, 0.4], [1], ["0", "1"]]
)
def test_invalid_weights_are_not_silently_repaired(weights):
    with pytest.raises(ValueError):
        validate_weights(weights, 1)


def test_cash_hhi_and_two_equal_weight_definitions():
    risky = np.r_[np.full(12, 1 / 12), 0]
    all_equal = np.full(13, 1 / 13)
    a = allocation_statistics(risky, risky)
    b = allocation_statistics(all_equal, all_equal, all_equal)
    assert a["target_cash"] == 0 and a["risky_equal_l1"] == 0
    assert a["hhi"] == pytest.approx(1 / 12)
    assert b["actual_cash"] == pytest.approx(1 / 13)
    assert b["hhi"] == pytest.approx(1 / 13) and b["all_equal_l1"] == 0
    assert b["risky_equal_l1"] > 0 and b["target_change_l1"] == 0
    concentrated = np.r_[1.0, np.zeros(12)]
    assert allocation_statistics(concentrated, concentrated)["hhi"] == 1


def test_allocation_drift_and_summaries_are_observations(case):
    result = diagnose(case)
    history = result.allocation_history
    np.testing.assert_allclose(history.filter(like="target__").sum(axis=1), 1)
    np.testing.assert_allclose(history.filter(like="actual__").sum(axis=1), 1)
    assert not np.allclose(history.actual__ETF0, history.target__ETF0)
    assert set(result.allocation_summary.asset) == {*case[0].returns.columns, "CASH"}
    assert set(result.allocation_summary.regime) == {"all", "unknown"}


@pytest.mark.parametrize(
    "nav,terminated", [(0.85, False), (np.nextafter(0.85, 0), True), (0.851, False)]
)
def test_existing_safeguard_boundary_is_strictly_greater_than_15_percent(nav, terminated):
    result = MDDSafeguard(1, "diagnostic-test", 0.15).update(float(nav))
    assert result.terminated is terminated
    if nav == 0.85:
        assert result.drawdown == 0.15


def test_incomplete_guarded_evaluation_not_filled_and_same_model_sensitivity(case):
    data, config, periods, manifest = copy.deepcopy(case)
    data.returns.iloc[22] = -0.5
    result = diagnose((data, config, periods, manifest))
    guarded = result.period_metrics.query("period == 'test' and mode == 'configured'").iloc[0]
    full = result.period_metrics.query(
        "period == 'test' and mode == 'no_early_stop_inference'"
    ).iloc[0]
    assert not guarded.complete and guarded.observations == 3
    assert full.complete and full.observations == 12
    event = result.safeguard_events.iloc[0]
    assert event.date == data.returns.index[22].isoformat()
    assert event.drawdown > 0.15 and event.nav < 0.85
    trace = result.allocation_history.query("period == 'test' and mode == 'configured'")
    assert trace.iloc[-1].actual_cash < 1  # termination did not liquidate to cash
    assert config.mdd_limit == 0.15  # native default not changed


def test_safeguard_on_last_observation_is_an_event_but_not_early(case):
    case[0].returns.iloc[-1] = -0.5
    result = diagnose(case, sensitivity=False)
    detail = result.summary["periods"][-1]
    assert detail["complete"] and detail["reason"] == "mdd_limit_exceeded"
    assert detail["early_terminated"] is False
    assert result.safeguard_events.iloc[-1].date == case[0].returns.index[-1].isoformat()


def references(case):
    data, config, _, _ = case
    dates = data.returns.index[20:]
    starts = pd.Series(data.returns.index[19:-1], index=dates)
    return {
        "test": {
            "benchmark": ReferenceReturns(
                pd.Series(0.0, index=dates),
                config.commission_rate,
                config.slippage_rate,
                "KRW",
                return_starts=starts,
            )
        }
    }


def test_matched_reference_and_missing_baseline_are_distinguished(case):
    result = diagnose(case, references=references(case))
    detail = next(
        p for p in result.summary["periods"] if p["period"] == "test" and p["mode"] == "configured"
    )
    assert detail["comparisons"]["benchmark"]["status"] == "matched"
    assert detail["comparisons"]["equal_weight"]["status"] == "unavailable"
    value = result.period_metrics.query("period == 'test' and mode == 'configured'").iloc[0]
    assert value.benchmark_excess_cumulative_return == pytest.approx(value.cumulative_return)


@pytest.mark.parametrize(
    "defect", ["dates", "cost", "currency", "gross", "interval", "missing", "nan"]
)
def test_different_comparison_conditions_rejected(case, defect):
    refs = references(case)
    ref = refs["test"]["benchmark"]
    args = dict(
        returns=ref.returns.copy(),
        commission_rate=ref.commission_rate,
        slippage_rate=ref.slippage_rate,
        currency="KRW",
        return_starts=ref.return_starts.copy(),
    )
    if defect == "dates":
        args["returns"] = ref.returns.iloc[1:]
    elif defect == "cost":
        args["commission_rate"] = 0
    elif defect == "currency":
        args["currency"] = "USD"
    elif defect == "gross":
        args["cost_basis"] = "gross"
    elif defect == "interval":
        args["return_starts"].iloc[0] -= pd.Timedelta(days=1)
    elif defect == "missing":
        args["return_starts"] = None
    else:
        args["returns"].iloc[0] = np.nan
    refs["test"]["benchmark"] = ReferenceReturns(**args)
    with pytest.raises(ValueError):
        diagnose(case, references=refs)


def test_early_end_refuses_full_period_excess_comparison(case):
    case[0].returns.iloc[22] = -0.5
    result = diagnose(case, references=references(case))
    detail = next(
        p for p in result.summary["periods"] if p["period"] == "test" and p["mode"] == "configured"
    )
    assert detail["comparisons"]["benchmark"]["status"] == "not_comparable_incomplete"
    assert detail["comparisons"]["benchmark"]["excess_cumulative_return"] is None


@pytest.mark.leakage
def test_regime_uses_pre_decision_history_not_current_or_future_returns(case):
    dates = case[0].returns.index
    benchmark = pd.Series(0.01, index=dates)
    rules = {
        "block_anchor": dates[0].isoformat(),
        "block_size": 1,
        "min_blocks_per_regime": 1,
        "lookback": 3,
        "threshold": 0.02,
    }
    before = diagnose(case, regime_benchmark=benchmark, regime_rules=rules)
    benchmark.iloc[21:] = -0.8
    after = diagnose(case, regime_benchmark=benchmark, regime_rules=rules)
    a = before.allocation_history
    b = after.allocation_history
    assert a.loc[a.date <= dates[22], "regime"].equals(b.loc[b.date <= dates[22], "regime"])
    assert "up" in set(a.regime)


def test_training_episode_counts_and_missing_dates_remain_explicit(tmp_path):
    monitor = tmp_path / "monitor"
    monitor.mkdir()
    (monitor / "0.monitor.csv").write_text(
        "#{}\nr,l,t,reason\n1,5,1,mdd_limit_exceeded\n2,8,2,data_exhausted\n", encoding="utf-8"
    )
    summary, events = summarize_training_episodes(tmp_path)
    assert summary["episodes"] == 2 and summary["early_end_rate"] == 0.5
    assert events[0]["date"] is None and events[0]["nav"] is None
    assert summary["convergence"]["status"] == "insufficient_episodes"
    (monitor / "0.monitor.csv").write_text("#{}\nr,l,t\n1,5,1\n", encoding="utf-8")
    assert summarize_training_episodes(tmp_path)[0]["early_end_rate"] is None
    assert summarize_training_episodes(None)[0]["episodes"] is None


def test_outputs_reproducible_hashed_never_approved_and_no_overwrite(case, tmp_path):
    original = case[0].returns.copy(deep=True)
    result = diagnose(case)
    first = save_diagnostics(result, tmp_path / "first")
    second = save_diagnostics(diagnose(case), tmp_path / "second")
    assert len(list(first.iterdir())) == 7
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["output_sha256"].items():
        assert hashlib.sha256((first / name).read_bytes()).hexdigest() == digest
    assert not manifest["training_approved"] and result.summary["status"] == "unverified"
    with pytest.raises(FileExistsError):
        save_diagnostics(result, first)
    assert_frame_equal(original, case[0].returns)


def make_spec(case, tmp_path):
    data, _, periods, manifest = copy.deepcopy(case)

    def pin(name, content):
        (tmp_path / name).write_bytes(content)
        return {"path": name, "sha256": hashlib.sha256(content).hexdigest()}

    files = {}
    for name in ("returns", "rsi", "macd"):
        content = getattr(data, name).to_parquet()
        files[name + ".parquet"] = pin(name + ".parquet", content)
    manifest["data_file_sha256"] = {name: item["sha256"] for name, item in files.items()}
    canonical = json.dumps(
        manifest["config"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    manifest["config_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    spec = {
        "schema_version": 1,
        "kind": "synthetic",
        "training_manifest": pin("training.json", json.dumps(manifest).encode()),
        "checkpoint": {
            **pin("model.zip", b"SYNTHETIC TEST DOUBLE NOT A REAL MODEL"),
            "run_id": manifest["run_id"],
        },
        "dataset": files,
        "policy_version": "test-only-v1",
        "currency": "KRW",
        "source_audit": pin("audit.json", b'{"status":"failed","training_approved":false}'),
        "periods": [
            {"name": p.name, "start": p.start.isoformat(), "end": p.end.isoformat()}
            for p in periods
        ],
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path, spec


def test_file_entrypoint_pins_inputs_and_preserves_failed_source(case, tmp_path, monkeypatch):
    path, spec = make_spec(case, tmp_path)
    monkeypatch.setattr(inputs.PPO, "load", lambda *args, **kwargs: ConstantPolicy())
    output = inputs.run_from_spec(path, root=tmp_path, output_dir=tmp_path / "diagnostic")
    result = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert result["source_validation"]["reported_status"] == "failed"
    assert not result["training_approved"] and result["status"] == "unverified"
    assert result["provenance"]["checkpoint"]["sha256"] == spec["checkpoint"]["sha256"]
    assert result["provenance"]["dataset_inputs"] == spec["dataset"]
    assert result["kind"] == "synthetic"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "synthetic"
    csv = pd.read_csv(output / "period_metrics.csv")
    assert csv.result_kind.eq("synthetic").all()


@pytest.mark.parametrize("defect", ["checkpoint", "data", "manifest", "run_id", "config", "seed"])
def test_file_entrypoint_rejects_hashes_and_model_manifest_mismatch_before_output(
    case, tmp_path, monkeypatch, defect
):
    path, spec = make_spec(case, tmp_path)
    monkeypatch.setattr(
        inputs.PPO, "load", lambda *args, **kwargs: pytest.fail("must fail before model load")
    )
    if defect == "checkpoint":
        (tmp_path / "model.zip").write_bytes(b"tampered")
    elif defect == "data":
        (tmp_path / "returns.parquet").write_bytes(b"tampered")
    elif defect == "manifest":
        (tmp_path / "training.json").write_bytes(b"tampered")
    elif defect == "run_id":
        spec["checkpoint"]["run_id"] = "another-model"
    else:
        file = tmp_path / "training.json"
        manifest = json.loads(file.read_bytes())
        manifest["config"]["env"]["mdd_limit"] = 1 if defect == "config" else 0.15
        if defect == "seed":
            manifest["seed"] = 7
        content = json.dumps(manifest).encode()
        file.write_bytes(content)
        spec["training_manifest"]["sha256"] = hashlib.sha256(content).hexdigest()
    path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError):
        inputs.run_from_spec(path, root=tmp_path, output_dir=tmp_path / "diagnostic")
    assert not (tmp_path / "diagnostic").exists()


@pytest.mark.parametrize("path", ["../model.zip", "/absolute/model.zip"])
def test_pinned_inputs_cannot_escape_root(tmp_path, path):
    with pytest.raises(ValueError):
        inputs.read_pinned(tmp_path, {"path": path, "sha256": "0" * 64})


def test_experiment_condition_mismatches_are_explicit_without_performance_claim(case):
    first = diagnose(case)
    changed = copy.deepcopy(case)
    changed[0].returns.iloc[22] += 0.01
    other = diagnose(changed, policy_version="different-policy")
    comparison = compare_experiment_settings(first, other)
    assert comparison["performance_comparison"] == "not_performed"
    assert set(comparison["mismatches"]) == {"policy_version", "dataset_fingerprints"}
    assert compare_experiment_settings(first, diagnose(case))["same_conditions"]


@pytest.mark.leakage
def test_policy_observation_contains_past_window_not_ensuing_return(case):
    model = ObservationPolicy()
    trace_policy(model, case[0], case[1])
    np.testing.assert_array_equal(
        model.observations[0][:36], case[0].returns.iloc[:3].to_numpy().ravel()
    )
    assert not np.array_equal(
        model.observations[0][:36], case[0].returns.iloc[1:4].to_numpy().ravel()
    )


def test_failed_output_write_removes_only_new_output(case, tmp_path, monkeypatch):
    result = diagnose(case)
    original = pd.DataFrame.to_csv
    sentinel = tmp_path / "original.txt"
    sentinel.write_text("keep", encoding="utf-8")

    def fail(self, *args, **kwargs):
        if "target__ETF0" in self.columns:
            raise OSError("simulated disk error")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail)
    with pytest.raises(OSError, match="disk error"):
        save_diagnostics(result, tmp_path / "new")
    assert not (tmp_path / "new").exists()
    assert sentinel.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("defect", ["NaN", "Inf", "negative_return", "RSI", "columns"])
def test_bad_dataset_never_produces_a_success_report(case, defect):
    changed = copy.deepcopy(case)
    if defect == "NaN":
        changed[0].returns.iloc[15, 0] = np.nan
    elif defect == "Inf":
        changed[0].macd.iloc[15, 0] = np.inf
    elif defect == "negative_return":
        changed[0].returns.iloc[15, 0] = -1.1
    elif defect == "RSI":
        changed[0].rsi.iloc[15, 0] = 101
    else:
        changed[0].macd.columns = list(reversed(changed[0].macd.columns))
    with pytest.raises(ValueError):
        diagnose(changed)


def test_empty_future_test_is_unavailable_not_zero_performance(case):
    data, config, periods, manifest = case
    end = data.returns.index[-1] + pd.Timedelta(days=1)
    new_periods = [*periods[:2], DiagnosticPeriod("test", end, end + pd.Timedelta(days=30))]
    result = diagnose((data, config, new_periods, manifest))
    rows = result.period_metrics.query("period == 'test'")
    assert not rows.complete.any() and rows.observations.eq(0).all()
    assert rows.cumulative_return.isna().all()


def test_source_claim_cannot_promote_diagnostic_to_training_approved(case, tmp_path, monkeypatch):
    path, spec = make_spec(case, tmp_path)
    raw = b'{"status":"verified","training_approved":true}'
    (tmp_path / "audit.json").write_bytes(raw)
    spec["source_audit"]["sha256"] = hashlib.sha256(raw).hexdigest()
    path.write_text(json.dumps(spec), encoding="utf-8")
    monkeypatch.setattr(inputs.PPO, "load", lambda *args, **kwargs: ConstantPolicy())
    output = inputs.run_from_spec(path, root=tmp_path, output_dir=tmp_path / "new")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "unverified" and summary["training_approved"] is False
    assert summary["source_validation"]["status"] == "unverified"


def test_direct_api_keeps_source_claim_separate_from_diagnostic_status(case):
    result = diagnose(
        case,
        provenance={
            "kind": "synthetic",
            "run_id": case[3]["run_id"],
            "source_validation": {"status": "verified", "training_approved": True},
        },
    )
    expected = {
        "reported_status": "verified",
        "reported_training_approved": True,
        "status": "unverified",
        "training_approved": False,
    }
    assert result.summary["source_validation"] == expected
    assert result.summary["provenance"]["source_validation"] == expected


@pytest.mark.parametrize("field,value", [("status", "verified"), ("training_approved", True)])
def test_export_rejects_manually_promoted_result(case, tmp_path, field, value):
    result = diagnose(case)
    result.summary[field] = value
    with pytest.raises(ValueError, match="cannot certify"):
        save_diagnostics(result, tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_non_utc_input_is_equivalent_and_does_not_fake_incomplete_coverage(case):
    before = diagnose(case)
    changed = copy.deepcopy(case)
    for frame in (changed[0].returns, changed[0].rsi, changed[0].macd):
        frame.index = frame.index.tz_convert("Asia/Seoul")
    after = diagnose(changed)
    assert_frame_equal(before.period_metrics, after.period_metrics)
    assert_frame_equal(before.allocation_history, after.allocation_history)


def test_reference_same_instants_in_another_timezone_are_comparable(case):
    refs = references(case)
    ref = refs["test"]["benchmark"]
    values = ref.returns.tz_convert("America/New_York")
    starts = ref.return_starts.tz_convert("America/New_York")
    refs["test"]["benchmark"] = ReferenceReturns(
        values, ref.commission_rate, ref.slippage_rate, "KRW", return_starts=starts
    )
    result = diagnose(case, references=refs)
    assert (
        result.period_metrics.query("period == 'test'")
        .benchmark_excess_cumulative_return.notna()
        .all()
    )


@pytest.mark.parametrize("defect", ["seed", "action_space", "observation_space"])
def test_checkpoint_metadata_matches_actual_loaded_model(case, tmp_path, monkeypatch, defect):
    path, _ = make_spec(case, tmp_path)
    model = ConstantPolicy()
    setattr(model, defect, 7 if defect == "seed" else SimpleNamespace(shape=(2,)))
    monkeypatch.setattr(inputs.PPO, "load", lambda *args, **kwargs: model)
    with pytest.raises(ValueError, match="checkpoint"):
        inputs.run_from_spec(path, root=tmp_path, output_dir=tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_serialized_untrained_synthetic_policy_can_run_inference_without_learning(case, tmp_path):
    from stable_baselines3 import PPO
    from robo_advisor.rl.training import _make_env

    path, spec = make_spec(case, tmp_path)
    env = _make_env(case[0], case[1], seed=42)
    try:
        # No learn(): this is only an offline serialization/inference contract test.
        model = PPO("MlpPolicy", env, seed=42, n_steps=2, batch_size=2, device="cpu")
        model.save(tmp_path / "model.zip")
    finally:
        env.close()
    spec["checkpoint"]["sha256"] = hashlib.sha256((tmp_path / "model.zip").read_bytes()).hexdigest()
    path.write_text(json.dumps(spec), encoding="utf-8")
    output = inputs.run_from_spec(path, root=tmp_path, output_dir=tmp_path / "new")
    assert (output / "allocation_history.csv").exists()


@pytest.mark.leakage
def test_risk_available_at_is_not_visible_before_the_decision_time(case):
    from robo_advisor.schemas.risk import RiskTag

    data = case[0]
    dates = data.returns.index
    tag = RiskTag(
        event_type="sharp_move",
        severity=1,
        confidence=1,
        asset="ETF0",
        published_at=dates[14].to_pydatetime(),
        available_at=dates[15].to_pydatetime(),
        source_url="https://example.com/synthetic-test",
        trace_id="synthetic-risk",
    )
    tagged = TrainingDataset(data.returns, data.rsi, data.macd, (tag,), data.file_sha256)
    model = ObservationPolicy()
    trace_policy(model, tagged, case[1])
    for step, observation in enumerate(model.observations):
        assert observation[-12] == (0 if step + 2 < 15 else 1)


@pytest.mark.parametrize(
    "nav,reached,exceeded",
    [(0.85, True, False), (np.nextafter(0.85, 0), True, True), (0.851, False, False)],
)
def test_report_15pct_crossings_use_nav_boundary_in_both_sensitivity_modes(nav, reached, exceeded):
    dates = pd.date_range("2024-01-01", periods=1, tz="UTC")
    # At the exact boundary the unrestricted guard reports a rounded value > 0.15.
    trace = pd.DataFrame(
        {"nav": [float(nav)], "peak_nav": [1.0], "drawdown": [1 - nav]}, index=dates
    )
    result = _drawdown_crossings(trace)
    assert result["first_reached_15pct"] == (dates[0].isoformat() if reached else None)
    assert result["first_exceeded_15pct"] == (dates[0].isoformat() if exceeded else None)


@pytest.mark.parametrize("complete", [False, True])
def test_native_training_rate_requires_all_parallel_environment_monitors(case, tmp_path, complete):
    changed = copy.deepcopy(case)
    changed[3]["n_envs"] = changed[3]["config"]["n_envs"] = 2
    monitor = tmp_path / "monitor"
    monitor.mkdir()
    (monitor / "env-000.monitor.csv").write_text(
        "#{}\nr,l,t,reason\n1,5,1,mdd_limit_exceeded\n", encoding="utf-8"
    )
    if not complete:
        with pytest.raises(ValueError, match="all native environment Monitor"):
            diagnose(changed, training_run_dir=tmp_path)
    else:
        (monitor / "env-001.monitor.csv").write_text(
            "#{}\nr,l,t,reason\n2,8,2,data_exhausted\n", encoding="utf-8"
        )
        result = diagnose(changed, training_run_dir=tmp_path)
        assert result.summary["training_episodes"]["early_end_rate"] == 0.5
        assert result.summary["training_episodes"]["scope"] == "provided_completed_monitor_episodes"
