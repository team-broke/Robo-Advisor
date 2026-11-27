import hashlib
import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from robo_advisor.backtest.comparison import (
    ComparisonPolicy,
    _schedule_intervals,
    compare_window,
    composite_benchmark,
    main,
    prepare_anova_reward_samples,
    save_comparison,
)
from robo_advisor.backtest.retrain import _metrics, equal_weight_policy
from robo_advisor.config import Settings
from robo_advisor.diagnostics.ppo_diagnostics import DiagnosticPeriod
from robo_advisor.rl import training
from robo_advisor.rl.evaluation import evaluate_policy
from robo_advisor.rl.training import EnvironmentConfig, TrainingDataset


@pytest.fixture
def case():
    dates = pd.bdate_range("2024-01-02", periods=45, tz="UTC") + pd.Timedelta(hours=6.5)
    returns = pd.DataFrame(
        np.random.default_rng(12).normal(0.001, 0.005, (45, 3)),
        index=dates,
        columns=["A", "B", "C"],
    )
    config = EnvironmentConfig(lookback=3, commission_rate=0, slippage_rate=0)
    periods = [
        DiagnosticPeriod("train", dates[0], dates[12]),
        DiagnosticPeriod("validation", dates[12], dates[20]),
        DiagnosticPeriod("test", dates[20], dates[-1] + pd.Timedelta(days=1)),
    ]
    manifest = {
        "status": "completed",
        "run_id": "synthetic",
        "seed": 42,
        "config": {
            "seed": 42,
            "env": config.model_dump(mode="json"),
            "data": {"train_start": dates[0].isoformat(), "train_end": dates[19].isoformat()},
            "validation": {"start": dates[12].isoformat()},
        },
        "data_cutoff": {
            "assets": list(returns),
            "actual_first_index": dates[0].isoformat(),
            "actual_last_index": dates[11].isoformat(),
        },
        "data_file_sha256": {"returns.parquet": "synthetic-fixture"},
    }
    data = TrainingDataset(returns, returns * 0 + 50, returns * 0, (), manifest["data_file_sha256"])
    policy = ComparisonPolicy("draft", "include_first_session", "observe_only", "PR", "smbs", 5)
    session = [d.date().isoformat() for d in dates]
    audit = {
        "status": "unverified",
        "training_approved": False,
        "metrics": {
            "fx": {"source": "smbs"},
            "sources": {name: {"return_type": "PR"} for name in ("KOSPI", "SP500")},
            "rows": [
                {
                    "evaluation_at": d.isoformat(),
                    "session_date": session[i],
                    "return_start_date": session[i - 1] if i else None,
                    "kospi_simple_return": 0.006 if i else None,
                    "sp500_krw_simple_return": -0.002 if i else None,
                }
                for i, d in enumerate(dates)
            ],
        },
    }
    return {
        "model": equal_weight_policy(3),
        "dataset": data,
        "config": config,
        "schedule": dates,
        "periods": periods,
        "benchmark_audit": audit,
        "policy": policy,
        "policy_version": "draft",
        "currency": "KRW",
        "training_manifest": manifest,
        "provenance": {"kind": "synthetic", "run_id": "synthetic"},
    }


def test_backtest_paths_and_output_hashes(case, tmp_path):
    result = compare_window(**case)
    assert result.manifest["comparable"] is True
    assert result.manifest["policy_status"] == "candidate"
    assert result.manifest["benchmark_source_status"] == "unverified"
    assert result.manifest["training_approved"] is False
    daily = result.tables["daily_returns"]
    benchmark = daily.query("strategy == 'benchmark'")
    assert benchmark["return"].tolist() == pytest.approx([0.002] * 25)
    assert benchmark.iloc[0].return_start == case["schedule"][19]
    assert benchmark.iloc[0].return_end == case["schedule"][20]
    assert daily.groupby("strategy").size().eq(25).all()
    out = save_comparison(result, tmp_path / "comparison")
    pinned = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert len(pinned["output_sha256"]) == 9
    repeated = save_comparison(compare_window(**case), tmp_path / "repeated")
    for name, digest in pinned["output_sha256"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
        assert (out / name).read_bytes() == (repeated / name).read_bytes()
        assert b"\r\n" not in (out / name).read_bytes()
    assert (out / "manifest.json").read_bytes() == (repeated / "manifest.json").read_bytes()
    assert pd.read_csv(out / "daily_returns.csv").result_kind.eq("synthetic").all()
    with pytest.raises(FileExistsError):
        save_comparison(result, out)
    result.manifest["training_approved"] = True
    with pytest.raises(ValueError, match="cannot certify"):
        save_comparison(result, tmp_path / "approved")


def test_mdd_and_first_cost_use_initial_nav(case):
    case["config"] = case["config"].model_copy(
        update={"commission_rate": 0.02, "slippage_rate": 0.01}
    )
    case["training_manifest"]["config"]["env"] = case["config"].model_dump(mode="json")
    case["dataset"].returns.iloc[20:] = 0.0
    case["periods"][-1] = replace(case["periods"][-1], end=case["schedule"][22])
    result = compare_window(**case)
    metrics = result.tables["strategy_metrics"].set_index("strategy")
    history = result.tables["nav_history"]
    trades = result.tables["rebalance_history"]
    for strategy in ("ppo", "mvo", "equal_weight"):
        nav = history[history.strategy.eq(strategy)]
        assert list(nav.evaluation_at[:2]) == list(case["schedule"][19:21])
        assert nav.nav.iloc[0] == 1.0
        assert nav.nav.iloc[1] == pytest.approx(0.97)
        assert trades.loc[trades.strategy.eq(strategy), "cost"].iloc[0] == pytest.approx(0.03)
        assert metrics.loc[strategy, "mdd"] == pytest.approx(0.03)


def test_missing_benchmark_does_not_change_any_strategy_path_or_create_relative_statistics(case):
    full = compare_window(**case)
    case["benchmark_audit"]["metrics"]["rows"][22]["sp500_krw_simple_return"] = None
    partial = compare_window(**case)
    assert not partial.manifest["comparable"]
    assert partial.tables["strategy_metrics"]["information_ratio"].isna().all()
    for table in ("daily_returns", "nav_history", "rebalance_history"):
        pd.testing.assert_frame_equal(
            full.tables[table].query("strategy != 'benchmark'").reset_index(drop=True),
            partial.tables[table].query("strategy != 'benchmark'").reset_index(drop=True),
        )
    assert pd.isna(
        partial.tables["daily_returns"].query("strategy == 'benchmark'").iloc[2]["return"]
    )


@pytest.mark.parametrize("defect", ["start", "end", "return_type", "fx"])
def test_benchmark_rejects_mixed_basis_or_mismatched_transitions(case, defect):
    audit = case["benchmark_audit"]["metrics"]
    if defect == "start":
        audit["rows"][20]["return_start_date"] = audit["rows"][18]["session_date"]
    elif defect == "end":
        audit["rows"][20]["evaluation_at"] = (
            case["schedule"][20] + pd.Timedelta(minutes=1)
        ).isoformat()
    elif defect == "return_type":
        audit["sources"]["SP500"]["return_type"] = "TR"
    else:
        audit["fx"]["source"] = "alfred"
    test = case["periods"][-1]
    intervals = _schedule_intervals(
        case["schedule"], test.start, test.end, case["policy"].first_return
    )
    with pytest.raises(ValueError):
        composite_benchmark(case["benchmark_audit"], intervals, case["policy"])


def test_guarded_ppo_is_incomplete_while_mvo_completes_and_sensitivity_completes(case):
    case["dataset"].returns.iloc[20] = -0.15
    case["dataset"].returns.iloc[21] = -0.01
    no_stop = compare_window(**case)
    case["policy"] = replace(case["policy"], safeguard="configured")
    stopped = compare_window(**case)
    assert no_stop.manifest["comparable"]
    assert not stopped.manifest["comparable"]
    metrics = stopped.tables["strategy_metrics"].set_index("strategy")
    assert not metrics.loc["ppo", "complete"] and metrics.loc["mvo", "complete"]
    assert pd.isna(metrics.loc["ppo", "excess_vs_equal_weight"])
    event = stopped.tables["safeguard_events"].set_index("strategy").loc["ppo"]
    assert event.first_exceeded_15pct == case["schedule"][21].isoformat()
    assert case["config"].mdd_limit == 0.15


def test_comparison_can_start_after_first_observation_without_changing_validation(case):
    include = compare_window(**case)
    case["policy"] = replace(case["policy"], first_return="after_first_observation")
    exclude = compare_window(**case)
    assert exclude.manifest["first_return_end"] == case["schedule"][21]
    a, b = [
        r.tables["ppo_period_metrics"].query("period == 'validation'") for r in (include, exclude)
    ]
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.leakage
def test_excluding_first_return_cannot_hide_requested_test_overlap(case):
    case["policy"] = replace(case["policy"], first_return="after_first_observation")
    test = case["periods"][-1]
    case["periods"][-1] = replace(test, start=case["schedule"][19])
    with pytest.raises(ValueError, match="nonoverlapping"):
        compare_window(**case)


def test_missing_asset_row_is_not_joined_across_gap(case):
    data = case["dataset"]
    hole = case["schedule"][21]
    case["dataset"] = replace(
        data, returns=data.returns.drop(hole), rsi=data.rsi.drop(hole), macd=data.macd.drop(hole)
    )
    with pytest.raises(ValueError, match="gaps cannot be rolled"):
        compare_window(**case)


def test_audit_calendar_exposes_shared_asset_and_schedule_gap(case):
    data, hole = case["dataset"], case["schedule"][22]
    case["dataset"] = replace(
        data, returns=data.returns.drop(hole), rsi=data.rsi.drop(hole), macd=data.macd.drop(hole)
    )
    case["schedule"] = case["schedule"].drop(hole)
    rows = case["benchmark_audit"]["metrics"]["rows"]
    for position in (22, 23):
        rows[position]["sp500_krw_simple_return"] = None
    with pytest.raises(ValueError, match="audit calendar contains missing comparison sessions"):
        compare_window(**case)


@pytest.mark.leakage
def test_future_market_returns_do_not_change_past_trade(case):
    before = compare_window(**case).tables["rebalance_history"]
    cutoff = case["schedule"][23]
    case["dataset"].returns.loc[case["schedule"] > cutoff] = 0.25
    after = compare_window(**case).tables["rebalance_history"]
    for table in (before, after):
        assert (table.return_end <= cutoff).any()
    pd.testing.assert_frame_equal(
        before.loc[before.return_end <= cutoff].reset_index(drop=True),
        after.loc[after.return_end <= cutoff].reset_index(drop=True),
    )


@pytest.mark.parametrize("missing", [True, False])
def test_retrain_does_not_report_ir_against_zero_fallback(case, missing):
    native = evaluate_policy(case["model"], case["dataset"], case["config"])
    benchmark = native.returns.iloc[1:] if missing else None
    result = _metrics(native, benchmark)
    assert all(result[name] is None for name in ("alpha", "beta", "information_ratio"))
    assert result["cumulative_return"] is not None


def test_anova_rejects_synthetic_and_explains_missing_cells(case):
    result = compare_window(**case)
    summary = prepare_anova_reward_samples(
        {("return", 42, "W1"): result},
        rewards=["return", "sharpe"],
        seeds=[42],
        windows=["W1", "W2"],
    )
    assert summary["status"] == "insufficient" and summary["samples"] is None
    assert summary["incomplete"] == [("return", 42, "W1")]
    assert len(summary["missing"]) == 3


def test_anova_builds_candidate_samples_only_after_validating_complete_design(case):
    # Exercise the gate with test doubles. No statistical test or real model is involved.
    result = compare_window(**case)
    runs = {}
    for reward in ("return", "sharpe"):
        for seed in (42, 43):
            run = deepcopy(result)
            run.manifest["kind"] = "checkpoint_inference"
            metadata = run.manifest["ppo"]
            metadata["provenance"]["run_id"] = f"fixture-{reward}-{seed}"
            metadata["environment"]["reward_variant"] = reward
            metadata["seed"] = seed
            runs[reward, seed, "W1"] = run
    design = {"rewards": ["return", "sharpe"], "seeds": [42, 43], "windows": ["W1"]}
    collected = prepare_anova_reward_samples(runs, **design, selected_reward="return", block_size=5)
    assert collected["status"] == "candidate_samples"
    assert len(collected["samples"]) == 4
    assert len(collected["blocks"]) == 15
    runs["sharpe", 43, "W1"].manifest["ppo"]["seed"] = 42
    with pytest.raises(ValueError, match="matching reward and seed"):
        prepare_anova_reward_samples(runs, **design)


def test_synthetic_cpu_smoke_train_reload_and_export(case, tmp_path, monkeypatch):
    def pin(path):
        return {
            "path": path.relative_to(tmp_path).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    data = case["dataset"]
    dates = case["schedule"]
    files = {}
    for name in ("returns", "rsi", "macd"):
        path = tmp_path / (name + ".parquet")
        getattr(data, name).to_parquet(path)
        files[path.name] = pin(path)
    settings = Settings(
        _env_file=None, models_dir=tmp_path / "models", artifacts_dir=tmp_path / "artifacts"
    )
    monkeypatch.setattr(training, "get_settings", lambda: settings)
    config = training.TrainingConfig.model_validate(
        {
            "device": "cpu",
            "seed": 42,
            "vec_env": "dummy",
            "n_envs": 1,
            "total_timesteps": 16,
            "ppo": {"n_steps": 8, "batch_size": 8, "n_epochs": 1},
            "env": case["config"].model_dump(mode="json"),
            "data": {"dataset_dir": tmp_path, "train_start": dates[0], "train_end": dates[19]},
            "validation": {"start": dates[12], "eval_freq": 8},
        },
        context={"smoke": True},
    )
    run = training.run_training(config, smoke=True, run_id="synthetic-only")
    spec = tmp_path / "diagnostic.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "synthetic",
                "training_manifest": pin(run.manifest_path),
                "checkpoint": {**pin(run.final_model_path), "run_id": "synthetic-only"},
                "dataset": files,
                "periods": [
                    {"name": p.name, "start": p.start.isoformat(), "end": p.end.isoformat()}
                    for p in case["periods"]
                ],
                "policy_version": "draft",
                "currency": "KRW",
            }
        ),
        encoding="utf-8",
    )
    schedule = tmp_path / "schedule.csv"
    pd.DataFrame({"evaluation_at": dates}).to_csv(schedule, index=False)

    def csv(name, **columns):
        path = tmp_path / (name + ".csv")
        pd.DataFrame(columns).to_csv(path, index=False)
        return pin(path)

    seed_date = dates[0] - pd.Timedelta(days=1)
    us_dates = dates.insert(0, seed_date)
    kr_days = dates.strftime("%Y-%m-%d")
    us_days = us_dates.strftime("%Y-%m-%d")
    benchmark_spec = tmp_path / "benchmark.json"
    benchmark_spec.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "start": kr_days[0],
                "end_exclusive": (dates[-1] + pd.Timedelta(days=1)).date().isoformat(),
                "calendars": {
                    "KRX": csv("kr_calendar", session_date=kr_days, close=dates),
                    "US": csv(
                        "us_calendar",
                        session_date=us_days,
                        close=(pd.to_datetime(us_days) + pd.Timedelta(hours=16)).tz_localize(
                            "America/New_York"
                        ),
                    ),
                },
                "evaluation": csv("evaluation", session_date=kr_days, evaluation_at=dates),
                "benchmarks": {
                    "KOSPI": {
                        **csv(
                            "kospi",
                            Date=kr_days,
                            Close=np.arange(len(dates)) + 3000,
                            Ticker="^KS11",
                            Source="Yahoo Finance",
                        ),
                        "index_id": "KOSPI",
                        "source": "yahoo",
                        "symbol": "^KS11",
                        "currency": "KRW",
                        "return_type": "PR",
                    },
                    "SP500": {
                        **csv("spx", Date=us_days, Close=np.arange(len(us_dates)) + 4000),
                        "index_id": "SP500",
                        "source": "unknown",
                        "currency": "USD",
                        "return_type": "PR",
                    },
                },
                "fx": {
                    **csv(
                        "fx",
                        date=kr_days,
                        mar_krw_per_usd=1000,
                        source_identity="smbs:USD_KRW_MAR:public_period_table",
                        actual_available_at=dates - pd.Timedelta(hours=3),
                    ),
                    "source": "smbs",
                    "max_age_days": 7,
                },
            }
        ),
        encoding="utf-8",
    )
    integrated = tmp_path / "integrated.json"
    integrated.write_text(
        json.dumps(
            {
                "diagnostic": pin(spec),
                "schedule": pin(schedule),
                "benchmark": pin(benchmark_spec),
                "policy": case["policy"].__dict__,
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "output"
    main(["--spec", str(integrated), "--root", str(tmp_path), "--output", str(out)])
    report = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert report["kind"] == "synthetic" and report["comparable"], report["benchmark"]
    assert not report["training_approved"]
    assert report["comparison_spec_sha256"] == pin(integrated)["sha256"]
    original_spec = spec.read_text(encoding="utf-8")
    original_integrated = integrated.read_text(encoding="utf-8")
    changed = tmp_path / "changed.parquet"
    (data.returns * 2).to_parquet(changed)
    diagnostic = json.loads(original_spec)
    diagnostic["dataset"]["returns.parquet"] = pin(changed)
    spec.write_text(json.dumps(diagnostic), encoding="utf-8")
    re_pinned = {**json.loads(original_integrated), "diagnostic": pin(spec)}
    integrated.write_text(json.dumps(re_pinned), encoding="utf-8")
    with pytest.raises(ValueError, match="dataset/manifest version mismatch"):
        main(
            [
                "--spec",
                str(integrated),
                "--root",
                str(tmp_path),
                "--output",
                str(tmp_path / "changed-input"),
            ]
        )
    spec.write_text(original_spec, encoding="utf-8")
    integrated.write_text(original_integrated, encoding="utf-8")
    schedule.write_text("evaluation_at\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        main(
            [
                "--spec",
                str(integrated),
                "--root",
                str(tmp_path),
                "--output",
                str(tmp_path / "tampered"),
            ]
        )
