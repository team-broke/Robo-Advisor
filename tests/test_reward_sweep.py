"""두 단계 실험의 선별·재개·컷오프 차단 계약 검증."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from robo_advisor.config import Settings
from robo_advisor.experiments import reward_sweep as sweep
from robo_advisor.rl import training
from robo_advisor.rl.evaluation import EvaluationResult


@pytest.fixture
def setup(tmp_path, monkeypatch):
    directory = tmp_path / "data"
    directory.mkdir()
    dates = pd.date_range("2026-01-01", periods=12, tz="UTC")
    returns = pd.DataFrame(0.001, index=dates, columns=["A", "B"])
    for name, frame in (("returns", returns), ("rsi", returns * 0 + 50), ("macd", returns * 0)):
        frame.to_parquet(directory / f"{name}.parquet")
    settings = Settings(
        _env_file=None,
        models_dir=tmp_path / "models",
        artifacts_dir=tmp_path / "artifacts",
    )
    monkeypatch.setattr(sweep, "get_settings", lambda: settings)
    monkeypatch.setattr(training, "get_settings", lambda: settings)
    base_path = tmp_path / "training.yaml"
    base_path.write_text(
        yaml.safe_dump(
            {
                "device": "cpu",
                "vec_env": "dummy",
                "n_envs": 2,
                "total_timesteps": 64,
                "ppo": {"n_steps": 16, "batch_size": 16, "n_epochs": 1},
                "env": {"lookback": 2},
                "data": {
                    "dataset_dir": str(directory),
                    "train_start": "2026-01-01T00:00:00Z",
                    "train_end": "2026-01-08T00:00:00Z",
                },
            }
        )
    )
    config_path = tmp_path / "sweep.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "base_training_config": "training.yaml",
                "sweep_id": "test-sweep",
                "stage1": {"seeds": [42]},
                "stage2": {"seeds": [42, 43, 44, 45, 46]},
            }
        )
    )
    config = sweep.load_sweep_config(config_path)
    calls = []
    loads = []

    def run(cfg, *, run_id, smoke):
        assert smoke
        model_dir = settings.models_dir / run_id
        artifact_dir = settings.artifacts_dir / "training" / run_id
        if model_dir.exists() or artifact_dir.exists():
            raise FileExistsError(run_id)
        model_dir.mkdir(parents=True)
        artifact_dir.mkdir(parents=True)
        model_path = model_dir / "final_model.zip"
        model_path.write_text("saved model")
        manifest = {
            "run_id": run_id,
            "status": "completed",
            "smoke": smoke,
            "requested_device": cfg.device,
            "config": cfg.model_dump(mode="json"),
        }
        (artifact_dir / "manifest.json").write_text(json.dumps(manifest))
        calls.append((run_id, cfg))
        return SimpleNamespace(model=SimpleNamespace(run_id=run_id))

    def load(path, device):
        loads.append(Path(path))
        return SimpleNamespace(run_id=Path(path).parent.name)

    def evaluate(model, dataset, config):
        index = dataset.returns.index[config.lookback :]
        values = pd.Series([0.01, -0.005] * 3, index=index, name="portfolio_return")
        nav = pd.Series(np.r_[1, np.cumprod(1 + values)], index=dataset.returns.index[1:])
        return EvaluationResult(values, nav, False, True, "data_exhausted", len(values))

    monkeypatch.setattr(sweep, "run_training", run)
    monkeypatch.setattr(sweep.PPO, "load", load)
    monkeypatch.setattr(sweep, "evaluate_policy", evaluate)
    return SimpleNamespace(
        config=config,
        calls=calls,
        loads=loads,
        settings=settings,
        dataset_dir=directory,
        base_path=base_path,
        config_path=config_path,
        dates=dates,
    )


def test_stage1_grid_ids_tie_break_and_recorded_evaluation(setup):
    path = sweep.run_stage1(setup.config, smoke=True)
    result = json.loads(path.read_text())
    assert len(setup.calls) == 5
    assert result["selected_lambdas"] == [0.5, 1.0]
    assert result["tie_break"] == "smaller_lambda_first"
    assert result["selection_rule"]["metric"] == "sharpe"
    assert len({run_id for run_id, _ in setup.calls}) == 5
    assert "lam0p5" in setup.calls[0][0]
    for record in result["runs"]:
        assert record["evaluation"]["observation_count"] == 6
        assert pd.Timestamp(record["evaluation"]["end"]) <= setup.dates[7]
        assert record["evaluation"]["reason"] == "data_exhausted"
    manifest = json.loads((path.parent / "manifest.json").read_text())
    assert manifest["stages"]["1"]["status"] == "completed"
    assert manifest["stages"]["1"]["run_ids"] == [run_id for run_id, _ in setup.calls]
    assert manifest["finished_at"] is not None


def test_stage2_has_four_groups_times_five_seeds(setup):
    selection = sweep.run_stage1(setup.config, smoke=True)
    path = sweep.run_stage2(setup.config, selection, smoke=True)
    jobs = setup.calls[5:]
    assert len(jobs) == 20
    assert len({run_id for run_id, _ in setup.calls}) == 25
    groups = {}
    for _, cfg in jobs:
        key = (cfg.env.reward_variant.value, cfg.env.penalty_lambda)
        groups.setdefault(key, []).append(cfg.seed)
        assert cfg.total_timesteps == 64
    assert set(groups) == {
        ("return", 1.0),
        ("sharpe", 1.0),
        ("mdd_penalty", 0.5),
        ("mdd_penalty", 1.0),
    }
    assert all(seeds == [42, 43, 44, 45, 46] for seeds in groups.values())
    manifest = json.loads(path.read_text())
    assert manifest["stages"]["2"]["status"] == "completed"
    assert len(manifest["stages"]["2"]["run_ids"]) == 20
    assert manifest["stages"]["2"]["selection_sha256"]


def test_stage2_requires_selection_file_before_training(setup):
    with pytest.raises(FileNotFoundError):
        sweep.run_stage2(setup.config, setup.config_path.parent / "missing.json", smoke=True)
    assert not setup.calls


@pytest.mark.parametrize(
    "field",
    ["sweep_id", "base_config_sha256", "data_cutoff", "selected_lambdas", "selection_rule"],
)
def test_stage2_rejects_mismatched_selection(setup, field):
    path = sweep.run_stage1(setup.config, smoke=True)
    selected = json.loads(path.read_text())
    if field == "selected_lambdas":
        selected[field] = [2.0, 3.0]
    elif field == "data_cutoff":
        selected[field]["train_end"] = "2026-01-09T00:00:00Z"
    elif field == "selection_rule":
        selected[field]["metric"] = "cagr"
    else:
        selected[field] = "mismatch"
    path.write_text(json.dumps(selected))
    with pytest.raises(ValueError):
        sweep.run_stage2(setup.config, path, smoke=True)
    assert len(setup.calls) == 5


def test_base_configuration_change_blocks_stage2(setup):
    path = sweep.run_stage1(setup.config, smoke=True)
    base = yaml.safe_load(setup.base_path.read_text())
    base["ppo"]["learning_rate"] = 0.001
    setup.base_path.write_text(yaml.safe_dump(base))
    with pytest.raises(ValueError, match="selection file"):
        sweep.run_stage2(setup.config, path, smoke=True)
    assert len(setup.calls) == 5


def test_resume_skips_completed_runs_and_refuses_implicit_overwrite(setup):
    path = sweep.run_stage1(setup.config, smoke=True)
    with pytest.raises(FileExistsError):
        sweep.run_stage1(setup.config, smoke=True)
    assert sweep.run_stage1(setup.config, smoke=True, resume=True) == path
    assert len(setup.calls) == 5
    assert len(setup.loads) == 5
    sweep.run_stage2(setup.config, path, smoke=True)
    sweep.run_stage2(setup.config, path, smoke=True, resume=True)
    assert len(setup.calls) == 25
    assert len(setup.loads) == 25


@pytest.mark.parametrize("change", ["failed", "wrong_config", "missing_model"])
def test_resume_rejects_incomplete_or_incompatible_run(setup, change):
    sweep.run_stage1(setup.config, smoke=True)
    run_id = setup.calls[0][0]
    path = setup.settings.artifacts_dir / "training" / run_id / "manifest.json"
    manifest = json.loads(path.read_text())
    if change == "failed":
        manifest["status"] = "failed"
    elif change == "wrong_config":
        manifest["config"]["seed"] = 99
    else:
        (setup.settings.models_dir / run_id / "final_model.zip").unlink()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="cannot resume"):
        sweep.run_stage1(setup.config, smoke=True, resume=True)
    assert len(setup.calls) == 5


@pytest.mark.leakage
def test_future_poisoning_keeps_selection_and_training_fingerprint(setup):
    first = json.loads(sweep.run_stage1(setup.config, smoke=True).read_text())
    for name in ("returns", "rsi", "macd"):
        path = setup.dataset_dir / f"{name}.parquet"
        frame = pd.read_parquet(path)
        frame.loc[frame.index > setup.dates[7]] = 1e9
        frame.to_parquet(path)
    # 같은 실행을 명시적으로 재개해 미래 파일 내용만 바뀐 경우도 확인한다.
    second = json.loads(sweep.run_stage1(setup.config, smoke=True, resume=True).read_text())
    assert first["selected_lambdas"] == second["selected_lambdas"]
    assert first["training_data_sha256"] == second["training_data_sha256"]
    assert first["runs"] == second["runs"]


@pytest.mark.leakage
def test_selection_refuses_even_one_future_row(setup):
    base = training.load_training_config(setup.base_path, smoke=True)
    data = training.load_training_dataset(base.data)
    frames = []
    for frame in (data.returns, data.rsi, data.macd):
        extra = frame.iloc[[-1]].copy()
        extra.index = pd.DatetimeIndex([setup.dates[8]])
        frames.append(pd.concat([frame, extra]))
    contaminated = training.TrainingDataset(*frames, (), {})
    with pytest.raises(ValueError, match="cutoff"):
        sweep.evaluate_selection(None, contaminated, base, setup.config.selection)


@pytest.mark.parametrize("higher,expected", [(True, [5.0, 1.0]), (False, [0.5, 1.0])])
def test_metric_direction_and_invalid_score_handling(higher, expected):
    records = [
        sweep.SelectionRun(
            lambda_value=penalty,
            run_id=str(penalty),
            score=score,
            metrics={},
            evaluation={},
        )
        for penalty, score in [(0.5, -1), (1.0, 0), (2.0, 0), (3.0, None), (5.0, 1)]
    ]
    assert sweep._rank(records, sweep.SelectionRule(higher_is_better=higher)) == expected
    with pytest.raises(ValueError, match="two finite"):
        sweep._rank(records[-2:-1], sweep.SelectionRule())


@pytest.mark.parametrize(
    "section,field,value",
    [
        (None, "unknown", 1),
        (None, "sweep_id", "../escape"),
        ("stage1", "lambdas", [1.0, 2.0]),
        ("stage1", "variant", "return"),
        ("stage1", "seeds", [1, 2]),
        ("stage2", "seeds", [1, 2]),
        ("stage2", "seeds", [1, 1, 2, 3, 4]),
        ("stage2", "n_candidates", 3),
        ("selection", "metric", "unknown"),
        ("selection", "metric", "alpha"),
    ],
)
def test_invalid_sweep_configuration(setup, section, field, value):
    raw = setup.config.model_dump()
    (raw if section is None else raw[section])[field] = value
    with pytest.raises(ValidationError):
        sweep.SweepConfig.model_validate(raw)


def test_cli_requires_selection_and_forwards_smoke_resume(setup, monkeypatch):
    captured = {}

    def stage1(config, **kwargs):
        captured.update(kwargs)
        return Path("selection.json")

    monkeypatch.setattr(sweep, "run_stage1", stage1)
    sweep.main(["stage1", "--config", str(setup.config_path), "--smoke", "--resume"])
    assert captured == {"smoke": True, "resume": True}
    with pytest.raises(SystemExit):
        sweep.main(["stage2", "--config", str(setup.config_path)])


def test_undefined_short_selection_fails_without_filling(setup, monkeypatch):
    def short_rollout(model, dataset, config):
        index = dataset.returns.index[2:3]
        returns = pd.Series([-0.2], index=index)
        nav = pd.Series([1, 0.8], index=dataset.returns.index[1:3])
        return EvaluationResult(returns, nav, True, False, "mdd_limit_exceeded", 1)

    monkeypatch.setattr(sweep, "evaluate_policy", short_rollout)
    with pytest.raises(ValueError, match="two finite"):
        sweep.run_stage1(setup.config, smoke=True)
    directory = setup.settings.artifacts_dir / "experiments" / setup.config.sweep_id
    assert not (directory / "stage1_selection.json").exists()
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["stages"]["1"]["status"] == "failed"
    assert len(setup.calls) == 5


def test_explicit_benchmark_metric_and_future_benchmark_rejection(setup):
    path = setup.config_path.parent / "benchmark.parquet"
    benchmark = pd.DataFrame({"benchmark": [0.004, -0.002] * 4}, index=setup.dates[:8])
    benchmark.to_parquet(path)
    setup.config.selection = sweep.SelectionRule(metric="alpha", benchmark_path=path)
    selection = json.loads(sweep.run_stage1(setup.config, smoke=True).read_text())
    assert all(record["score"] is not None for record in selection["runs"])
    extended = benchmark.reindex(setup.dates[:9], fill_value=0)
    extended.to_parquet(path)
    with pytest.raises(ValueError, match="training dates"):
        sweep.run_stage1(setup.config, smoke=True, resume=True)
    assert len(setup.calls) == 5


@pytest.mark.leakage
def test_changed_training_slice_blocks_stage2(setup):
    selection = sweep.run_stage1(setup.config, smoke=True)
    path = setup.dataset_dir / "returns.parquet"
    frame = pd.read_parquet(path)
    frame.iloc[0, 0] = 0.3
    frame.to_parquet(path)
    with pytest.raises(ValueError, match="selection file"):
        sweep.run_stage2(setup.config, selection, smoke=True)
    assert len(setup.calls) == 5


def test_stage2_cli_forwards_selection_path(setup, monkeypatch):
    captured = {}

    def stage2(config, selection, **kwargs):
        captured.update(selection=selection, **kwargs)
        return Path("manifest.json")

    monkeypatch.setattr(sweep, "run_stage2", stage2)
    sweep.main(
        [
            "stage2",
            "--config",
            str(setup.config_path),
            "--selection",
            "selection.json",
            "--smoke",
        ]
    )
    assert captured == {"selection": "selection.json", "smoke": True, "resume": False}
