"""학습 구간 내 검증 분리와 최적 모델 선택 검증."""

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
import torch
from pydantic import ValidationError
from stable_baselines3 import PPO

from robo_advisor.config import Settings
from robo_advisor.rl import training, validation
from robo_advisor.rl.evaluation import EvaluationResult

PERIODS = 40


def write_dataset(directory):
    directory.mkdir(parents=True, exist_ok=True)
    index = pd.date_range("2026-01-01", periods=PERIODS, tz="UTC")
    values = np.random.default_rng(3).normal(0, 0.002, (PERIODS, 2))
    returns = pd.DataFrame(values, index=index, columns=["A", "B"])
    for name, frame in (("returns", returns), ("rsi", returns * 0 + 50), ("macd", returns * 0)):
        frame.to_parquet(directory / f"{name}.parquet")
    return index


def payload(directory, **validation_changes):
    raw = {
        "seed": 7,
        "device": "cpu",
        "total_timesteps": 64,
        "n_envs": 2,
        "vec_env": "dummy",
        "ppo": {"n_steps": 16, "batch_size": 16, "n_epochs": 1, "net_arch": {"pi": [8], "vf": [8]}},
        "env": {"lookback": 3, "mdd_limit": 1},
        "data": {"dataset_dir": str(directory), "train_end": "2026-02-09T00:00:00Z"},
        "checkpoint": {"save_freq": 64},
        "validation": {"start": "2026-01-31T00:00:00Z", "eval_freq": 32},
    }
    raw["validation"].update(validation_changes)
    return raw


def config(directory, **validation_changes):
    return training.TrainingConfig.model_validate(
        payload(directory, **validation_changes), context={"smoke": True}
    )


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    directory = tmp_path / "data"
    write_dataset(directory)
    settings = Settings(
        _env_file=None, models_dir=tmp_path / "models", artifacts_dir=tmp_path / "artifacts"
    )
    monkeypatch.setattr(training, "get_settings", lambda: settings)
    return directory


@pytest.mark.parametrize(
    "changes",
    [
        {"start": "2026-02-10T00:00:00Z"},
        {"start": "2026-01-31T00:00:00"},
        {"eval_freq": 0},
        {"metric": "alpha"},
        {"unknown": 1},
    ],
)
def test_invalid_validation_settings(tmp_path, changes):
    with pytest.raises(ValidationError):
        config(tmp_path, **changes)


def test_validation_start_must_follow_train_start(tmp_path):
    raw = payload(tmp_path)
    raw["data"]["train_start"] = "2026-01-31T00:00:00Z"
    with pytest.raises(ValidationError, match="validation start"):
        training.TrainingConfig.model_validate(raw, context={"smoke": True})


@pytest.mark.leakage
def test_split_keeps_validation_rows_out_of_training(dataset):
    cfg = config(dataset)
    full = training.load_training_dataset(cfg.data)
    train, held_out = validation.split_validation(full, cfg.validation.start, cfg.env.lookback)
    start = pd.Timestamp(cfg.validation.start)
    assert train.returns.index.max() < start
    assert held_out.returns.index[cfg.env.lookback - 1] >= start
    assert held_out.returns.index[cfg.env.lookback - 2] < start
    assert held_out.returns.index[-1] == full.returns.index[-1]
    assert len(train.returns) + len(held_out.returns) - (cfg.env.lookback - 1) == len(full.returns)


@pytest.mark.parametrize("start", ["2026-01-03T00:00:00Z", "2026-02-09T00:00:00Z"])
def test_split_rejects_too_short_periods(dataset, start):
    cfg = config(dataset)
    full = training.load_training_dataset(cfg.data)
    with pytest.raises(ValueError):
        validation.split_validation(
            full, datetime.fromisoformat(start.replace("Z", "+00:00")), cfg.env.lookback
        )


def test_run_records_history_and_saves_best_model(dataset):
    result = training.run_training(config(dataset), run_id="validated", smoke=True)
    manifest = json.loads(result.manifest_path.read_text())
    record = manifest["validation"]
    assert record["metric"] == "sharpe" and record["higher_is_better"] is True
    assert record["eval_interval_timesteps"] == 32
    assert [item["timesteps"] for item in record["history"]] == [32, 64]
    assert record["train_last_index"] < record["first_index"]
    scores = [item["score"] for item in record["history"] if item["score"] is not None]
    assert record["best"]["score"] == max(scores)
    best = PPO.load(record["best"]["path"], device="cpu")
    assert best.policy is not None
    assert manifest["status"] == "completed"


@pytest.mark.leakage
def test_poisoned_validation_rows_do_not_change_training(dataset):
    clean = training.run_training(config(dataset), run_id="clean", smoke=True)
    start = pd.Timestamp("2026-01-31T00:00:00Z")
    for name in ("returns", "macd"):
        path = dataset / f"{name}.parquet"
        frame = pd.read_parquet(path)
        frame.loc[frame.index >= start] = 0.5
        frame.to_parquet(path)
    poisoned = training.run_training(config(dataset), run_id="poisoned", smoke=True)
    for name, value in clean.model.policy.state_dict().items():
        assert torch.equal(value, poisoned.model.policy.state_dict()[name])
    left = json.loads(clean.manifest_path.read_text())["validation"]["history"]
    right = json.loads(poisoned.manifest_path.read_text())["validation"]["history"]
    assert [item["metrics"] for item in left] != [item["metrics"] for item in right]


def fake_rollout(returns):
    index = pd.date_range("2026-02-01", periods=len(returns), tz="UTC")
    series = pd.Series(returns, index=index)
    nav = (1 + series).cumprod()
    return EvaluationResult(series, nav, False, True, "data_exhausted", len(series))


def test_lower_is_better_and_ties_keep_first(dataset, monkeypatch):
    outcomes = iter(
        [[-0.05, 0.01, 0.0], [-0.01, 0.01, 0.0], [-0.01, 0.01, 0.0], [-0.03, 0.01, 0.0]]
    )
    monkeypatch.setattr(validation, "evaluate_policy", lambda *args: fake_rollout(next(outcomes)))
    cfg = config(dataset, metric="mdd", higher_is_better=False, eval_freq=16)
    result = training.run_training(cfg, run_id="lower", smoke=True)
    record = json.loads(result.manifest_path.read_text())["validation"]
    assert [item["timesteps"] for item in record["history"]] == [16, 32, 48, 64]
    mdds = [item["score"] for item in record["history"]]
    assert mdds[1] == pytest.approx(mdds[2]) < mdds[3] < mdds[0]
    assert record["best"]["timesteps"] == 32


def test_undefined_scores_never_become_best(dataset, monkeypatch):
    monkeypatch.setattr(validation, "evaluate_policy", lambda *args: fake_rollout([0.01]))
    result = training.run_training(config(dataset), run_id="undefined", smoke=True)
    record = json.loads(result.manifest_path.read_text())["validation"]
    assert all(item["score"] is None for item in record["history"])
    assert record["best"] is None
    assert not (result.final_model_path.parent / "best_model.zip").exists()


def test_without_validation_manifest_is_unchanged(dataset):
    raw = payload(dataset)
    raw.pop("validation")
    cfg = training.TrainingConfig.model_validate(raw, context={"smoke": True})
    result = training.run_training(cfg, run_id="plain", smoke=True)
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["validation"] is None
    assert manifest["data_cutoff"]["rows"] == PERIODS
    assert datetime.fromisoformat(manifest["finished_at"]) <= datetime.now(timezone.utc)


def test_short_validation_works_with_random_start_training(dataset):
    raw = payload(dataset)
    raw["env"].update(random_start=True, min_episode_steps=12, max_episode_steps=20)
    cfg = training.TrainingConfig.model_validate(raw, context={"smoke": True})
    result = training.run_training(cfg, run_id="random-start", smoke=True)
    record = json.loads(result.manifest_path.read_text())["validation"]
    assert record["rows"] < 12
    assert [item["observation_count"] for item in record["history"]] == [record["rows"] - 1] * 2
