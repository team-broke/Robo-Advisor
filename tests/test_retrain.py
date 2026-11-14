"""Window별 완전 재학습의 경계·독립성·재개 계약 검증."""

import json
import os

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError
from stable_baselines3 import PPO

from robo_advisor.backtest import retrain
from robo_advisor.config import Settings
from robo_advisor.rl import training


@pytest.fixture
def setup(tmp_path, monkeypatch):
    directory = tmp_path / "data"
    directory.mkdir()
    dates = pd.bdate_range("2020-01-01", "2023-12-31", tz="UTC") + pd.Timedelta(hours=7)
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(
        rng.normal(0.0003, 0.01, (len(dates), 2)), index=dates, columns=["A", "B"]
    )
    returns.to_parquet(directory / "returns.parquet")
    (returns * 0 + 50).to_parquet(directory / "rsi.parquet")
    (returns * 0).to_parquet(directory / "macd.parquet")
    settings = Settings(
        _env_file=None, models_dir=tmp_path / "models", artifacts_dir=tmp_path / "artifacts"
    )
    for module in (retrain, training):
        monkeypatch.setattr(module, "get_settings", lambda: settings)
    (tmp_path / "training.yaml").write_text(
        yaml.safe_dump(
            {
                "device": "cpu",
                "vec_env": "dummy",
                "n_envs": 2,
                "total_timesteps": 64,
                "ppo": {"n_steps": 16, "batch_size": 16, "n_epochs": 1},
                "env": {"lookback": 2},
                "data": {"train_end": "2020-12-31T00:00:00Z"},
            }
        )
    )
    payload = {
        "base_training_config": "training.yaml",
        "experiment_id": "wf-test",
        "datasets": {"toy": "data"},
        "split": {
            "start_date": "2020-01-01T00:00:00+09:00",
            "train_years": 2,
            "test_years": 1,
            "coverage_start": "2020-01-01T00:00:00+09:00",
            "coverage_end": "2024-01-01T00:00:00+09:00",
        },
        "validation_years": 1,
        "variants": [{"name": "return", "reward_variant": "return"}],
        "seeds": [42],
    }
    path = tmp_path / "walk_forward.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path, payload, settings


def test_jobs_follow_windows_and_never_train_on_test(setup):
    path, _, _ = setup
    config = retrain.load_retrain_config(path)
    base = training.load_training_config(config.base_training_config, smoke=True)
    jobs = retrain.build_jobs(config)
    assert [job.run_id for job in jobs] == [
        "wf-test-toy-w1-return-s42",
        "wf-test-toy-w2-return-s42",
    ]
    job = jobs[0]
    assert job.test_start == pd.Timestamp("2022-01-01T00:00:00+09:00")
    assert job.validation_start == pd.Timestamp("2021-01-01T00:00:00+09:00")
    config_for_job = retrain.job_training_config(base, config, job, smoke=True)
    assert config_for_job.data.train_end < job.test_start
    assert config_for_job.validation.start == job.validation_start
    dataset = training.load_training_dataset(config_for_job.data)
    assert dataset.returns.index[-1] < job.test_start
    test = retrain.load_test_dataset(config_for_job, job)
    lookback = config_for_job.env.lookback
    assert (test.returns.index[lookback - 1 :] >= job.test_start).all()
    assert (test.returns.index[: lookback - 1] < job.test_start).all()
    assert test.returns.index[-1] < job.test_end


def test_seed_is_outer_loop(setup):
    path, payload, _ = setup
    payload["seeds"] = [1, 2]
    payload["variants"].append(
        {"name": "mdd", "reward_variant": "mdd_penalty", "penalty_lambda": 2}
    )
    path.write_text(yaml.safe_dump(payload))
    jobs = retrain.build_jobs(retrain.load_retrain_config(path))
    assert [job.seed for job in jobs] == [1, 1, 1, 1, 2, 2, 2, 2]
    assert jobs[1].variant.penalty_lambda == 2


def test_rejects_validation_as_long_as_training(setup):
    path, payload, _ = setup
    payload["validation_years"] = 2
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValidationError):
        retrain.load_retrain_config(path)


def test_runs_fresh_models_and_resumes(setup, monkeypatch):
    path, payload, settings = setup
    payload["split"]["train_years"] = 1
    payload["validation_years"] = 0
    path.write_text(yaml.safe_dump(payload))
    config = retrain.load_retrain_config(path)
    loaded = []
    original = PPO.load.__func__

    def spy(cls, model_path, *args, **kwargs):
        loaded.append(str(model_path))
        return original(cls, model_path, *args, **kwargs)

    monkeypatch.setattr(PPO, "load", classmethod(spy))
    manifest_path = retrain.run_experiment(config, smoke=True)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "completed"
    assert [job["status"] for job in manifest["jobs"]] == ["completed"] * 3
    run_ids = [job["run_id"] for job in manifest["jobs"]]
    # 각 작업은 자기 run_id의 최종 모델만 평가용으로 읽는다.
    assert len(loaded) == 3
    for run_id, model_path in zip(run_ids, loaded):
        assert f"/{run_id}/" in model_path
    for run_id in run_ids:
        result = json.loads((manifest_path.parent / "jobs" / run_id / "result.json").read_text())
        assert result["initial_weights"] == "random"
        assert result["evaluated_model"] == "final"
        frame = pd.read_csv(
            manifest_path.parent / "jobs" / run_id / "test_returns.csv",
            index_col=0,
            parse_dates=True,
        )
        assert (frame.index >= pd.Timestamp(result["test_start"])).all()
        assert (frame.index < pd.Timestamp(result["test_end"])).all()
        assert result["ppo"]["metrics"]["sharpe"] is not None
    summary = pd.read_csv(manifest_path.parent / "summary.csv")
    assert list(summary["window"]) == [1, 2, 3]

    with pytest.raises(FileExistsError):
        retrain.run_experiment(config, smoke=True)
    calls = []
    monkeypatch.setattr(retrain, "run_job", lambda *args, **kwargs: calls.append(args))
    retrain.run_experiment(config, smoke=True, resume=True)
    manifest = json.loads(manifest_path.read_text())
    assert calls == []
    assert {job["status"] for job in manifest["jobs"]} == {"skipped_completed"}


def test_failure_is_recorded_and_next_job_runs(setup, monkeypatch):
    path, payload, _ = setup
    payload["split"]["train_years"] = 1
    payload["validation_years"] = 0
    path.write_text(yaml.safe_dump(payload))
    config = retrain.load_retrain_config(path)
    calls = []

    def fake(job, *args, **kwargs):
        calls.append(job.run_id)
        if job.window == 1:
            raise RuntimeError("boom")
        return {
            "ppo": {"metrics": {"sharpe": 1.0}},
            "equal_weight": {"metrics": {"sharpe": 0.5}},
        }

    monkeypatch.setattr(retrain, "run_job", fake)
    manifest = json.loads(retrain.run_experiment(config, smoke=True).read_text())
    assert len(calls) == 3
    assert manifest["status"] == "completed_with_failures"
    assert manifest["jobs"][0]["error"] == "RuntimeError: boom"
    assert [job["status"] for job in manifest["jobs"][1:]] == ["completed"] * 2


def test_stale_attempt_is_archived_before_retraining(setup, monkeypatch):
    path, payload, settings = setup
    payload["split"]["train_years"] = 1
    payload["validation_years"] = 0
    path.write_text(yaml.safe_dump(payload))
    config = retrain.load_retrain_config(path)
    stale = settings.models_dir / "wf-test-toy-w1-return-s42"
    stale.mkdir(parents=True)
    (stale / "final_model.zip").write_text("old")
    manifest = json.loads(retrain.run_experiment(config, smoke=True).read_text())
    assert manifest["jobs"][0]["status"] == "completed"
    archived = [entry for entry in settings.models_dir.iterdir() if ".stale-" in entry.name]
    assert len(archived) == 1


def test_lock_refuses_live_owner(tmp_path):
    lock = tmp_path / "run.lock"
    lock.write_text(str(os.getpid()))
    with pytest.raises(RuntimeError):
        with retrain.RunLock(lock):
            pass
    lock.write_text("999999999")
    with retrain.RunLock(lock):
        assert lock.read_text() == str(os.getpid())
    assert not lock.exists()


def test_equal_weight_policy_puts_no_weight_in_cash():
    logits, _ = retrain.equal_weight_policy(3).predict(None)
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    assert weights[-1] < 1e-8
    assert np.allclose(weights[:-1], 1 / 3, atol=1e-8)
