"""학습곡선 산출물과 사전 정의 수렴 판정 검증."""

import csv
import json

import numpy as np
import pandas as pd
import pytest

from robo_advisor.config import Settings
from robo_advisor.rl import curves, training


def write_monitor(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="\n") as stream:
        stream.write('#{"t_start": 0}\n')
        writer = csv.DictWriter(stream, fieldnames=("r", "l", "t", "reason"))
        writer.writeheader()
        writer.writerows(rows)


def monitor_run(root, run_id="run", status="completed", episodes=60):
    run_dir = root / run_id
    for env in range(2):
        rows = [
            {
                "r": 1.0 + 0.1 * (-1) ** i,
                "l": 10,
                "t": round(0.01 * (2 * i + env + 1), 6),
                "reason": "mdd_limit_exceeded" if i % 4 == 0 else "data_exhausted",
            }
            for i in range(episodes // 2)
        ]
        write_monitor(run_dir / "monitor" / f"env-{env:03d}.monitor.csv", rows)
    (run_dir / "manifest.json").write_text(json.dumps({"run_id": run_id, "status": status}))
    return run_dir


def test_episodes_are_merged_in_elapsed_order_with_cumulative_steps(tmp_path):
    run_dir = monitor_run(tmp_path, episodes=8)
    episodes = curves.load_episodes(run_dir)
    assert list(episodes["env"]) == ["env-000", "env-001"] * 4
    assert list(episodes["episode"]) == list(range(1, 9))
    assert list(episodes["timesteps"]) == list(range(10, 90, 10))
    assert list(episodes["safeguard"]) == [1, 1, 0, 0, 0, 0, 0, 0]


def test_missing_monitor_files_are_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        curves.load_episodes(tmp_path)


def test_smoothing_keeps_raw_values_and_uses_available_prefix():
    frame = pd.DataFrame({"reward": [1.0, 3.0, 5.0], "length": [2, 4, 6], "safeguard": [1, 0, 0]})
    smoothed = curves.smooth_curves(frame, window=2)
    assert list(smoothed["reward"]) == [1.0, 3.0, 5.0]
    assert list(smoothed["reward_smoothed"]) == [1.0, 2.0, 4.0]
    assert list(smoothed["safeguard_smoothed"]) == [1.0, 0.5, 0.0]


def test_flat_noisy_rewards_converge():
    rewards = np.random.default_rng(1).normal(0, 1, 200)
    result = curves.judge_convergence(rewards)
    assert result["status"] == "converged"
    assert result["tail_episodes"] == 40


def test_rising_tail_is_not_converged():
    rewards = np.concatenate([np.zeros(160), np.linspace(0, 10, 40)])
    rewards += np.random.default_rng(2).normal(0, 0.1, 200)
    assert curves.judge_convergence(rewards)["status"] == "not_converged"


@pytest.mark.parametrize("count", [0, 49, 99])
def test_too_few_episodes_are_not_judged(count):
    result = curves.judge_convergence(np.zeros(count))
    assert result["status"] == "insufficient_episodes"
    assert result["tail_mean"] is None


def test_run_outputs_and_convergence_table(tmp_path):
    monitor_run(tmp_path, "done", episodes=120)
    monitor_run(tmp_path, "running", status="running")
    summary = curves.write_run_curves(tmp_path, "done")
    assert summary["episodes"] == 120 and summary["status"] == "converged"
    assert summary["safeguard_rate"] == pytest.approx(0.25)
    assert summary["criteria"]["tolerance"] == curves.CRITERIA.tolerance
    saved = pd.read_csv(tmp_path / "done" / "curves.csv")
    for column in ("reward", "length", "safeguard"):
        assert {column, f"{column}_smoothed"} <= set(saved.columns)
    assert (tmp_path / "curves" / "done.png").stat().st_size > 0
    table = pd.read_csv(curves.write_convergence_table(tmp_path))
    assert list(table["run_id"]) == ["done"]
    assert list(table.columns) == list(curves.CONVERGENCE_COLUMNS)
    assert table.loc[0, "status"] == "converged"


def test_table_regenerates_missing_judgements_and_cli(tmp_path):
    monitor_run(tmp_path, "first")
    monitor_run(tmp_path, "second")
    curves.main(["--training-dir", str(tmp_path)])
    table = pd.read_csv(tmp_path / "convergence.csv")
    assert list(table["run_id"]) == ["first", "second"]
    assert (tmp_path / "curves" / "second.png").exists()


def test_training_run_records_monitor_curves_and_safeguard(tmp_path, monkeypatch):
    index = pd.date_range("2026-01-01", periods=16, tz="UTC")
    returns = pd.DataFrame(
        np.tile([[0.05, 0.05], [-0.08, -0.08]], (8, 1)), index=index, columns=["A", "B"]
    )
    directory = tmp_path / "data"
    directory.mkdir()
    for name, frame in (("returns", returns), ("rsi", returns * 0 + 50), ("macd", returns * 0)):
        frame.to_parquet(directory / f"{name}.parquet")
    settings = Settings(
        _env_file=None, models_dir=tmp_path / "models", artifacts_dir=tmp_path / "artifacts"
    )
    monkeypatch.setattr(training, "get_settings", lambda: settings)
    cfg = training.TrainingConfig.model_validate(
        {
            "device": "cpu",
            "total_timesteps": 64,
            "n_envs": 2,
            "vec_env": "dummy",
            "ppo": {
                "n_steps": 16,
                "batch_size": 16,
                "n_epochs": 1,
                "net_arch": {"pi": [8], "vf": [8]},
            },
            "env": {"lookback": 2, "mdd_limit": 0.02},
            "data": {"dataset_dir": str(directory), "train_end": "2026-01-16T00:00:00Z"},
            "checkpoint": {"save_freq": 64},
        },
        context={"smoke": True},
    )
    result = training.run_training(cfg, run_id="curves", smoke=True)
    manifest = json.loads(result.manifest_path.read_text())
    run_dir = result.manifest_path.parent
    assert sorted(path.name for path in (run_dir / "monitor").iterdir()) == [
        "env-000.monitor.csv",
        "env-001.monitor.csv",
    ]
    assert manifest["learning_curves"]["status"] == "insufficient_episodes"
    assert manifest["learning_curves"]["episodes"] > 0
    episodes = pd.read_csv(manifest["learning_curves"]["curves_csv"])
    assert (episodes["reason"] == "mdd_limit_exceeded").any()
    assert episodes["safeguard"].sum() == (episodes["reason"] == "mdd_limit_exceeded").sum()
    assert (run_dir.parent / "curves" / "curves.png").exists()
    table = pd.read_csv(curves.write_convergence_table(run_dir.parent))
    assert list(table["run_id"]) == ["curves"]


def test_curve_failure_keeps_completed_training(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("plot backend unavailable")

    monkeypatch.setattr(training, "write_run_curves", broken)
    index = pd.date_range("2026-01-01", periods=8, tz="UTC")
    returns = pd.DataFrame(0.001, index=index, columns=["A"])
    directory = tmp_path / "data"
    directory.mkdir()
    for name, frame in (("returns", returns), ("rsi", returns * 0 + 50), ("macd", returns * 0)):
        frame.to_parquet(directory / f"{name}.parquet")
    settings = Settings(
        _env_file=None, models_dir=tmp_path / "models", artifacts_dir=tmp_path / "artifacts"
    )
    monkeypatch.setattr(training, "get_settings", lambda: settings)
    cfg = training.TrainingConfig.model_validate(
        {
            "device": "cpu",
            "total_timesteps": 16,
            "n_envs": 1,
            "vec_env": "dummy",
            "ppo": {
                "n_steps": 16,
                "batch_size": 16,
                "n_epochs": 1,
                "net_arch": {"pi": [4], "vf": [4]},
            },
            "env": {"lookback": 2},
            "data": {"dataset_dir": str(directory), "train_end": "2026-01-08T00:00:00Z"},
            "checkpoint": {"save_freq": 16},
        },
        context={"smoke": True},
    )
    result = training.run_training(cfg, run_id="broken-curves", smoke=True)
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["status"] == "completed"
    assert manifest["learning_curves"] == {"error": "RuntimeError: plot backend unavailable"}
    assert result.final_model_path.exists()
