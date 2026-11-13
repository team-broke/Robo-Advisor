"""실험 대기열 실행기와 학습 상태 파일 검증."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from robo_advisor.ops import queue as queue_module


def write_queue(path, jobs, **extra):
    payload = {"queue_id": "nightly", "jobs": jobs, **extra}
    path.write_text(yaml.safe_dump(payload, allow_unicode=True))
    return path


class FakeRunner:
    """학습 대신 manifest를 만들어 성공·실패를 흉내 낸다."""

    def __init__(self, artifacts, fail=()):
        self.artifacts = artifacts
        self.fail = set(fail)
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append(command)
        if kwargs.get("shell"):
            return subprocess.CompletedProcess(command, 1 if "false" in command else 0)
        run_id = command[command.index("--run-id") + 1]
        status = "failed" if run_id in self.fail else "completed"
        run_dir = self.artifacts / "training" / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(json.dumps({"status": status}))
        kwargs["stdout"].write(f"training {run_id} {status}\n")
        return subprocess.CompletedProcess(command, 0 if status == "completed" else 1)


@pytest.fixture
def base_config(tmp_path):
    path = tmp_path / "base.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "seed": 1,
                "total_timesteps": 1000000,
                "ppo": {"n_steps": 512, "net_arch": {"pi": [256]}},
            }
        )
    )
    return path


def test_deep_merge_keeps_unrelated_nested_values():
    base = {"a": 1, "ppo": {"n_steps": 512, "net_arch": {"pi": [256], "vf": [256]}}}
    merged = queue_module.deep_merge(base, {"ppo": {"net_arch": {"pi": [64]}}, "b": 2})
    assert merged == {
        "a": 1,
        "b": 2,
        "ppo": {"n_steps": 512, "net_arch": {"pi": [64], "vf": [256]}},
    }
    assert base["ppo"]["net_arch"]["pi"] == [256]


@pytest.mark.parametrize(
    "payload",
    [
        {"queue_id": "q", "jobs": []},
        {"queue_id": "bad id", "jobs": [{"run_id": "a", "config": "x"}]},
        {"queue_id": "q", "jobs": [{"run_id": "a", "config": "x"}, {"run_id": "a", "config": "y"}]},
        {"queue_id": "q", "jobs": [{"run_id": "../a", "config": "x"}]},
        {"queue_id": "q", "jobs": [{"run_id": "a", "config": "x", "unknown": 1}]},
        {"queue_id": "q", "defaults": {"device": "tpu"}, "jobs": [{"run_id": "a", "config": "x"}]},
    ],
)
def test_invalid_queue_files(tmp_path, payload):
    path = tmp_path / "q.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValidationError):
        queue_module.load_queue(path)


def test_runs_in_order_with_merged_configs_and_state(tmp_path, base_config):
    artifacts = tmp_path / "artifacts"
    path = write_queue(
        tmp_path / "queue.yaml",
        [
            {"run_id": "a", "config": "base.yaml", "overrides": {"total_timesteps": 200000}},
            {
                "run_id": "b",
                "config": str(base_config),
                "overrides": {"ppo": {"net_arch": {"pi": [64]}}},
                "device": "cpu",
                "smoke": True,
                "after": ["echo {run_id} {artifact_dir}"],
            },
        ],
        defaults={"device": "cuda", "log_interval": 15},
    )
    runner = FakeRunner(artifacts)
    state = queue_module.run_queue(path, artifacts_dir=artifacts, python="py", runner=runner)
    assert state["state"] == "completed"
    assert [job["state"] for job in state["jobs"]] == ["completed", "completed"]
    first, second, hook = runner.calls
    assert first[:3] == ["py", "-m", "robo_advisor.rl.train"]
    assert first[first.index("--device") + 1] == "cuda" and "--smoke" not in first
    assert first[first.index("--log-interval") + 1] == "15.0"
    assert second[second.index("--device") + 1] == "cpu" and "--smoke" in second
    assert hook == f"echo b {artifacts / 'training' / 'b'}"
    config_a = yaml.safe_load(Path(state["jobs"][0]["config_path"]).read_text())
    assert config_a["total_timesteps"] == 200000 and config_a["ppo"]["n_steps"] == 512
    config_b = yaml.safe_load(Path(state["jobs"][1]["config_path"]).read_text())
    assert config_b["ppo"]["net_arch"] == {"pi": [64]}
    saved = json.loads((artifacts / "queue/nightly/state.json").read_text())
    assert saved["schema_version"] == 1 and saved["state"] == "completed"
    assert saved["finished_at"] and all(job["finished_at"] for job in saved["jobs"])
    log = (artifacts / "queue/nightly/logs/b.log").read_text()
    assert "training b completed" in log and "$ echo b" in log
    assert not (artifacts / "queue/nightly/lock").exists()


def test_completed_runs_are_skipped_and_existing_incomplete_runs_fail(tmp_path, base_config):
    artifacts = tmp_path / "artifacts"
    for run_id, status in (("done", "completed"), ("stuck", "running")):
        run_dir = artifacts / "training" / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(json.dumps({"status": status}))
    path = write_queue(
        tmp_path / "queue.yaml",
        [{"run_id": name, "config": "base.yaml"} for name in ("done", "stuck", "new")],
    )
    runner = FakeRunner(artifacts)
    state = queue_module.run_queue(path, artifacts_dir=artifacts, python="py", runner=runner)
    assert [job["state"] for job in state["jobs"]] == ["skipped", "failed", "completed"]
    assert "status running" in state["jobs"][1]["error"]
    assert state["state"] == "failed"
    assert len(runner.calls) == 1


def test_failure_continues_or_stops(tmp_path, base_config):
    jobs = [{"run_id": name, "config": "base.yaml"} for name in ("x", "y")]
    artifacts = tmp_path / "continue"
    state = queue_module.run_queue(
        write_queue(tmp_path / "q1.yaml", jobs),
        artifacts_dir=artifacts,
        python="py",
        runner=FakeRunner(artifacts, fail={"x"}),
    )
    assert [job["state"] for job in state["jobs"]] == ["failed", "completed"]
    assert "training x failed" in state["jobs"][0]["error"]
    artifacts = tmp_path / "stop"
    state = queue_module.run_queue(
        write_queue(tmp_path / "q2.yaml", jobs, stop_on_failure=True),
        artifacts_dir=artifacts,
        python="py",
        runner=FakeRunner(artifacts, fail={"x"}),
    )
    assert [job["state"] for job in state["jobs"]] == ["failed", "pending"]


def test_failed_after_command_is_recorded_without_failing_training(tmp_path, base_config):
    artifacts = tmp_path / "artifacts"
    path = write_queue(
        tmp_path / "queue.yaml",
        [{"run_id": "a", "config": "base.yaml", "after": ["false", "echo never"]}],
    )
    runner = FakeRunner(artifacts)
    state = queue_module.run_queue(path, artifacts_dir=artifacts, python="py", runner=runner)
    job = state["jobs"][0]
    assert job["state"] == "completed" and job["error"] == "after command failed: false"
    assert runner.calls[-1] == "false"


def test_dry_run_writes_configs_without_training(tmp_path, base_config):
    artifacts = tmp_path / "artifacts"
    path = write_queue(tmp_path / "queue.yaml", [{"run_id": "a", "config": "base.yaml"}])
    runner = FakeRunner(artifacts)
    state = queue_module.run_queue(
        path, artifacts_dir=artifacts, python="py", runner=runner, dry_run=True
    )
    assert runner.calls == []
    assert state["jobs"][0]["error"].startswith("dry run: py -m robo_advisor.rl.train")
    assert Path(state["jobs"][0]["config_path"]).exists()


def test_live_lock_blocks_and_stale_lock_is_taken_over(tmp_path, base_config):
    artifacts = tmp_path / "artifacts"
    path = write_queue(tmp_path / "queue.yaml", [{"run_id": "a", "config": "base.yaml"}])
    lock = artifacts / "queue/nightly/lock"
    lock.parent.mkdir(parents=True)
    lock.write_text(str(os.getpid()))
    with pytest.raises(RuntimeError, match="already running"):
        queue_module.run_queue(path, artifacts_dir=artifacts, runner=FakeRunner(artifacts))
    lock.write_text("999999999")
    state = queue_module.run_queue(path, artifacts_dir=artifacts, runner=FakeRunner(artifacts))
    assert state["state"] == "completed" and not lock.exists()


def test_interrupt_marks_queue_and_job(tmp_path, base_config):
    artifacts = tmp_path / "artifacts"
    path = write_queue(tmp_path / "queue.yaml", [{"run_id": "a", "config": "base.yaml"}])

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        queue_module.run_queue(path, artifacts_dir=artifacts, runner=interrupted)
    saved = json.loads((artifacts / "queue/nightly/state.json").read_text())
    assert saved["state"] == "interrupted" and saved["jobs"][0]["state"] == "failed"
    assert not (artifacts / "queue/nightly/lock").exists()


def test_status_cli(tmp_path, base_config, capsys):
    artifacts = tmp_path / "artifacts"
    path = write_queue(tmp_path / "queue.yaml", [{"run_id": "a", "config": "base.yaml"}])
    queue_module.main(["status", str(path), "--artifacts-dir", str(artifacts)])
    assert "아직 실행 기록 없음" in capsys.readouterr().out
    queue_module.run_queue(path, artifacts_dir=artifacts, runner=FakeRunner(artifacts))
    queue_module.main(["status", str(path), "--artifacts-dir", str(artifacts)])
    out = capsys.readouterr().out
    assert "nightly: completed" in out and "completed a" in out


def test_end_to_end_real_training_subprocess_writes_status(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    index = pd.date_range("2026-01-01", periods=20, tz="UTC")
    returns = pd.DataFrame(
        np.random.default_rng(1).normal(0, 0.002, (20, 2)), index=index, columns=["A", "B"]
    )
    for name, frame in (("returns", returns), ("rsi", returns * 0 + 50), ("macd", returns * 0)):
        frame.to_parquet(data / f"{name}.parquet")
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "device": "cpu",
                "total_timesteps": 32,
                "n_envs": 2,
                "vec_env": "dummy",
                "ppo": {
                    "n_steps": 16,
                    "batch_size": 16,
                    "n_epochs": 1,
                    "net_arch": {"pi": [8], "vf": [8]},
                },
                "env": {"lookback": 2, "mdd_limit": 1},
                "data": {"dataset_dir": str(data), "train_end": "2026-01-20T00:00:00Z"},
                "checkpoint": {"save_freq": 32},
            }
        )
    )
    artifacts = tmp_path / "artifacts"
    path = write_queue(
        tmp_path / "queue.yaml",
        [{"run_id": "e2e", "config": "base.yaml"}],
        defaults={"smoke": True, "log_interval": 1},
    )
    env = {
        **os.environ,
        "ARTIFACTS_DIR": str(artifacts),
        "MODELS_DIR": str(tmp_path / "models"),
    }
    state = queue_module.run_queue(path, artifacts_dir=artifacts, python=sys.executable, env=env)
    assert state["state"] == "completed", (artifacts / "queue/nightly/logs/e2e.log").read_text()
    status = json.loads((artifacts / "training/e2e/status.json").read_text())
    assert status["schema_version"] == 1 and status["run_id"] == "e2e"
    assert status["state"] == "completed" and status["progress"] == 1.0
    assert status["eta_seconds"] is None and status["validation"] is None
    assert status["episodes"]["count_total"] >= 0
