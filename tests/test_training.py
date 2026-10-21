"""학습 설정·컷오프·재현성 및 체크포인트 저장 계약 검증."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from pydantic import ValidationError
from stable_baselines3 import PPO

from robo_advisor.config import Settings
from robo_advisor.rl import train, training
from robo_advisor.schemas.risk import RiskTag


def write_dataset(directory):
    directory.mkdir(parents=True, exist_ok=True)
    index = pd.date_range("2026-01-01", periods=16, tz="UTC")
    values = np.random.default_rng(7).normal(0, 0.001, (16, 2))
    returns = pd.DataFrame(values, index=index, columns=["A", "B"])
    for name, frame in (("returns", returns), ("rsi", returns * 0 + 50), ("macd", returns * 0)):
        frame.to_parquet(directory / f"{name}.parquet")
    tag = RiskTag(
        event_type="sharp_move",
        severity=0.8,
        confidence=0.5,
        asset="A",
        published_at=index[1].to_pydatetime(),
        available_at=index[2].to_pydatetime(),
        source_url="https://example.com/event",
        trace_id="training-test",
    ).model_dump(mode="json")
    pd.DataFrame([tag]).to_parquet(directory / "risk_tags.parquet")
    return index


def payload(directory):
    return {
        "seed": 42,
        "device": "cpu",
        "total_timesteps": 128,
        "n_envs": 2,
        "vec_env": "dummy",
        "ppo": {
            "n_steps": 16,
            "batch_size": 16,
            "n_epochs": 1,
            "net_arch": {"pi": [16], "vf": [16]},
        },
        "env": {"lookback": 2, "mdd_limit": 1},
        "data": {
            "dataset_dir": str(directory),
            "train_start": "2026-01-02T00:00:00Z",
            "train_end": "2026-01-12T00:00:00Z",
        },
        "checkpoint": {"save_freq": 64},
    }


def config(directory):
    return training.TrainingConfig.model_validate(payload(directory), context={"smoke": True})


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    directory = tmp_path / "data"
    write_dataset(directory)
    settings = Settings(
        _env_file=None,
        models_dir=tmp_path / "models",
        artifacts_dir=tmp_path / "artifacts",
        data_processed_dir=directory,
    )
    monkeypatch.setattr(training, "get_settings", lambda: settings)
    return directory


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("training")
    directory = root / "data"
    write_dataset(directory)
    settings = Settings(
        _env_file=None,
        models_dir=root / "models",
        artifacts_dir=root / "artifacts",
    )
    cfg = config(directory)
    data = training.load_training_dataset(cfg.data)
    env = training._make_env(data, cfg.env, cfg.seed)
    observation, _ = env.reset(seed=42)
    env.close()
    snapshots = {}
    callback_class = training.CheckpointCallback

    class ObservedCheckpoint(callback_class):
        def _on_step(self):
            result = super()._on_step()
            if self.n_calls % self.save_freq == 0:
                prediction = self.model.predict(observation, deterministic=True)[0]
                snapshots[self.model.num_timesteps] = prediction.copy()
            return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(training, "get_settings", lambda: settings)
        patch.setattr(training, "CheckpointCallback", ObservedCheckpoint)
        first = training.run_training(cfg, run_id="first", smoke=True)
        original_snapshots = {key: value.copy() for key, value in snapshots.items()}
        second = training.run_training(cfg, run_id="second", smoke=True)
        yield first, second, observation, cfg, original_snapshots


def test_same_seed_reproduces_policy_and_prediction(trained):
    first, second, obs, _, _ = trained
    for name, value in first.model.policy.state_dict().items():
        assert torch.equal(value, second.model.policy.state_dict()[name])
    np.testing.assert_array_equal(
        first.model.predict(obs, deterministic=True)[0],
        second.model.predict(obs, deterministic=True)[0],
    )


def test_final_and_checkpoint_roundtrip(trained):
    first, _, obs, _, snapshots = trained
    restored = PPO.load(first.final_model_path, device="cpu")
    np.testing.assert_array_equal(
        first.model.predict(obs, deterministic=True)[0],
        restored.predict(obs, deterministic=True)[0],
    )
    assert len(first.checkpoint_paths) == 2
    for path in first.checkpoint_paths:
        saved_step = int(path.stem.split("_")[-2])
        restored_checkpoint = PPO.load(path, device="cpu")
        np.testing.assert_array_equal(
            snapshots[saved_step], restored_checkpoint.predict(obs, deterministic=True)[0]
        )


def test_manifest_records_resolved_run(trained):
    first, second, _, cfg, _ = trained
    manifest = json.loads(first.manifest_path.read_text())
    assert manifest["status"] == "completed"
    assert manifest["run_id"] == "first"
    assert manifest["seed"] == 42
    assert manifest["env_seeds"] == [42, 43]
    assert manifest["smoke"] is True
    assert manifest["device"] == "cpu" and manifest["gpu_name"] is None
    assert manifest["requested_device"] == "cpu"
    assert manifest["n_envs"] == 2
    assert manifest["actual_timesteps"] == 128
    assert manifest["checkpoint_interval_timesteps"] == 64
    assert manifest["data_cutoff"]["rows"] == 11
    assert manifest["data_cutoff"]["assets"] == ["A", "B"]
    actual_last = pd.Timestamp(manifest["data_cutoff"]["actual_last_index"])
    assert actual_last == pd.Timestamp(cfg.data.train_end)
    assert manifest["config"]["data"]["dataset_dir"] == str(cfg.data.dataset_dir.resolve())
    assert manifest["checkpoint_paths"] == [str(path) for path in first.checkpoint_paths]
    assert manifest["final_model_path"] == str(first.final_model_path)
    assert set(manifest["package_versions"]) == {
        "python",
        "stable-baselines3",
        "torch",
        "gymnasium",
        "numpy",
        "pandas",
    }
    assert "git_commit" in manifest and "torch_cuda_version" in manifest
    assert pd.Timestamp(manifest["finished_at"]) >= pd.Timestamp(manifest["started_at"])
    canonical = json.dumps(
        manifest["config"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    assert manifest["config_sha256"] == hashlib.sha256(canonical.encode()).hexdigest()
    for name, digest in manifest["data_file_sha256"].items():
        assert digest == hashlib.sha256((cfg.data.dataset_dir / name).read_bytes()).hexdigest()
    second_manifest = json.loads(second.manifest_path.read_text())
    assert manifest["config_sha256"] == second_manifest["config_sha256"]


def test_duplicate_run_id_does_not_overwrite(trained):
    first, _, _, cfg, _ = trained
    original = first.manifest_path.read_bytes()
    with pytest.raises(FileExistsError):
        training.run_training(cfg, run_id="first", smoke=True)
    assert first.manifest_path.read_bytes() == original


@pytest.mark.leakage
def test_post_cutoff_poison_does_not_change_training_or_observations(dataset):
    cfg = config(dataset)
    before = training.load_training_dataset(cfg.data)
    original = training.run_training(cfg, run_id="clean", smoke=True)
    for name in ("returns", "rsi", "macd"):
        path = dataset / f"{name}.parquet"
        frame = pd.read_parquet(path)
        frame.loc[frame.index > cfg.data.train_end] = 1e9
        frame.to_parquet(path)
    tags = pd.read_parquet(dataset / "risk_tags.parquet")
    future = tags.iloc[0].to_dict()
    future.update(available_at="2026-01-15T00:00:00Z", severity=1e9)
    pd.concat([tags, pd.DataFrame([future])], ignore_index=True).to_parquet(
        dataset / "risk_tags.parquet"
    )
    after = training.load_training_dataset(cfg.data)
    assert before.risk_tags == after.risk_tags
    for name in ("returns", "rsi", "macd"):
        pd.testing.assert_frame_equal(getattr(before, name), getattr(after, name))
    left = training._make_env(before, cfg.env, 42)
    right = training._make_env(after, cfg.env, 42)
    try:
        np.testing.assert_array_equal(left.reset(seed=42)[0], right.reset(seed=42)[0])
        for _ in range(3):
            a, b = left.step(np.zeros(3)), right.step(np.zeros(3))
            np.testing.assert_array_equal(a[0], b[0])
            assert a[1:4] == b[1:4]
    finally:
        left.close()
        right.close()
    poisoned = training.run_training(cfg, run_id="poisoned", smoke=True)
    for name, value in original.model.policy.state_dict().items():
        assert torch.equal(value, poisoned.model.policy.state_dict()[name])
    assert before.file_sha256 != after.file_sha256


@pytest.mark.parametrize(
    "section,key,value",
    [
        (None, "unknown", 1),
        (None, "seed", -1),
        (None, "seed", 2**32 - 1),
        (None, "n_envs", 0),
        (None, "n_envs", True),
        (None, "device", "mps"),
        (None, "total_timesteps", 0),
        ("ppo", "batch_size", 33),
        ("ppo", "batch_size", 1),
        ("ppo", "gamma", 1.1),
        ("ppo", "learning_rate", float("nan")),
        ("ppo", "n_steps", 0),
        ("ppo", "net_arch", {"pi": [0], "vf": [16]}),
        ("env", "lookback", 0),
        ("env", "reward_variant", "unknown"),
        ("env", "commission_rate", -1),
        ("env", "mdd_limit", 0),
        ("data", "train_end", None),
        ("data", "train_end", "2026-01-01"),
        ("data", "train_start", "2026-02-01T00:00:00Z"),
        ("checkpoint", "save_freq", 0),
        ("checkpoint", "unknown", 1),
    ],
)
def test_invalid_configuration(section, key, value, tmp_path):
    raw = payload(tmp_path)
    (raw if section is None else raw[section])[key] = value
    with pytest.raises(ValidationError):
        training.TrainingConfig.model_validate(raw, context={"smoke": True})


def test_minimum_steps_requires_explicit_smoke(dataset):
    with pytest.raises(ValidationError, match="100000"):
        training.TrainingConfig.model_validate(payload(dataset))
    with pytest.raises(ValidationError, match="100000"):
        training.run_training(config(dataset), run_id="missing-smoke")
    raw = payload(dataset)
    raw["total_timesteps"] = 100000
    assert training.TrainingConfig.model_validate(raw).total_timesteps == 100000


@pytest.mark.parametrize("case", ["columns", "index", "naive", "duplicate", "unsorted", "empty"])
def test_dataset_contract_errors(dataset, case):
    path = dataset / "rsi.parquet"
    frame = pd.read_parquet(path)
    if case == "columns":
        frame = frame.iloc[:, ::-1]
    elif case == "index":
        frame.index += pd.Timedelta(hours=1)
    elif case == "naive":
        frame.index = frame.index.tz_localize(None)
    elif case == "duplicate":
        frame.index = pd.DatetimeIndex([frame.index[0]] * len(frame))
    elif case == "unsorted":
        frame = frame.iloc[::-1]
    else:
        frame = frame.iloc[:0]
    frame.to_parquet(path)
    with pytest.raises(ValueError):
        training.load_training_dataset(config(dataset).data)


def test_cuda_selection_requires_available_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert training.resolve_device("auto") == "cpu"
    assert training.resolve_device("cpu") == "cpu"
    with pytest.raises(ValueError, match="CUDA"):
        training.resolve_device("cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert training.resolve_device("auto") == "cuda"


def test_cli_overrides_are_validated_and_forwarded(dataset, tmp_path, monkeypatch):
    path = tmp_path / "training.yaml"
    raw = payload(tmp_path / "unused")
    raw["total_timesteps"] = 100000
    path.write_text(yaml.safe_dump(raw))
    captured = {}

    def capture(cfg, *, run_id, smoke):
        captured.update(config=cfg, run_id=run_id, smoke=smoke)
        return training.TrainingResult(None, Path("final.zip"), Path("manifest.json"), ())

    monkeypatch.setattr(train, "run_training", capture)
    train.main(
        [
            "--config",
            str(path),
            "--dataset-dir",
            str(dataset),
            "--device",
            "cpu",
            "--total-timesteps",
            "128",
            "--run-id",
            "cli",
            "--smoke",
        ]
    )
    assert captured["config"].data.dataset_dir == dataset
    assert captured["config"].device == "cpu"
    assert captured["config"].total_timesteps == 128
    assert captured["run_id"] == "cli" and captured["smoke"]
    with pytest.raises(ValidationError):
        training.load_training_config(path, total_timesteps=128)


def test_git_lookup_failure_is_nonfatal(monkeypatch):
    def missing(*args, **kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr(training.subprocess, "run", missing)
    assert training._git_commit() is None


def test_checkpoint_interval_rounding_and_actual_rollout_steps(dataset):
    raw = payload(dataset)
    raw["total_timesteps"] = 33
    raw["checkpoint"]["save_freq"] = 31
    cfg = training.TrainingConfig.model_validate(raw, context={"smoke": True})
    result = training.run_training(cfg, run_id="rounding", smoke=True)
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["actual_timesteps"] == 64
    assert manifest["checkpoint_interval_timesteps"] == 32


def test_run_id_cannot_escape_output_directories(dataset):
    with pytest.raises(ValueError, match="run_id"):
        training.run_training(config(dataset), run_id="../elsewhere", smoke=True)


def test_settings_defaults_and_individual_environment_seeds(dataset):
    defaults = training.get_settings()
    cfg = config(dataset)
    assert cfg.env.commission_rate == defaults.commission_rate
    assert cfg.env.slippage_rate == defaults.slippage_rate
    assert training.EnvironmentConfig().mdd_limit == defaults.safeguard_mdd_limit
    data = training.load_training_dataset(cfg.data)
    for seed in (cfg.seed, cfg.seed + 1):
        env = training._make_env(data, cfg.env, seed)
        try:
            assert env.unwrapped.np_random.random() == np.random.default_rng(seed).random()
        finally:
            env.close()


@pytest.mark.leakage
def test_inclusive_cutoff_and_pre_start_tags(dataset):
    cfg = config(dataset)
    path = dataset / "risk_tags.parquet"
    original = pd.read_parquet(path).iloc[0].to_dict()
    before = {**original, "published_at": "2026-01-01T00:00:00Z"}
    before["available_at"] = before["published_at"]
    boundary = {**original, "available_at": "2026-01-12T00:00:00Z"}
    future = {**original, "available_at": "2026-01-12T00:00:00.000001Z"}
    pd.DataFrame([before, boundary, future]).to_parquet(path)
    data = training.load_training_dataset(cfg.data)
    assert data.returns.index[0] == pd.Timestamp(cfg.data.train_start)
    assert data.returns.index[-1] == pd.Timestamp(cfg.data.train_end)
    assert len(data.risk_tags) == 2
    assert data.risk_tags[0].available_at < cfg.data.train_start
    assert data.risk_tags[1].available_at == cfg.data.train_end


def test_failed_learning_records_status_and_closes_vector(dataset, monkeypatch):
    closed = []
    original_close = training.DummyVecEnv.close

    def close(vec):
        closed.append(True)
        original_close(vec)

    def fail(*args, **kwargs):
        raise RuntimeError("learning failed")

    monkeypatch.setattr(training.DummyVecEnv, "close", close)
    monkeypatch.setattr(training.PPO, "learn", fail)
    with pytest.raises(RuntimeError, match="learning failed"):
        training.run_training(config(dataset), run_id="failure", smoke=True)
    path = training.get_settings().artifacts_dir / "training" / "failure" / "manifest.json"
    manifest = json.loads(path.read_text())
    assert manifest["status"] == "failed"
    assert manifest["finished_at"] is not None
    assert manifest["actual_timesteps"] == 0
    assert closed == [True]


def test_existing_artifact_directory_also_blocks_run(dataset):
    settings = training.get_settings()
    directory = settings.artifacts_dir / "training" / "reserved"
    directory.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        training.run_training(config(dataset), run_id="reserved", smoke=True)
    assert not (settings.models_dir / "reserved").exists()
