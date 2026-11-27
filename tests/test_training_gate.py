"""The PPO entrypoint rejects unreviewed inputs before creating an environment."""

from contextlib import redirect_stdout
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml
from stable_baselines3 import PPO

from robo_advisor.config import Settings
from robo_advisor.data.snapshot import read_snapshot, write_snapshot
from robo_advisor.features.dataset_io import write_feature_dataset
from robo_advisor.features.ppo_input import convert_ppo_inputs
from robo_advisor.ops import ppo_preflight, queue as queue_module
from robo_advisor.rl import train, training, training_gate
from tests import test_feature_dataset as feature_tests
from tests import test_ppo_reproducibility as handoff_tests
from tests.test_training import payload


def write_dataset(directory):
    directory.mkdir(parents=True)
    index = pd.date_range("2026-01-01", periods=16, tz="UTC")
    returns = pd.DataFrame(
        np.random.default_rng(7).normal(0, 0.001, (16, 2)),
        index=index,
        columns=["A", "B"],
    )
    for name, frame in (("returns", returns), ("rsi", returns * 0 + 50), ("macd", returns * 0)):
        frame.to_parquet(directory / f"{name}.parquet")


def _settings(root):
    return Settings(_env_file=None, models_dir=root / "models", artifacts_dir=root / "artifacts")


def _environment_reached(*args, **kwargs):
    raise RuntimeError("environment reached")


@pytest.fixture
def research(tmp_path, monkeypatch):
    directory = tmp_path / "data"
    write_dataset(directory)
    raw = payload(directory)
    raw["total_timesteps"] = 100000
    config = training.TrainingConfig.model_validate(raw)
    reviewed = config.model_copy(
        update={"data": config.data.model_copy(update={"dataset_dir": Path("data")})}
    )
    dataset = training.load_training_dataset(config.data)
    spec = tmp_path / "preflight.json"
    spec.write_text(json.dumps({"kind": "research", "dataset_id": "fixture-1"}))
    entry = {
        "dataset_id": "fixture-1",
        "file_sha256": dataset.file_sha256,
        "allowed_train_start": config.data.train_start.isoformat(),
        "allowed_train_end": config.data.train_end.isoformat(),
        "review_ref": "https://github.com/team-broke/Robo-Advisor/issues/138",
        "limitations": ["Synthetic test source; financial provenance is not established"],
    }
    registry = tmp_path / "reviewed.yaml"
    monkeypatch.setattr(training_gate, "RESEARCH_REGISTRY", registry)
    monkeypatch.setattr(
        ppo_preflight,
        "check_inputs",
        lambda *args, **kwargs: ppo_preflight.CheckedInputs(
            {"technical_status": "passed", "source_status": "unverified", "reasons": []},
            reviewed,
            dataset,
            [],
            {},
            {},
        ),
    )
    monkeypatch.setattr(
        training,
        "get_settings",
        lambda: _settings(tmp_path),
    )

    def record(items):
        registry.write_text(yaml.safe_dump({"schema_version": 1, "research_datasets": items}))

    record([entry])
    return config, spec, entry, record, tmp_path


def test_reviewed_research_reaches_environment_checks(research, monkeypatch):
    config, spec, entry, _, _ = research
    gate, checked = training_gate.authorize_research(config, spec)
    assert gate["dataset_id"] == entry["dataset_id"]
    assert gate["training_approved"] is False
    assert gate["source_status"] == "unverified"
    monkeypatch.setattr(
        training_gate, "validate_research_environments", lambda _: {"train": {"rows": 1}}
    )
    monkeypatch.setattr(training, "_make_env", _environment_reached)
    with pytest.raises(RuntimeError, match="environment reached"):
        training.run_training(config, run_id="reviewed", preflight_spec=spec)


def test_reviewed_research_cli_uses_the_same_gate(research, monkeypatch):
    config, spec, _, _, root = research
    config_path = root / "cli-config.yaml"
    config_path.write_text(yaml.safe_dump(config.model_dump(mode="json")))
    monkeypatch.setattr(training_gate, "validate_research_environments", lambda _: {})
    monkeypatch.setattr(training, "_make_env", _environment_reached)
    with pytest.raises(RuntimeError, match="environment reached"):
        train.main(
            [
                "--config",
                str(config_path),
                "--preflight-spec",
                str(spec),
                "--run-id",
                "cli-reviewed",
            ]
        )


def test_changed_native_input_is_rejected_before_environment(research, monkeypatch):
    config, spec, _, _, root = research
    original = training.load_training_dataset(config.data)
    changed = training.TrainingDataset(
        original.returns,
        original.rsi,
        original.macd,
        original.risk_tags,
        {**original.file_sha256, "returns.parquet": "0" * 64},
    )
    monkeypatch.setattr(training, "load_training_dataset", lambda _: changed)
    monkeypatch.setattr(
        training_gate,
        "validate_research_environments",
        lambda _: pytest.fail("environment checks ran"),
    )
    with pytest.raises(ValueError, match="changed after authorization"):
        training.run_training(config, run_id="changed", preflight_spec=spec)
    assert not (root / "models").exists()


@pytest.mark.parametrize(
    "defect", ["absent", "hash", "period", "source", "config", "claim", "rogue_registry", "smoke"]
)
def test_research_denials_make_no_environment_or_output(research, monkeypatch, defect):
    config, spec, entry, record, root = research
    if defect == "absent":
        record([])
    elif defect == "hash":
        record([{**entry, "file_sha256": {**entry["file_sha256"], "returns.parquet": "0" * 64}}])
    elif defect == "period":
        record([{**entry, "allowed_train_end": "2026-01-03T00:00:00+00:00"}])
    elif defect == "source":
        monkeypatch.setattr(
            ppo_preflight,
            "check_inputs",
            lambda *a, **kw: ppo_preflight.CheckedInputs(
                {"technical_status": "passed", "source_status": "failed", "reasons": []},
                config,
                training.load_training_dataset(config.data),
                [],
                {},
                {},
            ),
        )
    elif defect == "config":
        config = config.model_copy(update={"seed": config.seed + 1})
    elif defect == "rogue_registry":
        record([])
        rogue = root / "requester-registry.yaml"
        rogue.write_text(yaml.safe_dump({"schema_version": 1, "research_datasets": [entry]}))
        spec.write_text(
            json.dumps(
                {
                    "kind": "research",
                    "dataset_id": "fixture-1",
                    "registry_path": str(rogue),
                }
            )
        )
    else:
        record([])
        if defect == "claim":
            spec.write_text(
                json.dumps(
                    {
                        "kind": "research",
                        "dataset_id": "fixture-1",
                        "training_approved": True,
                    }
                )
            )
    monkeypatch.setattr(
        training, "_make_env", lambda *a, **kw: pytest.fail("environment was created")
    )
    with pytest.raises(ValueError):
        training.run_training(config, run_id="denied", preflight_spec=spec, smoke=defect == "smoke")
    assert not (root / "models" / "denied").exists()
    assert not (root / "artifacts" / "training" / "denied").exists()


def test_smoke_and_cli_cannot_claim_synthetic_fixture(tmp_path, monkeypatch):
    directory = tmp_path / "data"
    write_dataset(directory)
    raw = payload(directory)
    config = training.TrainingConfig.model_validate(raw, context={"smoke": True})
    monkeypatch.setattr(
        training,
        "get_settings",
        lambda: _settings(tmp_path),
    )
    monkeypatch.setattr(
        training, "_make_env", lambda *a, **kw: pytest.fail("environment was created")
    )
    with pytest.raises(ValueError, match="trusted test fixture|reviewed repository test"):
        training.run_training(config, run_id="direct", smoke=True)
    config_path = tmp_path / "training.yaml"
    config_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="reviewed repository test"):
        train.main(["--config", str(config_path), "--smoke", "--run-id", "cli"])
    claimed = tmp_path / "claimed-preflight.json"
    claimed.write_text(json.dumps({"kind": "synthetic", "training_approved": True}))
    with pytest.raises(ValueError, match="reviewed repository test"):
        train.main(
            [
                "--config",
                str(config_path),
                "--smoke",
                "--run-id",
                "cli-claimed",
                "--preflight-spec",
                str(claimed),
            ]
        )
    with pytest.raises(ValueError, match="reviewed repository test"):
        training.run_training(config, run_id="claimed", smoke=True, preflight_spec=claimed)
    monkeypatch.delenv("PYTEST_CURRENT_TEST")
    with pytest.raises(SystemExit, match="2"):
        train.main(["--config", str(config_path), "--smoke", "--run-id", "operating"])
    with pytest.raises(SystemExit, match="2"):
        train.main(
            [
                "--config",
                str(config_path),
                "--smoke",
                "--run-id",
                "operating-claimed",
                "--preflight-spec",
                str(claimed),
            ]
        )
    assert not (tmp_path / "models").exists()
    assert not (tmp_path / "artifacts").exists()


def _native_research_inputs(tmp_path, monkeypatch):
    """Build research-shaped inputs from local synthetic bytes and real preflight evidence."""
    monkeypatch.setattr(handoff_tests, "register_synthetic_test_fixture", lambda _: None)
    root, spec, _ = handoff_tests.inputs.__wrapped__(tmp_path)
    alignment = feature_tests._toy_alignment()
    snapshot = write_snapshot(
        feature_tests._results(alignment), root=root / "snapshots", snapshot_id="review-fixture"
    )
    feature = feature_tests._build(alignment, (snapshot, read_snapshot(snapshot).manifest_sha256))
    feature_path = write_feature_dataset(feature, root / "features", dataset_id="review-fixture")
    output = convert_ppo_inputs(feature_path, root / "converted")
    manifest = ppo_preflight.object_json((output / "manifest.json").read_bytes())
    mapping = manifest["date_evaluation_mapping"]
    dates = [entry["evaluation_at"] for entry in mapping]
    partitions = manifest["windows"]["wf-1"]["partitions"]
    bounds = [partitions[name]["first_evaluation_at"] for name in ("train", "validation", "test")]
    bounds.append(str(partitions["test"]["end_exclusive"]) + "T00:00:00+09:00")
    spec.update(kind="research", dataset_id="reviewed-fixture", window_id="wf-1")
    spec["dataset"] = {
        name: handoff_tests._pin(root, f"converted/{name}") for name in manifest["files"]
    }
    spec["lineage"] = handoff_tests._edit(
        root,
        handoff_tests._pin(root, feature_path.relative_to(root) / "lineage.json"),
        dataset_kind="research_only",
    )
    spec["dataset_manifest"] = handoff_tests._edit(
        root,
        handoff_tests._pin(root, "converted/manifest.json"),
        dataset_kind="research_only",
        source_lineage_sha256=spec["lineage"]["sha256"],
    )
    spec["source_audit"] = handoff_tests._edit(
        root,
        spec["source_audit"],
        status="unverified",
        source_lineage_sha256=spec["lineage"]["sha256"],
    )
    original = ppo_preflight.object_json((root / spec["training_config"]["path"]).read_bytes())
    original["data"] = {
        "dataset_dir": "converted",
        "train_start": bounds[0],
        "train_end": partitions["validation"]["last_included_evaluation_at"],
    }
    original["validation"]["start"] = bounds[1]
    original["total_timesteps"] = 100000
    spec["training_config"] = handoff_tests._write(root, "config.json", original)
    spec["calendar"] = handoff_tests._edit(
        root,
        spec["calendar"],
        status="unverified",
        evaluation_at=dates,
        start=bounds[0],
        end=bounds[-1],
    )
    spec["evaluation_policy"] = handoff_tests._edit(
        root,
        spec["evaluation_policy"],
        status="draft",
        calendar_sha256=spec["calendar"]["sha256"],
    )
    for period, start, stop in zip(spec["periods"], bounds, bounds[1:]):
        period.update(start=start, end=stop)
    spec_path = root / "preflight.json"
    spec_path.write_bytes(ppo_preflight.json_bytes(spec))
    checked = ppo_preflight.check_inputs(spec, root, check_environments=False)
    assert checked.result["technical_status"] == "passed", checked.result
    registry = tmp_path / "reviewed.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "research_datasets": [
                    {
                        "dataset_id": "reviewed-fixture",
                        "file_sha256": checked.dataset.file_sha256,
                        "allowed_train_start": original["data"]["train_start"],
                        "allowed_train_end": original["data"]["train_end"],
                        "review_ref": "https://github.com/team-broke/Robo-Advisor/issues/138",
                        "limitations": ["Only synthetic source data was used for this gate test"],
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(training_gate, "RESEARCH_REGISTRY", registry)
    config = training.load_training_config(root / "config.json", dataset_dir=output)
    monkeypatch.setattr(
        training,
        "get_settings",
        lambda: _settings(tmp_path),
    )
    return config, spec_path, checked


def test_native_research_preflight_and_entrypoint_agree(tmp_path, monkeypatch):
    """Use the real converter and preflight; stop only after the gate admits the input."""
    config, spec_path, _ = _native_research_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(training, "_make_env", _environment_reached)
    with pytest.raises(RuntimeError, match="environment reached"):
        training.run_training(config, run_id="reviewed", preflight_spec=spec_path)
    assert not (tmp_path / "models").exists()


def test_reviewed_research_trains_saves_and_reloads(tmp_path, monkeypatch):
    """Run real CPU PPO on reviewed test evidence, capping only the test's learn call."""
    config, spec_path, checked = _native_research_inputs(tmp_path, monkeypatch)
    assert checked.result["technical_status"] == "passed"
    assert checked.result["source_status"] == "unverified"
    original_learn = PPO.learn

    def short_learn(model, total_timesteps, **kwargs):
        assert total_timesteps == 100000
        return original_learn(model, total_timesteps=128, **kwargs)

    monkeypatch.setattr(PPO, "learn", short_learn)
    result = training.run_training(config, run_id="reviewed-positive", preflight_spec=spec_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["actual_timesteps"] >= 128
    assert manifest["training_approved"] is False
    assert manifest["research_use"] is True
    assert manifest["input_gate"]["technical_status"] == "passed"
    assert manifest["input_gate"]["source_status"] == "unverified"
    assert manifest["input_gate"]["file_sha256"] == checked.dataset.file_sha256
    assert result.final_model_path.is_file()
    assert PPO.load(result.final_model_path).num_timesteps >= 128


def test_queue_passes_reviewed_research_to_cli_and_training(tmp_path, monkeypatch):
    """Queue configuration reaches the same research gate and a real short PPO learn call."""
    config, spec_path, checked = _native_research_inputs(tmp_path, monkeypatch)
    base = tmp_path / "queue-base.yaml"
    base.write_text(yaml.safe_dump(config.model_dump(mode="json")))
    queue_path = tmp_path / "queue.yaml"
    queue_path.write_text(
        yaml.safe_dump(
            {
                "queue_id": "reviewed",
                "jobs": [
                    {
                        "run_id": "queue-research",
                        "config": base.name,
                        "preflight_spec": str(spec_path.relative_to(tmp_path)),
                    }
                ],
            }
        )
    )
    original_learn = PPO.learn

    def short_learn(model, total_timesteps, **kwargs):
        assert total_timesteps == 100000
        return original_learn(model, total_timesteps=128, **kwargs)

    monkeypatch.setattr(PPO, "learn", short_learn)

    def run_in_process(command, **kwargs):
        assert command[:3] == [sys.executable, "-m", "robo_advisor.rl.train"]
        assert command[command.index("--preflight-spec") + 1] == str(spec_path)
        with redirect_stdout(kwargs["stdout"]):
            train.main(command[3:])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.delenv("PYTEST_CURRENT_TEST")
    state = queue_module.run_queue(
        queue_path, artifacts_dir=tmp_path / "artifacts", runner=run_in_process
    )
    assert state["state"] == "completed"
    manifest_path = tmp_path / "artifacts/training/queue-research/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["training_approved"] is False
    assert manifest["research_use"] is True
    assert manifest["input_gate"]["file_sha256"] == checked.dataset.file_sha256
    assert (tmp_path / "models/queue-research/final_model.zip").is_file()


def test_failed_real_source_audit_blocks_before_environment(tmp_path, monkeypatch):
    config, spec_path, _ = _native_research_inputs(tmp_path, monkeypatch)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["source_audit"] = handoff_tests._edit(
        spec_path.parent, spec["source_audit"], status="failed"
    )
    spec_path.write_bytes(ppo_preflight.json_bytes(spec))
    checked = ppo_preflight.check_inputs(spec, spec_path.parent, check_environments=False)
    assert checked.result["technical_status"] == "passed"
    assert checked.result["source_status"] == "failed"
    monkeypatch.setattr(training, "_make_env", lambda *a, **kw: pytest.fail("environment created"))
    with pytest.raises(ValueError, match="research preflight rejected"):
        training.run_training(config, run_id="failed-source", preflight_spec=spec_path)
    assert not (tmp_path / "models").exists()
    assert not (tmp_path / "artifacts").exists()
