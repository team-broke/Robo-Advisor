"""Portable PPO handoff tests using only synthetic market data."""

from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from robo_advisor.config import Settings
from robo_advisor.data.snapshot import read_snapshot, write_snapshot
from robo_advisor.features.dataset_io import write_feature_dataset
from robo_advisor.features.ppo_input import convert_ppo_inputs
from robo_advisor.ops.ppo_preflight import check_inputs, json_bytes, object_json, sha256
from robo_advisor.ops.ppo_reproducibility import (
    MANIFEST,
    SUMS,
    diagnose_run,
    main,
    pack_run,
    verify_run,
)
from robo_advisor.rl import training
from tests import test_feature_dataset as feature_tests


def _pin(root, path):
    relative = Path(path)
    return {"path": relative.as_posix(), "sha256": sha256((root / relative).read_bytes())}


def _write(root, path, value):
    (root / path).write_bytes(json_bytes(value))
    return _pin(root, path)


def _edit(root, descriptor, **fields):
    path = descriptor["path"]
    value = object_json((root / path).read_bytes())
    return _write(root, path, {**value, **fields})


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / "input"
    (root / "data").mkdir(parents=True)
    dates = pd.bdate_range("2024-03-01", periods=32, tz="UTC") + pd.Timedelta(hours=6, minutes=30)
    assets = ["SYNTH_A", "SYNTH_B"]
    values = {
        "returns.parquet": np.reshape(np.sin(np.arange(64)) * 0.005, (32, 2)),
        "rsi.parquet": 50.0,
        "macd.parquet": 0.005,
    }
    files = {}
    for name, value in values.items():
        path = f"data/{name}"
        pd.DataFrame(value, index=dates, columns=assets).to_parquet(root / path)
        files[name] = _pin(root, path)
    config = {
        "seed": 42,
        "device": "cpu",
        "total_timesteps": 64,
        "n_envs": 2,
        "vec_env": "dummy",
        "ppo": {
            "n_steps": 8,
            "batch_size": 8,
            "n_epochs": 1,
            "net_arch": {"pi": [8], "vf": [8]},
        },
        "env": {"lookback": 3, "min_episode_steps": 2, "max_episode_steps": 4},
        "data": {
            "dataset_dir": "data",
            "train_start": dates[0].isoformat(),
            "train_end": dates[23].isoformat(),
        },
        "validation": {"start": dates[15].isoformat(), "eval_freq": 16},
    }
    end = (dates[-1] + pd.offsets.BDay()).isoformat()
    bounds = [dates[0].isoformat(), dates[15].isoformat(), dates[24].isoformat(), end]
    calendar = _write(
        root,
        "calendar.json",
        {
            "schema_version": 1,
            "source": "synthetic test calendar",
            "status": "synthetic",
            "start": bounds[0],
            "end": end,
            "evaluation_at": [date.isoformat() for date in dates],
        },
    )
    lineage = _write(
        root, "lineage.json", {"dataset_kind": "synthetic_test", "training_approved": False}
    )
    spec = {
        "schema_version": 1,
        "kind": "synthetic",
        "currency": "KRW",
        "dataset": files,
        "training_config": _write(root, "config.json", config),
        "dataset_manifest": _write(
            root,
            "dataset-manifest.json",
            {
                "schema_version": 1,
                "dataset_kind": "synthetic_test",
                "asset_order": assets,
                "files": files,
                "training_approved": False,
                "source_lineage_sha256": lineage["sha256"],
            },
        ),
        "lineage": lineage,
        "source_audit": _write(
            root,
            "audit.json",
            {
                "status": "unverified",
                "training_approved": False,
                "source_lineage_sha256": lineage["sha256"],
            },
        ),
        "calendar": calendar,
        "evaluation_policy": _write(
            root,
            "policy.json",
            {
                "schema_version": 1,
                "version": "synthetic-protocol-v1",
                "status": "synthetic",
                "calendar_sha256": calendar["sha256"],
            },
        ),
        "periods": [
            {"name": name, "start": start, "end": stop}
            for name, start, stop in zip(("train", "validation", "test"), bounds, bounds[1:])
        ],
    }
    return root, spec, dates


def _add_run(root, spec, monkeypatch):
    run_id = "synthetic-repro-smoke"
    settings = Settings(
        _env_file=None, models_dir=root / "models", artifacts_dir=root / "artifacts"
    )
    monkeypatch.setattr(training, "get_settings", lambda: settings)
    config = training.load_training_config(
        root / spec["training_config"]["path"], dataset_dir=root / "data", smoke=True
    )
    training.run_training(config, run_id=run_id, smoke=True)
    run = f"artifacts/training/{run_id}"
    spec.update(
        run_id=run_id,
        training_manifest=_pin(root, f"{run}/manifest.json"),
        checkpoint={
            **_pin(root, f"models/{run_id}/final_model.zip"),
            "run_id": run_id,
            "kind": "final",
        },
        monitor_files=[
            _pin(root, f"{run}/monitor/env-{rank:03d}.monitor.csv") for rank in range(config.n_envs)
        ],
        curves={
            "curves_csv": _pin(root, f"{run}/curves.csv"),
            "convergence": _pin(root, f"{run}/convergence.json"),
            "plot": _pin(root, f"artifacts/training/curves/{run_id}.png"),
        },
    )


@pytest.fixture
def completed(inputs, monkeypatch):
    root, spec, _ = inputs
    _add_run(root, spec, monkeypatch)
    return root, spec


def test_preflight_does_not_train_and_keeps_scope_synthetic(inputs, monkeypatch, capsys):
    root, spec, _ = inputs

    def forbidden(*args, **kwargs):
        pytest.fail("preflight started training")

    monkeypatch.setattr(training, "run_training", forbidden)
    _write(root, "spec.json", spec)
    assert main(["preflight", "--spec", str(root / "spec.json"), "--root", str(root)]) == 0
    report = object_json(capsys.readouterr().out)
    assert report["status"] == "ready"
    assert report["scope"] == "synthetic_smoke"
    assert report["training_approved"] is False and report["formal_ready"] is False
    assert report["source_status"] == "unverified"
    assert report["source_audit_lineage_binding"] == "hash_matched"
    assert report["coverage"] == {
        "train": {"rows": 15, "context_only_rows": 3},
        "validation": {"rows": 9, "context_only_rows": 0},
        "test": {"rows": 8, "context_only_rows": 0},
    }


@pytest.mark.leakage
@pytest.mark.parametrize("problem", ["overlap", "test_in_cutoff", "warmup", "missing_session"])
def test_preflight_refuses_invalid_time_contracts(inputs, problem):
    root, spec, dates = inputs
    if problem == "overlap":
        spec["periods"][2]["start"] = dates[23].isoformat()
    elif problem == "test_in_cutoff":
        descriptor = spec["training_config"]
        config = object_json((root / descriptor["path"]).read_bytes())
        config["data"]["train_end"] = dates[24].isoformat()
        spec["training_config"] = _write(root, descriptor["path"], config)
    else:
        names = spec["dataset"]
        if problem == "warmup":
            name = "rsi.parquet"
            frame = pd.read_parquet(root / names[name]["path"])
            frame.iloc[15, 0] = np.nan
            frame.to_parquet(root / names[name]["path"])
            names[name] = _pin(root, names[name]["path"])
        else:
            for name, descriptor in names.items():
                frame = pd.read_parquet(root / descriptor["path"]).drop(index=dates[26])
                frame.to_parquet(root / descriptor["path"])
                names[name] = _pin(root, descriptor["path"])
        spec["dataset_manifest"] = _edit(root, spec["dataset_manifest"], files=names)
    report = check_inputs(spec, root).result
    assert report["technical_status"] == "failed"
    detail = report["reasons"][0]["detail"]
    assert {
        "overlap": "nonoverlapping",
        "test_in_cutoff": "cutoffs",
        "warmup": "rsi observations",
        "missing_session": dates[26].strftime("%Y-%m-%d"),
    }[problem] in detail


@pytest.mark.parametrize("artifact", ["dataset_manifest", "source_audit", "lineage"])
@pytest.mark.parametrize("claim", [True, None])
def test_missing_or_claimed_approval_is_blocked(inputs, artifact, claim):
    root, spec, _ = inputs
    path = spec[artifact]["path"]
    value = object_json((root / path).read_bytes())
    if claim is None:
        del value["training_approved"]
    else:
        value["training_approved"] = claim
    spec[artifact] = _write(root, path, value)
    if artifact == "lineage":
        lineage_digest = spec[artifact]["sha256"]
        for key in ("source_audit", "dataset_manifest"):
            spec[key] = _edit(root, spec[key], source_lineage_sha256=lineage_digest)
    report = check_inputs(spec, root).result
    assert report["status"] == "blocked"
    assert report["training_approved"] is False
    detail = report["reasons"][0]["detail"]
    assert "training_approved=false" in detail or "approval mismatch" in detail


@pytest.mark.parametrize(
    "path",
    [
        "../config.json",
        "/config.json",
        "C:/config.json",
        "data\\config.json",
        "data/./config.json",
        "data/CON.json",
        "data/too-late. ",
    ],
)
def test_rejects_nonportable_or_escaping_paths(inputs, path):
    root, spec, _ = inputs
    spec["training_config"]["path"] = path
    report = check_inputs(spec, root).result
    assert report["status"] == "blocked"
    assert "portable relative path" in report["reasons"][0]["detail"]


def test_missing_dataset_manifest_is_unavailable(inputs):
    root, spec, _ = inputs
    (root / spec["dataset_manifest"]["path"]).unlink()
    assert check_inputs(spec, root).result["status"] == "unavailable"


def test_input_tampering_is_detected_even_when_operator_repins(inputs):
    root, spec, _ = inputs
    descriptor = spec["dataset"]["returns.parquet"]
    path = root / descriptor["path"]
    frame = pd.read_parquet(path)
    frame.iloc[-1, 0] = 0.25
    frame.to_parquet(path)
    assert "SHA-256 mismatch" in check_inputs(spec, root).result["reasons"][0]["detail"]
    spec["dataset"]["returns.parquet"] = _pin(root, descriptor["path"])
    assert "manifest file hashes" in check_inputs(spec, root).result["reasons"][0]["detail"]


def test_policy_is_pinned_to_exact_calendar(inputs):
    root, spec, _ = inputs
    spec["calendar"] = _edit(root, spec["calendar"], source="a different calendar")
    report = check_inputs(spec, root).result
    assert "policy calendar SHA-256" in report["reasons"][0]["detail"]


def test_native_trainer_would_read_unlisted_optional_file(inputs):
    root, spec, _ = inputs
    (root / "data/risk_tags.parquet").write_bytes(b"untracked")
    assert "unlisted risk_tags" in check_inputs(spec, root).result["reasons"][0]["detail"]


def test_failed_source_is_explicit_even_when_inputs_are_technically_valid(inputs):
    root, spec, _ = inputs
    spec["source_audit"] = _edit(root, spec["source_audit"], status="failed")
    report = check_inputs(spec, root).result
    assert report["technical_status"] == "passed"
    assert report["status"] == "blocked"
    assert report["source_status"] == "failed"
    assert "SOURCE_AUDIT_FAILED" in {issue["code"] for issue in report["reasons"]}


@pytest.mark.leakage
@pytest.mark.parametrize("scenario", ["synthetic", "broken_link", "reported_approval"])
def test_native_converter_manifest_and_source_lineage_match(inputs, scenario):
    root, spec, _ = inputs
    alignment = feature_tests._toy_alignment()
    snapshot = write_snapshot(
        feature_tests._results(alignment), root=root / "snapshots", snapshot_id="synthetic-team12"
    )
    result = feature_tests._build(alignment, (snapshot, read_snapshot(snapshot).manifest_sha256))
    feature_path = write_feature_dataset(result, root / "features", dataset_id="toy")
    output = convert_ppo_inputs(feature_path, root / "converted")
    manifest = object_json((output / "manifest.json").read_bytes())
    mapping = manifest["date_evaluation_mapping"]
    dates = pd.DatetimeIndex([entry["evaluation_at"] for entry in mapping])
    partitions = manifest["windows"]["wf-1"]["partitions"]
    bounds = [partitions[name]["first_evaluation_at"] for name in ("train", "validation", "test")]
    bounds.append(pd.Timestamp(partitions["test"]["end_exclusive"], tz="Asia/Seoul").isoformat())
    spec["window_id"] = "wf-1"
    spec["dataset"] = {name: _pin(root, f"converted/{name}") for name in manifest["files"]}
    spec["dataset_manifest"] = _pin(root, "converted/manifest.json")
    if scenario == "broken_link":
        manifest["row_metadata"]["data"][1]["next_date"] = None
        spec["dataset_manifest"] = _write(root, "converted/manifest.json", manifest)
    spec["lineage"] = _pin(root, feature_path.relative_to(root) / "lineage.json")
    spec["source_audit"] = _edit(
        root, spec["source_audit"], source_lineage_sha256=spec["lineage"]["sha256"]
    )
    config = object_json((root / spec["training_config"]["path"]).read_bytes())
    config["data"] = {
        "dataset_dir": "converted",
        "train_start": bounds[0],
        "train_end": partitions["validation"]["last_included_evaluation_at"],
    }
    config["validation"]["start"] = bounds[1]
    spec["training_config"] = _write(root, "config.json", config)
    spec["calendar"] = _edit(
        root,
        spec["calendar"],
        evaluation_at=[date.isoformat() for date in dates],
        start=bounds[0],
        end=bounds[-1],
    )
    spec["evaluation_policy"] = _edit(
        root, spec["evaluation_policy"], calendar_sha256=spec["calendar"]["sha256"]
    )
    for period, start, stop in zip(spec["periods"], bounds, bounds[1:]):
        period.update(start=start, end=stop)
    if scenario == "reported_approval":
        spec["kind"] = "financial"
        spec["lineage"] = _edit(root, spec["lineage"], dataset_kind="research_only")
        lineage_sha = spec["lineage"]["sha256"]
        spec["dataset_manifest"] = _edit(
            root,
            spec["dataset_manifest"],
            dataset_kind="research_only",
            source_lineage_sha256=lineage_sha,
        )
        spec["source_audit"] = _edit(
            root, spec["source_audit"], status="passed", source_lineage_sha256=lineage_sha
        )
        spec["training_config"] = _edit(root, spec["training_config"], total_timesteps=100000)
        spec["calendar"] = _edit(
            root, spec["calendar"], status="official", evidence_ref="fixture://calendar"
        )
        spec["evaluation_policy"] = _edit(
            root,
            spec["evaluation_policy"],
            status="approved",
            approval_ref="fixture://policy",
            calendar_sha256=spec["calendar"]["sha256"],
        )
    report = check_inputs(spec, root).result
    if scenario == "broken_link":
        assert "discontinuous interval" in report["reasons"][0]["detail"]
        assert report["status"] == "blocked"
    elif scenario == "reported_approval":
        assert report["technical_status"] == "passed", report
        assert report["status"] == "blocked"
        assert report["formal_ready"] is False
        assert report["training_approved"] is False
    else:
        assert report["status"] == "ready", report
        assert report["assets"] == list(feature_tests.ASSETS)


def test_synthetic_cpu_handoff_survives_relocation_and_runs_native_diagnostics(
    completed, tmp_path, monkeypatch
):
    root, spec = completed
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_summary = pack_run(spec, root, first)
    assert pack_run(spec, root, second) == first_summary
    assert (first / SUMS).read_bytes() == (second / SUMS).read_bytes()
    assert b"\r" not in (first / SUMS).read_bytes()
    best_spec = deepcopy(spec)
    best_spec["checkpoint"] = {
        **_pin(root, f"models/{spec['run_id']}/best_model.zip"),
        "run_id": spec["run_id"],
        "kind": "best",
    }
    best = tmp_path / "best"
    assert pack_run(best_spec, root, best)["checkpoint"]["kind"] == "best"
    assert verify_run(best)["run_id"] == spec["run_id"]
    with pytest.raises(FileExistsError, match="already exists"):
        pack_run(spec, root, first)
    pin = sha256((first / MANIFEST).read_bytes())
    moved = tmp_path / "relocated package" / "renamed"
    moved.parent.mkdir()
    shutil.move(first, moved)
    shutil.move(root, tmp_path / "original disabled")
    monkeypatch.chdir(tmp_path)
    assert verify_run(moved, expected_manifest_sha256=pin) == first_summary
    assert first_summary["checkpoint"]["sha256_observed"] == "at_packaging"
    assert first_summary["checkpoint"]["sha256_at_training"] == "unavailable_in_native_manifest"
    spec_path = moved / "diagnostic.json"
    diagnostic = object_json(spec_path.read_bytes())
    assert diagnostic["kind"] == "synthetic"
    assert diagnostic["checkpoint"]["run_id"] == spec["run_id"]
    output = tmp_path / "diagnosed"
    assert (
        main(
            [
                "diagnose",
                "--package",
                str(moved),
                "--manifest-sha256",
                pin,
                "--output",
                str(output),
            ]
        )
        == 0
    )
    summary = object_json((output / "summary.json").read_bytes())
    assert summary["kind"] == "synthetic"
    assert summary["training_approved"] is False
    assert "unverified" in (output / "diagnostic_report.md").read_text(encoding="utf-8")
    assert (output / "period_metrics.csv").is_file()


def test_completed_run_packaging_rejects_mixed_incomplete_or_modified_inputs(completed, tmp_path):
    root, spec = completed
    out = tmp_path / "must-not-exist"
    for key, value, expected in (
        ("checkpoint", {**spec["checkpoint"], "sha256": "0" * 64}, "SHA-256 mismatch"),
        ("checkpoint", {**spec["checkpoint"], "run_id": "another"}, "run_id mismatch"),
        ("monitor_files", spec["monitor_files"][:-1], "parallel environment Monitor"),
        ("run_id", "other-run", "run_id mismatch"),
    ):
        bad_spec = {**deepcopy(spec), key: value}
        with pytest.raises(ValueError, match=expected):
            pack_run(bad_spec, root, out)
        assert not out.exists()
    original = (root / spec["training_manifest"]["path"]).read_bytes()
    bad_spec = deepcopy(spec)
    bad_spec["training_manifest"] = _edit(root, spec["training_manifest"], status="running")
    with pytest.raises(ValueError, match="completed run"):
        pack_run(bad_spec, root, out)
    (root / spec["training_manifest"]["path"]).write_bytes(original)
    (root / spec["checkpoint"]["path"]).unlink()
    with pytest.raises(FileNotFoundError):
        pack_run(spec, root, out)
    assert not out.exists()


def test_repacked_manifest_with_forged_training_approval_is_rejected(completed, tmp_path):
    root, spec = completed
    spec["training_manifest"] = _edit(root, spec["training_manifest"], training_approved=True)
    with pytest.raises(ValueError, match="unsupported approval claim"):
        pack_run(spec, root, tmp_path / "forged")


def test_checkpoint_from_another_run_with_reused_id_is_rejected(completed, tmp_path, monkeypatch):
    root, spec = completed
    foreign = root / "another-execution"
    shutil.copytree(root / "data", foreign / "data")
    shutil.copyfile(root / "config.json", foreign / "config.json")
    foreign_spec = {"training_config": _pin(foreign, "config.json")}
    returns = foreign / "data" / "returns.parquet"
    frame = pd.read_parquet(returns)
    (-frame).to_parquet(returns)
    _add_run(foreign, foreign_spec, monkeypatch)
    foreign_model = foreign / foreign_spec["checkpoint"]["path"]
    bad = deepcopy(spec)
    bad["checkpoint"].update(_pin(root, foreign_model.relative_to(root)))
    assert bad["checkpoint"]["sha256"] != spec["checkpoint"]["sha256"]
    with pytest.raises(ValueError, match="checkpoint path differs"):
        pack_run(bad, root, tmp_path / "mixed")
    bad = deepcopy(spec)
    foreign_monitor = foreign / foreign_spec["monitor_files"][0]["path"]
    bad["monitor_files"][0] = _pin(root, foreign_monitor.relative_to(root))
    with pytest.raises(ValueError, match="Monitor path differs"):
        pack_run(bad, root, tmp_path / "mixed")


def test_verification_refuses_unlisted_directory_links(completed, tmp_path):
    root, spec = completed
    out = tmp_path / "bundle"
    pack_run(spec, root, out)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = out / "external"
    if os.name == "nt":
        subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(outside)],
            check=True,
            capture_output=True,
        )
    else:
        link.symlink_to(outside, target_is_directory=True)
    try:
        with pytest.raises(ValueError, match="filesystem alias: external"):
            verify_run(out)
    finally:
        if os.name == "nt":
            link.rmdir()
        else:
            link.unlink()


def test_diagnosis_checks_entire_package_before_model_load(completed, tmp_path, monkeypatch):
    root, spec = completed
    bundle = tmp_path / "bundle"
    pack_run(spec, root, bundle)
    pin = sha256((bundle / MANIFEST).read_bytes())
    output = tmp_path / "diagnosed"

    def forbidden(*args, **kwargs):
        pytest.fail("diagnosis reached model loading with unverified inputs")

    monkeypatch.setattr("robo_advisor.diagnostics.inputs.PPO.load", forbidden)
    with pytest.raises(ValueError, match="trusted external manifest"):
        diagnose_run(bundle, output, expected_manifest_sha256=None)
    with pytest.raises(ValueError, match="external pin"):
        diagnose_run(bundle, output, expected_manifest_sha256="0" * 64)
    monitor = next(bundle.glob("runs/*/monitor/*.monitor.csv"))
    monitor.unlink()
    with pytest.raises(FileNotFoundError):
        diagnose_run(bundle, output, expected_manifest_sha256=pin)
    assert not output.exists()


def test_verification_detects_tampered_reports_and_external_pin(completed, tmp_path):
    root, spec = completed
    out = tmp_path / "bundle"
    pack_run(spec, root, out)
    with pytest.raises(ValueError, match="external pin"):
        verify_run(out, expected_manifest_sha256="0" * 64)
    diagnostic = object_json((out / "diagnostic.json").read_bytes())
    _write(out, "diagnostic.json", {**diagnostic, "kind": "checkpoint_inference"})
    with pytest.raises(ValueError, match="diagnostic.json"):
        verify_run(out)
