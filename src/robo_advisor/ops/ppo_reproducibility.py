"""Check PPO inputs, package a completed run and verify a relocated handoff."""

from __future__ import annotations

import argparse
import io
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from robo_advisor.diagnostics.inputs import run_from_spec
from robo_advisor.diagnostics.ppo_diagnostics import _timestamp, _validate_contract
from robo_advisor.ops.ppo_preflight import (
    check_inputs,
    json_bytes,
    object_json,
    pinned,
    portable_path,
    sha256,
)
from robo_advisor.rl.training import TrainingConfig

CURVE_FILES = ("curves_csv", "plot", "convergence")
CHECKPOINTS = {"best": "best_model.zip", "final": "final_model.zip"}
INPUT_FIELDS = {"schema_version", "kind", "window_id", "periods", "currency"}
MANIFEST = "bundle-manifest.json"
SUMS = "SHA256SUMS.txt"


def _descriptor(path, content):
    return {"path": path, "sha256": sha256(content)}


def _native_path(value, *suffix):
    if (
        not isinstance(value, str)
        or tuple(value.replace("\\", "/").split("/")[-len(suffix) :]) != suffix
    ):
        raise ValueError(f"run file must end in {'/'.join(suffix)}")


def _run_file(root, descriptor, *suffix):
    _native_path(descriptor["path"], *suffix)
    return pinned(root, descriptor)


def _require_origin(root, descriptor, recorded, label):
    path = root.joinpath(*portable_path(descriptor["path"]).parts)
    if path != recorded:
        raise ValueError(f"{label} path differs from the original run")


def _validate_run(manifest, config, checked, spec):
    run_id = manifest["run_id"]
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
        raise ValueError("unsafe run_id")
    portable_path(run_id)
    if spec["run_id"] != run_id:
        raise ValueError("training manifest and handoff run_id mismatch")
    if manifest.get("status") != "completed":
        raise ValueError("training manifest must describe a completed run")
    if "training_approved" in manifest and manifest["training_approved"] is not False:
        raise ValueError("native training manifest contains an unsupported approval claim")
    if manifest.get("smoke") is not (spec["kind"] == "synthetic"):
        raise ValueError("requested kind conflicts with the recorded smoke flag")
    recorded = manifest["config"]
    if sha256(json_bytes(recorded, canonical=True)) != manifest["config_sha256"]:
        raise ValueError("native training configuration SHA-256 mismatch")
    effective = TrainingConfig.model_validate(recorded, context={"smoke": manifest["smoke"]})
    supplied = config.model_dump(mode="json")
    supplied["data"]["dataset_dir"] = str(effective.data.dataset_dir)
    if supplied["device"] == "auto":
        supplied["device"] = effective.device
    if supplied != effective.model_dump(mode="json"):
        raise ValueError("supplied training config differs from effective run configuration")
    count = effective.n_envs
    if manifest["n_envs"] != count or manifest["env_seeds"] != list(
        range(config.seed, config.seed + count)
    ):
        raise ValueError("recorded parallel environment count or seeds mismatch")
    if (
        type(manifest["actual_timesteps"]) is not int
        or manifest["actual_timesteps"] < effective.total_timesteps
        or _timestamp(manifest["finished_at"]) < _timestamp(manifest["started_at"])
    ):
        raise ValueError("training completion evidence is incomplete")
    _validate_contract(checked.dataset, manifest, effective.env, checked.periods)
    cutoff = manifest["data_cutoff"]
    if (
        cutoff["rows"] != checked.result["coverage"]["train"]["rows"]
        or _timestamp(cutoff["train_start"]) != _timestamp(effective.data.train_start)
        or _timestamp(cutoff["train_end"]) != _timestamp(effective.data.train_end)
    ):
        raise ValueError("training cutoff or row count differs from the supplied input")
    validation = manifest["validation"]
    selection_index = checked.dataset.returns.index
    selection_index = selection_index[
        (selection_index >= effective.validation.start)
        & (selection_index <= effective.data.train_end)
    ]
    if (
        not isinstance(validation["history"], list)
        or not validation["history"]
        or _timestamp(validation["start"]) != _timestamp(effective.validation.start)
        or _timestamp(validation["end"]) != _timestamp(effective.data.train_end)
        or validation["rows"] != len(selection_index)
        or _timestamp(validation["first_index"]) != selection_index[0]
        or _timestamp(validation["last_index"]) != selection_index[-1]
        or _timestamp(validation["train_last_index"]) != _timestamp(cutoff["actual_last_index"])
        or validation["train_rows"] != cutoff["rows"]
        or validation["metric"] != effective.validation.metric
        or validation["higher_is_better"] is not effective.validation.higher_is_better
    ):
        raise ValueError("native model-selection evidence is incomplete or inconsistent")
    versions = manifest["package_versions"]
    for name in ("python", "stable-baselines3", "torch", "gymnasium", "numpy", "pandas"):
        if not isinstance(versions.get(name), str) or not versions[name]:
            raise ValueError(f"required runtime version missing: {name}")
    commit = manifest["git_commit"]
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit):
        raise ValueError("the original training Git commit is missing or invalid")
    return run_id


def _checkpoint(root, descriptor, manifest, run_id, *, relocated):
    if descriptor["run_id"] != run_id:
        raise ValueError("checkpoint and training manifest run_id mismatch")
    kind = descriptor["kind"]
    if kind not in CHECKPOINTS:
        raise ValueError("checkpoint kind must be best or final")
    name = CHECKPOINTS[kind]
    if kind == "best":
        best = manifest["validation"].get("best")
        if not best:
            raise ValueError("the run has no recorded best checkpoint")
        recorded_path = best["path"]
    else:
        recorded_path = manifest["final_model_path"]
    _native_path(recorded_path, run_id, name)
    if not relocated:
        _require_origin(root, descriptor, Path(recorded_path), "checkpoint")
    content = _run_file(root, descriptor, run_id, name)
    # JSON metadata can be checked without unpickling or running the model.
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        if archive.testzip() is not None or len(set(archive.namelist())) != len(archive.namelist()):
            raise ValueError("invalid SB3 checkpoint ZIP")
        metadata = object_json(archive.read("data"))
        version = archive.read("_stable_baselines3_version").decode("ascii").strip()
    config = manifest["config"]
    assets = len(manifest["data_cutoff"]["assets"])
    observation = [assets * (config["env"]["lookback"] + 4) + 1]
    if (
        metadata["seed"] != manifest["seed"]
        or metadata["n_envs"] != manifest["n_envs"]
        or metadata["num_timesteps"]
        != (manifest["actual_timesteps"] if kind == "final" else best["timesteps"])
        or metadata["observation_space"]["_shape"] != observation
        or metadata["action_space"]["_shape"] != [assets + 1]
        or version != manifest["package_versions"]["stable-baselines3"]
    ):
        raise ValueError("checkpoint metadata differs from the recorded run")
    return name, content


def _monitors(root, descriptors, manifest, run_id):
    names = [f"env-{rank:03d}.monitor.csv" for rank in range(manifest["n_envs"])]
    files = {portable_path(entry["path"]).name: entry for entry in descriptors}
    if len(descriptors) != len(names) or set(files) != set(names):
        raise ValueError("every parallel environment Monitor is required exactly once")
    contents = {}
    episodes = []
    for name in names:
        content = _run_file(root, files[name], run_id, "monitor", name)
        first, _, csv = content.partition(b"\n")
        if not first.startswith(b"#"):
            raise ValueError(f"invalid Monitor file: {name}")
        started = object_json(first[1:])["t_start"]
        if (
            type(started) not in {float, int}
            or not _timestamp(manifest["started_at"]).timestamp()
            <= started
            <= _timestamp(manifest["finished_at"]).timestamp()
        ):
            raise ValueError(f"Monitor start time falls outside the run: {name}")
        frame = pd.read_csv(io.BytesIO(csv))
        if list(frame.columns) != ["r", "l", "t", "reason"]:
            raise ValueError(f"invalid Monitor columns: {name}")
        if not frame.empty:
            values = frame[["r", "l", "t"]].to_numpy(dtype=float)
            if (
                not np.isfinite(values).all()
                or not (frame["l"] > 0).all()
                or not (frame["l"] % 1 == 0).all()
                or not (frame["t"] >= 0).all()
                or not frame["t"].is_monotonic_increasing
            ):
                raise ValueError(f"invalid Monitor episode data: {name}")
            episodes.append(
                frame.rename(columns={"r": "reward", "l": "length", "t": "elapsed_seconds"})
                .assign(env=name.removesuffix(".monitor.csv"))
                .infer_objects()
            )
        contents[name] = content
    return contents, episodes


def _curves(root, descriptors, manifest, run_id, episodes):
    native = manifest["learning_curves"]
    if not isinstance(native, dict) or "error" in native or set(descriptors) != set(CURVE_FILES):
        raise ValueError("completed native curves, convergence summary and plot are required")
    _native_path(native["curves_csv"], run_id, "curves.csv")
    _native_path(native["plot"], "curves", f"{run_id}.png")
    contents = {
        "curves_csv": _run_file(root, descriptors["curves_csv"], run_id, "curves.csv"),
        "convergence": _run_file(root, descriptors["convergence"], run_id, "convergence.json"),
        "plot": _run_file(root, descriptors["plot"], "curves", f"{run_id}.png"),
    }
    convergence = object_json(contents["convergence"])
    curve = pd.read_csv(io.BytesIO(contents["curves_csv"]))
    columns = ["reward", "length", "elapsed_seconds", "reason", "env"]
    expected = pd.concat(episodes, ignore_index=True) if episodes else pd.DataFrame(columns=columns)
    expected = expected.sort_values(["elapsed_seconds", "env"], kind="stable").reset_index(
        drop=True
    )
    if (
        convergence["run_id"] != run_id
        or convergence["status"] != native["status"]
        or convergence["episodes"] != native["episodes"]
        or len(curve) != len(expected)
        or len(curve) != convergence["episodes"]
        or convergence["total_timesteps"] != expected["length"].sum()
        or convergence["total_timesteps"] > manifest["actual_timesteps"]
        or not contents["plot"].startswith(b"\x89PNG\r\n\x1a\n")
    ):
        raise ValueError("learning curves, Monitor episodes or run identity differ")
    try:
        pd.testing.assert_frame_equal(
            curve[columns], expected[columns], check_dtype=False, rtol=1e-10, atol=1e-12
        )
    except AssertionError as exc:
        raise ValueError("curve episode data differs from the Monitor files") from exc
    return contents


def _bundle(spec, root, *, relocated=False):
    root = Path(root).resolve(strict=True)
    checked = check_inputs(spec, root, relocated=relocated)
    report = checked.result
    if report["technical_status"] != "passed":
        raise ValueError(f"preflight {report['status']}: {report['reasons']}")
    if report["source_status"] == "failed":
        raise ValueError("failed source audit blocks run handoff")
    run_content = pinned(root, spec["training_manifest"])
    manifest = object_json(run_content)
    run_id = _validate_run(manifest, checked.config, checked, spec)
    _native_path(spec["training_manifest"]["path"], run_id, "manifest.json")
    if not relocated:
        native = manifest["learning_curves"]
        if isinstance(native, dict) and "error" not in native:
            run_directory = Path(native["curves_csv"]).parent
            _require_origin(
                root, spec["training_manifest"], run_directory / "manifest.json", "manifest"
            )
            for descriptor in spec["monitor_files"]:
                name = portable_path(descriptor["path"]).name
                _require_origin(root, descriptor, run_directory / "monitor" / name, "Monitor")
            for name, recorded in (
                ("curves_csv", Path(native["curves_csv"])),
                ("plot", Path(native["plot"])),
                ("convergence", run_directory / "convergence.json"),
            ):
                _require_origin(root, spec["curves"][name], recorded, name)
    model_name, model = _checkpoint(root, spec["checkpoint"], manifest, run_id, relocated=relocated)
    monitors, episodes = _monitors(root, spec["monitor_files"], manifest, run_id)
    curves = _curves(root, spec["curves"], manifest, run_id, episodes)

    files = {}

    def add(path, content):
        files[path] = content
        return _descriptor(path, content)

    prefix = f"runs/{run_id}"
    handoff = {key: spec[key] for key in INPUT_FIELDS & spec.keys()}
    handoff["run_id"] = run_id
    evidence_names = {
        "training_config": "training-config.yaml",
        "dataset_manifest": "manifest.json",
    }
    for key, content in checked.evidence.items():
        handoff[key] = add(f"evidence/{evidence_names.get(key, key + '.json')}", content)
    handoff["dataset"] = {
        name: add(f"dataset/{name}", content) for name, content in checked.files.items()
    }
    handoff["training_manifest"] = add(f"{prefix}/manifest.json", run_content)
    handoff["checkpoint"] = {
        **add(f"{prefix}/{model_name}", model),
        "run_id": run_id,
        "kind": spec["checkpoint"]["kind"],
    }
    handoff["monitor_files"] = [
        add(f"{prefix}/monitor/{name}", content) for name, content in monitors.items()
    ]
    handoff["curves"] = {
        "curves_csv": add(f"{prefix}/curves.csv", curves["curves_csv"]),
        "plot": add(f"curves/{run_id}.png", curves["plot"]),
        "convergence": add(f"{prefix}/convergence.json", curves["convergence"]),
    }
    add("handoff.json", json_bytes(handoff))
    add("training-config-effective.json", json_bytes(manifest["config"]))
    diagnostic = {
        key: handoff[key]
        for key in ("training_manifest", "dataset", "monitor_files", "periods", "currency")
    }
    diagnostic.update(
        schema_version=1,
        kind="synthetic" if spec["kind"] == "synthetic" else "checkpoint_inference",
        policy_version=report["evaluation_policy"]["version"],
        checkpoint={key: handoff["checkpoint"][key] for key in ("path", "sha256", "run_id")},
    )
    if "source_audit" in handoff:
        diagnostic["source_audit"] = handoff["source_audit"]
    add("diagnostic.json", json_bytes(diagnostic))
    summary = {
        "schema_version": 1,
        "kind": spec["kind"],
        "run_id": run_id,
        "git_commit": manifest["git_commit"],
        "package_versions": manifest["package_versions"],
        "periods": handoff["periods"],
        "preflight": report,
        "training_approved": False,
        "checkpoint": {
            **handoff["checkpoint"],
            "sha256_observed": "at_packaging",
            "sha256_at_training": "unavailable_in_native_manifest",
        },
        "files": {name: sha256(content) for name, content in files.items()},
    }
    add(MANIFEST, json_bytes(summary))
    files[SUMS] = "".join(f"{sha256(files[name])}  {name}\n" for name in sorted(files)).encode(
        "utf-8"
    )
    return files


def _write_files(directory, files):
    for name, content in files.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as target:
            target.write(content)


def pack_run(spec, root, output):
    """Write an independent copy of the verified inputs and completed run."""
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("output directory already exists")
    files = _bundle(spec, root)
    output.mkdir(parents=True, exist_ok=False)
    try:
        _write_files(output, files)
    except BaseException:
        shutil.rmtree(output)
        raise
    return object_json(files[MANIFEST])


def _package_files(package):
    pending = [package]
    files = set()
    while pending:
        for path in pending.pop().iterdir():
            relative = path.relative_to(package).as_posix()
            if path.is_symlink() or path.resolve(strict=True) != path:
                raise ValueError(f"package may not use a filesystem alias: {relative}")
            if path.is_dir():
                pending.append(path)
            elif path.is_file():
                files.add(relative)
            else:
                raise ValueError(
                    f"package entries must be regular files or directories: {relative}"
                )
    return files


def _verified_files(package, expected_manifest_sha256):
    package = Path(package).resolve(strict=True)
    found = _package_files(package)
    manifest_bytes = (package / MANIFEST).read_bytes()
    if expected_manifest_sha256 is not None and sha256(manifest_bytes) != expected_manifest_sha256:
        raise ValueError("bundle manifest SHA-256 differs from the external pin")
    spec = object_json((package / "handoff.json").read_bytes())
    derived = _bundle(spec, package, relocated=True)
    if derived[MANIFEST] != manifest_bytes:
        raise ValueError("bundle manifest differs from the verified inputs")
    if found != set(derived):
        raise ValueError("package file set differs from the manifest")
    for name, content in derived.items():
        if pinned(package, _descriptor(name, content)) != content:
            raise ValueError(f"package bytes differ: {name}")
    return derived


def verify_run(package, *, expected_manifest_sha256=None):
    """Recheck a closed handoff file set and regenerate its derived records."""
    return object_json(_verified_files(package, expected_manifest_sha256)[MANIFEST])


def diagnose_run(package, output, *, expected_manifest_sha256):
    """Run the existing diagnostic only on a verified, trusted snapshot of a handoff."""
    if not isinstance(expected_manifest_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_manifest_sha256
    ):
        raise ValueError("diagnosis requires a trusted external manifest SHA-256")
    files = _verified_files(package, expected_manifest_sha256)
    # The diagnostic must consume the bytes that passed verification, even if the
    # original package is subsequently changed.
    with tempfile.TemporaryDirectory(prefix="ppo-handoff-") as temporary:
        root = Path(temporary)
        spec = object_json(files["diagnostic.json"])
        descriptors = [
            spec["training_manifest"],
            spec["checkpoint"],
            *spec["dataset"].values(),
            *spec["monitor_files"],
        ]
        if "source_audit" in spec:
            descriptors.append(spec["source_audit"])
        snapshot = {"diagnostic.json": files["diagnostic.json"]}
        snapshot.update({item["path"]: files[item["path"]] for item in descriptors})
        _write_files(root, snapshot)
        return run_from_spec(root / "diagnostic.json", root=root, output_dir=output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "pack"):
        command = commands.add_parser(name)
        command.add_argument("--spec", required=True, type=Path)
        command.add_argument("--root", required=True, type=Path)
        if name == "pack":
            command.add_argument("--output", required=True, type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("--package", required=True, type=Path)
    verify.add_argument("--manifest-sha256")
    diagnose = commands.add_parser("diagnose")
    diagnose.add_argument("--package", required=True, type=Path)
    diagnose.add_argument("--output", required=True, type=Path)
    diagnose.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            result = verify_run(args.package, expected_manifest_sha256=args.manifest_sha256)
        elif args.command == "diagnose":
            output = diagnose_run(
                args.package, args.output, expected_manifest_sha256=args.manifest_sha256
            )
            print(json_bytes({"status": "ready", "output": str(output)}).decode("utf-8"), end="")
            return 0
        else:
            spec = object_json(args.spec.read_bytes())
            if args.command == "preflight":
                report = check_inputs(spec, args.root).result
                print(json_bytes(report).decode("utf-8"), end="")
                return 0 if report["status"] == "ready" else 2
            result = pack_run(spec, args.root, args.output)
    except (KeyError, FileNotFoundError) as exc:
        result = {"status": "unavailable", "detail": str(exc)}
    except (ValueError, OSError, TypeError, AttributeError, zipfile.BadZipFile) as exc:
        result = {"status": "blocked", "detail": str(exc)}
    else:
        print(
            json_bytes({"status": "ready", "scope": "bundle_integrity", "bundle": result}).decode(
                "utf-8"
            ),
            end="",
        )
        return 0
    print(json_bytes(result).decode("utf-8"), end="")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
