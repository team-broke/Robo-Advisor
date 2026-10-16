"""Immutable native Parquet feature artifacts with verified data lineage.

These artifacts are research outputs. Successful storage never grants training
approval and does not change the validation status supplied by the builder.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import uuid

import pandas as pd

from robo_advisor.data.snapshot import read_frame, write_frame


class FeatureDatasetIOError(ValueError):
    """A feature artifact has an invalid schema or cannot be stored safely."""


class FeatureDatasetIntegrityError(FeatureDatasetIOError):
    """A feature artifact does not match its recorded or trusted digests."""


_ROOT_FRAMES = {"features", "rewards", "rows", "regimes"}
_REQUIRED_FRAMES = {"features", "rewards", "rows"}
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_SECRET_KEY = re.compile(
    r"(?:api.?key|access.?token|refresh.?token|password|authorization|secret|credential|cookie)",
    re.I,
)
_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


def _safe_id(value, label):
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value)
        or value.endswith(".")
        or value.split(".", 1)[0].upper() in _RESERVED
    ):
        raise FeatureDatasetIOError(f"Unsafe {label}")
    return value


def _relative_frame_path(key):
    if key in _ROOT_FRAMES:
        return f"{key}.parquet"
    if not isinstance(key, str):
        raise FeatureDatasetIOError("Frame keys must be strings")
    parts = key.split("/")
    if (
        len(parts) != 3
        or parts[0] != "windows"
        or parts[2] not in {"features", "splits", "rewards"}
    ):
        raise FeatureDatasetIOError("Unsupported feature frame key")
    _safe_id(parts[1], "walk-forward window ID")
    return f"windows/{parts[1]}/{parts[2]}.parquet"


def _check_frame_keys(frames):
    if not isinstance(frames, Mapping) or not _REQUIRED_FRAMES.issubset(frames):
        raise FeatureDatasetIOError("Feature artifacts require features, rewards and rows")
    paths = {}
    windows = {}
    for key in frames:
        relative = _relative_frame_path(key)
        if relative.casefold() in paths:
            raise FeatureDatasetIOError("Feature frame paths collide ignoring case")
        paths[relative.casefold()] = key
        if key.startswith("windows/"):
            _, window, kind = key.split("/")
            windows.setdefault(window, set()).add(kind)
    if any(not {"features", "splits"}.issubset(kinds) for kinds in windows.values()):
        raise FeatureDatasetIOError("Each walk-forward window needs features and splits")


def _json_value(value):
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        copied = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise FeatureDatasetIOError("Lineage JSON keys must be strings")
            if _SECRET_KEY.search(key):
                raise FeatureDatasetIOError("Credentials must not be written into lineage")
            copied[key] = _json_value(item)
        return copied
    raise FeatureDatasetIOError("Lineage requires finite native JSON values")


def _check_lineage(lineage):
    if not isinstance(lineage, dict):
        raise FeatureDatasetIOError("Feature lineage must be a JSON object")
    lineage = _json_value(lineage)
    if lineage.get("training_approved") is not False:
        raise FeatureDatasetIOError("Feature artifacts must explicitly withhold training approval")
    if lineage.get("dataset_kind") not in {
        "synthetic_test",
        "research_only",
        "validated_research",
    }:
        raise FeatureDatasetIOError("Dataset kind must distinguish synthetic and research data")
    if type(lineage.get("schema_version")) is not int or lineage["schema_version"] != 1:
        raise FeatureDatasetIOError("Unsupported feature lineage schema")
    if (
        not isinstance(lineage.get("feature_version"), str)
        or not lineage["feature_version"].strip()
    ):
        raise FeatureDatasetIOError("Feature lineage requires a nonempty feature version")
    source = lineage.get("source_snapshot")
    if (
        not isinstance(source, dict)
        or not isinstance(source.get("manifest_sha256"), str)
        or not _DIGEST.fullmatch(source["manifest_sha256"])
    ):
        raise FeatureDatasetIOError("Feature lineage requires the source snapshot manifest SHA256")
    for field in ("asset_order", "feature_columns", "reward_columns"):
        columns = lineage.get(field)
        if (
            not isinstance(columns, list)
            or not columns
            or any(not isinstance(value, str) or not value for value in columns)
            or len(set(columns)) != len(columns)
        ):
            raise FeatureDatasetIOError(f"Feature lineage requires unique ordered {field}")
    if len(lineage["asset_order"]) != 12 or lineage["reward_columns"] != lineage["asset_order"]:
        raise FeatureDatasetIOError("Reward columns must preserve the twelve-asset order")
    for field in (
        "data_period",
        "warmup",
        "alignment",
        "validation",
        "windows",
        "market_regimes",
    ):
        if not isinstance(lineage.get(field), dict):
            raise FeatureDatasetIOError(f"Feature lineage requires {field} metadata")
    cutoff = lineage.get("cutoff_at")
    try:
        if not isinstance(cutoff, str):
            raise ValueError("Explicit timestamp required")
        timestamp = pd.Timestamp(cutoff)
        if pd.isna(timestamp) or timestamp.tzinfo is None:
            raise ValueError("Aware timestamp required")
    except (ValueError, TypeError, OverflowError) as exc:
        raise FeatureDatasetIOError("Feature lineage requires an aware cutoff_at") from exc
    return lineage


def _check_declared_frames(frames, lineage):
    """Reject stale column/order/row declarations before storage or after decoding."""
    features, rewards, rows = (frames[key] for key in ("features", "rewards", "rows"))
    if features.columns.tolist() != lineage["feature_columns"]:
        raise FeatureDatasetIOError("Feature columns disagree with lineage column order")
    if rewards.columns.tolist() != lineage["reward_columns"]:
        raise FeatureDatasetIOError("Reward columns disagree with lineage asset order")
    index = features.index
    if (
        not isinstance(index, pd.DatetimeIndex)
        or index.empty
        or index.tz is not None
        or index.hasnans
        or not index.is_unique
        or not index.is_monotonic_increasing
        or not index.equals(index.normalize())
    ):
        raise FeatureDatasetIOError("Feature rows require ordered unique naive daily labels")
    if not rewards.index.equals(index) or not rows.index.equals(index):
        raise FeatureDatasetIOError("Feature, reward and row-metadata indexes must match")
    period = lineage["data_period"]
    expected_period = {
        "start": index.min().date().isoformat(),
        "end_inclusive": index.max().date().isoformat(),
        "rows": len(index),
    }
    if (
        any(period.get(key) != value for key, value in expected_period.items())
        or type(period.get("rows")) is not int
    ):
        raise FeatureDatasetIOError("Feature data period disagrees with stored rows")
    window_ids = {key.split("/")[1] for key in frames if key.startswith("windows/")}
    if window_ids != set(lineage["windows"]):
        raise FeatureDatasetIOError("Window frames disagree with lineage window declarations")
    for window in window_ids:
        window_features = frames[f"windows/{window}/features"]
        splits = frames[f"windows/{window}/splits"]
        if (
            window_features.columns.tolist() != lineage["feature_columns"]
            or not window_features.index.equals(splits.index)
            or not window_features.index.isin(index).all()
        ):
            raise FeatureDatasetIOError("Window feature columns or split indexes disagree")
        window_rewards = frames.get(f"windows/{window}/rewards")
        if window_rewards is not None and (
            window_rewards.columns.tolist() != lineage["reward_columns"]
            or not window_rewards.index.equals(window_features.index)
        ):
            raise FeatureDatasetIOError("Window reward columns or indexes disagree")
    if "regimes" in frames and not frames["regimes"].index.equals(index):
        raise FeatureDatasetIOError("Market regime indexes must match feature rows")


def _bytes(value):
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_symlink_ancestors(path):
    for candidate in (path, *path.parents):
        if candidate.is_symlink():
            raise FeatureDatasetIOError("Feature artifact paths cannot traverse symlinks")


def _remove_owned(path, root):
    if path.parent.resolve() != root or path.resolve().parent != root:
        raise FeatureDatasetIOError("Refusing cleanup outside the feature artifact root")
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _existing_id(root, dataset_id):
    return any(child.name.casefold() == dataset_id.casefold() for child in root.iterdir())


def write_feature_dataset(result, root, *, dataset_id=None) -> Path:
    """Write ``result.frames`` and ``result.lineage`` without replacing any artifact.

    Frames are native Parquet. The final ``lineage.json`` rename is the atomic
    completion marker; incomplete directories cannot be read as complete outputs.
    A new UTC/UUID ID is generated only when ``dataset_id`` is omitted.
    """
    frames = getattr(result, "frames", None)
    _check_frame_keys(frames)
    if any(not isinstance(frame, pd.DataFrame) for frame in frames.values()):
        raise FeatureDatasetIOError("Every feature frame must be a pandas DataFrame")
    lineage = _check_lineage(getattr(result, "lineage", None))
    if "artifact" in lineage:
        raise FeatureDatasetIOError("The artifact lineage field is reserved for storage")
    _check_declared_frames(frames, lineage)
    if dataset_id is None:
        dataset_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex
    _safe_id(dataset_id, "feature dataset ID")
    root = Path(root).absolute()
    _reject_symlink_ancestors(root)
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    final = root / dataset_id
    lock = root / f".feature-{dataset_id.casefold()}.lock"
    lock_fd = None
    stage = None
    own_final = False
    try:
        lock_fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        if _existing_id(root, dataset_id):
            raise FileExistsError(f"Feature dataset ID already exists: {dataset_id}")
        stage = Path(tempfile.mkdtemp(prefix=f".staging-{dataset_id}-", dir=root))
        files = {}
        for key, frame in frames.items():
            relative = _relative_frame_path(key)
            destination = stage.joinpath(*relative.split("/"))
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor = write_frame(frame, destination)
            files[key] = {"path": relative, **descriptor}
        lineage["artifact"] = {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "status": "complete",
            "files": files,
        }
        with (stage / "lineage.json").open("xb") as handle:
            handle.write(_bytes(lineage))
            handle.flush()
            os.fsync(handle.fileno())
        final.mkdir(exist_ok=False)
        own_final = True
        for child in stage.iterdir():
            if child.name != "lineage.json":
                os.replace(child, final / child.name)
        os.replace(stage / "lineage.json", final / "lineage.json")
        return final
    except Exception:
        if own_final:
            _remove_owned(final, root)
        raise
    finally:
        if stage is not None:
            _remove_owned(stage, root)
        if lock_fd is not None:
            os.close(lock_fd)
            lock.unlink()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise FeatureDatasetIOError("Duplicate lineage JSON key")
        result[key] = value
    return result


def _payload(root, relative):
    path = root.joinpath(*relative.split("/"))
    _reject_symlink_ancestors(path)
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise FeatureDatasetIntegrityError("Missing or unsafe feature payload path")
    return path


def _check_artifact(lineage, root):
    artifact = lineage.get("artifact")
    required = {"schema_version", "dataset_id", "created_at", "status", "files"}
    if not isinstance(artifact, dict) or set(artifact) != required:
        raise FeatureDatasetIOError("Missing or unknown feature artifact fields")
    if type(artifact["schema_version"]) is not int or artifact["schema_version"] != 1:
        raise FeatureDatasetIOError("Unsupported feature artifact schema")
    if artifact["status"] != "complete":
        raise FeatureDatasetIOError("Feature artifact is incomplete")
    if _safe_id(artifact["dataset_id"], "feature dataset ID") != root.name:
        raise FeatureDatasetIOError("Feature dataset ID does not match its directory")
    try:
        created_at = datetime.fromisoformat(artifact["created_at"].replace("Z", "+00:00"))
        if created_at.tzinfo is None or created_at.utcoffset().total_seconds() != 0:
            raise ValueError("UTC required")
    except (AttributeError, TypeError, ValueError) as exc:
        raise FeatureDatasetIOError("Feature artifact creation time must be UTC") from exc
    _check_frame_keys(artifact["files"])
    for key, descriptor in artifact["files"].items():
        if not isinstance(descriptor, dict):
            raise FeatureDatasetIOError("Feature payload descriptor must be a JSON object")
        if descriptor.get("path") != _relative_frame_path(key):
            raise FeatureDatasetIOError("Feature payload path is not canonical")
        if any(
            not isinstance(descriptor.get(field), str) or not _DIGEST.fullmatch(descriptor[field])
            for field in ("sha256", "fingerprint")
        ):
            raise FeatureDatasetIOError("Feature payload digest is invalid")
    return artifact


def read_feature_dataset(path, *, expected_lineage_sha256=None) -> dict:
    """Verify all bytes/content and return frames, lineage and its byte digest.

    Pin ``expected_lineage_sha256`` to a trusted external record to authenticate
    lineage. Internal checks alone detect accidental changes, not a coordinated
    rewrite of both payloads and their digests.
    """
    root = Path(path).absolute()
    if root.name == "lineage.json":
        root = root.parent
    _reject_symlink_ancestors(root)
    if not root.is_dir():
        raise FeatureDatasetIOError("Feature artifact directory does not exist")
    root = root.resolve()
    lineage_path = _payload(root, "lineage.json")
    lineage_sha256 = _sha256(lineage_path)
    if expected_lineage_sha256 is not None:
        if (
            not isinstance(expected_lineage_sha256, str)
            or not _DIGEST.fullmatch(expected_lineage_sha256)
            or lineage_sha256 != expected_lineage_sha256
        ):
            raise FeatureDatasetIntegrityError("Feature lineage SHA256 mismatch")
    try:
        lineage = json.loads(
            lineage_path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise FeatureDatasetIOError("Invalid feature lineage JSON") from exc
    lineage = _check_lineage(lineage)
    artifact = _check_artifact(lineage, root)
    expected_files = {"lineage.json"}
    payloads = {}
    for key, descriptor in artifact["files"].items():
        relative = descriptor["path"]
        expected_files.add(relative)
        payload = _payload(root, relative)
        if _sha256(payload) != descriptor["sha256"]:
            raise FeatureDatasetIntegrityError(f"Feature payload SHA256 mismatch: {key}")
        payloads[key] = payload
    actual_files = set()
    for child in root.rglob("*"):
        if child.is_symlink():
            raise FeatureDatasetIntegrityError("Feature artifact contains a symlink")
        if child.is_file():
            actual_files.add(child.relative_to(root).as_posix())
    if actual_files != expected_files:
        raise FeatureDatasetIntegrityError("Feature artifact has missing or unlisted files")
    frames = {}
    for key, payload in payloads.items():
        try:
            frames[key] = read_frame(payload, artifact["files"][key])
        except (ValueError, TypeError, OSError) as exc:
            raise FeatureDatasetIntegrityError(f"Feature payload integrity failed: {key}") from exc
    try:
        _check_declared_frames(frames, lineage)
    except FeatureDatasetIOError as exc:
        raise FeatureDatasetIntegrityError(
            "Stored feature declarations disagree with payloads"
        ) from exc
    return {
        "path": root,
        "frames": frames,
        "lineage": lineage,
        "lineage_sha256": lineage_sha256,
    }
