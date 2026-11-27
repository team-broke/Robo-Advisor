"""Read-only PPO input checks using the native training and evaluation contracts."""

from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import yaml

from robo_advisor.diagnostics.inputs import read_pinned
from robo_advisor.diagnostics.ppo_diagnostics import DiagnosticPeriod, _period_dataset, _timestamp
from robo_advisor.features.dataset import TEAM12_ASSETS
from robo_advisor.features.dataset_io import _check_lineage, _unique_object
from robo_advisor.features.ppo_input import CONVERSION_VERSION
from robo_advisor.rl.evaluation import evaluation_env_config
from robo_advisor.rl.training import TrainingConfig, TrainingDataset, _make_env
from robo_advisor.rl.validation import split_validation
from robo_advisor.schemas.risk import RiskTag

NATIVE_FILES = ("returns.parquet", "rsi.parquet", "macd.parquet")
RISK_FILE = "risk_tags.parquet"
PERIODS = ("train", "validation", "test")
PORTABLE_SEGMENT = re.compile(r"[^<>:\"/\\|?*\x00-\x1f]+")
WINDOWS_DEVICE = re.compile(r"CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³]", re.I)


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def json_bytes(value, *, canonical=False):
    options = {"separators": (",", ":")} if canonical else {"indent": 2}
    value = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, **options)
    return (value if canonical else value + "\n").encode("utf-8")


def object_json(content):
    value = json.loads(content, object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def portable_path(value):
    if not isinstance(value, str):
        raise ValueError("input path must be a portable relative path")
    parts = value.split("/")
    if any(
        not PORTABLE_SEGMENT.fullmatch(part)
        or part.endswith((" ", "."))
        or WINDOWS_DEVICE.fullmatch(part.split(".")[0])
        for part in parts
    ):
        raise ValueError(f"input path must be a portable relative path: {value!r}")
    return PurePosixPath(value)


def pinned(root, descriptor):
    relative = portable_path(descriptor["path"])
    # Reject aliases as well as escaping symlinks, including Windows junctions.
    root = Path(root).resolve(strict=True)
    path = root.joinpath(*relative.parts)
    if path.resolve(strict=True) != path:
        raise ValueError(f"input may not use a filesystem alias: {relative}")
    return read_pinned(root, descriptor)


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


@dataclass
class CheckedInputs:
    result: dict
    config: TrainingConfig | None
    dataset: TrainingDataset | None
    periods: list[DiagnosticPeriod]
    evidence: dict[str, bytes]
    files: dict[str, bytes]


def _periods(entries):
    periods = [DiagnosticPeriod(**entry) for entry in entries]
    if [period.name for period in periods] != list(PERIODS):
        raise ValueError("provide train, validation, test in that order")
    bounds = [(_timestamp(period.start), _timestamp(period.end)) for period in periods]
    if any(start >= end for start, end in bounds):
        raise ValueError("period start must precede its exclusive end")
    if any(end != start for (_, end), (start, _) in zip(bounds, bounds[1:])):
        raise ValueError("periods must be contiguous and nonoverlapping")
    return periods, bounds


def _load_dataset(files):
    hashes = {name: sha256(content) for name, content in files.items()}
    frames = [pd.read_parquet(io.BytesIO(files[name])) for name in NATIVE_FILES]
    dates, assets = frames[0].index, frames[0].columns
    if (
        not isinstance(dates, pd.DatetimeIndex)
        or dates.tz is None
        or dates.hasnans
        or dates.empty
        or not dates.is_unique
        or not dates.is_monotonic_increasing
    ):
        raise ValueError("dataset dates must be ordered, unique and timezone-aware")
    if (
        assets.empty
        or not assets.is_unique
        or "CASH" in assets
        or any(not isinstance(asset, str) or not asset for asset in assets)
    ):
        raise ValueError("dataset asset order must have unique non-CASH names")
    for frame in frames:
        if not frame.index.equals(dates) or not frame.columns.equals(assets):
            raise ValueError("dataset dates and asset order must match")
        if any(dtype.kind not in "fiu" for dtype in frame.dtypes):
            raise ValueError("dataset must be numeric")
    returns = frames[0].to_numpy()
    if not np.isfinite(returns).all() or (returns < -1).any():
        raise ValueError("simple returns must be finite and at least -1")
    tags = ()
    if RISK_FILE in files:
        raw = pd.read_parquet(io.BytesIO(files[RISK_FILE]))
        if not raw.columns.is_unique or set(raw.columns) != set(RiskTag.model_fields):
            raise ValueError("risk_tags columns must match RiskTag fields")
        tags = tuple(RiskTag.model_validate(row) for row in raw.to_dict(orient="records"))
        if any(tag.asset not in assets for tag in tags):
            raise ValueError("risk_tags must reference assets in the dataset")
    return TrainingDataset(*frames, tags, hashes)


def _check_dataset_manifest(manifest, dataset, kind, evidence, bounds, window_id):
    native = manifest.get("conversion_version") == CONVERSION_VERSION
    if not native and (
        kind != "synthetic"
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
    ):
        raise ValueError("expected the native converted dataset manifest")
    declared_kind = manifest["dataset_kind"]
    allowed = {"synthetic_test"} if kind == "synthetic" else {"research_only", "validated_research"}
    if declared_kind not in allowed:
        raise ValueError("dataset kind disagrees with the requested scope")
    if manifest.get("training_approved") is not False:
        raise ValueError("dataset manifest must explicitly record training_approved=false")
    if manifest["asset_order"] != list(dataset.returns.columns):
        raise ValueError("dataset manifest asset order mismatch")
    hashes = {name: entry["sha256"] for name, entry in manifest["files"].items()}
    if hashes != dataset.file_sha256:
        raise ValueError("dataset manifest file hashes mismatch")
    if not native:
        if evidence.get("lineage") is not None:
            lineage = object_json(evidence["lineage"])
            if (
                lineage.get("dataset_kind") != declared_kind
                or lineage.get("training_approved") is not False
            ):
                raise ValueError("synthetic lineage kind or approval mismatch")
            if manifest.get("source_lineage_sha256") != sha256(evidence["lineage"]):
                raise ValueError("synthetic source lineage SHA-256 mismatch")
        return
    if list(dataset.returns.columns) != list(TEAM12_ASSETS):
        raise ValueError("native inputs must use the established Team12 asset order")
    lineage = _check_lineage(object_json(evidence["lineage"]))
    if (
        sha256(evidence["lineage"]) != manifest["source_lineage_sha256"]
        or lineage["dataset_kind"] != declared_kind
        or lineage["asset_order"] != manifest["asset_order"]
        or lineage["source_snapshot"]["manifest_sha256"]
        != manifest["source_snapshot"]["manifest_sha256"]
        or lineage["validation"] != manifest["validation"]
        or lineage["artifact"]["files"] != manifest["source_files"]
    ):
        raise ValueError("source lineage differs from the converted dataset manifest")
    dates = dataset.returns.index.tz_convert("UTC")
    mapping = manifest["date_evaluation_mapping"]
    mapped = pd.DatetimeIndex([_timestamp(entry["evaluation_at"]) for entry in mapping])
    labels = dates.tz_convert("Asia/Seoul")
    if (
        not dates.equals(mapped)
        or [entry["date"] for entry in mapping] != labels.strftime("%Y-%m-%d").tolist()
        or not ((labels.hour == 15) & (labels.minute == 30) & (labels == labels.floor("min"))).all()
    ):
        raise ValueError("dataset index differs from recorded Seoul 15:30 evaluation mapping")
    row_metadata = manifest["row_metadata"]
    rows = pd.read_json(io.BytesIO(json_bytes(row_metadata)), orient="table")
    daily = labels.tz_localize(None).normalize()
    if (
        not rows.index.equals(daily)
        or not rows["feature_ready"].eq(True).all()
        or not rows["warmup"].eq(False).all()
        or not rows["return_interval_contiguous"].eq(True).all()
        or rows["segment_id"].isna().any()
        or rows["segment_id"].nunique() != 1
        or not pd.DatetimeIndex(rows["next_date"].iloc[:-1]).equals(daily[1:])
    ):
        raise ValueError("native row metadata reports warmup or a discontinuous interval")
    window = manifest["windows"][_text(window_id, "window_id")]
    if window["source"] != lineage["windows"][window_id]:
        raise ValueError("converted Window definition differs from source lineage")
    partitions = window["partitions"]
    for name, (start, end) in zip(PERIODS, bounds):
        partition = partitions[name]
        expected = dates[(dates >= start) & (dates < end)]
        if (
            expected.empty
            or partition.get("complete") is not True
            or partition["rows"] != len(expected)
            or _timestamp(partition["first_evaluation_at"]) != expected[0]
            or _timestamp(partition["last_included_evaluation_at"]) != expected[-1]
        ):
            raise ValueError(f"{name} coverage differs from the converted Window partition")
        if start.tz_convert("Asia/Seoul").date().isoformat() != partition["start"] or (
            end.tz_convert("Asia/Seoul").date().isoformat() != partition["end_exclusive"]
        ):
            raise ValueError(f"{name} date bounds differ from the converted Window")


def _check_schedule(calendar, dataset, bounds, kind):
    if type(calendar.get("schema_version")) is not int or calendar["schema_version"] != 1:
        raise ValueError("unsupported calendar schema")
    source = _text(calendar["source"], "calendar source")
    status = calendar["status"]
    allowed = {"synthetic"} if kind == "synthetic" else {"official", "unverified"}
    if status not in allowed:
        raise ValueError("calendar status disagrees with the requested scope")
    if status == "official":
        _text(calendar["evidence_ref"], "official calendar evidence reference")
    start, end = bounds[0][0], bounds[-1][1]
    if (_timestamp(calendar["start"]), _timestamp(calendar["end"])) != (start, end):
        raise ValueError("calendar coverage must equal the requested period bounds")
    expected = pd.DatetimeIndex([_timestamp(value) for value in calendar["evaluation_at"]])
    if (
        expected.empty
        or not expected.is_unique
        or not expected.is_monotonic_increasing
        or expected[0] < start
        or expected[-1] >= end
    ):
        raise ValueError("calendar must contain ordered unique sessions within coverage")
    dates = dataset.returns.index.tz_convert("UTC")
    observed = dates[(dates >= start) & (dates < end)]
    if not observed.equals(expected):
        missing = expected.difference(observed).strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
        extra = observed.difference(expected).strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
        raise ValueError(f"calendar coverage mismatch; missing={missing}, unexpected={extra}")
    return {"source": source, "reported_status": status, "sessions": len(expected)}


def _check_policy(policy, calendar, kind):
    if type(policy.get("schema_version")) is not int or policy["schema_version"] != 1:
        raise ValueError("unsupported evaluation policy record schema")
    version = _text(policy["version"], "evaluation policy version")
    status = policy["status"]
    allowed = {"synthetic"} if kind == "synthetic" else {"draft", "approved", "unverified"}
    if status not in allowed:
        raise ValueError("policy status disagrees with the requested scope")
    if policy["calendar_sha256"] != sha256(calendar):
        raise ValueError("evaluation policy calendar SHA-256 mismatch")
    if status == "approved":
        _text(policy["approval_ref"], "evaluation policy approval reference")
    return {"version": version, "reported_status": status}


def _check_environments(dataset, config, bounds):
    index = dataset.returns.index
    start, validation_start = bounds[0]
    validation_end, test_start = bounds[1][1], bounds[2][0]
    cutoff = pd.Timestamp(config.data.train_end)
    if (
        config.data.train_start is None
        or pd.Timestamp(config.data.train_start) != start
        or config.validation is None
        or pd.Timestamp(config.validation.start) != validation_start
        or not validation_start <= cutoff < test_start
    ):
        raise ValueError("configured cutoffs disagree with train, validation or test boundaries")
    selection = index[(index >= validation_start) & (index < validation_end)]
    if selection.empty or cutoff != selection[-1]:
        raise ValueError("training cutoff must equal the last model-selection session")
    mask = (index >= start) & (index <= cutoff)
    frames = (dataset.returns, dataset.rsi, dataset.macd)
    limited = TrainingDataset(
        *(frame.loc[mask] for frame in frames),
        tuple(tag for tag in dataset.risk_tags if tag.available_at <= cutoff),
        dataset.file_sha256,
    )
    train, validation = split_validation(limited, validation_start, config.env.lookback)
    _make_env(train, config.env, config.seed).close()
    evaluation = evaluation_env_config(config.env)
    _make_env(validation, evaluation, config.seed).close()
    coverage = {}
    for name, (start, end) in zip(PERIODS, bounds):
        subset, requested, missing = _period_dataset(dataset, start, end, config.env.lookback)
        if subset is None or (name != "train" and len(missing)):
            raise ValueError(f"{name} lacks preceding lookback context or a usable return")
        if name == "train":
            # Native training starts inside the partition, even if older input rows exist.
            missing = train.returns.index[: config.env.lookback]
        else:
            if len(requested) < config.env.min_episode_steps:
                raise ValueError(f"{name} is shorter than the configured minimum episode")
            _make_env(subset, evaluation, config.seed).close()
        coverage[name] = {"rows": len(requested), "context_only_rows": len(missing)}
    return coverage


def _check_config_location(root, config, descriptors):
    original = config.data.dataset_dir
    directory = (root / original).resolve()
    for name, descriptor in descriptors.items():
        if root.joinpath(*portable_path(descriptor["path"]).parts) != directory / name:
            raise ValueError(f"training config dataset directory differs from pinned {name}")
    if (directory / RISK_FILE).exists() and RISK_FILE not in descriptors:
        raise ValueError("unlisted risk_tags.parquet would be loaded by training")


def check_inputs(spec, root, *, relocated=False, check_environments=True):
    """Return explicit evidence and technical status; never construct a learner."""
    root = Path(root).resolve()
    kind = spec.get("kind")
    result = {
        "status": "blocked",
        "technical_status": "unavailable",
        "kind": kind,
        "formal_ready": False,
        "training_approved": False,
        "source_status": "unverified",
        "reasons": [],
    }
    checked = CheckedInputs(result, None, None, [], {}, {})
    try:
        if type(spec.get("schema_version")) is not int or spec["schema_version"] != 1:
            raise ValueError("unsupported specification version")
        if kind not in {"synthetic", "research", "financial"}:
            raise ValueError("kind must be synthetic, research or financial")
        _text(spec["currency"], "currency")
        descriptors = spec["dataset"]
        required = set(NATIVE_FILES)
        if set(descriptors) not in (required, required | {RISK_FILE}):
            raise ValueError("dataset must list the native PPO input filenames")
        for key in (
            "training_config",
            "dataset_manifest",
            "calendar",
            "evaluation_policy",
            "lineage",
            "source_audit",
        ):
            if key not in spec and key in {"lineage", "source_audit"} and kind == "synthetic":
                continue
            checked.evidence[key] = pinned(root, spec[key])
        evidence = checked.evidence
        if "source_audit" in evidence:
            audit = object_json(evidence["source_audit"])
            if audit.get("training_approved") is not False:
                raise ValueError("source audit must explicitly record training_approved=false")
            if audit.get("status") not in {"passed", "unverified", "failed"}:
                raise ValueError("source audit must record an explicit result")
            result["source_status"] = audit["status"]
            if audit.get("source_lineage_sha256") is not None and (
                "lineage" not in evidence
                or audit["source_lineage_sha256"] != sha256(evidence["lineage"])
            ):
                raise ValueError("source audit and data lineage SHA-256 mismatch")
            result["source_audit_lineage_binding"] = (
                "hash_matched" if audit.get("source_lineage_sha256") else "operator_supplied"
            )
        for name, descriptor in descriptors.items():
            checked.files[name] = pinned(root, descriptor)
        config = TrainingConfig.model_validate(
            yaml.safe_load(evidence["training_config"]), context={"smoke": kind == "synthetic"}
        )
        checked.config = config
        if not relocated:
            _check_config_location(root, config, descriptors)
        checked.dataset = _load_dataset(checked.files)
        checked.periods, bounds = _periods(spec["periods"])
        _check_dataset_manifest(
            object_json(evidence["dataset_manifest"]),
            checked.dataset,
            kind,
            evidence,
            bounds,
            spec.get("window_id"),
        )
        result["calendar"] = _check_schedule(
            object_json(evidence["calendar"]), checked.dataset, bounds, kind
        )
        result["evaluation_policy"] = _check_policy(
            object_json(evidence["evaluation_policy"]), evidence["calendar"], kind
        )
        if check_environments:
            result["coverage"] = _check_environments(checked.dataset, config, bounds)
        else:
            result["coverage"] = "deferred_until_training_allowed"
        result["assets"] = list(checked.dataset.returns.columns)
        result["data_file_sha256"] = checked.dataset.file_sha256
    except (KeyError, FileNotFoundError) as exc:
        result.update(status="unavailable", technical_status="unavailable")
        result["reasons"].append({"code": "MISSING_INPUT", "detail": str(exc)})
        return checked
    except (ValueError, TypeError, OSError, AttributeError, ImportError, yaml.YAMLError) as exc:
        result["technical_status"] = "failed"
        result["reasons"].append({"code": "INPUT_INVALID", "detail": str(exc)})
        return checked
    result["technical_status"] = "passed"
    result["reasons"].append(
        {"code": "TRAINING_APPROVAL_WITHHELD", "detail": "No supported source approval evidence"}
    )
    result["scope"] = "synthetic_smoke" if kind == "synthetic" else "research_diagnostics"
    if result["source_status"] == "failed":
        result["reasons"].append({"code": "SOURCE_AUDIT_FAILED"})
    elif kind == "synthetic":
        result["status"] = "ready"
    elif result["source_status"] == "unverified":
        result["reasons"].append({"code": "SOURCE_UNVERIFIED"})
    if kind != "synthetic":
        if result["calendar"]["reported_status"] != "official":
            result["reasons"].append({"code": "OFFICIAL_CALENDAR_UNCONFIRMED"})
        if result["evaluation_policy"]["reported_status"] != "approved":
            result["reasons"].append({"code": "EVALUATION_POLICY_UNAPPROVED"})
    return checked
