"""Immutable CollectionResult snapshots with native Parquet and verified lineage.

The manifest is the atomic completion marker. Source verification states are
preserved and never imply training approval. No network or price transformations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from urllib.parse import urlsplit, urlunsplit
import uuid
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_timedelta64_dtype,
)

from .collectors import CollectionResult

FINGERPRINT_FORMAT = "pandas-frame-v1"


def _json_bytes(value):
    return json.dumps(
        value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _name(value):
    if value is not None and not isinstance(value, str):
        raise ValueError("Snapshot index and column level names must be strings or None")
    return value


def _scalar(value):
    """Canonical typed JSON scalar; float hex preserves finite values and -0."""
    if value is None:
        return ["none"]
    if value is pd.NA:
        return ["pd.NA"]
    if value is pd.NaT:
        return ["NaT"]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        return ["float", value.hex()]
    if isinstance(value, str):
        return ["str", value]
    raise ValueError(f"Unsupported snapshot scalar type: {type(value).__name__}")


def _dtype(dtype):
    if isinstance(dtype, pd.DatetimeTZDtype):
        return {"kind": "datetime", "dtype": str(dtype), "tz": str(dtype.tz)}
    if isinstance(dtype, np.dtype):
        if dtype.kind not in "biufmMO":
            raise ValueError(f"Unsupported snapshot dtype: {dtype}")
        if dtype.kind == "f" and dtype.itemsize > 8:
            raise ValueError("Extended-precision floats are not supported")
        return {"kind": "numpy", "dtype": dtype.str}
    if isinstance(dtype, pd.StringDtype):
        # str(dtype) alone omits the Python/Arrow storage distinction.
        return {"kind": "nullable_string", "dtype": str(dtype), "storage": dtype.storage}
    if str(dtype) in {
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
        "Float32",
        "Float64",
        "boolean",
    } and (is_integer_dtype(dtype) or is_float_dtype(dtype) or is_bool_dtype(dtype)):
        return {"kind": "nullable", "dtype": str(dtype)}
    raise ValueError(f"Unsupported snapshot dtype: {dtype}")


def _values(values, dtype):
    if is_datetime64_any_dtype(dtype) or is_timedelta64_dtype(dtype):
        # The dtype records precision and timezone. asi8 also preserves NaT.
        return (["datetime_i8", str(int(value))] for value in values.asi8)
    if isinstance(dtype, np.dtype) and dtype.kind == "O":
        for value in values:
            if value is not None and type(value) is not str:
                raise ValueError("Object snapshots support only Python strings and None")
    return (_scalar(value) for value in values)


def _index(index, *, column_level=False):
    if isinstance(index, pd.MultiIndex):
        if not column_level or index.nlevels != 2 or index.has_duplicates:
            raise ValueError("Only unique two-level Yahoo MultiIndex columns are supported")
        if any((codes < 0).any() for codes in index.codes):
            raise ValueError("Missing MultiIndex column labels are not supported")
        if any(type(value) is not str for level in index.levels for value in level):
            raise ValueError("Yahoo MultiIndex column labels must be strings")
        return {
            "kind": "MultiIndex",
            "names": [_name(name) for name in index.names],
            "levels": [_index(level, column_level=True) for level in index.levels],
            "codes": [codes.tolist() for codes in index.codes],
            "sortorder": index.sortorder,
        }
    info = {"name": _name(index.name), "dtype": _dtype(index.dtype)}
    if isinstance(index, pd.RangeIndex):
        return {
            **info,
            "kind": "RangeIndex",
            "start": index.start,
            "stop": index.stop,
            "step": index.step,
        }
    if isinstance(index, pd.DatetimeIndex):
        return {
            **info,
            "kind": "DatetimeIndex",
            "freq": index.freqstr,
            "values": list(_values(index.array, index.dtype)),
        }
    if type(index) is not pd.Index:
        raise ValueError(f"Unsupported snapshot index: {type(index).__name__}")
    return {**info, "kind": "Index", "values": list(_values(index.array, index.dtype))}


def _schema(frame):
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Snapshot data must be a pandas DataFrame")
    if not frame.columns.is_unique:
        raise ValueError("Duplicate snapshot columns cannot be represented safely in Parquet")
    if not isinstance(frame.columns, pd.MultiIndex) and any(
        type(label) is not str for label in frame.columns
    ):
        raise ValueError("Snapshot column labels must be strings or Yahoo string tuples")
    return {
        "version": FINGERPRINT_FORMAT,
        "shape": list(frame.shape),
        "index": _index(frame.index),
        "columns": _index(frame.columns, column_level=True),
        "dtypes": [_dtype(dtype) for dtype in frame.dtypes],
    }


def _dataframe_fingerprint(frame) -> str:
    """Return SHA256 of canonical structure and typed values, excluding attrs."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Snapshot data must be a pandas DataFrame")
    # Column extraction also propagates/deep-copies attrs in pandas. Start from
    # a plain wrapper so even nonserializable attrs never enter this path.
    frame = pd.DataFrame(frame, copy=False)
    digest = hashlib.sha256()
    digest.update(_json_bytes(_schema(frame)))
    for position in range(len(frame.columns)):
        digest.update(b"\ncolumn:")
        digest.update(str(position).encode("ascii"))
        digest.update(b"\n")
        series = frame.iloc[:, position]
        for value in _values(series.array, series.dtype):
            digest.update(_json_bytes(value))
            digest.update(b"\n")
    return digest.hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_arrow():
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise ImportError("Snapshot Parquet IO requires the project's locked pyarrow") from exc


def _reconstruction(frame):
    index = frame.index
    if isinstance(index, pd.RangeIndex):
        return {"kind": "RangeIndex", "start": index.start, "stop": index.stop, "step": index.step}
    if isinstance(index, pd.DatetimeIndex):
        return {"kind": "DatetimeIndex", "freq": index.freqstr}
    return {"kind": "Index"}


def _restore_index(frame, spec):
    if not isinstance(spec, dict):
        raise ValueError("Snapshot index reconstruction must be a JSON object")
    kind = spec.get("kind")
    if kind == "RangeIndex":
        if set(spec) != {"kind", "start", "stop", "step"} or any(
            type(spec.get(key)) is not int for key in ("start", "stop", "step")
        ):
            raise ValueError("Invalid RangeIndex reconstruction")
        expected = pd.RangeIndex(spec["start"], spec["stop"], spec["step"], name=frame.index.name)
        if len(expected) != len(frame) or not frame.index.equals(expected):
            raise ValueError("Stored index values disagree with RangeIndex reconstruction")
        frame.index = expected
    elif kind == "DatetimeIndex":
        if set(spec) != {"kind", "freq"} or not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError("Invalid DatetimeIndex reconstruction")
        if spec["freq"] is not None and not isinstance(spec["freq"], str):
            raise ValueError("DatetimeIndex frequency must be a string or None")
        frame.index = pd.DatetimeIndex(frame.index, freq=spec["freq"], name=frame.index.name)
    elif kind != "Index" or set(spec) != {"kind"}:
        raise ValueError("Unsupported index reconstruction")
    return frame


def read_frame(path, descriptor) -> pd.DataFrame:
    """Verify byte hash and content fingerprint while restoring pandas metadata."""
    if not isinstance(descriptor, dict) or (
        descriptor.get("format") != "parquet"
        or descriptor.get("fingerprint_format") != FINGERPRINT_FORMAT
    ):
        raise ValueError("Unsupported snapshot Parquet descriptor")
    if _file_sha256(path) != descriptor.get("sha256"):
        raise ValueError("Snapshot Parquet file SHA256 mismatch")
    _require_arrow()
    frame = pd.read_parquet(path, engine="pyarrow")
    frame = _restore_index(frame, descriptor.get("index_reconstruction"))
    columns = descriptor.get("columns_reconstruction")
    if columns is not None:
        if not isinstance(columns, dict) or set(columns) != {"start", "stop", "step", "name"}:
            raise ValueError("Invalid empty columns reconstruction")
        if any(type(columns[key]) is not int for key in ("start", "stop", "step")):
            raise ValueError("Invalid empty columns RangeIndex")
        expected = pd.RangeIndex(
            columns["start"], columns["stop"], columns["step"], name=_name(columns["name"])
        )
        if len(expected) or not frame.columns.equals(expected):
            raise ValueError("Stored columns disagree with empty RangeIndex reconstruction")
        frame.columns = expected
    if list(frame.shape) != [descriptor.get("row_count"), descriptor.get("column_count")]:
        raise ValueError("Snapshot Parquet shape mismatch")
    if _dataframe_fingerprint(frame) != descriptor.get("fingerprint"):
        raise ValueError("Snapshot DataFrame fingerprint mismatch")
    return frame


def write_frame(frame, path) -> dict:
    """Write native typed Parquet only when a strict lossless roundtrip succeeds."""
    fingerprint = _dataframe_fingerprint(frame)
    _require_arrow()
    # DataFrame.copy() deep-copies attrs even for deep=False in pandas 2.3.
    # The plain constructor shares data without copying or persisting attrs.
    clean = pd.DataFrame(frame, copy=False)
    try:
        clean.to_parquet(path, engine="pyarrow", index=True)
        descriptor = {
            "format": "parquet",
            "fingerprint_format": FINGERPRINT_FORMAT,
            "sha256": _file_sha256(path),
            "fingerprint": fingerprint,
            "row_count": len(frame),
            "column_count": len(frame.columns),
            "index_reconstruction": _reconstruction(frame),
            "columns_reconstruction": (
                {
                    "start": frame.columns.start,
                    "stop": frame.columns.stop,
                    "step": frame.columns.step,
                    "name": _name(frame.columns.name),
                }
                if isinstance(frame.columns, pd.RangeIndex)
                else None
            ),
        }
        restored = read_frame(path, descriptor)
        pd.testing.assert_frame_equal(
            clean,
            restored,
            check_dtype=True,
            check_index_type=True,
            check_column_type=True,
            check_names=True,
            check_exact=True,
            check_categorical=True,
            check_freq=True,
        )
    except (AssertionError, TypeError, ValueError, NotImplementedError) as exc:
        raise ValueError("DataFrame cannot be represented losslessly by native Parquet") from exc
    return descriptor


class SnapshotError(ValueError):
    """A snapshot could not be safely stored or read."""


class SnapshotSchemaError(SnapshotError):
    """The input/manifest schema is missing, unsupported or unsafe."""


class SnapshotIntegrityError(SnapshotError):
    """A stored payload does not match its manifest or trusted digest."""


class SnapshotChangedWarning(UserWarning):
    """A new snapshot differs from the supplied previous snapshot."""


@dataclass(frozen=True)
class Snapshot:
    path: Path
    manifest: dict
    datasets: dict[str, CollectionResult]
    manifest_sha256: str


_ROLES = {"etf", "benchmark", "fx", "risk_free"}
_STATUSES = {"passed", "failed", "unverified"}
_KINDS = {
    "data": "data.parquet",
    "raw_data": "raw_data.parquet",
    "raw_text": "raw_text.txt",
    "available_at": "available_at.parquet",
}
_META_FIELDS = {
    "dataset_id",
    "symbol",
    "source",
    "role",
    "market",
    "currency",
    "timezone",
    "unit",
    "asset_class",
    "selection_status",
    "rationale",
    "requested_start",
    "requested_end_exclusive",
    "actual_start",
    "actual_end",
    "row_count",
    "collected_at",
    "auto_adjust",
    "price_basis",
    "availability_status",
    "issues",
    "status",
    "source_url",
    "fx_direction",
    "fx_direction_status",
    "fx_quote_evidence",
}
_ENTRY_REQUIRED = {
    "dataset_id",
    "ticker",
    "source",
    "role",
    "market",
    "currency",
    "collected_at",
    "actual_start",
    "actual_end",
    "row_count",
    "requested_start",
    "requested_end_exclusive",
    "price_basis",
    "auto_adjust",
    "status",
    "validation_status",
    "validation",
    "unverified_reasons",
    "availability_status",
    "has_available_at",
    "files",
    "metadata",
}
_SECRET_KEY = re.compile(r"api.?key|token|password|authorization|secret|credential|cookie", re.I)
_SECRET_VALUE = re.compile(
    r"(?i)(?:bearer\s+\S+)|"
    r"(?:(?:api[_-]?key|token|password|secret|authorization)\s*[=:]\s*[^\s&,;]+)"
)
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def dataframe_fingerprint(frame) -> str:
    """Hash ordered values, missingness and supported pandas schema; exclude attrs."""
    try:
        return _dataframe_fingerprint(frame)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SnapshotSchemaError("Unsupported DataFrame fingerprint schema") from exc


def _safe_name(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value):
        raise SnapshotSchemaError(f"Unsafe {label}")
    reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    if value.endswith(".") or value.split(".")[0].upper() in reserved:
        raise SnapshotSchemaError(f"Unsafe {label}")
    return value


def _safe_json(value):
    """Keep JSON metadata only, removing credential keys and known credential syntax."""
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise SnapshotSchemaError("Metadata keys must be strings")
        return {key: _safe_json(item) for key, item in value.items() if not _SECRET_KEY.search(key)}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, str):
        value = _SECRET_VALUE.sub("[REDACTED]", value)

        # Strip credentials, query strings and fragments from URLs even inside issue messages.
        def clean_url(match):
            try:
                parts = urlsplit(match.group())
                host = parts.hostname or ""
                if parts.port:
                    host += f":{parts.port}"
                return urlunsplit((parts.scheme, host, parts.path, "", ""))
            except ValueError:
                return "[REDACTED_URL]"

        return re.sub(r"https?://[^\s<>]+", clean_url, value)
    raise SnapshotSchemaError("Metadata must contain finite JSON values only")


def _utc(value, label, *, optional=False):
    if value is None and optional:
        return
    try:
        if not isinstance(value, str):
            raise ValueError("ISO timestamp required")
        stamp = pd.Timestamp(value)
        if pd.isna(stamp) or stamp.tzinfo is None or stamp.utcoffset().total_seconds() != 0:
            raise ValueError("UTC timestamp required")
    except (ValueError, TypeError, OverflowError) as exc:
        raise SnapshotSchemaError(f"{label} must be an ISO UTC timestamp") from exc


def _daily(value, label, *, optional=False):
    if value is None and optional:
        return
    try:
        if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
            raise ValueError("ISO date required")
    except (TypeError, ValueError) as exc:
        raise SnapshotSchemaError(f"{label} must be an ISO date") from exc


def _metadata(result):
    if not isinstance(result, CollectionResult) or not isinstance(result.metadata, dict):
        raise SnapshotSchemaError("Expected CollectionResult objects")
    metadata = _safe_json(
        {key: value for key, value in result.metadata.items() if key in _META_FIELDS}
    )
    required = (
        _ENTRY_REQUIRED
        - {
            "ticker",
            "validation_status",
            "validation",
            "unverified_reasons",
            "has_available_at",
            "files",
            "metadata",
        }
    ) | {"symbol", "issues"}
    if not required.issubset(metadata):
        raise SnapshotSchemaError("Collection metadata is missing required fields")
    _safe_name(metadata["dataset_id"], "dataset_id")
    if metadata["role"] not in _ROLES or metadata["status"] not in _STATUSES:
        raise SnapshotSchemaError("Unsupported role or collection status")
    if metadata["availability_status"] not in {"verified", "unverified", "failed"}:
        raise SnapshotSchemaError("Invalid availability status")
    if type(metadata["row_count"]) is not int or metadata["row_count"] < 0:
        raise SnapshotSchemaError("row_count must be a nonnegative integer")
    if metadata["auto_adjust"] is not None and type(metadata["auto_adjust"]) is not bool:
        raise SnapshotSchemaError("auto_adjust must be bool or None")
    for key in ("symbol", "source", "market", "currency", "price_basis"):
        if not isinstance(metadata[key], str) or not metadata[key].strip():
            raise SnapshotSchemaError(f"{key} must be a nonempty string")
    _utc(metadata["collected_at"], "collected_at", optional=True)
    for key in ("actual_start", "actual_end", "requested_start", "requested_end_exclusive"):
        _daily(metadata[key], key, optional=key.startswith("actual_"))
    if metadata["requested_start"] >= metadata["requested_end_exclusive"]:
        raise SnapshotSchemaError("Requested period must have an exclusive end after start")
    if (metadata["actual_start"] is None) != (metadata["actual_end"] is None):
        raise SnapshotSchemaError("Actual period endpoints must both be absent or present")
    if metadata["actual_start"] is not None and metadata["actual_start"] > metadata["actual_end"]:
        raise SnapshotSchemaError("Actual period endpoints are reversed")
    if not isinstance(metadata["issues"], list):
        raise SnapshotSchemaError("issues must be a list")
    return metadata


def _write_frame(frame, path):
    try:
        return write_frame(frame, path)
    except ImportError as exc:
        raise SnapshotError(
            "Parquet IO requires the configured pyarrow test/runtime dependency"
        ) from exc
    except (TypeError, ValueError, NotImplementedError) as exc:
        raise SnapshotSchemaError(
            "DataFrame cannot be stored losslessly as native Parquet"
        ) from exc


def _read_frame(path, descriptor):
    try:
        return read_frame(path, descriptor)
    except ImportError as exc:
        raise SnapshotError(
            "Parquet IO requires the configured pyarrow test/runtime dependency"
        ) from exc
    except Exception as exc:
        raise SnapshotIntegrityError("Parquet payload cannot be verified or decoded") from exc


def _check_data_metadata(frame, metadata):
    if metadata["row_count"] != len(frame):
        raise SnapshotSchemaError("Source row_count differs from normalized data")
    if metadata["actual_start"] is not None:
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.hasnans or frame.empty:
            raise SnapshotSchemaError("Actual period requires complete valid daily labels")
        dates = frame.index
        if dates.tz is not None and metadata.get("timezone"):
            dates = dates.tz_convert(metadata["timezone"])
        if (dates.min().date().isoformat(), dates.max().date().isoformat()) != (
            metadata["actual_start"],
            metadata["actual_end"],
        ):
            raise SnapshotSchemaError("Source actual period differs from normalized data")


def _prepare(results):
    if not isinstance(results, Mapping) or not results:
        raise SnapshotSchemaError("results must be a nonempty mapping")
    prepared, seen = [], set()
    for dataset_id, result in results.items():
        _safe_name(dataset_id, "dataset_id")
        if dataset_id.casefold() in seen:
            raise SnapshotSchemaError("Dataset IDs collide under case-insensitive filesystems")
        seen.add(dataset_id.casefold())
        metadata = _metadata(result)
        if metadata["dataset_id"] != dataset_id:
            raise SnapshotSchemaError("Mapping ID and source dataset_id differ")
        dataframe_fingerprint(result.data)
        _check_data_metadata(result.data, metadata)
        if result.raw_data is not None:
            dataframe_fingerprint(result.raw_data)
        if result.raw_text is not None and not isinstance(result.raw_text, str):
            raise SnapshotSchemaError("raw_text must be a string or None")
        available = result.metadata.get("available_at")
        if available is None and metadata["availability_status"] == "verified":
            raise SnapshotSchemaError("Verified availability requires the supplied Series")
        if available is not None:
            if not isinstance(available, pd.Series) or (
                available.name is not None and not isinstance(available.name, str)
            ):
                raise SnapshotSchemaError("available_at must be a Series with a string/None name")
            dataframe_fingerprint(available.to_frame(name="available_at"))
        validation = _safe_json(result.validation)
        if not isinstance(validation, dict):
            raise SnapshotSchemaError("validation must be a mapping")
        prepared.append((result, metadata, validation, available))
    return prepared


def _entry(stage, prepared):
    result, metadata, validation, available = prepared
    relative = Path(metadata["role"]) / metadata["dataset_id"]
    directory = stage / relative
    directory.mkdir(parents=True)
    files = {key: None for key in _KINDS}
    for kind, frame in (
        ("data", result.data),
        ("raw_data", result.raw_data),
        (
            "available_at",
            available.to_frame(name="available_at") if available is not None else None,
        ),
    ):
        if frame is None:
            continue
        path = directory / _KINDS[kind]
        files[kind] = {**_write_frame(frame, path), "path": (relative / path.name).as_posix()}
        if kind == "available_at":
            files[kind]["series_name"] = available.name
    if result.raw_text is not None:
        path = directory / _KINDS["raw_text"]
        path.write_bytes(result.raw_text.encode("utf-8"))
        digest = _file_sha256(path)
        files["raw_text"] = {
            "path": (relative / path.name).as_posix(),
            "format": "utf-8",
            "sha256": digest,
            "fingerprint": digest,
        }
    fields = {key: metadata[key] for key in _ENTRY_REQUIRED if key in metadata}
    return {
        **fields,
        "ticker": metadata["symbol"],
        "metadata": metadata,
        "validation_status": metadata["status"],
        "validation": validation,
        "unverified_reasons": [
            item
            for item in metadata["issues"]
            if isinstance(item, dict) and item.get("severity") == "unverified"
        ],
        "has_available_at": available is not None,
        "files": files,
    }


def _remove_owned(path, root):
    # Recursive deletion is limited to an exclusively created direct child of this root.
    if path.is_symlink() or path.resolve().parent != root.resolve():
        raise SnapshotError("Refusing cleanup outside the snapshot root")
    if path.exists():
        shutil.rmtree(path)


def write_snapshot(results, root="data/raw/snapshots", *, snapshot_id=None, previous=None) -> Path:
    """Save a new immutable snapshot; publish manifest only after every payload succeeds."""
    if snapshot_id is None:
        snapshot_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ_") + uuid.uuid4().hex
    snapshot_id = _safe_name(snapshot_id, "snapshot_id")
    prepared = _prepare(results)
    previous_manifest = _as_snapshot(previous).manifest if previous is not None else None
    root = Path(root).absolute()
    if root.is_symlink():
        raise SnapshotSchemaError("Snapshot root cannot be a symbolic link")
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SnapshotError("Snapshot root is inaccessible or is not a directory") from exc
    root = root.resolve()
    final = root / snapshot_id
    lock = root / f".snapshot-{snapshot_id.casefold()}.lock"
    stage, owns_final, owns_lock = None, False, False
    manifest = None
    try:
        if any(child.name.casefold() == snapshot_id.casefold() for child in root.iterdir()):
            raise SnapshotError("snapshot_id already exists; overwriting is forbidden")
        with lock.open("x", encoding="utf-8") as handle:
            owns_lock = True
            handle.write(snapshot_id)
        if final.exists() or final.is_symlink():
            raise SnapshotError("snapshot_id already exists")
        stage = Path(tempfile.mkdtemp(prefix=f".staging-{snapshot_id}-", dir=root))
        manifest = {
            "schema_version": 1,
            "snapshot_id": snapshot_id,
            "snapshot_created_at": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "fingerprint_algorithm": FINGERPRINT_FORMAT,
            "datasets": [_entry(stage, item) for item in prepared],
            "comparison": None,
        }
        if previous_manifest is not None:
            manifest["comparison"] = compare_manifests(previous_manifest, manifest)
        _validate_manifest(manifest)
        payload = _json_bytes(manifest)
        with (stage / "manifest.json").open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # Exclusive mkdir is portable and never replaces even an existing empty directory.
        # Until the last atomic manifest rename, read_snapshot rejects the incomplete directory.
        final.mkdir(exist_ok=False)
        owns_final = True
        for child in stage.iterdir():
            if child.name != "manifest.json":
                child.rename(final / child.name)
        (stage / "manifest.json").rename(final / "manifest.json")
        stage.rmdir()
        stage = None
    except Exception as exc:
        if owns_final:
            _remove_owned(final, root)
        if stage is not None:
            _remove_owned(stage, root)
        if isinstance(exc, SnapshotError):
            raise
        raise SnapshotError("Snapshot write failed; no complete snapshot was published") from exc
    finally:
        if owns_lock:
            lock.unlink()
    comparison = manifest["comparison"]
    if comparison and (comparison["changed"] or comparison["added"] or comparison["removed"]):
        warnings.warn(
            f"Snapshot source data changed: {json.dumps(comparison, sort_keys=True)}",
            SnapshotChangedWarning,
            stacklevel=2,
        )
    return final


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SnapshotSchemaError("Duplicate manifest JSON key")
        result[key] = value
    return result


def _validate_manifest(manifest):
    required = {
        "schema_version",
        "snapshot_id",
        "snapshot_created_at",
        "status",
        "fingerprint_algorithm",
        "datasets",
    }
    if not isinstance(manifest, dict) or not required.issubset(manifest):
        raise SnapshotSchemaError("Manifest is missing required fields")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise SnapshotSchemaError("Unsupported snapshot schema_version")
    _safe_name(manifest["snapshot_id"], "snapshot_id")
    _utc(manifest["snapshot_created_at"], "snapshot_created_at")
    if manifest["status"] != "complete" or manifest["fingerprint_algorithm"] != FINGERPRINT_FORMAT:
        raise SnapshotSchemaError(
            "Snapshot is incomplete or uses an unsupported fingerprint format"
        )
    if not isinstance(manifest["datasets"], list) or not manifest["datasets"]:
        raise SnapshotSchemaError("Manifest datasets must be a nonempty list")
    seen = set()
    for entry in manifest["datasets"]:
        if not isinstance(entry, dict) or not _ENTRY_REQUIRED.issubset(entry):
            raise SnapshotSchemaError("Dataset manifest is missing required fields")
        dataset_id = _safe_name(entry["dataset_id"], "dataset_id")
        if dataset_id.casefold() in seen:
            raise SnapshotSchemaError("Duplicate/colliding manifest dataset IDs")
        seen.add(dataset_id.casefold())
        if not isinstance(entry["metadata"], dict):
            raise SnapshotSchemaError("Manifest source metadata must be a mapping")
        source = CollectionResult(pd.DataFrame(), None, entry["metadata"], entry["validation"])
        metadata = _metadata(source)
        if metadata != entry["metadata"]:
            raise SnapshotSchemaError("Manifest metadata contains unsupported or secret fields")
        for key in _ENTRY_REQUIRED & metadata.keys():
            if entry[key] != metadata[key]:
                raise SnapshotSchemaError("Manifest/source metadata disagree")
        if (
            entry["ticker"] != metadata["symbol"]
            or entry["validation_status"] != metadata["status"]
        ):
            raise SnapshotSchemaError("Manifest/source ticker or validation status disagree")
        if not isinstance(entry["validation"], dict):
            raise SnapshotSchemaError("Manifest validation must be a mapping")
        if _safe_json(entry["validation"]) != entry["validation"]:
            raise SnapshotSchemaError("Manifest validation contains unsupported or secret metadata")
        expected_reasons = [
            item
            for item in metadata["issues"]
            if isinstance(item, dict) and item.get("severity") == "unverified"
        ]
        if entry["unverified_reasons"] != expected_reasons:
            raise SnapshotSchemaError("Unverified reasons disagree with source findings")
        if type(entry["has_available_at"]) is not bool:
            raise SnapshotSchemaError("has_available_at must be boolean")
        files = entry["files"]
        if not isinstance(files, dict) or set(files) != set(_KINDS) or files["data"] is None:
            raise SnapshotSchemaError("Manifest must explicitly describe all payload kinds")
        if entry["has_available_at"] != (files["available_at"] is not None):
            raise SnapshotSchemaError("Historical availability presence and payload disagree")
        if entry["availability_status"] == "verified" and files["available_at"] is None:
            raise SnapshotSchemaError("Verified availability requires an available_at payload")
        for kind, descriptor in files.items():
            if descriptor is None:
                continue
            if not isinstance(descriptor, dict):
                raise SnapshotSchemaError("File descriptor must be an object")
            expected = f"{entry['role']}/{dataset_id}/{_KINDS[kind]}"
            if descriptor.get("path") != expected:
                raise SnapshotSchemaError("Invalid/noncanonical payload path")
            for key in ("sha256", "fingerprint"):
                if not isinstance(descriptor.get(key), str) or not _DIGEST.fullmatch(
                    descriptor[key]
                ):
                    raise SnapshotSchemaError("Invalid SHA-256 or fingerprint")
            if descriptor.get("format") != ("utf-8" if kind == "raw_text" else "parquet"):
                raise SnapshotSchemaError("Invalid payload format")
            if kind != "raw_text" and descriptor.get("fingerprint_format") != FINGERPRINT_FORMAT:
                raise SnapshotSchemaError("Invalid payload fingerprint version")
    return manifest


def _payload_path(root, relative):
    candidate = root
    for part in Path(relative).parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise SnapshotIntegrityError("Symbolic links are not valid snapshot payloads")
    if not candidate.resolve().is_relative_to(root.resolve()) or not candidate.is_file():
        raise SnapshotIntegrityError("Snapshot payload is missing or outside its snapshot")
    return candidate


def read_snapshot(path, *, expected_manifest_sha256=None) -> Snapshot:
    """Verify a complete manifest, all file hashes and all DataFrame content fingerprints."""
    path = Path(path).absolute()
    if path.name == "manifest.json" and path.is_file():
        path = path.parent
    if path.is_symlink() or not path.is_dir():
        raise SnapshotIntegrityError("Snapshot directory is missing or symbolic")
    path = path.resolve()
    manifest_path = _payload_path(path, "manifest.json")
    manifest_hash = _file_sha256(manifest_path)
    if expected_manifest_sha256 is not None and manifest_hash != expected_manifest_sha256:
        raise SnapshotIntegrityError("Manifest SHA-256 differs from its trusted digest")
    try:
        manifest = _validate_manifest(
            json.loads(
                manifest_path.read_text(encoding="utf-8"),
                object_pairs_hook=_unique_object,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    SnapshotSchemaError("Nonfinite JSON")
                ),
            )
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SnapshotSchemaError("Snapshot manifest is missing or invalid JSON") from exc
    if manifest["snapshot_id"] != path.name:
        raise SnapshotSchemaError("Snapshot directory and manifest ID disagree")
    expected_paths = {"manifest.json"}
    verified = {}
    # Verify every payload before decoding any file; never expose a partially verified result.
    for entry in manifest["datasets"]:
        for kind, descriptor in entry["files"].items():
            if descriptor is None:
                continue
            file = _payload_path(path, descriptor["path"])
            if _file_sha256(file) != descriptor["sha256"]:
                raise SnapshotIntegrityError("Snapshot payload SHA-256 mismatch")
            expected_paths.add(descriptor["path"])
            verified[(entry["dataset_id"], kind)] = file
    for item in path.rglob("*"):
        if item.is_symlink():
            raise SnapshotIntegrityError("Snapshot includes a symbolic link")
        if item.is_file() and item.relative_to(path).as_posix() not in expected_paths:
            raise SnapshotIntegrityError("Snapshot contains an unlisted payload")
    datasets = {}
    for entry in manifest["datasets"]:
        frames, raw_text = {}, None
        for kind, descriptor in entry["files"].items():
            if descriptor is None:
                frames[kind] = None
                continue
            file = verified[(entry["dataset_id"], kind)]
            if kind == "raw_text":
                payload = file.read_bytes()
                if hashlib.sha256(payload).hexdigest() != descriptor["fingerprint"]:
                    raise SnapshotIntegrityError("Raw text fingerprint mismatch")
                try:
                    raw_text = payload.decode("utf-8")
                except UnicodeError as exc:
                    raise SnapshotIntegrityError("Raw text is not UTF-8") from exc
            else:
                frames[kind] = _read_frame(file, descriptor)
        metadata = dict(entry["metadata"])
        try:
            _check_data_metadata(frames["data"], metadata)
        except SnapshotSchemaError as exc:
            raise SnapshotIntegrityError(
                "Source period/row count differs from stored data"
            ) from exc
        metadata["available_at"] = None
        if frames.get("available_at") is not None:
            available = frames["available_at"]
            descriptor = entry["files"]["available_at"]
            if tuple(available.columns) != ("available_at",) or "series_name" not in descriptor:
                raise SnapshotSchemaError("Invalid available_at Series descriptor")
            if descriptor["series_name"] is not None and not isinstance(
                descriptor["series_name"], str
            ):
                raise SnapshotSchemaError("Invalid available_at name")
            metadata["available_at"] = available.iloc[:, 0].rename(descriptor["series_name"])
        datasets[entry["dataset_id"]] = CollectionResult(
            frames["data"], frames.get("raw_data"), metadata, entry["validation"], raw_text
        )
    return Snapshot(path, manifest, datasets, manifest_hash)


def _as_snapshot(value):
    if isinstance(value, Snapshot):
        return read_snapshot(value.path, expected_manifest_sha256=value.manifest_sha256)
    return read_snapshot(value)


def compare_manifests(previous, new) -> dict:
    """Compare logical contents and declared quality/provenance; do not verify disk files."""
    previous, new = _validate_manifest(previous), _validate_manifest(new)
    before = {entry["dataset_id"]: entry for entry in previous["datasets"]}
    after = {entry["dataset_id"]: entry for entry in new["datasets"]}
    result = {
        "unchanged": [],
        "added": sorted(after.keys() - before.keys()),
        "removed": sorted(before.keys() - after.keys()),
        "changed": [],
    }
    for dataset_id in sorted(before.keys() & after.keys()):
        old, current = before[dataset_id], after[dataset_id]
        changes, fingerprints = [], {}
        for kind in _KINDS:
            left = (old["files"][kind] or {}).get("fingerprint")
            right = (current["files"][kind] or {}).get("fingerprint")
            fingerprints[kind] = {"previous": left, "new": right}
            if left != right:
                changes.append(f"{kind}_fingerprint")
        groups = {
            "row_count": ("row_count",),
            "date_range": (
                "actual_start",
                "actual_end",
                "requested_start",
                "requested_end_exclusive",
            ),
            "status": ("status",),
            "provider": ("source", "ticker", "role", "market"),
            "price_definition": ("price_basis", "auto_adjust", "currency"),
            "availability": ("availability_status", "has_available_at"),
            "validation": ("validation", "unverified_reasons"),
        }
        for code, keys in groups.items():
            if any(old[key] != current[key] for key in keys):
                changes.append(code)
        for code, keys in {
            "provider": ("source_url", "timezone"),
            "price_definition": (
                "unit",
                "fx_direction",
                "fx_direction_status",
                "fx_quote_evidence",
            ),
            "validation": ("issues", "selection_status"),
        }.items():
            if code not in changes and any(
                old["metadata"].get(key) != current["metadata"].get(key) for key in keys
            ):
                changes.append(code)
        if changes:
            result["changed"].append(
                {
                    "dataset_id": dataset_id,
                    "changes": changes,
                    "previous_fingerprint": old["files"]["data"]["fingerprint"],
                    "new_fingerprint": current["files"]["data"]["fingerprint"],
                    "fingerprints": fingerprints,
                }
            )
        else:
            result["unchanged"].append(dataset_id)
    return result


def compare_snapshots(previous, new) -> dict:
    """Re-read and verify both snapshots, then compare manifests without modifying either."""
    return compare_manifests(_as_snapshot(previous).manifest, _as_snapshot(new).manifest)
