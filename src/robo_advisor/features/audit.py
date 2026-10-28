"""Read-only integrity, numerical and leakage audits for team12 artifacts.

The explicitly versioned experimental schema is inspected without weakening the
strict #15 reader or source readiness gate. No audit grants training approval.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

from robo_advisor.data.snapshot import read_frame
from .dataset import TEAM12_ASSETS, WARMUP_ROWS
from .dataset_io import read_feature_dataset
from .indicators import calculate_macd, calculate_rsi
from .scaling import TrainOnlyStandardScaler

_EXPERIMENTAL = "experimental-real-team12-v1"
_KINDS = ("log_return", "rsi14", "macd", "macd_signal", "macd_histogram", "volatility20")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_FRAME = re.compile(
    r"(?:features|rewards|rows|windows/[A-Za-z0-9_-]+/(?:features|rewards|splits))\Z"
)


def _issue(report, code, message, *, severity="error", **details):
    report["issues"].append({"code": code, "severity": severity, "message": message, **details})


def _finish(report):
    severities = {issue["severity"] for issue in report["issues"]}
    report["status"] = (
        "failed"
        if "error" in severities
        else "unverified" if "unverified" in severities else "passed"
    )
    return report


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate lineage JSON key")
        result[key] = value
    return result


def _safe_path(path):
    if any(candidate.is_symlink() for candidate in (path, *path.parents)):
        raise ValueError("Artifact paths must not traverse symlinks")


def _load(path, expected, report):
    root = Path(path).absolute()
    if root.name == "lineage.json":
        root = root.parent
    _safe_path(root)
    lineage_path = root / "lineage.json"
    _safe_path(lineage_path)
    digest = _sha(lineage_path)
    report["metrics"]["lineage_sha256"] = digest
    if expected is None:
        _issue(
            report,
            "LINEAGE_TRUST_ANCHOR_MISSING",
            "Internal hashes do not authenticate lineage without an external trusted digest.",
            severity="unverified",
        )
    elif not isinstance(expected, str) or not _DIGEST.fullmatch(expected):
        raise ValueError("expected_lineage_sha256 must be a lowercase SHA-256 or None")
    elif digest != expected:
        raise ValueError("Feature lineage SHA-256 mismatch")
    lineage = json.loads(
        lineage_path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")),
    )
    if not isinstance(lineage, dict):
        raise ValueError("Lineage must be an object")
    if type(lineage.get("schema_version")) is int and lineage["schema_version"] == 1:
        loaded = read_feature_dataset(root, expected_lineage_sha256=expected)
        return loaded["frames"], loaded["lineage"], False
    if lineage.get("schema_version") != _EXPERIMENTAL:
        raise ValueError("Unsupported feature lineage schema")
    if lineage.get("dataset_kind") != "experimental_real":
        raise ValueError("Experimental schema requires experimental_real dataset kind")
    descriptors = lineage.get("files")
    if (
        not isinstance(descriptors, dict)
        or not {"features", "rewards", "rows"} <= descriptors.keys()
    ):
        raise ValueError("Experimental lineage requires features, rewards and rows")
    frames, paths = {}, {"lineage.json"}
    root = root.resolve()
    for name, descriptor in descriptors.items():
        if not isinstance(name, str) or not _FRAME.fullmatch(name):
            raise ValueError("Unsupported experimental frame name")
        relative = f"{name}.parquet"
        if not isinstance(descriptor, dict) or descriptor.get("path") != relative:
            raise ValueError("Experimental frame path must be canonical")
        candidate = root / relative
        _safe_path(candidate)
        if not candidate.resolve().is_relative_to(root):
            raise ValueError("Feature payload escaped artifact directory")
        frames[name] = read_frame(candidate, descriptor)
        paths.add(relative)
    actual = set()
    for child in root.rglob("*"):
        if child.is_symlink():
            raise ValueError("Artifact contains a symlink")
        if child.is_file():
            actual.add(child.relative_to(root).as_posix())
    if actual != paths:
        raise ValueError("Artifact has unlisted or missing files")
    return frames, lineage, True


def _index(index):
    return (
        isinstance(index, pd.DatetimeIndex)
        and not index.empty
        and index.tz is None
        and not index.hasnans
        and index.is_unique
        and index.is_monotonic_increasing
        and index.equals(index.normalize())
    )


def _numeric(frame):
    return all(
        is_numeric_dtype(dtype) and not is_bool_dtype(dtype) and not is_complex_dtype(dtype)
        for dtype in frame.dtypes
    )


def _times(series, field, report, *, missing=False):
    values = []
    for value in series:
        if pd.isna(value) and missing:
            values.append(pd.NaT)
            continue
        try:
            if isinstance(value, (int, float, bool)):
                raise ValueError("Timestamp required")
            timestamp = pd.Timestamp(value)
            if pd.isna(timestamp) or timestamp.tzinfo is None:
                raise ValueError("Aware timestamp required")
            values.append(timestamp.tz_convert("UTC"))
        except (ValueError, TypeError, OverflowError):
            _issue(report, "INVALID_TIMESTAMP", "Expected an aware timestamp.", field=field)
            return None
    return pd.Series(pd.DatetimeIndex(values, tz="UTC"), index=series.index)


def _different(actual, expected):
    left, right = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    return left.shape != right.shape or not np.allclose(
        left, right, rtol=1e-9, atol=1e-11, equal_nan=True
    )


def _frame_checks(frames, lineage, experimental, report):
    features, rewards, rows = (frames[name] for name in ("features", "rewards", "rows"))
    if any(
        not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique
        for frame in (features, rewards, rows)
    ):
        raise ValueError("Audit frames require DataFrames with unique columns")
    version = "experimental-observed-team12-price-return-v1" if experimental else "team12-causal-v1"
    if lineage.get("feature_version") != version:
        raise ValueError("Unsupported numerical feature definition version")
    kinds = tuple(
        "macd_hist" if experimental and name == "macd_histogram" else name for name in _KINDS
    )
    columns = [f"{asset}__{kind}" for asset in TEAM12_ASSETS for kind in kinds]
    if (
        features.columns.tolist() != columns
        or lineage.get("feature_columns") != columns
        or lineage.get("asset_order") != list(TEAM12_ASSETS)
        or lineage.get("reward_columns") != list(TEAM12_ASSETS)
        or rewards.columns.tolist() != list(TEAM12_ASSETS)
    ):
        raise ValueError("Canonical 12-asset and 72-feature column order is required")
    if not all(_index(frame.index) for frame in (features, rewards, rows)):
        raise ValueError("Frames require ordered unique naive daily indexes")
    if (
        not features.index.equals(rewards.index)
        or not features.index.isin(rows.index).all()
        or (not experimental and not features.index.equals(rows.index))
    ):
        raise ValueError("Feature, reward and row metadata indexes disagree")
    if not _numeric(features) or not _numeric(rewards):
        raise ValueError("Features and rewards must be real numeric values")
    report["metrics"].update(
        feature_rows=len(features),
        feature_count=len(features.columns),
        row_metadata_rows=len(rows),
        asset_order=list(TEAM12_ASSETS),
        missing_feature_cells=int(features.isna().sum().sum()),
        missing_feature_ratio=float(features.isna().to_numpy().mean()),
        missing_reward_cells=int(rewards.isna().sum().sum()),
        dataset_kind=lineage.get("dataset_kind"),
        schema_version=lineage.get("schema_version"),
    )
    if (
        np.isinf(features.to_numpy(dtype=float)).any()
        or np.isinf(rewards.to_numpy(dtype=float)).any()
    ):
        _issue(report, "NONFINITE_FEATURE_OR_REWARD", "Infinity must not enter feature artifacts.")
    eligible_name = "feature_eligible" if experimental else "feature_ready"
    position_name = "segment_row" if experimental else "warmup_position"
    required = {"evaluation_at", eligible_name, position_name, "segment_id", "reward_known"}
    required.add("return_start_date" if experimental else "return_interval_contiguous")
    if not required <= set(rows):
        raise ValueError("Required row chronology and warmup evidence is missing")
    if any(not is_bool_dtype(rows[name].dtype) for name in (eligible_name, "reward_known")):
        raise ValueError("Eligibility and reward-known fields must be boolean")
    eligible = rows[eligible_name]
    if experimental:
        if not rows.index[eligible].equals(features.index) or features.isna().any().any():
            _issue(
                report,
                "FEATURE_ELIGIBILITY_MISMATCH",
                "Feature rows must equal eligible finite rows.",
            )
    else:
        finite = pd.Series(np.isfinite(features.to_numpy()).all(axis=1), index=features.index)
        if not np.array_equal(eligible, finite):
            _issue(
                report,
                "FEATURE_ELIGIBILITY_MISMATCH",
                "Feature-ready flags disagree with finite values.",
            )
    if experimental:
        starts = pd.to_datetime(rows["return_start_date"], errors="raise")
        if starts.dt.tz is not None or not starts.dropna().eq(starts.dropna().dt.normalize()).all():
            raise ValueError("Return start dates must be naive daily dates")
        observation_dates = pd.Series(rows.index, index=rows.index)
        if (starts.notna() & starts.ge(observation_dates)).any():
            _issue(
                report,
                "INVALID_RETURN_INTERVAL",
                "Return intervals must start strictly before their evaluation date.",
            )
        contiguous = starts.eq(observation_dates.shift(1))
    else:
        if not is_bool_dtype(rows["return_interval_contiguous"].dtype):
            raise ValueError("Return-contiguous flags must be boolean")
        contiguous = rows["return_interval_contiguous"].copy()
    contiguous.iloc[0] = False
    groups = (~contiguous).cumsum()
    position = groups.groupby(groups).cumcount()
    actual_group_changes = rows["segment_id"].ne(rows["segment_id"].shift())
    if not np.array_equal(actual_group_changes, ~contiguous):
        _issue(
            report, "SEGMENT_BOUNDARY_MISMATCH", "Segment changes disagree with return continuity."
        )
    if not np.array_equal(rows[position_name].to_numpy(), position.to_numpy()):
        _issue(report, "WARMUP_POSITION_MISMATCH", "Warmup positions must restart after every gap.")
    warmup = position < WARMUP_ROWS
    if (eligible & warmup).any():
        _issue(
            report, "WARMUP_ROW_EXPOSED", "Initial 33 observations per segment cannot be features."
        )
    if not experimental and features.loc[warmup].notna().any().any():
        _issue(report, "WARMUP_ROW_EXPOSED", "Warmup rows must retain missing feature values.")
    expected_known = contiguous.shift(-1, fill_value=False)
    if not np.array_equal(rows["reward_known"], expected_known):
        _issue(
            report,
            "REWARD_CONTINUITY_MISMATCH",
            "Known rewards must end in the next contiguous interval.",
        )
    unknown = ~expected_known.reindex(rewards.index)
    if rewards.loc[unknown].notna().any().any():
        _issue(report, "UNKNOWN_REWARD_EXPOSED", "Final or gap-crossing rewards must be missing.")
    evaluation = _times(rows["evaluation_at"], "evaluation_at", report)
    if evaluation is not None:
        evaluation_days = pd.DatetimeIndex(
            evaluation.dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
        ).normalize()
        if not evaluation_days.equals(rows.index):
            _issue(
                report,
                "EVALUATION_DATE_MISMATCH",
                "Evaluation timestamps must belong to their Korean daily label.",
            )
        if not evaluation.is_monotonic_increasing:
            _issue(report, "EVALUATION_ORDER_INVALID", "Evaluation timestamps must be increasing.")
        for asset in TEAM12_ASSETS:
            date_field = f"{asset}__source_dates"
            if date_field not in rows:
                raise ValueError("Asset source-date evidence is missing")
            dates = pd.to_datetime(rows[date_field], errors="raise")
            if (
                dates.isna().any()
                or dates.dt.tz is not None
                or not dates.eq(dates.dt.normalize()).all()
            ):
                raise ValueError("Asset source dates must be complete naive daily dates")
            day = pd.Series(rows.index, index=rows.index)
            future = dates.ge(day) if asset in TEAM12_ASSETS[:10] else dates.gt(day)
            if future.any():
                _issue(
                    report,
                    "FUTURE_SOURCE_DATE",
                    "Source date violates market-lag chronology.",
                    asset=asset,
                )
            fields = (
                ("actual_available_at", "assumed_usable_at") if experimental else ("available_at",)
            )
            for suffix in fields:
                field = f"{asset}__{suffix}"
                if field not in rows:
                    raise ValueError("Asset availability evidence is missing")
                values = _times(rows[field], field, report, missing=experimental)
                if values is None:
                    continue
                timezone = "America/New_York" if asset in TEAM12_ASSETS[:10] else "Asia/Seoul"
                source_start = dates.dt.tz_localize(timezone).dt.tz_convert("UTC")
                if values.lt(source_start).any():
                    _issue(
                        report,
                        "AVAILABILITY_PREDATES_OBSERVATION",
                        "Availability precedes the source observation day.",
                        field=field,
                    )
                if values.gt(evaluation).any():
                    _issue(
                        report,
                        "FUTURE_AVAILABILITY",
                        "Data is usable only after evaluation.",
                        field=field,
                    )
                if suffix != "assumed_usable_at" and values.isna().any():
                    report["metrics"]["actual_availability_missing_cells"] = report["metrics"].get(
                        "actual_availability_missing_cells", 0
                    ) + int(values.isna().sum())
    if report["metrics"].get("actual_availability_missing_cells"):
        _issue(
            report,
            "ACTUAL_AVAILABILITY_UNVERIFIED",
            "Research assumptions do not prove actual release times.",
            severity="unverified",
        )
    for asset in TEAM12_ASSETS:
        rsi = features[f"{asset}__rsi14"].dropna()
        volatility = features[f"{asset}__volatility20"].dropna()
        if not rsi.between(0, 100).all():
            _issue(
                report, "RSI_RANGE_INVALID", "Unscaled RSI must be between 0 and 100.", asset=asset
            )
        if (volatility < 0).any():
            _issue(report, "NEGATIVE_VOLATILITY", "Volatility cannot be negative.", asset=asset)
        if _different(
            features[f"{asset}__{kinds[4]}"],
            features[f"{asset}__macd"] - features[f"{asset}__macd_signal"],
        ):
            _issue(
                report,
                "MACD_HISTOGRAM_MISMATCH",
                "Histogram must equal MACD minus Signal.",
                asset=asset,
            )
    report["metrics"]["warmup_rows"] = int(warmup.sum())
    report["metrics"]["segment_count"] = int(groups.nunique())
    return groups, eligible, expected_known, kinds


def _windows(frames, lineage, experimental, eligible, known, report):
    records = lineage.get("windows")
    if experimental:
        if (
            not isinstance(records, list)
            or not records
            or any(not isinstance(item, dict) for item in records)
        ):
            raise ValueError("Experimental window declarations must be a nonempty list")
        pairs = [(item.get("window_id"), item) for item in records]
    else:
        if not isinstance(records, dict) or not records:
            raise ValueError("Window declarations must be a nonempty mapping")
        pairs = list(records.items())
    if len({name for name, _ in pairs}) != len(pairs):
        raise ValueError("Window IDs must be unique")
    expected_frames = {"features", "rewards", "rows"}
    if "regimes" in frames:
        expected_frames.add("regimes")
    results = {}
    features, rows = frames["features"], frames["rows"]
    for name, record in pairs:
        if not isinstance(record, dict):
            raise ValueError("Window records must be objects")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("Window IDs must be safe strings")
        prefix = f"windows/{name}"
        required = {f"{prefix}/{kind}" for kind in ("features", "splits", "rewards")}
        expected_frames |= required
        if not required <= frames.keys():
            raise ValueError("Every audited window requires feature, split and reward payloads")
        scaled, splits, rewards = (
            frames[f"{prefix}/{kind}"] for kind in ("features", "splits", "rewards")
        )
        if (
            scaled.columns.tolist() != features.columns.tolist()
            or rewards.columns.tolist() != list(TEAM12_ASSETS)
            or not scaled.index.equals(splits.index)
            or not rewards.index.equals(splits.index)
            or not scaled.index.isin(features.index).all()
            or "split" not in splits
            or not _numeric(scaled)
        ):
            raise ValueError("Window payload indexes or columns disagree")
        cutoffs = record if experimental else record.get("cutoffs", {})
        boundary_names = ("train_start", "train_end_exclusive", "test_end_exclusive")
        if any(field not in cutoffs for field in boundary_names):
            raise ValueError("Window cutoff evidence is missing")
        begin, train_end, end = (pd.Timestamp(cutoffs[field]) for field in boundary_names)
        if not experimental and "validation_end_exclusive" not in cutoffs:
            raise ValueError("Formal windows require a validation cutoff")
        validation_end = (
            train_end if experimental else pd.Timestamp(cutoffs["validation_end_exclusive"])
        )
        days = (begin, train_end, validation_end, end)
        if (
            any(pd.isna(day) or day.tzinfo is not None or day != day.normalize() for day in days)
            or not begin < train_end <= validation_end < end
        ):
            raise ValueError("Window boundaries must be ordered naive daily cutoffs")
        index = features.index[(features.index >= begin) & (features.index < end)]
        labels = pd.Series(
            np.where(
                index < train_end, "train", np.where(index < validation_end, "validation", "test")
            ),
            index=index,
        )
        if not index.equals(splits.index) or not np.array_equal(splits["split"], labels):
            _issue(
                report,
                "SPLIT_BOUNDARY_MISMATCH",
                "Split labels disagree with declared date cutoffs.",
                window=name,
            )
            continue
        if not experimental and (
            "eligible_observation" not in splits
            or not is_bool_dtype(splits["eligible_observation"].dtype)
            or not np.array_equal(splits["eligible_observation"], eligible.reindex(index))
        ):
            _issue(
                report,
                "OBSERVATION_ELIGIBILITY_MISMATCH",
                "Window observation eligibility must match the root feature-ready mask.",
                window=name,
            )
        fit_index = index[(labels == "train") & eligible.reindex(index)]
        if fit_index.empty:
            raise ValueError("Each window requires finite eligible training rows")
        scaler = TrainOnlyStandardScaler().fit(features.loc[fit_index])
        statistics = record.get("scaler", {})
        if not isinstance(statistics, dict) or not {"mean", "scale"} <= statistics.keys():
            raise ValueError("Window scaler statistics are missing")
        if _different(statistics["mean"], scaler.mean_) or _different(
            statistics["scale"], scaler.scale_
        ):
            _issue(
                report,
                "TRAIN_SCALER_STATISTICS_MISMATCH",
                "Stored scaler statistics do not match this window's Train only.",
                window=name,
            )
        if _different(scaled, scaler.transform(features.loc[index])):
            _issue(
                report,
                "SCALED_FEATURE_MISMATCH",
                "Window features disagree with its Train-only scaler.",
                window=name,
            )
        fit_start, fit_end = fit_index.min().date().isoformat(), fit_index.max().date().isoformat()
        if experimental:
            if (
                record.get("scaler_fit_start") != fit_start
                or record.get("scaler_fit_end") != fit_end
            ):
                _issue(
                    report,
                    "SCALER_FIT_RANGE_MISMATCH",
                    "Recorded fit range differs from actual Train.",
                    window=name,
                )
        elif (
            statistics.get("fit_rows") != len(fit_index)
            or statistics.get("fit_start") != fit_start
            or statistics.get("fit_end") != fit_end
            or statistics.get("fit_partition") != "train_only"
            or statistics.get("ddof") != 0
        ):
            _issue(
                report,
                "SCALER_FIT_RANGE_MISMATCH",
                "Recorded fit metadata differs from actual Train.",
                window=name,
            )
        next_label = labels.reindex(rows.index).shift(-1).reindex(index)
        permitted = labels.eq(next_label) & known.reindex(index)
        if not experimental:
            permitted &= eligible.reindex(index)
        if not experimental and (
            "reward_eligible" not in splits
            or not np.array_equal(splits["reward_eligible"], permitted)
        ):
            _issue(
                report,
                "REWARD_ELIGIBILITY_MISMATCH",
                "Reward flags cross a split or unavailable endpoint.",
                window=name,
            )
        expected_rewards = frames["rewards"].loc[index].where(permitted, axis=0)
        if _different(rewards, expected_rewards):
            _issue(
                report,
                "SPLIT_REWARD_MISMATCH",
                "Window rewards must mask final, gap and split boundary outcomes.",
                window=name,
            )
        results[name] = {
            "train_rows": len(fit_index),
            "fit_start": fit_start,
            "fit_end": fit_end,
            "reward_rows": int(permitted.sum()),
        }
    if set(frames) != expected_frames:
        raise ValueError("Window payloads and declarations disagree")
    report["metrics"]["windows"] = results


def _recompute(frames, prices, groups, kinds, report):
    rows, features = frames["rows"], frames["features"]
    if not isinstance(prices, pd.DataFrame) or (
        not prices.index.equals(rows.index)
        or prices.columns.tolist() != list(TEAM12_ASSETS)
        or not _numeric(prices)
        or not np.isfinite(prices.to_numpy(dtype=float)).all()
        or not prices.gt(0).all().all()
    ):
        raise ValueError(
            "prices_krw must match all row dates and canonical positive 12-asset prices"
        )
    expected = pd.DataFrame(np.nan, index=rows.index, columns=features.columns)
    trailing = np.log(prices).diff()
    for _, dates in groups.groupby(groups, sort=False):
        index = dates.index
        for asset in TEAM12_ASSETS:
            close = prices.loc[index, asset]
            returns = np.log(close).diff()
            macd = calculate_macd(close)
            values = (
                returns,
                calculate_rsi(close),
                macd["macd"],
                macd["macd_signal"],
                macd["macd"] - macd["macd_signal"],
                returns.rolling(20, min_periods=20).std(ddof=0),
            )
            for kind, value in zip(kinds, values):
                expected.loc[index, f"{asset}__{kind}"] = value
        expected.loc[index[:WARMUP_ROWS]] = np.nan
    if _different(features, expected.loc[features.index]):
        _issue(
            report,
            "FEATURE_SOURCE_RECOMPUTATION_MISMATCH",
            "RSI/MACD/volatility/log returns disagree with supplied source prices.",
        )
    next_contiguous = groups.eq(groups.shift(-1))
    rewards = trailing.shift(-1).where(next_contiguous, axis=0).reindex(features.index)
    if _different(frames["rewards"], rewards):
        _issue(
            report,
            "REWARD_SOURCE_RECOMPUTATION_MISMATCH",
            "Rewards disagree with next contiguous source-price log returns.",
        )
    report["metrics"]["source_price_recomputation"] = "checked"


def audit_feature_artifact(path, *, expected_lineage_sha256, prices_krw=None) -> dict:
    """Inspect hashes, chronology, Train scalers and optional source arithmetic.

    A None digest records the missing external trust anchor. Source prices must
    contain every row-metadata date, including warmup. Passing numerical checks
    never verifies the truth of financial source evidence or grants approval.
    """
    report = {"status": "unverified", "training_approved": False, "issues": [], "metrics": {}}
    try:
        frames, lineage, experimental = _load(path, expected_lineage_sha256, report)
        report["metrics"]["integrity"] = "passed"
        if lineage.get("training_approved") is not False:
            raise ValueError("Audited artifacts must explicitly withhold training approval")
        if experimental:
            _issue(
                report,
                "EXPERIMENTAL_SOURCE_UNVERIFIED",
                "The experimental schema does not establish source or historical release approval.",
                severity="unverified",
            )
        elif lineage.get("dataset_kind") == "synthetic_test":
            _issue(
                report,
                "SYNTHETIC_DATA_ONLY",
                "Synthetic tests are not evidence of real financial-source validity.",
                severity="unverified",
            )
        groups, eligible, known, kinds = _frame_checks(frames, lineage, experimental, report)
        _windows(frames, lineage, experimental, eligible, known, report)
        if prices_krw is None:
            _issue(
                report,
                "SOURCE_PRICE_RECOMPUTATION_UNAVAILABLE",
                "Source prices were not supplied; indicator arithmetic has not been replayed.",
                severity="unverified",
            )
            report["metrics"]["source_price_recomputation"] = "unverified"
        else:
            _recompute(frames, prices_krw, groups, kinds, report)
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        _issue(report, "FEATURE_ARTIFACT_INVALID", str(exc))
    return _finish(report)
