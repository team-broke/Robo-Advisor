"""Causal team12 features from an independently validated alignment result.

No alignment implementation, FX conversion, filling or source approval occurs
here. Even valid research artifacts require a separate training approval policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

from robo_advisor.data.snapshot import Snapshot, dataframe_fingerprint, read_snapshot
from robo_advisor.experiments.regimes import label_market_regimes
from .indicators import calculate_macd, calculate_rsi
from .scaling import TrainOnlyStandardScaler

TEAM12_ASSETS = (
    "SPY",
    "IWM",
    "EFA",
    "EEM",
    "IEF",
    "TIP",
    "LQD",
    "GLD",
    "DBC",
    "VNQ",
    "KODEX_200TR",
    "KTB10Y",
)
FEATURE_VERSION = "team12-causal-v1"
WARMUP_ROWS = 33
_FEATURES = ("log_return", "rsi14", "macd", "macd_signal", "macd_histogram", "volatility20")
_POLICY = {
    "status": "passed",
    "calendar_status": "verified",
    "availability_status": "verified",
    "price_basis": "market_prices_krw",
    "calendar_policy": "KRX_US_intersection",
    "overseas_lag_sessions": 1,
    "lag_axis": "full_us_sessions_before_intersection",
    "max_forward_fill_sessions": 5,
    "max_fx_age_days": 7,
}


class DatasetInputError(ValueError):
    """Input schema, chronology or declared policy contradicts the dataset contract."""


class DatasetBlockedError(DatasetInputError):
    """Real source/availability evidence does not permit model dataset construction."""

    def __init__(self, report):
        super().__init__("Real feature dataset is blocked by source/alignment verification")
        self.report = report


class AlignmentInput(Protocol):
    """Structural PR81 contract; no runtime dependency on the unmerged module."""

    prices_krw: pd.DataFrame
    log_returns: pd.DataFrame
    source_dates: pd.DataFrame
    available_at: pd.DataFrame
    forward_filled: pd.DataFrame
    fx: pd.DataFrame
    evaluation_at: pd.Series
    return_start_dates: pd.Series
    diagnostics: dict


@dataclass(frozen=True)
class DatasetWindow:
    window_id: str
    train_start: object
    train_end: object
    validation_end: object
    test_end: object


@dataclass
class BenchmarkInput:
    returns: pd.Series
    trading_calendar: pd.DatetimeIndex
    rules: dict
    evidence: dict


@dataclass
class FeatureDataset:
    frames: dict[str, pd.DataFrame]
    lineage: dict


def _day(value):
    try:
        if isinstance(value, (bool, int, float)):
            raise ValueError("Daily timestamp required")
        result = pd.Timestamp(value)
        if pd.isna(result) or result.tzinfo is not None or result != result.normalize():
            raise ValueError("Timezone-naive daily timestamp required")
        return result
    except (ValueError, TypeError, OverflowError) as exc:
        raise DatasetInputError("Expected an explicit timezone-naive daily date") from exc


def _aware(values, name):
    converted = []
    for value in values:
        try:
            if isinstance(value, (bool, int, float)):
                raise ValueError("Timestamp required")
            timestamp = pd.Timestamp(value)
            if pd.isna(timestamp) or timestamp.tzinfo is None:
                raise ValueError("Timezone-aware timestamp required")
            converted.append(timestamp.tz_convert("UTC"))
        except (ValueError, TypeError, OverflowError) as exc:
            raise DatasetInputError(
                f"{name} requires actual aware timestamps, without NaT"
            ) from exc
    return pd.DatetimeIndex(converted)


def _snapshot(source_snapshot, expected):
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise DatasetInputError("Expected the trusted source manifest SHA-256")
    path = source_snapshot.path if isinstance(source_snapshot, Snapshot) else source_snapshot
    return read_snapshot(path, expected_manifest_sha256=expected)


def _policy_reasons(evidence, digest, fx_source):
    if not isinstance(evidence, dict):
        return [{"code": "ALIGNMENT_EVIDENCE_MISSING"}]
    reasons = []
    for field, value in {
        **_POLICY,
        "source_manifest_sha256": digest,
        "fx_source": fx_source,
    }.items():
        supplied = evidence.get(field)
        if supplied != value or (type(value) is int and type(supplied) is not int):
            reasons.append({"code": "ALIGNMENT_POLICY_UNVERIFIED", "field": field})
    return reasons


def _readiness(snapshot, evidence):
    reasons = []
    for asset in TEAM12_ASSETS:
        result = snapshot.datasets.get(asset)
        if result is None:
            reasons.append({"code": "SOURCE_MISSING", "dataset_id": asset})
            continue
        if result.status != "passed":
            reasons.append(
                {"code": "SOURCE_NOT_PASSED", "dataset_id": asset, "status": result.status}
            )
        expected_symbol = {"KODEX_200TR": "278530.KS", "KTB10Y": "148070.KS"}.get(asset, asset)
        if (
            result.metadata.get("source") != "yahoo"
            or result.metadata.get("symbol") != expected_symbol
            or result.metadata.get("role") != "etf"
            or result.metadata.get("currency")
            != ("KRW" if asset in {"KODEX_200TR", "KTB10Y"} else "USD")
        ):
            reasons.append({"code": "SOURCE_IDENTITY_UNVERIFIED", "dataset_id": asset})
        if (
            result.metadata.get("availability_status") != "verified"
            or result.metadata.get("available_at") is None
        ):
            reasons.append({"code": "SOURCE_AVAILABILITY_UNVERIFIED", "dataset_id": asset})
    fx = snapshot.datasets.get("USD_KRW")
    if fx is None or fx.metadata.get("symbol") != "KRW=X" or fx.metadata.get("source") != "yahoo":
        reasons.append({"code": "YAHOO_KRW_X_MISSING"})
    elif (
        fx.status != "passed"
        or fx.metadata.get("availability_status") != "verified"
        or fx.metadata.get("available_at") is None
    ):
        reasons.append({"code": "FX_SOURCE_UNVERIFIED"})
    reasons.extend(_policy_reasons(evidence, snapshot.manifest_sha256, "yahoo:KRW=X"))
    return {
        "status": "blocked" if reasons else "research_input_verified",
        "training_approved": False,
        "source_manifest_sha256": snapshot.manifest_sha256,
        "reasons": reasons,
    }


def assess_dataset_readiness(source_snapshot, *, expected_manifest_sha256, alignment_evidence=None):
    """Audit source evidence without constructing prices, features or rewards."""
    return _readiness(_snapshot(source_snapshot, expected_manifest_sha256), alignment_evidence)


def _index(index, name):
    if (
        not isinstance(index, pd.DatetimeIndex)
        or index.empty
        or index.tz is not None
        or (
            index.hasnans
            or not index.is_unique
            or not index.is_monotonic_increasing
            or not index.equals(index.normalize())
        )
    ):
        raise DatasetInputError(f"{name} must be ordered unique naive daily labels")


def _frame(value, index, name, *, numeric=False, boolean=False):
    if not isinstance(value, pd.DataFrame) or not value.index.equals(index):
        raise DatasetInputError(f"{name} index must match aligned prices exactly")
    if not value.columns.is_unique or set(value.columns) != set(TEAM12_ASSETS):
        raise DatasetInputError(f"{name} requires the exact 12 asset columns")
    result = value.loc[:, list(TEAM12_ASSETS)].copy(deep=True)
    if boolean and any(not is_bool_dtype(dtype) for dtype in result.dtypes):
        raise DatasetInputError(f"{name} must be boolean")
    if numeric:
        if any(
            not is_numeric_dtype(dtype) or is_bool_dtype(dtype) or is_complex_dtype(dtype)
            for dtype in result.dtypes
        ):
            raise DatasetInputError(f"{name} must be real numeric")
        if not np.isfinite(result.to_numpy(dtype=float)).all():
            raise DatasetInputError(f"{name} must be finite")
    if result.isna().any().any():
        raise DatasetInputError(f"{name} cannot contain missing values")
    return result


def _validated_alignment(alignment):
    names = (
        "prices_krw",
        "log_returns",
        "source_dates",
        "available_at",
        "forward_filled",
        "fx",
        "evaluation_at",
        "return_start_dates",
        "diagnostics",
    )
    if any(not hasattr(alignment, name) for name in names):
        raise DatasetInputError("Expected the complete AlignmentResult structural contract")
    prices = alignment.prices_krw
    if not isinstance(prices, pd.DataFrame):
        raise DatasetInputError("prices_krw must be a DataFrame")
    _index(prices.index, "alignment index")
    index = prices.index
    frames = {
        "prices_krw": _frame(prices, index, "prices_krw", numeric=True),
        "log_returns": _frame(alignment.log_returns, index, "log_returns", numeric=True),
        "source_dates": _frame(alignment.source_dates, index, "source_dates"),
        "available_at": _frame(alignment.available_at, index, "available_at"),
        "forward_filled": _frame(alignment.forward_filled, index, "forward_filled", boolean=True),
    }
    if (frames["prices_krw"] <= 0).any().any():
        raise DatasetInputError("Aligned KRW market prices must be positive")
    for name in ("evaluation_at", "return_start_dates"):
        value = getattr(alignment, name)
        if not isinstance(value, pd.Series) or not value.index.equals(index):
            raise DatasetInputError(f"{name} must match aligned dates exactly")
    evaluation = _aware(alignment.evaluation_at, "evaluation_at")
    expected_clock = (
        (index + pd.Timedelta(hours=15, minutes=30)).tz_localize("Asia/Seoul").tz_convert("UTC")
    )
    if not evaluation.equals(expected_clock):
        raise DatasetInputError("PR81 evaluation cutoff must be the actual Seoul 15:30 clock")
    starts = pd.DatetimeIndex([_day(value) for value in alignment.return_start_dates])
    if not (starts < index).all():
        raise DatasetInputError("Trailing return intervals must start before their evaluation")
    contiguous = np.r_[False, starts[1:] == index[:-1]]
    expected_returns = np.log(frames["prices_krw"]).diff()
    if not np.allclose(
        frames["log_returns"].iloc[np.flatnonzero(contiguous)],
        expected_returns.iloc[np.flatnonzero(contiguous)],
        rtol=1e-9,
        atol=1e-12,
    ):
        raise DatasetInputError("Trailing returns disagree with contiguous KRW price intervals")
    for asset in TEAM12_ASSETS:
        source_dates = pd.DatetimeIndex([_day(value) for value in frames["source_dates"][asset]])
        foreign = asset not in {"KODEX_200TR", "KTB10Y"}
        if not (source_dates < index if foreign else source_dates <= index).all():
            raise DatasetInputError("Same-day overseas or future source observations are forbidden")
        published = _aware(frames["available_at"][asset], f"{asset} available_at")
        if (published > evaluation).any():
            raise DatasetInputError("A source was unavailable at its evaluation cutoff")
        earliest = source_dates.tz_localize(
            "America/New_York" if foreign else "Asia/Seoul"
        ).tz_convert("UTC")
        if (published < earliest).any():
            raise DatasetInputError("Publication predates the actual source observation day")
        frames["source_dates"][asset] = source_dates
        frames["available_at"][asset] = published
    fx = alignment.fx
    required_fx = ("rate", "observation_date", "available_at", "forward_filled")
    if (
        not isinstance(fx, pd.DataFrame)
        or not fx.index.equals(index)
        or not fx.columns.is_unique
        or not set(required_fx).issubset(fx.columns)
    ):
        raise DatasetInputError("Expected aligned FX rate, observation, release and fill evidence")
    fx = fx.loc[:, list(required_fx)].copy(deep=True)
    if fx.isna().any().any():
        raise DatasetInputError("FX evidence cannot contain missing values")
    if (
        not is_numeric_dtype(fx["rate"].dtype)
        or is_bool_dtype(fx["rate"].dtype)
        or is_complex_dtype(fx["rate"].dtype)
    ):
        raise DatasetInputError("FX rates must be real numbers")
    if (
        not np.isfinite(fx["rate"].to_numpy(dtype=float)).all()
        or not (fx["rate"] > 0).all()
        or not is_bool_dtype(fx["forward_filled"].dtype)
    ):
        raise DatasetInputError("Invalid FX rates or fill flags")
    fx_days = pd.DatetimeIndex([_day(value) for value in fx["observation_date"]])
    fx_time = _aware(fx["available_at"], "FX available_at")
    fx_start = fx_days.tz_localize("Asia/Seoul").tz_convert("UTC")
    age = pd.Timedelta(days=7)
    if (
        (fx_time > evaluation).any()
        or (fx_time < fx_start).any()
        or (evaluation - fx_time > age).any()
        or (evaluation - fx_start > age).any()
    ):
        raise DatasetInputError("FX was unpublished, predates observation or is stale at cutoff")
    fx["observation_date"], fx["available_at"] = fx_days, fx_time
    if not isinstance(alignment.diagnostics, dict):
        raise DatasetInputError("Alignment diagnostics must be a mapping")
    if alignment.diagnostics.get("status", "passed") != "passed" or (
        isinstance(alignment.diagnostics.get("finite_check"), dict)
        and alignment.diagnostics["finite_check"].get("passed") is not True
    ):
        raise DatasetInputError("Upstream diagnostics report unverified or failed alignment")
    return frames, fx, pd.Series(evaluation, index=index, name="evaluation_at"), starts, contiguous


def _window_features(features, rows, windows):
    if not windows:
        raise DatasetInputError("At least one explicit Train/Validation/Test window is required")
    frames, records, seen = {}, {}, set()
    for window in windows:
        if (
            not isinstance(window, DatasetWindow)
            or not isinstance(window.window_id, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", window.window_id)
        ):
            raise DatasetInputError("Window IDs must be safe explicit identifiers")
        if window.window_id.casefold() in seen:
            raise DatasetInputError("Duplicate or colliding Walk-Forward window IDs")
        seen.add(window.window_id.casefold())
        bounds = [
            _day(getattr(window, key))
            for key in ("train_start", "train_end", "validation_end", "test_end")
        ]
        if not all(left < right for left, right in zip(bounds, bounds[1:])):
            raise DatasetInputError("Window cutoffs must be strictly increasing, end-exclusive")
        first, train_end, validation_end, test_end = bounds
        selected = (features.index >= first) & (features.index < test_end)
        index = features.index[selected]
        splits = pd.DataFrame(index=index)
        splits["split"] = np.where(
            index < train_end, "train", np.where(index < validation_end, "validation", "test")
        )
        splits["eligible_observation"] = rows.loc[index, "feature_ready"]
        if any(
            not ((splits["split"] == split) & splits["eligible_observation"]).any()
            for split in ("train", "validation", "test")
        ):
            raise DatasetInputError("Every split needs observations after technical warmup")
        fit_index = splits.index[(splits["split"] == "train") & splits["eligible_observation"]]
        scaler = TrainOnlyStandardScaler().fit(features.loc[fit_index])
        scaled = scaler.transform(features.loc[index])
        next_split = splits["split"].shift(-1)
        splits["reward_eligible"] = (
            splits["eligible_observation"]
            & rows.loc[index, "reward_known"]
            & splits["split"].eq(next_split)
            & rows.loc[index, "next_date"].lt(test_end)
        )
        # Each reward endpoint must be inside its own split and before its cutoff.
        split_ends = splits["split"].map(
            {"train": train_end, "validation": validation_end, "test": test_end}
        )
        splits["reward_eligible"] &= rows.loc[index, "next_date"].lt(split_ends)
        frames[f"windows/{window.window_id}/features"] = scaled
        frames[f"windows/{window.window_id}/splits"] = splits
        records[window.window_id] = {
            "cutoffs": {
                key: day.date().isoformat()
                for key, day in zip(
                    (
                        "train_start",
                        "train_end_exclusive",
                        "validation_end_exclusive",
                        "test_end_exclusive",
                    ),
                    bounds,
                )
            },
            "scaler": {
                "class": "TrainOnlyStandardScaler",
                "columns": features.columns.tolist(),
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
                "ddof": 0,
                "fit_rows": len(fit_index),
                "fit_start": fit_index.min().date().isoformat(),
                "fit_end": fit_index.max().date().isoformat(),
                "fit_partition": "train_only",
            },
            "split_rows": {
                split: int(splits["split"].eq(split).sum())
                for split in ("train", "validation", "test")
            },
            "eligible_reward_rows": int(splits["reward_eligible"].sum()),
            "rewards_must_use_eligibility_mask": True,
        }
    return frames, records


def build_feature_dataset(
    alignment: AlignmentInput,
    *,
    source_snapshot,
    expected_manifest_sha256,
    windows,
    alignment_evidence,
    dataset_kind="research_only",
    benchmark=None,
    cutoff_at=None,
):
    """Create research/test features, with real inputs blocked unless independently verified."""
    if dataset_kind not in {"synthetic_test", "research_only"}:
        raise DatasetInputError("dataset_kind must be synthetic_test or research_only")
    snapshot = _snapshot(source_snapshot, expected_manifest_sha256)
    if dataset_kind == "synthetic_test":
        if any(
            asset not in snapshot.datasets
            or snapshot.datasets[asset].metadata.get("source") != "synthetic"
            for asset in TEAM12_ASSETS
        ):
            raise DatasetInputError("Synthetic tests require explicitly synthetic source snapshots")
        reasons = _policy_reasons(alignment_evidence, snapshot.manifest_sha256, "synthetic")
        if reasons:
            raise DatasetInputError(f"Invalid synthetic alignment evidence: {reasons}")
    else:
        report = _readiness(snapshot, alignment_evidence)
        if report["status"] == "blocked":
            raise DatasetBlockedError(report)
    inputs, fx, evaluation, starts, contiguous = _validated_alignment(alignment)
    if cutoff_at is not None:
        cutoff = _aware([cutoff_at], "cutoff_at")[0]
        keep = evaluation <= cutoff
        if not keep.any():
            raise DatasetInputError("Cutoff leaves no observable data")
        inputs = {key: value.loc[keep].copy() for key, value in inputs.items()}
        fx, evaluation, starts, contiguous = (
            fx.loc[keep],
            evaluation.loc[keep],
            starts[keep],
            contiguous[keep],
        )
    else:
        cutoff = evaluation.iloc[-1]
    prices, returns = inputs["prices_krw"], inputs["log_returns"]
    index = prices.index
    segment_ids = np.cumsum(~contiguous) - 1
    features = pd.DataFrame(
        np.nan,
        index=index,
        columns=[f"{asset}__{name}" for asset in TEAM12_ASSETS for name in _FEATURES],
    )
    positions = pd.Series(index=index, dtype="int64")
    for segment in np.unique(segment_ids):
        dates = index[segment_ids == segment]
        positions.loc[dates] = np.arange(len(dates))
        for asset in TEAM12_ASSETS:
            close = prices.loc[dates, asset]
            trailing = returns.loc[dates, asset].copy()
            trailing.iloc[0] = np.nan
            macd = calculate_macd(close)
            values = (
                trailing,
                calculate_rsi(close),
                macd["macd"],
                macd["macd_signal"],
                macd["macd"] - macd["macd_signal"],
                trailing.rolling(20, min_periods=20).std(ddof=0),
            )
            for name, value in zip(_FEATURES, values):
                features.loc[dates, f"{asset}__{name}"] = value.to_numpy()
    positions = positions.astype("int64")
    warmup = positions < WARMUP_ROWS
    features.loc[warmup] = np.nan
    if np.isinf(features.to_numpy()).any():
        raise DatasetInputError("Feature arithmetic overflow must not enter an artifact")
    ready = pd.Series(np.isfinite(features.to_numpy()).all(axis=1), index=index)
    rewards = returns.shift(-1).copy()
    reward_known = np.r_[contiguous[1:], False]
    rewards.loc[~reward_known] = np.nan
    next_date = pd.Series(index, index=index).shift(-1)
    rows = pd.DataFrame(
        {
            "evaluation_at": evaluation,
            "feature_ready": ready,
            "warmup": warmup,
            "segment_id": segment_ids,
            "warmup_position": positions,
            "return_interval_contiguous": contiguous,
            "next_date": next_date,
            "next_evaluation_at": evaluation.shift(-1),
            "reward_known": reward_known,
        },
        index=index,
    )
    for asset in TEAM12_ASSETS:
        for field in ("source_dates", "available_at", "forward_filled"):
            rows[f"{asset}__{field}"] = inputs[field][asset]
    for field in fx:
        rows[f"fx__{field}"] = fx[field]
    frames = {"features": features, "rewards": rewards, "rows": rows}
    window_frames, window_records = _window_features(features, rows, windows)
    frames.update(window_frames)
    for window_id in window_records:
        splits = frames[f"windows/{window_id}/splits"]
        frames[f"windows/{window_id}/rewards"] = rewards.loc[splits.index].where(
            splits["reward_eligible"], axis=0
        )
    lineage = {
        "schema_version": 1,
        "feature_version": FEATURE_VERSION,
        "dataset_kind": dataset_kind,
        "training_approved": False,
        "validation": {
            "status": (
                "synthetic_only"
                if dataset_kind == "synthetic_test"
                else "verified_input_research_only"
            ),
            "source_statuses": {asset: snapshot.datasets[asset].status for asset in TEAM12_ASSETS},
            "approval_policy": "Storage and numeric checks never grant training approval",
        },
        "source_snapshot": {
            "path": str(snapshot.path),
            "snapshot_id": snapshot.manifest["snapshot_id"],
            "manifest_sha256": snapshot.manifest_sha256,
        },
        "asset_order": list(TEAM12_ASSETS),
        "feature_columns": features.columns.tolist(),
        "reward_columns": rewards.columns.tolist(),
        "reward_kind": "next_evaluation_log_return",
        "data_period": {
            "start": index.min().date().isoformat(),
            "end_inclusive": index.max().date().isoformat(),
            "rows": len(index),
        },
        "cutoff_at": cutoff.isoformat(),
        "cutoff_policy": "Only releases at/before each Seoul 15:30 evaluation; "
        "final reward unknown",
        "warmup": {
            "excluded_initial_rows_per_segment": WARMUP_ROWS,
            "required_price_observations": 34,
            "rsi_period": 14,
            "macd": {"fast": 12, "slow": 26, "signal": 9, "normalize": True},
            "volatility_period": 20,
            "volatility_ddof": 0,
            "policy": "Mask all first33 rows per contiguous segment; "
            "EWM initialization convention, not finite memory",
            "rows": int(warmup.sum()),
            "segment_count": int(len(np.unique(segment_ids))),
        },
        "alignment": {
            "contract": "PR81 AlignmentResult structural interface",
            "reference": "https://github.com/team-broke/Robo-Advisor/pull/81",
            "reference_revision": "c03df105f6c0b28475a03ea6097240f783a9dba8",
            "evidence": dict(alignment_evidence),
            "input_fingerprints": {
                key: dataframe_fingerprint(frame) for key, frame in inputs.items()
            },
            "fx_fingerprint": dataframe_fingerprint(fx),
            "diagnostics": alignment.diagnostics,
        },
        "windows": window_records,
        "market_regimes": {"status": "not_generated", "reason": "verified_benchmark_not_supplied"},
    }
    if benchmark is not None:
        if (
            not isinstance(benchmark, BenchmarkInput)
            or not isinstance(benchmark.evidence, dict)
            or benchmark.evidence.get("status") != "passed"
            or benchmark.evidence.get("source_manifest_sha256") != snapshot.manifest_sha256
            or benchmark.evidence.get("dataset_kind") != dataset_kind
        ):
            lineage["market_regimes"] = {
                "status": "not_generated",
                "reason": "benchmark_evidence_unverified",
            }
        else:
            labels = label_market_regimes(
                benchmark.returns,
                trading_calendar=benchmark.trading_calendar,
                evaluation_start=index[0],
                evaluation_end=index[-1] + pd.Timedelta(days=1),
                **benchmark.rules,
            )
            if not labels.labels.index.equals(index):
                raise DatasetInputError("Regime labels must match feature evaluation dates")
            frames["regimes"] = labels.labels.to_frame()
            lineage["market_regimes"] = {
                "status": (
                    "synthetic_only"
                    if dataset_kind == "synthetic_test"
                    else "verified_input_research_only"
                ),
                "evidence": benchmark.evidence,
                "rules": {
                    key: value.isoformat() if isinstance(value, pd.Timestamp) else value
                    for key, value in benchmark.rules.items()
                },
                "benchmark_fingerprint": dataframe_fingerprint(
                    benchmark.returns.to_frame(name="simple_return")
                ),
                "unknown_observation_count": labels.unknown_observation_count,
            }
    return FeatureDataset(frames, lineage)
