"""검증된 루트 피처를 단일 연속 구간의 PPO 원척도 입력으로 변환한다."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype

from robo_advisor.data.snapshot import read_snapshot, write_frame
from robo_advisor.features.dataset import TEAM12_ASSETS
from robo_advisor.features.dataset_io import read_feature_dataset

if TYPE_CHECKING:
    from robo_advisor.rl.training import DataConfig

CONVERSION_VERSION = "team12-ppo-input-v1"
_FEATURES = ("log_return", "rsi14", "macd", "macd_signal", "macd_histogram", "volatility20")
_FILES = ("returns.parquet", "rsi.parquet", "macd.parquet")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lengths(lookback, min_episode_steps):
    if any(type(value) is not int or value < 1 for value in (lookback, min_episode_steps)):
        raise ValueError("lookback and min_episode_steps must be positive integers")


def _evaluation_index(rows):
    required = {
        "evaluation_at",
        "feature_ready",
        "warmup",
        "segment_id",
        "return_interval_contiguous",
        "next_date",
        "reward_known",
    }
    if not required.issubset(rows.columns):
        raise ValueError("Missing continuity or evaluation row metadata")
    for name in ("feature_ready", "warmup", "return_interval_contiguous", "reward_known"):
        if not is_bool_dtype(rows[name].dtype) or rows[name].isna().any():
            raise ValueError(f"{name} must contain explicit boolean flags")
    if not is_integer_dtype(rows.segment_id.dtype) or rows.segment_id.isna().any():
        raise ValueError("segment_id must contain integers")
    values = [pd.Timestamp(value) for value in rows.evaluation_at]
    if any(pd.isna(value) or value.tzinfo is None for value in values):
        raise ValueError("evaluation_at must contain aware historical timestamps")
    evaluation = pd.DatetimeIndex(pd.to_datetime(values, utc=True), name="evaluation_at")
    expected = (
        (rows.index + pd.Timedelta(hours=15, minutes=30))
        .tz_localize("Asia/Seoul")
        .tz_convert("UTC")
    )
    if not evaluation.equals(expected):
        raise ValueError("evaluation_at must match each date's verified Seoul 15:30 cutoff")
    if evaluation.hasnans or not evaluation.is_unique or not evaluation.is_monotonic_increasing:
        raise ValueError("evaluation_at must be ordered and one-to-one with dates")
    for name in [f"{asset}__available_at" for asset in TEAM12_ASSETS] + ["fx__available_at"]:
        if name not in rows:
            raise ValueError(f"Missing publication evidence: {name}")
        releases = [pd.Timestamp(value) for value in rows[name]]
        if any(pd.isna(value) or value.tzinfo is None for value in releases):
            raise ValueError(f"{name} must contain aware publication timestamps")
        if (pd.DatetimeIndex(pd.to_datetime(releases, utc=True)) > evaluation).any():
            raise ValueError(f"Future publication at evaluation cutoff: {name}")
    return evaluation


def _runs(rows, frames):
    reasons, runs, current = {}, [], []
    for position, day in enumerate(rows.index):
        row = rows.iloc[position]
        reason = []
        if row.warmup:
            reason.append("warmup")
        if not row.feature_ready:
            reason.append("feature_not_ready")
        if not row.return_interval_contiguous:
            reason.append("noncontiguous_return")
        if not all(
            np.isfinite(frame.iloc[position].to_numpy(dtype=float)).all() for frame in frames
        ):
            reason.append("nonfinite_input")
        if ((frames[1].iloc[position] < 0) | (frames[1].iloc[position] > 100)).any():
            reason.append("rsi_out_of_range")
        if reason:
            reasons[position] = reason
        linked = position > 0 and (
            row.segment_id == rows.segment_id.iloc[position - 1]
            and row.return_interval_contiguous
            and rows.next_date.iloc[position - 1] == day
        )
        if current and (reason or not linked):
            runs.append(current)
            current = []
        if not reason:
            current.append(position)
    if current:
        runs.append(current)
    return runs, reasons


def _window_mapping(lineage, frames, dates, evaluation):
    mapping = {}
    for window_id, record in lineage["windows"].items():
        splits = frames[f"windows/{window_id}/splits"]
        cutoffs = record["cutoffs"]
        bounds = [
            pd.Timestamp(cutoffs[name])
            for name in (
                "train_start",
                "train_end_exclusive",
                "validation_end_exclusive",
                "test_end_exclusive",
            )
        ]
        if any(pd.isna(day) or day.tzinfo is not None or day != day.normalize() for day in bounds):
            raise ValueError("Window cutoffs must be naive daily labels")
        if not all(left < right for left, right in zip(bounds, bounds[1:])):
            raise ValueError("Window cutoffs must increase strictly")
        if not {"split", "eligible_observation", "reward_eligible"}.issubset(splits.columns):
            raise ValueError("Window split eligibility metadata is required")
        for flag in ("eligible_observation", "reward_eligible"):
            if not is_bool_dtype(splits[flag].dtype) or splits[flag].isna().any():
                raise ValueError("Window eligibility must contain explicit boolean flags")
        root_dates = frames["rows"].index
        if not splits.index.equals(
            root_dates[(root_dates >= bounds[0]) & (root_dates < bounds[-1])]
        ):
            raise ValueError("Window splits must cover their declared date range")
        partitions = {}
        for name, start, end in zip(("train", "validation", "test"), bounds, bounds[1:]):
            expected = splits.index[(splits.index >= start) & (splits.index < end)]
            if not splits.loc[expected, "split"].eq(name).all():
                raise ValueError("Window split labels disagree with cutoffs")
            eligible = expected[splits.loc[expected, "eligible_observation"].eq(True)]
            selected = dates[(dates >= start) & (dates < end)]
            if not selected.isin(eligible).all():
                raise ValueError("Selected rows disagree with Window observation eligibility")
            positions = np.flatnonzero((dates >= start) & (dates < end))
            partitions[name] = {
                "start": start.date().isoformat(),
                "end_exclusive": end.date().isoformat(),
                "complete": len(eligible) > 0 and eligible.equals(selected),
                "rows": len(selected),
                "first_evaluation_at": (
                    evaluation[positions[0]].isoformat() if len(positions) else None
                ),
                "last_included_evaluation_at": (
                    evaluation[positions[-1]].isoformat() if len(positions) else None
                ),
            }
        mapping[window_id] = {"source": record, "partitions": partitions}
    return mapping


def convert_ppo_inputs(
    feature_dataset,
    output_dir,
    *,
    source_snapshot=None,
    expected_lineage_sha256=None,
    run_id=None,
    lookback=20,
    min_episode_steps=1,
) -> Path:
    """해시 검증 후 선택한 연속 구간만 새 폴더에 저장한다.

    run_id는 유효 행을 원본 순서로 나눈 0부터 시작하는 구간 번호다. 구간이
    여럿이면 자동 선택하지 않는다. 워밍업과 결측 행은 채우지 않고 제외한다.
    출력 manifest는 학습 승인과 무관하며 원본 승인·검증 상태를 보존한다.
    """
    _lengths(lookback, min_episode_steps)
    output = Path(output_dir).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output directory already exists: {output}")
    loaded = read_feature_dataset(feature_dataset, expected_lineage_sha256=expected_lineage_sha256)
    lineage, source_frames = loaded["lineage"], loaded["frames"]
    source = lineage["source_snapshot"]
    snapshot = read_snapshot(
        source_snapshot if source_snapshot is not None else source["path"],
        expected_manifest_sha256=source["manifest_sha256"],
    )
    if output.resolve().is_relative_to(loaded["path"]) or output.resolve().is_relative_to(
        snapshot.path
    ):
        raise ValueError("Output directory cannot be inside a source artifact")
    columns = [f"{asset}__{kind}" for asset in TEAM12_ASSETS for kind in _FEATURES]
    if lineage["asset_order"] != list(TEAM12_ASSETS) or lineage["feature_columns"] != columns:
        raise ValueError("Expected the fixed team12 asset and feature order")
    rows, raw = source_frames["rows"], source_frames["features"]
    evaluation = _evaluation_index(rows)
    frames = []
    for kind in ("log_return", "rsi14", "macd"):
        frame = raw[[f"{asset}__{kind}" for asset in TEAM12_ASSETS]].astype(float)
        frame.columns = list(TEAM12_ASSETS)
        frames.append(frame)
    with np.errstate(over="ignore", invalid="ignore"):
        frames[0] = np.expm1(frames[0])
    runs, reasons = _runs(rows, frames)
    summaries = [
        {
            "run_id": number,
            "segment_id": int(rows.segment_id.iloc[run[0]]),
            "start": rows.index[run[0]].date().isoformat(),
            "end_inclusive": rows.index[run[-1]].date().isoformat(),
            "rows": len(run),
        }
        for number, run in enumerate(runs)
    ]
    if run_id is None:
        if len(runs) != 1:
            raise ValueError(f"Select an explicit run_id from continuous runs: {summaries}")
        run_id = 0
    if type(run_id) is not int or not 0 <= run_id < len(runs):
        raise ValueError(f"Unknown run_id; available continuous runs: {summaries}")
    selected = runs[run_id]
    if len(selected) < lookback + min_episode_steps:
        raise ValueError("Selected run lacks lookback and minimum episode rows")
    dates, times = rows.index[selected], evaluation[selected]
    windows = _window_mapping(lineage, source_frames, dates, times)
    kept = set(selected)
    excluded = [
        {
            "date": day.date().isoformat(),
            "evaluation_at": evaluation[position].isoformat(),
            "segment_id": int(rows.segment_id.iloc[position]),
            "reasons": reasons.get(position, ["unselected_run"]),
        }
        for position, day in enumerate(rows.index)
        if position not in kept
    ]
    mapping = [
        {"date": day.date().isoformat(), "evaluation_at": timestamp.isoformat()}
        for day, timestamp in zip(dates, times)
    ]
    manifest = {
        "conversion_version": CONVERSION_VERSION,
        "training_approved": lineage["training_approved"],
        "dataset_kind": lineage["dataset_kind"],
        "validation": lineage["validation"],
        "source_snapshot": {
            **source,
            "verified_path": str(snapshot.path),
            "manifest": snapshot.manifest,
        },
        "source_lineage_sha256": loaded["lineage_sha256"],
        "source_feature_dataset": {
            "path": str(loaded["path"]),
            "feature_version": lineage["feature_version"],
        },
        "source_files": lineage["artifact"]["files"],
        "asset_order": list(TEAM12_ASSETS),
        "feature_order": columns,
        "formulas": {
            "returns": "expm1(root.log_return)",
            "rsi": "root.rsi14 (0..100)",
            "macd": "root.macd = (EMA12 - EMA26) / current_price",
        },
        "alignment": lineage["alignment"],
        "cutoff_at": lineage["cutoff_at"],
        "cutoff_policy": lineage.get("cutoff_policy"),
        "warmup": lineage["warmup"],
        "date_evaluation_mapping": mapping,
        "mapping_summary": {
            "rows": len(mapping),
            "timezone": "UTC",
            "policy": "rows.evaluation_at; verified Seoul 15:30",
        },
        "windows": windows,
        "available_runs": summaries,
        "selected_run_id": run_id,
        "excluded_rows": excluded,
        "context_policy": "train: within partition; validation/test: borrow lookback-1 prior "
        "rows only within selected run; no refit",
        "boundary_policy": "[start, end) maps to first and last included evaluation_at",
        "minimum_length": {"lookback": lookback, "min_episode_steps": min_episode_steps},
        "row_metadata": json.loads(
            rows.loc[
                dates,
                [
                    "feature_ready",
                    "warmup",
                    "segment_id",
                    "return_interval_contiguous",
                    "next_date",
                    "reward_known",
                ],
            ].to_json(orient="table", date_format="iso")
        ),
        "window_masks": {
            key: json.loads(frame.to_json(orient="table", date_format="iso"))
            for key, frame in source_frames.items()
            if key.endswith("/splits")
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    try:
        files = {}
        for name, frame in zip(_FILES, frames):
            result = frame.iloc[selected].copy()
            result.index = times
            files[name] = write_frame(result, output / name)
        manifest["files"] = files
        (output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
    except Exception:
        shutil.rmtree(output)
        raise
    return output


def window_to_data_config(
    output_dir,
    window_id,
    *,
    split="train",
    lookback=20,
    min_episode_steps=1,
) -> DataConfig:
    """Window의 [start, end)를 실제 평가시각의 양쪽 포함 DataConfig로 연결한다.

    Train은 분할 내부 첫 lookback-1행을 문맥으로 사용한다. Validation/Test는
    선택 구간 안의 직전 lookback-1행만 빌리며 첫 관측은 분할 첫 행이다.
    불완전 분할이나 부족한 문맥·에피소드는 거부한다. 학습을 실행하지 않는다.
    """
    from robo_advisor.rl.training import DataConfig

    _lengths(lookback, min_episode_steps)
    output = Path(output_dir).expanduser().resolve()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest["conversion_version"] != CONVERSION_VERSION:
        raise ValueError("Unsupported PPO input conversion version")
    for name in _FILES:
        if _sha256(output / name) != manifest["files"][name]["sha256"]:
            raise ValueError(f"Converted input SHA256 mismatch: {name}")
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation or test")
    if window_id not in manifest["windows"]:
        raise ValueError(f"Unknown Window ID: {window_id}")
    partition = manifest["windows"][window_id]["partitions"][split]
    if not partition["complete"]:
        raise ValueError("Window partition crosses excluded rows or continuous runs")
    mapping = manifest["date_evaluation_mapping"]
    times = pd.DatetimeIndex([entry["evaluation_at"] for entry in mapping])
    first = times.get_loc(pd.Timestamp(partition["first_evaluation_at"]))
    last = times.get_loc(pd.Timestamp(partition["last_included_evaluation_at"]))
    start = first if split == "train" else first - (lookback - 1)
    if start < 0 or last - start + 1 < lookback + min_episode_steps:
        raise ValueError("Window partition lacks context or minimum episode rows")
    return DataConfig(
        dataset_dir=output,
        train_start=times[start].to_pydatetime(),
        train_end=times[last].to_pydatetime(),
    )


def main(argv=None):
    """학습 없이 원척도 PPO 입력과 계보 manifest를 생성한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feature_dataset", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source-snapshot", type=Path)
    parser.add_argument("--expected-lineage-sha256")
    parser.add_argument("--run-id", type=int)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--min-episode-steps", type=int, default=1)
    args = parser.parse_args(argv)
    print(convert_ppo_inputs(**vars(args)))


if __name__ == "__main__":
    main()
