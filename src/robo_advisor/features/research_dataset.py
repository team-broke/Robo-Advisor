"""Prepare immutable, unapproved Team12 research windows from a pinned archive."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .audit import read_audited_research_archive
from .dataset import (
    TEAM12_ASSETS,
    WARMUP_ROWS,
    DatasetInputError,
    DatasetWindow,
    FeatureDataset,
    _aware,
    _snapshot,
    _window_features,
)
from .dataset_io import _write_prepared_research

_FX_IDENTITIES = {
    "FX_ALFRED_DEXKOUS": ("alfred", "DEXKOUS"),
    "FX_SMBS_MAR": ("smbs", "USD_KRW_MAR"),
}


def _archive_sources(lineage, snapshot):
    provenance = lineage.get("provenance", {})
    evidence = provenance.get("evidence_sha", {})
    if evidence.get("selected_source_snapshot") != snapshot.manifest_sha256:
        raise DatasetInputError("Research archive and selected snapshot SHA-256 disagree")
    if not set(TEAM12_ASSETS).issubset(snapshot.datasets):
        raise DatasetInputError("Selected snapshot is missing a Team12 asset")
    fx = provenance.get("FX", {})
    fx_id = fx.get("dataset_id")
    if fx.get("source_file_sha256") != evidence.get("independent_FX") or not evidence.get(
        "independent_FX"
    ):
        raise DatasetInputError("Research FX requires its archived raw-file SHA-256")
    if fx_id not in _FX_IDENTITIES or fx_id not in snapshot.datasets:
        raise DatasetInputError("Research archive requires a separate pinned ALFRED or SMBS source")
    stored_fx = snapshot.datasets[fx_id].metadata
    if (stored_fx.get("source"), stored_fx.get("symbol")) != _FX_IDENTITIES[fx_id]:
        raise DatasetInputError("Selected FX snapshot identity disagrees with the archive scenario")
    return {
        "dataset_id": fx_id,
        "source": stored_fx["source"],
        "source_status": snapshot.datasets[fx_id].status,
        "archive_metadata": fx,
    }


def _verify_fx_rows(rows, source, age_days):
    if type(age_days) is not int or not 1 <= age_days <= 14:
        raise DatasetInputError("Unsupported archived FX age bound")
    required = [
        "FX__rate",
        "FX__observation_date",
        "FX__reference_at",
        "FX__actual_available_at",
        "FX__assumed_usable_at",
    ]
    if not set(required).issubset(rows):
        raise DatasetInputError("Archived row metadata is missing FX evidence")
    evaluation = _aware(rows["evaluation_at"], "evaluation_at")
    reference = _aware(rows["FX__reference_at"], "FX reference_at")
    actual = rows["FX__actual_available_at"]
    chosen = actual.where(actual.notna(), rows["FX__assumed_usable_at"])
    usable = _aware(chosen, "FX effective availability")
    if (usable < reference).any() or (usable > evaluation).any():
        raise DatasetInputError("Archived FX violates observation/publication order")
    if (evaluation - reference > pd.Timedelta(days=age_days)).any():
        raise DatasetInputError("Archived FX exceeds its recorded age bound")
    observations = pd.DatetimeIndex(rows["FX__observation_date"])
    if (
        observations.hasnans
        or observations.tz is not None
        or not observations.equals(observations.normalize())
        or not observations.isin(source.index).all()
    ):
        raise DatasetInputError("Archived FX observation dates do not match the selected snapshot")
    rates = rows["FX__rate"].to_numpy(dtype=float)
    if (
        not np.isfinite(rates).all()
        or (rates <= 0).any()
        or not np.array_equal(rates, source["rate"].reindex(observations).to_numpy(dtype=float))
    ):
        raise DatasetInputError("Archived FX rate does not match the selected snapshot")


def _reframe(archived):
    features, rewards, rows = (archived[key] for key in ("features", "rewards", "rows"))
    rows = rows.copy()
    index = rows.index
    date = pd.Series(index, index=index)
    contiguous = rows["return_start_date"].eq(date.shift())
    rows["archived_segment_id"] = rows["segment_id"]
    rows["segment_id"] = (~contiguous).cumsum()
    rows["return_interval_contiguous"] = contiguous
    rows["feature_ready"] = rows["feature_eligible"]
    rows["warmup_position"] = rows["segment_row"]
    rows["warmup"] = rows["warmup_position"].lt(WARMUP_ROWS)
    rows["next_date"] = date.shift(-1)
    rows["next_evaluation_at"] = rows["evaluation_at"].shift(-1)

    # The old archive stores values only on eligible dates. Retain unavailable
    # rows for coverage and warmup without manufacturing either observations or labels.
    features = features.rename(
        columns=lambda name: (
            name.removesuffix("__macd_hist") + "__macd_histogram"
            if name.endswith("__macd_hist")
            else name
        )
    ).reindex(index)
    rewards = rewards.reindex(index)
    rows["archived_reward_known"] = rows["reward_known"]
    rows["reward_known"] &= rows["feature_ready"] & rewards.notna().all(axis=1)
    rewards = rewards.where(rows["reward_known"], axis=0)
    return features, rewards, rows


def _coverage(rows):
    counts = rows.loc[rows["feature_ready"]].groupby("segment_id").size()
    return {
        "eligible_rows": int(counts.sum()),
        "segments": int(len(counts)),
        "longest_segment": int(counts.max()) if len(counts) else 0,
        "segments_with_20_lookback_and_60_steps": int(counts.ge(80).sum()),
        "longest_observed_segment_before_feature_mask": int(
            rows.groupby("segment_id").size().max()
        ),
    }


def prepare_research_dataset(
    archive,
    output_root,
    *,
    expected_lineage_sha256,
    source_snapshot,
    expected_manifest_sha256,
    windows,
    dataset_id=None,
) -> Path:
    """Store a historical research derivative without writing a PPO-native dataset.

    The archive pins 72 already computed causal features and their releases. The
    new artifact preserves those times and exclusions, rebuilds the three-way
    windows and refits only the Train scaler. No alignment policy is selected.
    """
    archived, lineage = read_audited_research_archive(
        archive, expected_lineage_sha256=expected_lineage_sha256
    )
    snapshot = _snapshot(source_snapshot, expected_manifest_sha256)
    fx = _archive_sources(lineage, snapshot)
    _verify_fx_rows(
        archived["rows"],
        snapshot.datasets[fx["dataset_id"]].data,
        fx["archive_metadata"].get("max_observation_age_days"),
    )
    output = Path(output_root).resolve()
    source = Path(archive).resolve()
    if source.name == "lineage.json":
        source = source.parent
    if any(output.is_relative_to(path) for path in (source, snapshot.path.resolve())):
        raise DatasetInputError("Research output directory cannot be inside a source artifact")

    features, rewards, rows = _reframe(archived)
    frames = {"features": features, "rewards": rewards, "rows": rows}
    window_frames, records = _window_features(features, rows, windows)
    frames.update(window_frames)
    for window in records:
        splits = frames[f"windows/{window}/splits"]
        frames[f"windows/{window}/rewards"] = rewards.loc[splits.index].where(
            splits["reward_eligible"], axis=0
        )
        records[window]["coverage_by_partition"] = {
            partition: _coverage(rows.loc[splits.index[splits["split"].eq(partition)]])
            for partition in ("train", "validation", "test")
        }

        test_days = rows.loc[
            splits.index[splits["split"].eq("test") & splits["eligible_observation"]]
        ]
        first_each_month = test_days.loc[~test_days.index.to_period("M").duplicated()]
        records[window]["mvo_previous_contiguous_rows_at_retained_test_month_starts"] = {
            date.date().isoformat(): int(position)
            for date, position in first_each_month["warmup_position"].items()
        }

    index = rows.index
    result = FeatureDataset(
        frames,
        {
            "schema_version": 1,
            "feature_version": lineage["feature_version"],
            "dataset_kind": "experimental_real",
            "training_approved": False,
            "validation": {
                "status": "unverified",
                "technical_status": "archive_integrity_and_chronology_checked",
                "native_ppo_status": "not_exported",
                "source_statuses": {
                    asset: snapshot.datasets[asset].status for asset in TEAM12_ASSETS
                },
            },
            "research_archive": {
                "schema_version": lineage["schema_version"],
                "lineage_sha256": expected_lineage_sha256,
                "files": {name: spec["sha256"] for name, spec in lineage["files"].items()},
            },
            "source_snapshot": {
                "snapshot_id": snapshot.manifest["snapshot_id"],
                "manifest_sha256": snapshot.manifest_sha256,
            },
            "asset_order": list(TEAM12_ASSETS),
            "feature_columns": features.columns.tolist(),
            "reward_columns": list(TEAM12_ASSETS),
            "reward_kind": "next_evaluation_log_price_return",
            "data_period": {
                "start": index[0].date().isoformat(),
                "end_inclusive": index[-1].date().isoformat(),
                "rows": len(index),
            },
            "cutoff_at": rows["evaluation_at"].iloc[-1].isoformat(),
            "alignment": {
                "policy": "preserve_archived_research_evaluation_and_releases",
                "provenance": lineage["provenance"],
                "fx": fx,
            },
            "warmup": lineage.get("warmup", {"rows_per_segment": WARMUP_ROWS}),
            "windows": records,
            "coverage": _coverage(rows),
            "market_regimes": {"status": "not_generated"},
        },
    )
    return _write_prepared_research(result, output, dataset_id=dataset_id)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--expected-lineage-sha256", required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--dataset-id")
    parser.add_argument(
        "--window",
        dest="windows",
        nargs=5,
        action="append",
        required=True,
        metavar=("ID", "TRAIN_START", "VALIDATION_START", "TEST_START", "END"),
    )
    args = vars(parser.parse_args(argv))
    args["windows"] = [DatasetWindow(*window) for window in args["windows"]]
    print(prepare_research_dataset(**args))


if __name__ == "__main__":
    main()
