"""Prepared archives preserve research boundaries and never emit learner inputs."""

from hashlib import sha256
from types import SimpleNamespace

import pandas as pd
import pytest

from robo_advisor.features.dataset import DatasetInputError, DatasetWindow, TEAM12_ASSETS
from robo_advisor.features.dataset_io import (
    FeatureDatasetIOError,
    read_feature_dataset,
    read_prepared_research_dataset,
)
from robo_advisor.features.ppo_input import convert_ppo_inputs
from robo_advisor.features import research_dataset
from tests.test_feature_audit import _save, _toy


@pytest.fixture
def preparation(tmp_path, monkeypatch):
    digest, fx_digest = "a" * 64, "b" * 64
    fx = SimpleNamespace(
        metadata={"source": "alfred", "symbol": "DEXKOUS", "unit": "KRW_per_USD"},
        data=pd.DataFrame({"rate": [1200.0]}, index=pd.DatetimeIndex(["2020-01-01"])),
        status="unverified",
    )
    snapshot = SimpleNamespace(
        path=tmp_path / "source",
        manifest={"snapshot_id": "test-only"},
        manifest_sha256=digest,
        datasets={
            **{asset: SimpleNamespace(status="unverified") for asset in TEAM12_ASSETS},
            "FX_ALFRED_DEXKOUS": fx,
        },
    )
    monkeypatch.setattr(research_dataset, "_snapshot", lambda *_: snapshot)

    def prepare(name, *, future_factor=1.0, mutation=None, pin=None):
        frames, lineage, _ = _toy(future_factor=future_factor)
        rows = frames["rows"]
        rows["FX__rate"] = 1200.0
        rows["FX__observation_date"] = pd.Series(
            pd.Timestamp("2020-01-01"), index=rows.index, dtype="datetime64[ns]"
        )
        rows["FX__reference_at"] = rows["evaluation_at"] - pd.Timedelta(hours=4)
        rows["FX__actual_available_at"] = pd.Series(
            pd.NaT, index=rows.index, dtype="datetime64[ns, UTC]"
        )
        rows["FX__assumed_usable_at"] = rows["evaluation_at"]
        lineage.update(
            validation_status="unverified",
            provenance={
                "evidence_sha": {"selected_source_snapshot": digest, "independent_FX": fx_digest},
                "FX": {
                    "dataset_id": "FX_ALFRED_DEXKOUS",
                    "source_file_sha256": fx_digest,
                    "max_observation_age_days": 14,
                    "unit": "KRW_per_USD",
                },
            },
        )
        if mutation == "future":
            rows.loc[rows.index[50], "FX__actual_available_at"] = rows["evaluation_at"].iloc[
                50
            ] + pd.Timedelta(seconds=1)
        elif mutation == "price":
            rows.loc[rows.index[50], "FX__rate"] = 1300.0
        elif mutation == "snapshot":
            lineage["provenance"]["evidence_sha"]["selected_source_snapshot"] = "c" * 64
        archive, archive_digest = _save(tmp_path / name, frames, lineage)
        index = rows.index
        window = DatasetWindow(
            "W1", index[0], index[90], index[120], index[-1] + pd.Timedelta(days=1)
        )
        output = research_dataset.prepare_research_dataset(
            archive,
            tmp_path / "derived",
            expected_lineage_sha256=archive_digest if pin is None else pin,
            source_snapshot=snapshot.path,
            expected_manifest_sha256=digest,
            windows=[window],
            dataset_id=name,
        )
        output_digest = sha256((output / "lineage.json").read_bytes()).hexdigest()
        return output, read_prepared_research_dataset(output, expected_lineage_sha256=output_digest)

    return prepare


def test_prepared_artifact_round_trip_and_native_converter_refusal(preparation, tmp_path):
    output, checked = preparation("test-archive")
    frames, lineage = checked["frames"], checked["lineage"]
    assert lineage["training_approved"] is False
    assert lineage["validation"]["status"] == "unverified"
    assert len(frames["features"].columns) == 72
    assert "SPY__macd_histogram" in frames["features"]
    rows = frames["rows"]
    assert rows.index.equals(frames["features"].index)
    assert frames["features"].loc[rows["warmup"]].isna().all().all()
    assert rows["FX__actual_available_at"].isna().all()
    assert rows["evaluation_at"].dt.tz_convert("Asia/Seoul").dt.hour.eq(16).all()
    splits = frames["windows/W1/splits"]
    last = splits.groupby("split").tail(1).index
    assert frames["windows/W1/rewards"].loc[last].isna().all().all()
    assert lineage["windows"]["W1"]["scaler"]["fit_rows"] == 57
    with pytest.raises(FeatureDatasetIOError):
        read_feature_dataset(output)
    with pytest.raises(FeatureDatasetIOError):
        convert_ppo_inputs(output, tmp_path / "native")
    assert not (tmp_path / "native").exists()


@pytest.mark.leakage
def test_validation_and_test_cannot_change_training_scaler(preparation):
    _, base = preparation("base")
    _, changed = preparation("changed", future_factor=1.5)
    assert (
        base["lineage"]["windows"]["W1"]["scaler"] == changed["lineage"]["windows"]["W1"]["scaler"]
    )
    key = "windows/W1/features"
    train = base["frames"]["windows/W1/splits"]["split"].eq("train")
    pd.testing.assert_frame_equal(base["frames"][key].loc[train], changed["frames"][key].loc[train])


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("future", "FX violates observation/publication order"),
        ("price", "FX rate does not match"),
        ("snapshot", "snapshot SHA-256 disagree"),
    ],
)
def test_inconsistent_source_evidence_is_rejected(preparation, tmp_path, mutation, message):
    with pytest.raises(DatasetInputError, match=message):
        preparation("bad", mutation=mutation)
    assert not (tmp_path / "derived").exists()


def test_wrong_external_lineage_pin_is_rejected(preparation, tmp_path):
    with pytest.raises(ValueError, match="lineage SHA-256 mismatch"):
        preparation("bad", pin="0" * 64)
    assert not (tmp_path / "derived").exists()
