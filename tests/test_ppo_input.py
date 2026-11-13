"""합성 산출물의 원척도·시간축·구간·계보 및 PPO 로더 연결 검증."""

from hashlib import sha256
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from tests import test_feature_dataset as feature_tests
from robo_advisor.data.snapshot import read_snapshot, write_snapshot
from robo_advisor.features import ppo_input
from robo_advisor.features.dataset_io import write_feature_dataset
from robo_advisor.features.ppo_input import convert_ppo_inputs, window_to_data_config
from robo_advisor.rl.environment import PortfolioEnv
from robo_advisor.rl.training import DataConfig, load_training_dataset


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    alignment = feature_tests._toy_alignment()
    path = write_snapshot(
        feature_tests._results(alignment),
        root=tmp_path_factory.mktemp("ppo-source"),
        snapshot_id="synthetic-team12",
    )
    return path, read_snapshot(path).manifest_sha256


@pytest.fixture
def built(source):
    return feature_tests._build(feature_tests._toy_alignment(), source)


def _store(built, tmp_path):
    return write_feature_dataset(built, tmp_path / "features", dataset_id="toy")


def _manifest(path):
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


def test_raw_formulas_order_time_and_loader(built, tmp_path):
    built.frames["features"].loc[:, "SPY__log_return"] = np.log(1.1)
    path = convert_ppo_inputs(_store(built, tmp_path), tmp_path / "ppo")
    rows = built.frames["rows"].iloc[33:]
    config = DataConfig(dataset_dir=path, train_end=rows.evaluation_at.iloc[-1])
    loaded = load_training_dataset(config)
    expected_index = pd.DatetimeIndex(rows.evaluation_at, name="evaluation_at")
    for name, kind in (("returns", "log_return"), ("rsi", "rsi14"), ("macd", "macd")):
        frame = getattr(loaded, name)
        assert frame.columns.tolist() == list(feature_tests.ASSETS)
        assert frame.index.equals(expected_index)
        expected = (
            built.frames["features"]
            .loc[rows.index, [f"{asset}__{kind}" for asset in feature_tests.ASSETS]]
            .to_numpy()
        )
        np.testing.assert_allclose(frame, np.expm1(expected) if name == "returns" else expected)
    np.testing.assert_allclose(loaded.returns.SPY, 0.1)
    assert ((loaded.rsi >= 0) & (loaded.rsi <= 100)).all().all()
    env = PortfolioEnv(loaded.returns, loaded.rsi, loaded.macd, lookback=20, n_assets=12)
    assert env.reset()[0].shape == (289,)
    env.close()


@pytest.mark.leakage
def test_future_targets_and_scaled_windows_cannot_change_inputs(built, tmp_path):
    first = convert_ppo_inputs(_store(built, tmp_path), tmp_path / "first")
    built.frames["rewards"].iloc[:] = 9999.0
    built.frames["windows/wf-1/rewards"].iloc[:] = -9999.0
    built.frames["windows/wf-1/features"].iloc[:] = 5555.0
    second_source = write_feature_dataset(built, tmp_path / "features", dataset_id="changed")
    second = convert_ppo_inputs(second_source, tmp_path / "second")
    for name in ppo_input._FILES:
        pd.testing.assert_frame_equal(pd.read_parquet(first / name), pd.read_parquet(second / name))


def test_manifest_links_original_hashes_metadata_and_exclusions(built, source, tmp_path):
    feature_path = _store(built, tmp_path)
    digest = sha256((feature_path / "lineage.json").read_bytes()).hexdigest()
    output = convert_ppo_inputs(feature_path, tmp_path / "ppo", expected_lineage_sha256=digest)
    manifest = _manifest(output)
    assert manifest["source_lineage_sha256"] == digest
    assert manifest["source_snapshot"]["manifest_sha256"] == source[1]
    assert (
        manifest["source_files"]
        == json.loads((feature_path / "lineage.json").read_text())["artifact"]["files"]
    )
    assert manifest["training_approved"] is False
    assert manifest["validation"] == built.lineage["validation"]
    assert manifest["windows"]["wf-1"]["source"] == built.lineage["windows"]["wf-1"]
    assert manifest["asset_order"] == list(feature_tests.ASSETS)
    assert manifest["feature_order"] == built.lineage["feature_columns"]
    assert manifest["conversion_version"] == ppo_input.CONVERSION_VERSION
    assert len(manifest["excluded_rows"]) == 33
    assert "warmup" in manifest["excluded_rows"][0]["reasons"]
    assert manifest["mapping_summary"]["rows"] == 127
    assert "windows/wf-1/splits" in manifest["window_masks"]
    for name in ppo_input._FILES:
        assert manifest["files"][name]["sha256"] == sha256((output / name).read_bytes()).hexdigest()


def test_discontinuous_segments_require_selection_and_never_bridge(source, tmp_path):
    alignment = feature_tests._toy_alignment()
    alignment.return_start_dates.iloc[80] = alignment.prices_krw.index[78]
    built = feature_tests._build(alignment, source)
    feature_path = _store(built, tmp_path)
    with pytest.raises(ValueError, match="explicit run_id"):
        convert_ppo_inputs(feature_path, tmp_path / "unspecified")
    for run_id, start, end in ((0, 33, 80), (1, 113, 160)):
        output = convert_ppo_inputs(feature_path, tmp_path / str(run_id), run_id=run_id)
        frame = pd.read_parquet(output / "returns.parquet")
        assert frame.index.equals(
            pd.DatetimeIndex(built.frames["rows"].evaluation_at.iloc[start:end])
        )
        assert len(_manifest(output)["available_runs"]) == 2
        if run_id == 0:
            config = window_to_data_config(output, "wf-1", split="train")
            assert config.train_end == built.frames["rows"].evaluation_at.iloc[79]
        else:
            with pytest.raises(ValueError, match="crosses excluded"):
                window_to_data_config(output, "wf-1", split="train")


@pytest.mark.parametrize(
    "boundary", ["segment_id", "return_interval_contiguous", "next_date", "feature_ready", "nan"]
)
def test_each_continuity_signal_splits_runs(built, tmp_path, boundary):
    rows = built.frames["rows"]
    if boundary == "segment_id":
        rows.loc[rows.index[80:], "segment_id"] = 1
    elif boundary == "return_interval_contiguous":
        rows.loc[rows.index[80], boundary] = False
    elif boundary == "next_date":
        rows.loc[rows.index[79], boundary] = pd.NaT
    elif boundary == "feature_ready":
        rows.loc[rows.index[80], boundary] = False
    else:
        built.frames["features"].iloc[80, 0] = np.nan
    output = convert_ppo_inputs(_store(built, tmp_path), tmp_path / "ppo", run_id=1)
    frame = pd.read_parquet(output / "returns.parquet")
    first = 81 if boundary in {"nan", "feature_ready", "return_interval_contiguous"} else 80
    assert frame.index[0] == rows.evaluation_at.iloc[first]
    assert np.isfinite(frame.to_numpy()).all()
    assert len(_manifest(output)["available_runs"]) == 2
    with pytest.raises(ValueError, match="crosses excluded"):
        window_to_data_config(output, "wf-1", split="train")
    if boundary in {"nan", "feature_ready"}:
        excluded = _manifest(output)["excluded_rows"]
        assert any(item["date"] == rows.index[80].date().isoformat() for item in excluded)


def test_window_inclusive_boundaries_and_context(built, tmp_path):
    output = convert_ppo_inputs(_store(built, tmp_path), tmp_path / "ppo")
    evaluation = built.frames["rows"].evaluation_at
    for split, start, end, first_observation in (
        ("train", 33, 89, 52),
        ("validation", 71, 119, 90),
        ("test", 101, 159, 120),
    ):
        config = window_to_data_config(output, "wf-1", split=split)
        assert config.train_start == evaluation.iloc[start]
        assert config.train_end == evaluation.iloc[end]
        loaded = load_training_dataset(config)
        assert loaded.returns.index[19] == evaluation.iloc[first_observation]
        assert loaded.returns.index[-1] == evaluation.iloc[end]
        assert config.train_start.tzinfo is not None
    mapping = _manifest(output)["windows"]["wf-1"]["partitions"]["train"]
    assert (
        mapping["end_exclusive"]
        == built.lineage["windows"]["wf-1"]["cutoffs"]["train_end_exclusive"]
    )
    assert mapping["last_included_evaluation_at"] == evaluation.iloc[89].isoformat()


def test_short_run_and_window_context_are_rejected(built, tmp_path):
    feature_path = _store(built, tmp_path)
    with pytest.raises(ValueError, match="minimum episode"):
        convert_ppo_inputs(feature_path, tmp_path / "short", lookback=127)
    output = convert_ppo_inputs(feature_path, tmp_path / "ppo")
    with pytest.raises(ValueError, match="context or minimum"):
        window_to_data_config(output, "wf-1", lookback=60)
    with pytest.raises(ValueError, match="context or minimum"):
        window_to_data_config(output, "wf-1", split="validation", lookback=60)


@pytest.mark.parametrize(
    "bad", ["naive", "missing", "duplicate", "midnight", "wrong_day", "future_release"]
)
@pytest.mark.leakage
def test_invalid_historical_evaluation_is_rejected(built, tmp_path, bad):
    rows = built.frames["rows"]
    if bad == "naive":
        rows["evaluation_at"] = rows.evaluation_at.dt.tz_localize(None)
    elif bad == "missing":
        rows.loc[rows.index[60], "evaluation_at"] = pd.NaT
    elif bad == "duplicate":
        rows.loc[rows.index[60], "evaluation_at"] = rows.evaluation_at.iloc[59]
    elif bad == "midnight":
        rows["evaluation_at"] = rows.index.tz_localize("UTC")
    elif bad == "wrong_day":
        rows["evaluation_at"] += pd.Timedelta(days=1)
    else:
        rows.loc[rows.index[60], "SPY__available_at"] = rows.evaluation_at.iloc[60] + pd.Timedelta(
            seconds=1
        )
    with pytest.raises(ValueError):
        convert_ppo_inputs(_store(built, tmp_path), tmp_path / "ppo")
    assert not (tmp_path / "ppo").exists()


def test_rsi_range_excludes_invalid_row_without_filling(built, tmp_path):
    built.frames["features"].loc[built.frames["rows"].index[80], "SPY__rsi14"] = 101.0
    output = convert_ppo_inputs(_store(built, tmp_path), tmp_path / "ppo", run_id=0)
    assert len(pd.read_parquet(output / "rsi.parquet")) == 47
    assert any("rsi_out_of_range" in row["reasons"] for row in _manifest(output)["excluded_rows"])


@pytest.mark.parametrize(
    "target", ["lineage_pin", "feature_bytes", "snapshot_bytes", "source_pin", "output_bytes"]
)
def test_integrity_checks_are_reused(built, source, tmp_path, target):
    feature_path = _store(built, tmp_path)
    output = tmp_path / "ppo"
    kwargs = {}
    if target == "lineage_pin":
        kwargs["expected_lineage_sha256"] = "f" * 64
    elif target == "feature_bytes":
        with (feature_path / "features.parquet").open("ab") as handle:
            handle.write(b"corrupt")
    elif target == "source_pin":
        built.lineage["source_snapshot"]["manifest_sha256"] = "f" * 64
        feature_path = write_feature_dataset(built, tmp_path / "features", dataset_id="bad-pin")
    elif target == "snapshot_bytes":
        copied = tmp_path / source[0].name
        import shutil

        shutil.copytree(source[0], copied)
        payload = next(copied.rglob("data.parquet"))
        with payload.open("ab") as handle:
            handle.write(b"corrupt")
        kwargs["source_snapshot"] = copied
    else:
        convert_ppo_inputs(feature_path, output)
        with (output / "returns.parquet").open("ab") as handle:
            handle.write(b"corrupt")
        with pytest.raises(ValueError, match="SHA256"):
            window_to_data_config(output, "wf-1")
        return
    with pytest.raises(ValueError):
        convert_ppo_inputs(feature_path, output, **kwargs)
    assert not output.exists()


def test_existing_output_and_sources_are_immutable(built, source, tmp_path):
    feature_path = _store(built, tmp_path)
    for output in (tmp_path / "ppo", feature_path, source[0]):
        if not output.exists():
            output.mkdir()
        before = {str(path): path.read_bytes() for path in output.rglob("*") if path.is_file()}
        with pytest.raises(FileExistsError):
            convert_ppo_inputs(feature_path, output)
        assert before == {
            str(path): path.read_bytes() for path in output.rglob("*") if path.is_file()
        }
    with pytest.raises(ValueError, match="inside a source"):
        convert_ppo_inputs(feature_path, feature_path / "nested")


def test_failed_write_removes_only_new_output(built, tmp_path, monkeypatch):
    feature_path = _store(built, tmp_path)

    def fail(*args, **kwargs):
        raise OSError("disk failure")

    monkeypatch.setattr(ppo_input, "write_frame", fail)
    with pytest.raises(OSError, match="disk failure"):
        convert_ppo_inputs(feature_path, tmp_path / "ppo")
    assert not (tmp_path / "ppo").exists()
    assert (feature_path / "lineage.json").exists()


def test_cli_creates_inputs_without_training(built, tmp_path):
    feature_path = _store(built, tmp_path)
    output = tmp_path / "ppo"
    result = subprocess.run(
        [sys.executable, "-m", "robo_advisor.features.ppo_input", str(feature_path), str(output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert set(path.name for path in output.iterdir()) == {*ppo_input._FILES, "manifest.json"}
    assert _manifest(output)["training_approved"] is False


@pytest.mark.parametrize("status", ["failed", "unverified"])
def test_original_source_status_and_approval_are_preserved(built, tmp_path, status):
    alignment = feature_tests._toy_alignment()
    source = write_snapshot(
        feature_tests._results(alignment, status=status, availability="unverified"),
        root=tmp_path / "sources",
        snapshot_id=status,
    )
    built.lineage["source_snapshot"] = {
        "path": str(source),
        "manifest_sha256": read_snapshot(source).manifest_sha256,
        "snapshot_id": status,
    }
    built.lineage["validation"]["status"] = status
    built.lineage["validation"]["source_statuses"] = {
        asset: status for asset in feature_tests.ASSETS
    }
    built.lineage["alignment"]["evidence"]["source_manifest_sha256"] = built.lineage[
        "source_snapshot"
    ]["manifest_sha256"]
    output = convert_ppo_inputs(_store(built, tmp_path), tmp_path / "ppo")
    manifest = _manifest(output)
    assert manifest["training_approved"] is False
    assert manifest["validation"]["status"] == status
    assert all(
        entry["validation_status"] == status
        for entry in manifest["source_snapshot"]["manifest"]["datasets"]
    )


@pytest.mark.parametrize("run_id", [-1, 2, True])
def test_unknown_run_id_is_rejected(built, tmp_path, run_id):
    with pytest.raises(ValueError, match="Unknown run_id"):
        convert_ppo_inputs(_store(built, tmp_path), tmp_path / "ppo", run_id=run_id)


def test_nontrading_exclusive_boundary_uses_last_actual_evaluation(source, tmp_path):
    alignment = feature_tests._toy_alignment()
    days = alignment.prices_krw.index
    window = feature_tests.DatasetWindow(
        "weekend",
        days[33],
        days[91] + pd.Timedelta(days=1),
        days[120],
        days[-1] + pd.Timedelta(days=1),
    )
    built = feature_tests._build(alignment, source, windows=[window])
    output = convert_ppo_inputs(_store(built, tmp_path), tmp_path / "ppo")
    config = window_to_data_config(output, "weekend")
    assert config.train_end == built.frames["rows"].evaluation_at.iloc[91]
    assert config.train_end.hour == 6 and config.train_end.minute == 30


def test_overflow_is_excluded_without_bridging(built, tmp_path):
    built.frames["features"].iloc[80, 0] = 1000.0
    output = convert_ppo_inputs(_store(built, tmp_path), tmp_path / "ppo", run_id=0)
    assert len(pd.read_parquet(output / "returns.parquet")) == 47
    assert any("nonfinite_input" in row["reasons"] for row in _manifest(output)["excluded_rows"])
