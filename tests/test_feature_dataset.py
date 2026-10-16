"""Offline feature, scaler, reward and lineage contracts using explicit toy data."""

from copy import deepcopy
from hashlib import sha256
import json
import socket
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from robo_advisor.data.collectors import CollectionResult
from robo_advisor.data.snapshot import read_snapshot, write_snapshot
from robo_advisor.features.dataset import (
    FEATURE_VERSION,
    TEAM12_ASSETS,
    WARMUP_ROWS,
    BenchmarkInput,
    DatasetBlockedError,
    DatasetWindow,
    assess_dataset_readiness,
    build_feature_dataset,
)
from robo_advisor.features import dataset_io
from robo_advisor.features.dataset_io import (
    FeatureDatasetIOError,
    FeatureDatasetIntegrityError,
    read_feature_dataset,
    write_feature_dataset,
)
from robo_advisor.features.indicators import calculate_macd, calculate_rsi

ASSETS = (
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
KINDS = ("log_return", "rsi14", "macd", "macd_signal", "macd_histogram", "volatility20")


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Feature dataset tests must not access external APIs")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def _toy_alignment():
    index = pd.bdate_range("2020-01-02", periods=160, name="date")
    t = np.arange(len(index), dtype=float)
    close = pd.DataFrame(
        {
            asset: (100.0 + position * 17.0)
            * np.exp(np.cumsum(0.001 + 0.007 * np.sin(t / 7.0 + position)))
            for position, asset in enumerate(ASSETS)
        },
        index=index,
    )
    decision = pd.Series(
        (index + pd.Timedelta(hours=15, minutes=30)).tz_localize("Asia/Seoul").tz_convert("UTC"),
        index=index,
    )
    dates = pd.DataFrame(
        {
            asset: index if asset in {"KODEX_200TR", "KTB10Y"} else index - pd.offsets.BDay(1)
            for asset in ASSETS
        },
        index=index,
    )
    availability = pd.DataFrame(
        {asset: decision - pd.Timedelta(hours=1) for asset in ASSETS}, index=index
    )
    fx = pd.DataFrame(
        {
            "rate": np.repeat(1200.0, len(index)),
            "observation_date": index,
            "available_at": decision - pd.Timedelta(hours=1),
            "forward_filled": False,
        },
        index=index,
    )
    return SimpleNamespace(
        prices_krw=close,
        log_returns=_trailing_returns(close),
        source_dates=dates,
        available_at=availability,
        forward_filled=pd.DataFrame(False, index=index, columns=ASSETS),
        fx=fx,
        evaluation_at=decision,
        return_start_dates=pd.Series(index - pd.offsets.BDay(1), index=index),
        diagnostics={"status": "passed", "synthetic": True},
    )


def _trailing_returns(close):
    returns = np.log(close / close.shift(1))
    # The aligned output has a prior warmup price outside this toy output slice.
    # Its explicit synthetic first return is unused by the technical warmup.
    returns.iloc[0] = 0.0
    return returns


def _results(alignment, *, source="synthetic", status="passed", availability="verified"):
    results = {}
    for asset in ASSETS:
        data = alignment.prices_krw[[asset]].rename(columns={asset: "Close"})
        dates = data.index
        issues = [] if status == "passed" else [{"code": "toy_source_issue", "severity": status}]
        metadata = {
            "dataset_id": asset,
            "symbol": asset,
            "source": source,
            "role": "etf",
            "market": "SYNTHETIC",
            "currency": "KRW",
            "collected_at": None,
            "actual_start": dates[0].date().isoformat(),
            "actual_end": dates[-1].date().isoformat(),
            "row_count": len(data),
            "requested_start": dates[0].date().isoformat(),
            "requested_end_exclusive": (dates[-1] + pd.Timedelta(days=1)).date().isoformat(),
            "price_basis": "synthetic_market_prices",
            "auto_adjust": True,
            "status": status,
            "issues": issues,
            "availability_status": availability,
            "available_at": alignment.available_at[asset].rename("available_at"),
        }
        results[asset] = CollectionResult(
            data=data,
            raw_data=data.copy(),
            metadata=metadata,
            validation={"status": status, "valid": status == "passed", "issues": issues},
        )
    return results


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    alignment = _toy_alignment()
    root = tmp_path_factory.mktemp("feature-source")
    path = write_snapshot(_results(alignment), root=root, snapshot_id="synthetic-team12")
    return path, read_snapshot(path).manifest_sha256


@pytest.fixture
def alignment():
    return _toy_alignment()


def _evidence(digest):
    return {
        "status": "passed",
        "source_manifest_sha256": digest,
        "calendar_status": "verified",
        "availability_status": "verified",
        "fx_source": "synthetic",
        "price_basis": "market_prices_krw",
        "calendar_policy": "KRX_US_intersection",
        "overseas_lag_sessions": 1,
        "lag_axis": "full_us_sessions_before_intersection",
        "max_forward_fill_sessions": 5,
        "max_fx_age_days": 7,
    }


def _window(index, name="wf-1", start=33, train=90, validation=120, end=160):
    return DatasetWindow(
        name,
        index[start],
        index[train],
        index[validation],
        index[end] if end < len(index) else index[-1] + pd.Timedelta(days=1),
    )


def _build(alignment, source, **overrides):
    path, digest = source
    kwargs = {
        "source_snapshot": path,
        "expected_manifest_sha256": digest,
        "windows": [_window(alignment.prices_krw.index)],
        "alignment_evidence": _evidence(digest),
        "dataset_kind": "synthetic_test",
    }
    kwargs.update(overrides)
    return build_feature_dataset(alignment, **kwargs)


def test_fixed_asset_order_and_feature_dimension(alignment, source):
    result = _build(alignment, source)
    assert TEAM12_ASSETS == ASSETS
    assert WARMUP_ROWS == 33
    assert FEATURE_VERSION == "team12-causal-v1"
    assert list(result.frames["features"].columns) == [
        f"{asset}__{kind}" for asset in ASSETS for kind in KINDS
    ]
    assert result.frames["features"].shape == (160, 72)
    assert list(result.frames["rewards"].columns) == list(ASSETS)


def test_existing_indicator_formula_and_causal_log_returns(alignment, source):
    result = _build(alignment, source)
    frame = result.frames["features"]
    close = alignment.prices_krw["SPY"]
    macd = calculate_macd(close)
    np.testing.assert_allclose(frame["SPY__rsi14"].iloc[33:], calculate_rsi(close).iloc[33:])
    np.testing.assert_allclose(frame["SPY__macd"].iloc[33:], macd["macd"].iloc[33:])
    np.testing.assert_allclose(
        frame["SPY__macd_histogram"].iloc[33:],
        (macd["macd"] - macd["macd_signal"]).iloc[33:],
    )
    np.testing.assert_allclose(
        frame["SPY__log_return"].iloc[33:], alignment.log_returns["SPY"].iloc[33:]
    )


def test_explicit_warmup_and_no_fill(alignment, source):
    result = _build(alignment, source)
    frame, rows = result.frames["features"], result.frames["rows"]
    assert frame.iloc[:33].isna().all().all()
    assert frame.iloc[33:].notna().all().all()
    assert rows["warmup"].iloc[:33].all()
    assert not rows["feature_ready"].iloc[:33].any()
    assert rows["feature_ready"].iloc[33:].all()


@pytest.mark.leakage
def test_future_prices_do_not_change_past_features(alignment, source):
    earlier = _build(alignment, source)
    modified = deepcopy(alignment)
    modified.prices_krw.iloc[120:] *= 1000.0
    modified.log_returns = _trailing_returns(modified.prices_krw)
    later = _build(modified, source)
    pd.testing.assert_frame_equal(
        earlier.frames["features"].iloc[:120], later.frames["features"].iloc[:120]
    )


@pytest.mark.leakage
def test_validation_test_do_not_change_train_scaler(alignment, source):
    earlier = _build(alignment, source)
    modified = deepcopy(alignment)
    modified.prices_krw.iloc[90:] *= np.linspace(2.0, 200.0, 70)[:, None]
    modified.log_returns = _trailing_returns(modified.prices_krw)
    later = _build(modified, source)
    pd.testing.assert_frame_equal(
        earlier.frames["windows/wf-1/features"].iloc[:57],
        later.frames["windows/wf-1/features"].iloc[:57],
    )
    assert earlier.lineage["windows"]["wf-1"]["scaler"] == (
        later.lineage["windows"]["wf-1"]["scaler"]
    )


def test_scaler_matches_train_only_reference(alignment, source):
    result = _build(alignment, source)
    raw = result.frames["features"].iloc[33:90]
    scaled = result.frames["windows/wf-1/features"].loc[raw.index]
    np.testing.assert_allclose(scaled.mean().to_numpy(), 0.0, atol=1e-11)
    np.testing.assert_allclose(scaled.std(ddof=0).to_numpy(), 1.0, atol=1e-11)
    stats = result.lineage["windows"]["wf-1"]["scaler"]
    np.testing.assert_allclose(stats["mean"], raw.mean().to_numpy(), atol=1e-12)
    assert stats["columns"] == list(raw.columns)


def test_each_walk_forward_window_uses_own_scaler(alignment, source):
    index = alignment.prices_krw.index
    result = _build(
        alignment,
        source,
        windows=[_window(index), _window(index, "wf-2", 55, 110, 135)],
    )
    first = result.lineage["windows"]["wf-1"]["scaler"]
    second = result.lineage["windows"]["wf-2"]["scaler"]
    assert not np.allclose(first["mean"], second["mean"])
    assert first["columns"] == second["columns"]


def test_rewards_separate_from_features_and_final_unknown(alignment, source):
    result = _build(alignment, source)
    expected = alignment.log_returns.shift(-1)
    pd.testing.assert_frame_equal(result.frames["rewards"], expected)
    assert result.frames["rewards"].iloc[-1].isna().all()
    assert not result.frames["rows"]["reward_known"].iloc[-1]
    assert not any("reward" in name for name in result.frames["features"].columns)


@pytest.mark.leakage
def test_reward_crossing_split_boundary_is_purged(alignment, source):
    result = _build(alignment, source)
    splits = result.frames["windows/wf-1/splits"]
    for position in (89, 119, 159):
        date = alignment.prices_krw.index[position]
        assert not splits.loc[date, "reward_eligible"]
        assert result.frames["windows/wf-1/rewards"].loc[date].isna().all()
    assert splits.loc[alignment.prices_krw.index[88], "reward_eligible"]
    assert result.frames["windows/wf-1/rewards"].loc[alignment.prices_krw.index[88]].notna().all()


def test_noncontiguous_return_interval_never_becomes_next_reward(alignment, source):
    alignment.return_start_dates.iloc[80] = alignment.prices_krw.index[78]
    result = _build(alignment, source)
    assert result.frames["rewards"].iloc[79].isna().all()
    assert not result.frames["rows"]["return_interval_contiguous"].iloc[80]
    assert not result.frames["rows"]["reward_known"].iloc[79]


def test_gap_starts_new_warmup_instead_of_bridging_history(alignment, source):
    alignment.return_start_dates.iloc[80] = alignment.prices_krw.index[78]
    result = _build(alignment, source)
    assert result.frames["features"].iloc[80:113].isna().all().all()
    assert result.frames["rows"]["warmup"].iloc[80:113].all()
    assert result.frames["features"].iloc[113:].notna().all().all()


@pytest.mark.leakage
def test_cutoff_truncates_future_before_rewards_and_scaler(alignment, source):
    index = alignment.prices_krw.index
    cutoff = alignment.evaluation_at.iloc[99] + pd.Timedelta(minutes=1)
    result = _build(
        alignment,
        source,
        cutoff_at=cutoff,
        windows=[_window(index, start=33, train=50, validation=70, end=100)],
    )
    assert len(result.frames["features"]) == 100
    assert result.frames["features"].index[-1] == index[99]
    assert result.frames["rewards"].iloc[-1].isna().all()
    assert not result.frames["rows"]["reward_known"].iloc[-1]


def test_inconsistent_supplied_log_return_is_rejected(alignment, source):
    alignment.log_returns.iloc[65, 0] += 0.1
    with pytest.raises(ValueError):
        _build(alignment, source)


def test_inputs_are_not_mutated(alignment, source):
    original = deepcopy(alignment)
    _build(alignment, source)
    for name in ("prices_krw", "log_returns", "source_dates", "available_at", "fx"):
        pd.testing.assert_frame_equal(getattr(alignment, name), getattr(original, name))
    pd.testing.assert_series_equal(alignment.return_start_dates, original.return_start_dates)


def test_synthetic_lineage_withholds_training_approval(alignment, source):
    result = _build(alignment, source)
    assert result.lineage["training_approved"] is False
    assert result.lineage["dataset_kind"] == "synthetic_test"
    assert result.lineage["validation"]["status"] == "synthetic_only"
    assert source[1] in json.dumps(result.lineage)
    assert list(ASSETS) == result.lineage["asset_order"]
    assert result.lineage["feature_version"] == FEATURE_VERSION


def test_no_benchmark_means_no_market_labels(alignment, source):
    result = _build(alignment, source)
    assert result.lineage["market_regimes"]["status"] == "not_generated"
    assert "regimes" not in result.frames


def _benchmark(alignment, digest, *, status="passed"):
    return BenchmarkInput(
        returns=np.expm1(alignment.log_returns["SPY"]).rename("simple_return"),
        trading_calendar=alignment.prices_krw.index,
        rules={
            "block_anchor": alignment.prices_krw.index[0],
            "block_size": 20,
            "min_blocks_per_regime": 1,
            "lookback": 10,
        },
        evidence={
            "status": status,
            "source_manifest_sha256": digest,
            "dataset_kind": "synthetic_test",
        },
    )


def test_explicit_synthetic_benchmark_labels_remain_test_only(alignment, source, tmp_path):
    result = _build(alignment, source, benchmark=_benchmark(alignment, source[1]))
    assert "regimes" in result.frames
    assert result.frames["regimes"].index.equals(alignment.prices_krw.index)
    assert result.lineage["market_regimes"]["status"] == "synthetic_only"
    assert result.lineage["training_approved"] is False
    path = write_feature_dataset(result, tmp_path, dataset_id="toy-regimes")
    loaded = read_feature_dataset(path)
    pd.testing.assert_frame_equal(loaded["frames"]["regimes"], result.frames["regimes"])


def test_unverified_benchmark_never_generates_labels(alignment, source):
    result = _build(
        alignment, source, benchmark=_benchmark(alignment, source[1], status="unverified")
    )
    assert "regimes" not in result.frames
    assert result.lineage["market_regimes"]["status"] == "not_generated"


@pytest.mark.parametrize("column", ["SPY", "KTB10Y"])
@pytest.mark.leakage
def test_future_availability_is_rejected(alignment, source, column):
    alignment.available_at.loc[alignment.prices_krw.index[75], column] = (
        alignment.evaluation_at.iloc[75] + pd.Timedelta(seconds=1)
    )
    with pytest.raises(ValueError):
        _build(alignment, source)


@pytest.mark.leakage
def test_future_fx_availability_is_rejected(alignment, source):
    alignment.fx.loc[alignment.prices_krw.index[75], "available_at"] = alignment.evaluation_at.iloc[
        75
    ] + pd.Timedelta(seconds=1)
    with pytest.raises(ValueError):
        _build(alignment, source)


@pytest.mark.parametrize("bad_value", [pd.NaT, pd.Timestamp("2020-04-01")])
def test_missing_or_naive_historical_availability_is_rejected(alignment, source, bad_value):
    alignment.available_at["SPY"] = alignment.available_at["SPY"].astype(object)
    alignment.available_at.loc[alignment.prices_krw.index[75], "SPY"] = bad_value
    with pytest.raises(ValueError):
        _build(alignment, source)


def test_missing_fx_forward_fill_evidence_is_rejected(alignment, source):
    alignment.fx["forward_filled"] = alignment.fx["forward_filled"].astype("boolean")
    alignment.fx.loc[alignment.prices_krw.index[75], "forward_filled"] = pd.NA
    with pytest.raises(ValueError):
        _build(alignment, source)


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "unverified"),
        ("calendar_status", "unverified"),
        ("availability_status", "unverified"),
        ("price_basis", "base100"),
        ("overseas_lag_sessions", 0),
        ("source_manifest_sha256", "f" * 64),
    ],
)
def test_unverified_or_incompatible_alignment_policy_is_rejected(alignment, source, field, value):
    evidence = _evidence(source[1])
    evidence[field] = value
    with pytest.raises(ValueError):
        _build(alignment, source, alignment_evidence=evidence)


def test_incorrect_pinned_source_hash_is_rejected(alignment, source):
    with pytest.raises(ValueError):
        _build(alignment, source, expected_manifest_sha256="f" * 64)


@pytest.mark.parametrize("which", ["missing", "extra"])
def test_wrong_asset_set_is_rejected(alignment, source, which):
    if which == "missing":
        alignment.prices_krw = alignment.prices_krw.drop(columns="KTB10Y")
    else:
        alignment.prices_krw["QQQ"] = alignment.prices_krw["SPY"]
    with pytest.raises(ValueError):
        _build(alignment, source)


@pytest.mark.parametrize("status", ["failed", "unverified"])
def test_real_source_status_is_never_promoted(alignment, tmp_path, status):
    path = write_snapshot(
        _results(alignment, source="legacy_csv", status=status, availability="unverified"),
        root=tmp_path / "sources",
        snapshot_id=status,
    )
    digest = read_snapshot(path).manifest_sha256
    evidence = _evidence(digest)
    evidence["fx_source"] = "yahoo:KRW=X"
    readiness = assess_dataset_readiness(
        path,
        expected_manifest_sha256=digest,
        alignment_evidence=evidence,
    )
    assert readiness["status"] == "blocked"
    assert readiness["reasons"]
    with pytest.raises(DatasetBlockedError):
        _build(alignment, (path, digest), dataset_kind="research_only", alignment_evidence=evidence)


def test_real_data_cannot_be_relabelled_synthetic(alignment, tmp_path):
    path = write_snapshot(
        _results(alignment, source="legacy_csv"),
        root=tmp_path,
        snapshot_id="real-source",
    )
    digest = read_snapshot(path).manifest_sha256
    with pytest.raises(ValueError):
        _build(alignment, (path, digest))


def test_invalid_or_overlapping_window_is_rejected(alignment, source):
    index = alignment.prices_krw.index
    with pytest.raises(ValueError):
        _build(
            alignment,
            source,
            windows=[DatasetWindow("bad", index[33], index[90], index[80], index[-1])],
        )


@pytest.fixture
def built(alignment, source):
    return _build(alignment, source)


def test_native_parquet_roundtrip_and_lineage_pin(built, tmp_path):
    path = write_feature_dataset(built, tmp_path, dataset_id="toy-team12")
    digest = sha256((path / "lineage.json").read_bytes()).hexdigest()
    loaded = read_feature_dataset(path, expected_lineage_sha256=digest)
    assert loaded["lineage_sha256"] == digest
    assert set(loaded["frames"]) == set(built.frames)
    for key, frame in built.frames.items():
        pd.testing.assert_frame_equal(loaded["frames"][key], frame)
    assert loaded["lineage"]["training_approved"] is False
    assert loaded["lineage"]["dataset_kind"] == "synthetic_test"


def test_existing_feature_dataset_is_immutable(built, tmp_path):
    path = write_feature_dataset(built, tmp_path, dataset_id="keep")
    original = (path / "lineage.json").read_bytes()
    with pytest.raises(FileExistsError):
        write_feature_dataset(built, tmp_path, dataset_id="keep")
    assert (path / "lineage.json").read_bytes() == original


@pytest.mark.parametrize("mutation", ["corrupt", "missing", "extra"])
def test_tampered_or_incomplete_parquet_artifact_fails(built, tmp_path, mutation):
    path = write_feature_dataset(built, tmp_path, dataset_id="artifact")
    target = path / "features.parquet"
    if mutation == "corrupt":
        target.write_bytes(target.read_bytes() + b"corruption")
    elif mutation == "missing":
        target.unlink()
    else:
        (path / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(FeatureDatasetIOError):
        read_feature_dataset(path)


def test_incorrect_pinned_lineage_hash_fails(built, tmp_path):
    path = write_feature_dataset(built, tmp_path, dataset_id="artifact")
    with pytest.raises(FeatureDatasetIntegrityError):
        read_feature_dataset(path, expected_lineage_sha256="f" * 64)


def test_manifest_path_escape_is_rejected(built, tmp_path):
    path = write_feature_dataset(built, tmp_path, dataset_id="artifact")
    lineage_path = path / "lineage.json"
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    lineage["artifact"]["files"]["features"]["path"] = "../outside.parquet"
    lineage_path.write_text(json.dumps(lineage), encoding="utf-8")
    with pytest.raises(FeatureDatasetIOError):
        read_feature_dataset(path)


def test_partial_write_is_not_published(built, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("mock disk error")

    monkeypatch.setattr(dataset_io, "write_frame", fail)
    with pytest.raises(OSError, match="mock disk error"):
        write_feature_dataset(built, tmp_path, dataset_id="partial")
    assert not (tmp_path / "partial").exists()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("dataset_id", ["../outside", "CON", "", "tail."])
def test_unsafe_feature_output_ids_are_rejected(built, tmp_path, dataset_id):
    with pytest.raises(FeatureDatasetIOError):
        write_feature_dataset(built, tmp_path, dataset_id=dataset_id)


def test_storage_never_accepts_training_approval(built, tmp_path):
    built.lineage["training_approved"] = True
    with pytest.raises(FeatureDatasetIOError):
        write_feature_dataset(built, tmp_path, dataset_id="not-approved")


def test_lineage_does_not_persist_credentials(built, tmp_path):
    built.lineage["api_key"] = "never-store-me"
    with pytest.raises(FeatureDatasetIOError):
        write_feature_dataset(built, tmp_path, dataset_id="not-secret")


def test_storage_requires_pinned_source_manifest_hash(built, tmp_path):
    del built.lineage["source_snapshot"]["manifest_sha256"]
    with pytest.raises(FeatureDatasetIOError):
        write_feature_dataset(built, tmp_path, dataset_id="missing-lineage")


def test_storage_rejects_feature_column_order_disagreeing_with_lineage(built, tmp_path):
    built.frames["features"] = built.frames["features"].iloc[:, ::-1]
    with pytest.raises(FeatureDatasetIOError):
        write_feature_dataset(built, tmp_path, dataset_id="wrong-order")


@pytest.mark.parametrize(
    "diagnostics", [{"status": "unverified"}, {"finite_check": {"passed": False}}]
)
def test_failed_upstream_diagnostics_cannot_be_overridden_by_evidence(
    alignment, source, diagnostics
):
    alignment.diagnostics = diagnostics
    with pytest.raises(ValueError, match="Upstream diagnostics"):
        _build(alignment, source)
