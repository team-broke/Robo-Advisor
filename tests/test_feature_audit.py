"""Offline artifact audits using explicitly synthetic financial-shaped values."""

from copy import deepcopy
import hashlib
import json
import socket

import numpy as np
import pandas as pd
import pytest

from robo_advisor.data.snapshot import write_frame
from robo_advisor.features.audit import audit_feature_artifact
from robo_advisor.features.dataset import FeatureDataset, TEAM12_ASSETS
from robo_advisor.features.dataset_io import write_feature_dataset
from robo_advisor.features.indicators import calculate_macd, calculate_rsi
from robo_advisor.features.scaling import TrainOnlyStandardScaler

KINDS = ("log_return", "rsi14", "macd", "macd_signal", "macd_hist", "volatility20")


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Feature audits must not access external APIs")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def _toy(*, future_factor=1.0):
    index = pd.bdate_range("2020-01-02", periods=180)
    t = np.arange(len(index))
    prices = pd.DataFrame(
        {
            asset: (100 + number * 20) * np.exp(np.cumsum(0.001 + 0.009 * np.sin(t / 5.0 + number)))
            for number, asset in enumerate(TEAM12_ASSETS)
        },
        index=index,
    )
    prices.iloc[110:] *= future_factor
    returns = np.log(prices).diff()
    features = pd.DataFrame(
        index=index,
        columns=[f"{asset}__{kind}" for asset in TEAM12_ASSETS for kind in KINDS],
        dtype=float,
    )
    for asset in TEAM12_ASSETS:
        close, trailing = prices[asset], returns[asset]
        macd = calculate_macd(close)
        values = (
            trailing,
            calculate_rsi(close),
            macd["macd"],
            macd["macd_signal"],
            macd["macd"] - macd["macd_signal"],
            trailing.rolling(20, min_periods=20).std(ddof=0),
        )
        for kind, value in zip(KINDS, values):
            features[f"{asset}__{kind}"] = value
    eligible = pd.Series(np.arange(len(index)) >= 33, index=index)
    features = features.loc[eligible].copy()
    known = pd.Series(True, index=index)
    known.iloc[-1] = False
    evaluation = pd.Series(
        (index + pd.Timedelta(hours=16)).tz_localize("Asia/Seoul").tz_convert("UTC"),
        index=index,
    )
    rows = pd.DataFrame(
        {
            "evaluation_at": evaluation,
            "return_start_date": pd.Series(index, index=index).shift(1),
            "segment_id": 1,
            "segment_row": np.arange(len(index)),
            "feature_eligible": eligible,
            "reward_known": known,
        },
        index=index,
    )
    for asset in TEAM12_ASSETS:
        rows[f"{asset}__source_dates"] = (
            index - pd.offsets.BDay(1) if asset in TEAM12_ASSETS[:10] else index
        )
        rows[f"{asset}__actual_available_at"] = evaluation - pd.Timedelta(hours=1)
        rows[f"{asset}__assumed_usable_at"] = evaluation
    rewards = returns.shift(-1).loc[features.index]
    labels = pd.Series(
        np.where(features.index < index[100], "train", "test"),
        index=features.index,
        name="split",
    )
    fit = features.loc[labels.eq("train")]
    scaler = TrainOnlyStandardScaler().fit(fit)
    window_rewards = rewards.where(labels.eq(labels.shift(-1)), axis=0)
    frames = {
        "features": features,
        "rewards": rewards,
        "rows": rows,
        "windows/wf-1/features": scaler.transform(features),
        "windows/wf-1/rewards": window_rewards,
        "windows/wf-1/splits": labels.to_frame(),
    }
    lineage = {
        "schema_version": "experimental-real-team12-v1",
        "dataset_kind": "experimental_real",
        "training_approved": False,
        "feature_version": "experimental-observed-team12-price-return-v1",
        "asset_order": list(TEAM12_ASSETS),
        "feature_columns": features.columns.tolist(),
        "reward_columns": list(TEAM12_ASSETS),
        "windows": [
            {
                "window_id": "wf-1",
                "train_start": index[33].date().isoformat(),
                "train_end_exclusive": index[100].date().isoformat(),
                "test_end_exclusive": (index[-1] + pd.Timedelta(days=1)).date().isoformat(),
                "scaler_fit_start": fit.index[0].date().isoformat(),
                "scaler_fit_end": fit.index[-1].date().isoformat(),
                "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
            }
        ],
    }
    return frames, lineage, prices


def _save(root, frames, lineage):
    root.mkdir()
    lineage = deepcopy(lineage)
    lineage["files"] = {}
    for name, frame in frames.items():
        path = root / f"{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        lineage["files"][name] = {"path": f"{name}.parquet", **write_frame(frame, path)}
    content = json.dumps(lineage, sort_keys=True, allow_nan=False).encode()
    (root / "lineage.json").write_bytes(content)
    return root, hashlib.sha256(content).hexdigest()


def _codes(report):
    return {issue["code"] for issue in report["issues"]}


def test_experimental_artifact_checks_do_not_grant_approval(tmp_path):
    frames, lineage, prices = _toy()
    path, digest = _save(tmp_path / "artifact", frames, lineage)
    original = {name: frame.copy(deep=True) for name, frame in frames.items()}
    report = audit_feature_artifact(path, expected_lineage_sha256=digest, prices_krw=prices)
    assert report["status"] == "unverified"
    assert report["training_approved"] is False
    assert report["metrics"]["integrity"] == "passed"
    assert report["metrics"]["source_price_recomputation"] == "checked"
    assert not [issue for issue in report["issues"] if issue["severity"] == "error"]
    assert report["metrics"]["feature_count"] == 72
    assert report["metrics"]["feature_rows"] == 147
    for name in frames:
        pd.testing.assert_frame_equal(frames[name], original[name])


def test_missing_external_pin_and_prices_are_unverified(tmp_path):
    frames, lineage, _ = _toy()
    path, _ = _save(tmp_path / "artifact", frames, lineage)
    report = audit_feature_artifact(path, expected_lineage_sha256=None)
    assert report["status"] == "unverified"
    assert {"LINEAGE_TRUST_ANCHOR_MISSING", "SOURCE_PRICE_RECOMPUTATION_UNAVAILABLE"} <= _codes(
        report
    )


@pytest.mark.parametrize("mutation", ["lineage_pin", "payload"])
def test_hash_failure_is_explicit(tmp_path, mutation):
    frames, lineage, _ = _toy()
    path, digest = _save(tmp_path / "artifact", frames, lineage)
    if mutation == "lineage_pin":
        digest = "0" * 64
    else:
        with (path / "features.parquet").open("ab") as handle:
            handle.write(b"tamper")
    report = audit_feature_artifact(path, expected_lineage_sha256=digest)
    assert report["status"] == "failed"
    assert "FEATURE_ARTIFACT_INVALID" in _codes(report)


@pytest.mark.parametrize("schema", ["experimental-other", 2])
def test_unknown_schema_never_falls_back_to_unchecked_parquet(tmp_path, schema):
    frames, lineage, _ = _toy()
    lineage["schema_version"] = schema
    path, digest = _save(tmp_path / "artifact", frames, lineage)
    report = audit_feature_artifact(path, expected_lineage_sha256=digest)
    assert report["status"] == "failed"


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("statistics", "TRAIN_SCALER_STATISTICS_MISMATCH"),
        ("transformed", "SCALED_FEATURE_MISMATCH"),
        ("boundary_reward", "SPLIT_REWARD_MISMATCH"),
        ("final_reward", "UNKNOWN_REWARD_EXPOSED"),
        ("future_release", "FUTURE_AVAILABILITY"),
        ("future_source", "FUTURE_SOURCE_DATE"),
        ("histogram", "MACD_HISTOGRAM_MISMATCH"),
        ("warmup_position", "WARMUP_POSITION_MISMATCH"),
        ("column_order", "FEATURE_ARTIFACT_INVALID"),
    ],
)
def test_semantic_tampering_is_detected_even_after_rehash(tmp_path, mutation, expected_code):
    frames, lineage, _ = _toy()
    if mutation == "statistics":
        leaked = TrainOnlyStandardScaler().fit(frames["features"])
        lineage["windows"][0]["scaler"]["mean"] = leaked.mean_.tolist()
        lineage["windows"][0]["scaler"]["scale"] = leaked.scale_.tolist()
    elif mutation == "transformed":
        frames["windows/wf-1/features"].iloc[0, 0] += 1
    elif mutation == "boundary_reward":
        boundary = frames["windows/wf-1/splits"].index[66]
        frames["windows/wf-1/rewards"].loc[boundary] = frames["rewards"].loc[boundary]
    elif mutation == "final_reward":
        frames["rewards"].iloc[-1] = 0.0
    elif mutation == "future_release":
        frames["rows"].iloc[
            70, frames["rows"].columns.get_loc("SPY__actual_available_at")
        ] += pd.Timedelta(days=1)
    elif mutation == "future_source":
        date = frames["rows"].index[70]
        frames["rows"].loc[date, "SPY__source_dates"] = date
    elif mutation == "histogram":
        frames["features"].iloc[20, 4] += 1
    elif mutation == "warmup_position":
        frames["rows"].iloc[50, frames["rows"].columns.get_loc("segment_row")] = 0
    elif mutation == "column_order":
        frames["features"] = frames["features"].iloc[:, ::-1]
        lineage["feature_columns"] = frames["features"].columns.tolist()
    path, digest = _save(tmp_path / "artifact", frames, lineage)
    report = audit_feature_artifact(path, expected_lineage_sha256=digest)
    assert report["status"] == "failed"
    assert expected_code in _codes(report)


@pytest.mark.leakage
def test_future_test_perturbation_preserves_train_statistics_and_scaled_values(tmp_path):
    first, first_lineage, first_prices = _toy()
    changed, changed_lineage, changed_prices = _toy(future_factor=1000)
    assert first_lineage["windows"][0]["scaler"] == changed_lineage["windows"][0]["scaler"]
    train = first["windows/wf-1/splits"]["split"].eq("train")
    pd.testing.assert_frame_equal(
        first["windows/wf-1/features"].loc[train],
        changed["windows/wf-1/features"].loc[train],
    )
    for name, frames, lineage, prices in (
        ("first", first, first_lineage, first_prices),
        ("changed", changed, changed_lineage, changed_prices),
    ):
        path, digest = _save(tmp_path / name, frames, lineage)
        report = audit_feature_artifact(path, expected_lineage_sha256=digest, prices_krw=prices)
        assert report["status"] == "unverified"
        assert not [issue for issue in report["issues"] if issue["severity"] == "error"]


@pytest.mark.leakage
def test_changed_source_prices_do_not_silently_validate_saved_features(tmp_path):
    frames, lineage, prices = _toy()
    path, digest = _save(tmp_path / "artifact", frames, lineage)
    prices.iloc[110:] *= 1000
    report = audit_feature_artifact(path, expected_lineage_sha256=digest, prices_krw=prices)
    assert report["status"] == "failed"
    assert "FEATURE_SOURCE_RECOMPUTATION_MISMATCH" in _codes(report)


def test_missing_actual_release_is_not_replaced_by_research_assumption(tmp_path):
    frames, lineage, _ = _toy()
    frames["rows"]["SPY__actual_available_at"] = pd.Series(
        pd.NaT, index=frames["rows"].index, dtype="datetime64[ns, UTC]"
    )
    path, digest = _save(tmp_path / "artifact", frames, lineage)
    report = audit_feature_artifact(path, expected_lineage_sha256=digest)
    assert report["status"] == "unverified"
    assert "ACTUAL_AVAILABILITY_UNVERIFIED" in _codes(report)
    assert report["metrics"]["actual_availability_missing_cells"] == 180


@pytest.mark.parametrize("observation_mask", ["valid", "false", "missing", "nonboolean"])
def test_formal_artifact_uses_existing_reader_and_checks_train_validation_test(
    tmp_path, observation_mask
):
    frames, experimental, prices = _toy()
    rows = (
        frames["rows"]
        .rename(
            columns={
                "feature_eligible": "feature_ready",
                "segment_row": "warmup_position",
            }
        )
        .drop(columns=["return_start_date"])
    )
    rows["return_interval_contiguous"] = True
    rows.iloc[0, rows.columns.get_loc("return_interval_contiguous")] = False
    for asset in TEAM12_ASSETS:
        rows[f"{asset}__available_at"] = rows[f"{asset}__actual_available_at"]
    rows["warmup"] = rows["warmup_position"] < 33
    index = rows.index
    features = (
        frames["features"]
        .rename(columns=lambda name: name.replace("__macd_hist", "__macd_histogram"))
        .reindex(index)
    )
    rewards = np.log(prices).diff().shift(-1)
    labels = pd.Series(
        np.where(
            index[33:] < index[100],
            "train",
            np.where(index[33:] < index[130], "validation", "test"),
        ),
        index=index[33:],
    )
    fit = features.loc[labels.index[labels.eq("train")]]
    scaler = TrainOnlyStandardScaler().fit(fit)
    permitted = labels.eq(labels.shift(-1))
    splits = pd.DataFrame(
        {"split": labels, "eligible_observation": True, "reward_eligible": permitted},
    )
    lineage = {
        "schema_version": 1,
        "dataset_kind": "synthetic_test",
        "training_approved": False,
        "feature_version": "team12-causal-v1",
        "source_snapshot": {"manifest_sha256": "1" * 64},
        "asset_order": list(TEAM12_ASSETS),
        "feature_columns": features.columns.tolist(),
        "reward_columns": list(TEAM12_ASSETS),
        "cutoff_at": rows["evaluation_at"].iloc[-1].isoformat(),
        "data_period": {
            "start": index[0].date().isoformat(),
            "end_inclusive": index[-1].date().isoformat(),
            "rows": len(index),
        },
        "warmup": {},
        "alignment": {},
        "validation": {},
        "market_regimes": {},
        "windows": {
            "wf-1": {
                "cutoffs": {
                    "train_start": index[33].date().isoformat(),
                    "train_end_exclusive": index[100].date().isoformat(),
                    "validation_end_exclusive": index[130].date().isoformat(),
                    "test_end_exclusive": (index[-1] + pd.Timedelta(days=1)).date().isoformat(),
                },
                "scaler": {
                    "mean": scaler.mean_.tolist(),
                    "scale": scaler.scale_.tolist(),
                    "ddof": 0,
                    "fit_rows": len(fit),
                    "fit_start": fit.index[0].date().isoformat(),
                    "fit_end": fit.index[-1].date().isoformat(),
                    "fit_partition": "train_only",
                },
            }
        },
    }
    formal = FeatureDataset(
        {
            "features": features,
            "rewards": rewards,
            "rows": rows,
            "windows/wf-1/features": scaler.transform(features.loc[labels.index]),
            "windows/wf-1/rewards": rewards.loc[labels.index].where(permitted, axis=0),
            "windows/wf-1/splits": splits,
        },
        lineage,
    )
    if observation_mask == "false":
        formal.frames["windows/wf-1/splits"]["eligible_observation"] = False
    elif observation_mask == "missing":
        del formal.frames["windows/wf-1/splits"]["eligible_observation"]
    elif observation_mask == "nonboolean":
        formal.frames["windows/wf-1/splits"]["eligible_observation"] = 1
    path = write_feature_dataset(formal, tmp_path, dataset_id="formal")
    digest = hashlib.sha256((path / "lineage.json").read_bytes()).hexdigest()
    report = audit_feature_artifact(path, expected_lineage_sha256=digest, prices_krw=prices)
    assert report["metrics"]["integrity"] == "passed"
    if observation_mask == "valid":
        assert report["status"] == "unverified"
        assert not [issue for issue in report["issues"] if issue["severity"] == "error"]
    else:
        assert report["status"] == "failed"
        assert "OBSERVATION_ELIGIBILITY_MISMATCH" in _codes(report)


@pytest.mark.parametrize("tamper", [False, True])
def test_gap_restarts_warmup_and_masks_next_reward(tmp_path, tamper):
    frames, lineage, prices = _toy()
    rows, index = frames["rows"], frames["rows"].index
    rows.loc[index[80], "return_start_date"] = index[78]
    rows.loc[index[80] :, "segment_id"] = 2
    rows.loc[index[80] :, "segment_row"] = np.arange(len(index) - 80)
    rows["feature_eligible"] = rows["segment_row"] >= 33
    known = rows["segment_id"].eq(rows["segment_id"].shift(-1))
    rows["reward_known"] = known
    features = pd.DataFrame(index=index, columns=frames["features"].columns, dtype=float)
    for _, group in prices.groupby(rows["segment_id"], sort=False):
        for asset in TEAM12_ASSETS:
            close = group[asset]
            trailing = np.log(close).diff()
            macd = calculate_macd(close)
            values = (
                trailing,
                calculate_rsi(close),
                macd["macd"],
                macd["macd_signal"],
                macd["macd"] - macd["macd_signal"],
                trailing.rolling(20, min_periods=20).std(ddof=0),
            )
            for kind, values in zip(KINDS, values):
                features.loc[group.index, f"{asset}__{kind}"] = values
    features = features.loc[rows["feature_eligible"]]
    rewards = np.log(prices).diff().shift(-1).where(known, axis=0).loc[features.index]
    labels = pd.Series(
        np.where(features.index < index[100], "train", "test"),
        index=features.index,
        name="split",
    )
    fit = features.loc[labels.eq("train")]
    scaler = TrainOnlyStandardScaler().fit(fit)
    permitted = labels.eq(labels.reindex(index).shift(-1).reindex(features.index)) & known.reindex(
        features.index
    )
    frames.update(
        {
            "features": features,
            "rewards": rewards,
            "windows/wf-1/features": scaler.transform(features),
            "windows/wf-1/rewards": rewards.where(permitted, axis=0),
            "windows/wf-1/splits": labels.to_frame(),
        }
    )
    lineage["windows"][0]["scaler"] = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    lineage["windows"][0]["scaler_fit_end"] = fit.index[-1].date().isoformat()
    if tamper:
        frames["rewards"].loc[index[79]] = 0.0
    path, digest = _save(tmp_path / "artifact", frames, lineage)
    report = audit_feature_artifact(path, expected_lineage_sha256=digest, prices_krw=prices)
    if tamper:
        assert report["status"] == "failed"
        assert "UNKNOWN_REWARD_EXPOSED" in _codes(report)
    else:
        assert report["status"] == "unverified"
        assert report["metrics"]["segment_count"] == 2
        assert report["metrics"]["warmup_rows"] == 66
        assert not [issue for issue in report["issues"] if issue["severity"] == "error"]


def test_feature_definition_version_is_explicit(tmp_path):
    frames, lineage, _ = _toy()
    lineage["feature_version"] = "unknown-formula-v2"
    path, digest = _save(tmp_path / "artifact", frames, lineage)
    report = audit_feature_artifact(path, expected_lineage_sha256=digest)
    assert report["status"] == "failed"
    assert "FEATURE_ARTIFACT_INVALID" in _codes(report)


@pytest.mark.leakage
@pytest.mark.parametrize("offset", [0, 1])
def test_empty_or_future_return_interval_is_rejected_after_rehash(tmp_path, offset):
    frames, lineage, prices = _toy()
    first = frames["rows"].index[0]
    frames["rows"].loc[first, "return_start_date"] = first + pd.Timedelta(days=offset)
    path, digest = _save(tmp_path / "artifact", frames, lineage)
    report = audit_feature_artifact(path, expected_lineage_sha256=digest, prices_krw=prices)
    assert report["status"] == "failed"
    assert "INVALID_RETURN_INTERVAL" in _codes(report)
