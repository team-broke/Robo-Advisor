"""사전 계산 SHAP의 정확성·가법성·재현성과 저장·조회·플롯 추적을 검증한다."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import shap
import torch
from stable_baselines3 import PPO

from robo_advisor.explain.background import sample_training_background
from robo_advisor.explain.policy import ObservationLayout, PPOPolicyWrapper
from robo_advisor.explain.shap_values import (
    compute_kernel_shap,
    group_contributions,
    load_shap_result,
    model_fingerprint,
    save_shap_result,
)
from robo_advisor.rl.environment import PortfolioEnv

LAYOUT = ObservationLayout(("A", "B"), lookback=2)


@pytest.fixture(scope="module")
def wrapper():
    index = pd.date_range("2026-01-01", periods=6, tz="UTC")
    returns = pd.DataFrame([[0.01, 0.02]] * 6, index=index, columns=["A", "B"])
    env = PortfolioEnv(returns, returns * 0 + 50, returns * 0 + 0.03, lookback=2, n_assets=2)
    model = PPO(
        "MlpPolicy",
        env,
        seed=17,
        device="cpu",
        n_steps=4,
        batch_size=4,
        policy_kwargs={"net_arch": {"pi": [], "vf": []}},
    )
    # Linear means: A = x0 + 25, B = 2 * x7 (rsi of A), CASH = -25.
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.weight[0, 0] = 1
        model.policy.action_net.weight[1, 7] = 2
        model.policy.action_net.bias.copy_(torch.tensor([25.0, 0.0, -25.0]))
    env.close()
    return PPOPolicyWrapper(model, LAYOUT)


@pytest.fixture(scope="module")
def background():
    rng = np.random.default_rng(3)
    index = pd.date_range("2026-01-01", periods=12, tz="UTC")
    observations = LAYOUT.to_frame(rng.uniform(0, 1, (12, 13)), index=index)
    strata = pd.Series(["up", "down"] * 6, index=index)
    return sample_training_background(
        observations,
        strata,
        layout=LAYOUT,
        train_start=index[0],
        train_end=index[-1],
        n_samples=8,
        seed=5,
        expected_strata=("up", "down"),
    )


@pytest.fixture(scope="module")
def targets():
    rng = np.random.default_rng(9)
    index = pd.date_range("2026-02-01", periods=3, tz="Asia/Seoul")
    return LAYOUT.to_frame(rng.uniform(0, 1, (3, 13)), index=index)


@pytest.fixture(scope="module")
def result(wrapper, background, targets):
    return compute_kernel_shap(wrapper, background, targets, model_id="model-x", seed=11)


def test_linear_policy_has_exact_attribution_and_additivity(result, background, targets):
    reference = background.observations.mean()
    a = result.values["A"]
    np.testing.assert_allclose(a.iloc[:, 0], targets.iloc[:, 0] - reference.iloc[0], atol=1e-5)
    np.testing.assert_allclose(a.drop(columns=a.columns[0]).to_numpy(), 0, atol=1e-5)
    b = result.values["B"]
    np.testing.assert_allclose(
        b.iloc[:, 7], 2 * (targets.iloc[:, 7] - reference.iloc[7]), atol=1e-5
    )
    np.testing.assert_allclose(result.values["CASH"].to_numpy(), 0, atol=1e-5)
    assert result.base_values["A"] == pytest.approx(reference.iloc[0] + 25)
    for output in result.outputs:
        total = result.base_values[output] + result.values[output].sum(axis=1)
        np.testing.assert_allclose(total, result.predictions[output], atol=1e-4)
        assert result.metadata["max_additivity_error"][output] <= 1e-4
    assert result.values["A"].index.tz is not None
    assert str(result.values["A"].index.tz) == "UTC"


def test_metadata_tracks_model_background_seed_and_time(result, background):
    meta = result.metadata
    assert meta["model_id"] == "model-x"
    assert meta["background_id"] == background.sample_id
    assert meta["seed"] == 11 and meta["nsamples"] == "auto"
    assert meta["outputs"] == ["A", "B", "CASH"]
    assert set(meta["compute_seconds"]) == {"A", "B", "CASH"}
    assert all(seconds >= 0 for seconds in meta["compute_seconds"].values())
    assert meta["shap_version"] == shap.__version__
    json.dumps(meta, allow_nan=False)


def test_same_seed_is_reproducible_and_global_rng_is_restored(wrapper, background, targets):
    np.random.seed(123)
    expected_next = np.random.random()
    np.random.seed(123)
    kwargs = dict(model_id="m", outputs=["A"], nsamples=40, seed=7)
    first = compute_kernel_shap(wrapper, background, targets, **kwargs)
    assert np.random.random() == expected_next
    second = compute_kernel_shap(wrapper, background, targets, **kwargs)
    pd.testing.assert_frame_equal(first.values["A"], second.values["A"])


def test_feature_groups_are_preserved_in_lookups(result, targets):
    explanation = result.explanation("A", targets.index[1])
    assert explanation["timestamp"] == targets.index[1].tz_convert("UTC").isoformat()
    groups = explanation["group_contributions"]
    assert list(groups.index) == ["returns", "weights", "rsi", "macd", "risk"]
    assert groups.sum() == pytest.approx(explanation["contributions"].sum())
    assert groups["returns"] == pytest.approx(explanation["contributions"].iloc[0])
    assert explanation["base_value"] + groups.sum() == pytest.approx(explanation["prediction"])
    with pytest.raises(ValueError, match="no precomputed"):
        result.explanation("A", "2030-01-01T00:00:00Z")
    with pytest.raises(ValueError, match="timezone"):
        result.explanation("A", "2026-02-01")
    with pytest.raises(ValueError, match="unknown"):
        result.explanation("Z", targets.index[0])
    with pytest.raises(ValueError, match="align"):
        group_contributions(explanation["contributions"], ("returns",))


def test_failed_additivity_is_rejected(wrapper, background, targets, monkeypatch):
    original = shap.KernelExplainer.shap_values

    def shifted(self, X, **kwargs):
        return np.asarray(original(self, X, **kwargs)) + 0.01

    monkeypatch.setattr(shap.KernelExplainer, "shap_values", shifted)
    with pytest.raises(ValueError, match="additivity"):
        compute_kernel_shap(wrapper, background, targets, model_id="m", outputs=["A"])


@pytest.mark.parametrize(
    "change",
    [
        {"model_id": " "},
        {"outputs": ["Z"]},
        {"outputs": "A"},
        {"outputs": ["A", "A"]},
        {"seed": -1},
        {"seed": True},
        {"nsamples": 0},
        {"tolerance": 0},
        {"tolerance": float("nan")},
    ],
)
def test_invalid_arguments_are_rejected(wrapper, background, targets, change):
    kwargs = {"model_id": "m", **change}
    with pytest.raises(ValueError):
        compute_kernel_shap(wrapper, background, targets, **kwargs)


def test_invalid_observations_are_rejected(wrapper, background, targets):
    naive = targets.copy()
    naive.index = naive.index.tz_localize(None)
    with pytest.raises(ValueError, match="timezone"):
        compute_kernel_shap(wrapper, background, naive, model_id="m")
    with pytest.raises(ValueError):
        compute_kernel_shap(wrapper, background, targets.iloc[:, ::-1], model_id="m")
    with pytest.raises(TypeError):
        compute_kernel_shap(wrapper, background.observations, targets, model_id="m")


def test_save_and_load_roundtrip_with_hash_checks(result, tmp_path):
    saved = save_shap_result(result, tmp_path / "shap")
    loaded = load_shap_result(saved)
    assert loaded.outputs == result.outputs
    for output in result.outputs:
        pd.testing.assert_frame_equal(loaded.values[output], result.values[output])
        assert loaded.base_values[output] == result.base_values[output]
    pd.testing.assert_frame_equal(loaded.predictions, result.predictions)
    pd.testing.assert_frame_equal(loaded.observations, result.observations)
    assert loaded.feature_groups == result.feature_groups
    assert loaded.metadata["model_id"] == "model-x"
    digest = loaded.metadata["manifest_sha256"]
    assert load_shap_result(saved, expected_manifest_sha256=digest).outputs == result.outputs
    with pytest.raises(ValueError, match="manifest"):
        load_shap_result(saved, expected_manifest_sha256="0" * 64)
    with pytest.raises(FileExistsError):
        save_shap_result(result, saved)
    target = saved / "shap__A.parquet"
    target.write_bytes(target.read_bytes() + b"x")
    with pytest.raises(ValueError, match="SHA256"):
        load_shap_result(saved)


def test_unsafe_output_names_leave_no_partial_directory(result, tmp_path):
    from dataclasses import replace

    unsafe = replace(result, values={"../A": result.values["A"]})
    with pytest.raises(ValueError):
        save_shap_result(unsafe, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_model_fingerprint_hashes_checkpoint_bytes(tmp_path):
    path = tmp_path / "model.zip"
    path.write_bytes(b"checkpoint")
    assert model_fingerprint(path) == hashlib.sha256(b"checkpoint").hexdigest()


def test_plots_record_model_time_asset_and_background(result, targets, tmp_path):
    from robo_advisor.explain.plots import save_force_plot, save_summary_plot

    summary = save_summary_plot(result, "A", tmp_path / "plots" / "summary_A.png")
    force = save_force_plot(result, "B", targets.index[0], tmp_path / "plots" / "force_B.png")
    for path, kind, output in ((summary, "summary", "A"), (force, "force", "B")):
        assert path.read_bytes().startswith(b"\x89PNG")
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        assert meta["kind"] == kind and meta["output"] == output
        assert meta["model_id"] == "model-x"
        assert meta["background_id"] == result.metadata["background_id"]
    force_meta = json.loads(force.with_suffix(".json").read_text(encoding="utf-8"))
    assert force_meta["timestamps"] == [targets.index[0].tz_convert("UTC").isoformat()]
    assert force_meta["prediction"] == pytest.approx(
        force_meta["base_value"] + result.values["B"].iloc[0].sum()
    )
    summary_meta = json.loads(summary.with_suffix(".json").read_text(encoding="utf-8"))
    assert len(summary_meta["timestamps"]) == len(targets)
    with pytest.raises(FileExistsError):
        save_summary_plot(result, "A", summary)
    with pytest.raises(ValueError, match="png"):
        save_summary_plot(result, "A", tmp_path / "summary.svg")
    with pytest.raises(ValueError, match="unknown"):
        save_summary_plot(result, "Z", tmp_path / "z.png")
