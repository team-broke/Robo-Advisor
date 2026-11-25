"""Exercise the real synthetic handoff and existing API preload contracts."""

import shutil

from fastapi.testclient import TestClient
import pandas as pd
import pytest

from robo_advisor.api.app import create_app
from robo_advisor.api.lifecycle import PreloadSettings
from robo_advisor.integration.artifact_export import _records, export_artifacts
from robo_advisor.ops.ppo_preflight import json_bytes, object_json, sha256
from scripts.run_integration_smoke import check_http, run


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory):
    destination = tmp_path_factory.mktemp("integration") / "synthetic"
    run(destination)
    return destination


def _json(path):
    return object_json(path.read_bytes())


def test_existing_api_loads_real_smoke_artifacts(synthetic_run, monkeypatch):
    root = synthetic_run / "api"
    acceptance = _json(root / "acceptance.json")
    settings = PreloadSettings(
        **{
            key.removeprefix("API_").lower(): (value if key.endswith("SHA256") else root / value)
            for key, value in acceptance["environment"].items()
        }
    )

    def forbidden(*args, **kwargs):
        pytest.fail("request attempted PPO training or full SHAP computation")

    monkeypatch.setattr("stable_baselines3.PPO.learn", forbidden)
    monkeypatch.setattr("shap.KernelExplainer", forbidden)
    with TestClient(create_app(preload_settings=settings)) as client:
        result = check_http(client, acceptance)
        future = client.post("/optimize", json={"as_of": "2030-01-01T06:30:00Z"})
        assert future.status_code == 404
    assert result["max_absolute_weight_error"] <= 1e-7
    assert result["training_approved"] is False
    assert acceptance["provenance"]["kind"] == "synthetic"


@pytest.mark.leakage
def test_exported_policy_and_shap_background_respect_causal_times(synthetic_run):
    root = synthetic_run / "api"
    policy = _json(root / "policy.json")
    obs = pd.read_parquet(root / "observations.parquet")
    tags = pd.read_parquet(root / "risk_tags.parquet")
    assert len(obs.columns) == len(policy["assets"]) * (policy["lookback"] + 4) + 1
    assert tags.trace_id.tolist() == ["synthetic-observed"]
    availability = pd.to_datetime(tags.available_at.iloc[0], utc=True)
    score = obs[f"risk_score__{tags.asset.iloc[0]}"]
    assert score[obs.index < availability].eq(0).all()
    assert score[obs.index >= availability].eq(0.2).all()
    shap = _json(root / "shap/manifest.json")
    source = _json(synthetic_run / "source/artifacts/training/synthetic-61/manifest.json")
    assert shap["model_id"] == policy["model_id"]
    assert pd.Timestamp(shap["background_train_end"]) == pd.Timestamp(
        source["data_cutoff"]["actual_last_index"]
    )
    assert pd.Timestamp(shap["background_train_end"]) < pd.Timestamp(shap["first_timestamp"])


@pytest.mark.parametrize("defect", ["model", "run", "dataset", "hash", "missing", "approved"])
def test_invalid_comparison_is_rejected_before_api_export(synthetic_run, tmp_path, defect):
    comparison = tmp_path / "comparison"
    shutil.copytree(synthetic_run / "comparison", comparison)
    manifest_path = comparison / "manifest.json"
    manifest = _json(manifest_path)
    if defect == "model":
        manifest["ppo"]["provenance"]["checkpoint"]["sha256"] = "0" * 64
    elif defect == "run":
        manifest["ppo"]["provenance"]["run_id"] = "another-training-run"
    elif defect == "dataset":
        manifest["ppo"]["provenance"]["dataset_inputs"]["returns.parquet"]["sha256"] = "0" * 64
    elif defect == "approved":
        manifest["training_approved"] = True
    elif defect == "hash":
        (comparison / "strategy_metrics.csv").write_bytes(b"tampered")
    else:
        (comparison / "daily_returns.csv").unlink()
    manifest_path.write_bytes(json_bytes(manifest))
    out = tmp_path / "api"
    with pytest.raises((ValueError, FileNotFoundError)):
        export_artifacts(
            synthetic_run / "relocated/handoff",
            _json(synthetic_run / "handoff-pin.json")["sha256"],
            comparison,
            sha256(manifest_path.read_bytes()),
            out,
            window="synthetic-smoke",
        )
    assert not out.exists()


def test_incomplete_metrics_stay_null(synthetic_run):
    comparison = synthetic_run / "comparison"
    rows = pd.read_csv(comparison / "strategy_metrics.csv")
    rows.loc[rows.strategy.eq("ppo"), "complete"] = False
    records = _records(
        {"strategy_metrics": rows}, "synthetic-smoke", _json(comparison / "manifest.json"), {}
    )
    ppo = next(record for record in records if record["strategy"] == "ppo")
    assert ppo["metrics"]["sharpe"] is None
    baseline = next(record for record in records if record["strategy"] == "benchmark")
    assert baseline["metrics"]["cumulative_return"] is not None
