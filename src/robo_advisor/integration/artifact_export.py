"""Export a verified PPO handoff and comparison for the existing API loaders."""

from __future__ import annotations

from contextlib import contextmanager
import io
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from robo_advisor.api.schemas import ArtifactProvenance, STRATEGIES
from robo_advisor.data.snapshot import write_frame
from robo_advisor.diagnostics.inputs import prepare_diagnostic_inputs
from robo_advisor.diagnostics.ppo_diagnostics import _period_dataset
from robo_advisor.explain.background import (
    collect_training_observations,
    sample_training_background,
)
from robo_advisor.explain.plots import save_force_plot, save_summary_plot
from robo_advisor.explain.policy import ObservationLayout, PPOPolicyWrapper
from robo_advisor.explain.shap_values import compute_kernel_shap, save_shap_result
from robo_advisor.ops.ppo_preflight import json_bytes, object_json, pinned, sha256
from robo_advisor.ops.ppo_reproducibility import _verified_files, _write_files
from robo_advisor.rl.evaluation import evaluation_env_config
from robo_advisor.rl.training import _make_env
from robo_advisor.schemas.common import PortfolioWeights

TABLES = (
    "daily_returns",
    "nav_history",
    "strategy_metrics",
    "rebalance_history",
    "safeguard_events",
    "evaluation_coverage",
)
LABELS = ["result_kind", "policy_version", "status", "training_approved"]


@contextmanager
def verified_snapshot(package, manifest_sha256):
    """Isolate exactly the bytes covered by the externally pinned handoff."""
    if not manifest_sha256:
        raise ValueError("a trusted handoff manifest SHA-256 is required")
    files = _verified_files(package, manifest_sha256)
    with tempfile.TemporaryDirectory(prefix="robo-e2e-") as directory:
        root = Path(directory)
        _write_files(root, files)
        yield root, files


def _comparison(directory, bundle, files, manifest_sha256):
    directory = Path(directory)
    manifest = object_json(pinned(directory, {"path": "manifest.json", "sha256": manifest_sha256}))
    diag = object_json(files["diagnostic.json"])
    ppo = manifest["ppo"]["provenance"]
    if (
        manifest["kind"] != bundle["kind"]
        or manifest["status"] != "unverified"
        or manifest["training_approved"] is not False
        or manifest["policy_status"] != "candidate"
        or manifest["performance_decision"] != "pending_protocol_and_evidence"
        or ppo["run_id"] != bundle["run_id"]
        or ppo["spec_sha256"] != sha256(files["diagnostic.json"])
        or ppo["checkpoint"] != diag["checkpoint"]
        or ppo["dataset_inputs"] != diag["dataset"]
    ):
        raise ValueError("comparison approval or model/data/run provenance mismatch")
    outputs = {
        name: pinned(directory, {"path": name, "sha256": digest})
        for name, digest in manifest["output_sha256"].items()
    }
    tables = {name: pd.read_csv(io.BytesIO(outputs[f"{name}.csv"])) for name in TABLES}
    for table in tables.values():
        if (
            not set(LABELS).issubset(table)
            or not (
                table.result_kind.eq(manifest["kind"])
                & table.status.eq("unverified")
                & table.training_approved.eq(False)
                & table.policy_version.eq(manifest["comparison_policy"]["version"])
            ).all()
        ):
            raise ValueError("comparison table lost its unverified status or policy version")
    metrics = tables["strategy_metrics"]
    if not metrics.strategy.is_unique or set(metrics.strategy) != set(STRATEGIES):
        raise ValueError("ambiguous or missing strategy records")
    coverage = tables["evaluation_coverage"]
    if not set(f"{strategy}_observed" for strategy in STRATEGIES).issubset(coverage):
        raise ValueError("missing strategy evaluation coverage")
    daily = tables["daily_returns"]
    for strategy in STRATEGIES:
        path = daily[daily.strategy.eq(strategy)]
        observed = coverage[f"{strategy}_observed"]
        if (
            not path.return_end.is_unique
            or path.loc[path["return"].notna(), "return_end"].tolist()
            != coverage.loc[observed, "return_end"].tolist()
            or bool(metrics.loc[metrics.strategy.eq(strategy), "complete"].iloc[0])
            != bool(observed.all())
        ):
            raise ValueError(f"strategy performance disagrees with evaluation coverage: {strategy}")
    return manifest, tables


def _ppo_observations(arguments, manifest, tables):
    config, dataset = arguments["config"], arguments["dataset"]
    model = arguments["model"]
    layout = ObservationLayout(tuple(dataset.returns.columns), config.lookback)
    wrapper = PPOPolicyWrapper(model, layout)
    first = pd.Timestamp(manifest["first_return_end"])
    end = pd.Timestamp(manifest["requested_test"]["end"])
    sliced, _, _ = _period_dataset(dataset, first, end, config.lookback)
    if sliced is None:
        raise ValueError("no evaluable PPO observations")
    effective = evaluation_env_config(config)
    if manifest["ppo_evaluation_mode"] == "no_early_stop_inference":
        effective = effective.model_copy(update={"mdd_limit": 1.0})
    env = _make_env(sliced, effective, seed=0)
    observations, decision_dates, nav_dates, nav = [], [], [], []
    try:
        observation, info = env.reset(seed=0)
        nav_dates.append(info["timestamp"])
        nav.append(info["nav"])
        while True:
            observations.append(observation.copy())
            decision_dates.append(info["timestamp"])
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(action)
            nav_dates.append(info["timestamp"])
            nav.append(info["nav"])
            if terminated or truncated:
                break
    finally:
        env.close()
    ppo_nav = tables["nav_history"].query("strategy == 'ppo'")
    if not pd.DatetimeIndex(pd.to_datetime(ppo_nav.evaluation_at, utc=True)).equals(
        pd.DatetimeIndex(nav_dates)
    ) or not np.allclose(ppo_nav.nav, nav, rtol=1e-10, atol=1e-12):
        raise ValueError("observation replay does not match comparison PPO NAV")
    observed = layout.to_frame(np.asarray(observations), index=pd.DatetimeIndex(decision_dates))
    return wrapper, observed, info


def _records(tables, window, manifest, proof):
    rows = tables["strategy_metrics"]
    numeric = rows.drop(columns=[*LABELS, "strategy", "complete", "benchmark_comparable"])
    records = []
    for position, row in rows.iterrows():
        metrics = {
            name: float(value) if row.complete and pd.notna(value) else None
            for name, value in numeric.loc[position].items()
        }
        records.append(
            {
                "strategy": row.strategy,
                "window": window,
                **manifest["requested_test"],
                "metrics": metrics,
                "provenance": proof,
            }
        )
    return records


def _write_json(root, name, value):
    content = json_bytes(value)
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)
    return {"path": name, "sha256": sha256(content)}


def _point(date, value, series):
    return {"timestamp": pd.Timestamp(date).isoformat(), "value": float(value), "series": series}


def _dashboard(root, arguments, tables, records, info, proof, curves, explained, stamp):
    nav = tables["nav_history"]
    ppo = nav.query("strategy == 'ppo'")
    portfolio = [_point(row.evaluation_at, row.nav - 1.0, "ppo") for row in ppo.itertuples()]
    comparison = [
        _point(row.evaluation_at, row.nav - 1.0, row.strategy) for row in nav.itertuples()
    ]
    weights = PortfolioWeights(
        asset_weights=[
            {"asset": asset, "weight": weight}
            for asset, weight in zip(arguments["dataset"].returns.columns, info["weights"][:-1])
        ],
        cash_weight=info["weights"][-1],
    )
    ppo_metrics = next(record["metrics"] for record in records if record["strategy"] == "ppo")
    events = []
    for row in tables["safeguard_events"].query("strategy == 'ppo'").to_dict("records"):
        for field in ("first_reached_15pct", "first_exceeded_15pct"):
            if pd.notna(row[field]):
                events.append(
                    {"timestamp": row[field], "reason": field, "trace_id": proof["run_id"]}
                )
    if len(events) == 2 and events[0]["timestamp"] == events[1]["timestamp"]:
        events = events[1:]
    output = explained["output"]
    pages = {
        "portfolio": {"weights": weights.model_dump(mode="json"), "cumulative_returns": portfolio},
        "rl_performance": {
            "learning_curve": [
                {"step": int(row.timesteps), "value": float(row.reward)}
                for row in curves.itertuples()
            ],
            "metrics": [{"strategy": r["strategy"], "metrics": r["metrics"]} for r in records],
            "comparison": comparison,
        },
        "risk_monitor": {
            "var_95": ppo_metrics.get("var_95"),
            "cvar_95": ppo_metrics.get("cvar_95"),
            "mdd_limit": arguments["config"].mdd_limit,
            "drawdown": [
                _point(date, value, "ppo")
                for date, value in zip(ppo.evaluation_at, 1 - ppo.nav / ppo.nav.cummax())
            ],
            "safeguard_events": events,
            "tags": [
                tag.model_dump(mode="json")
                for tag in arguments["dataset"].risk_tags
                if tag.available_at <= info["timestamp"]
            ],
        },
        "anova": {"diagnostics": [{"status": "unavailable", "reason": "single_synthetic_run"}]},
        "shap": {
            "output": output,
            "base_value": explained["base_value"],
            "prediction": explained["prediction"],
            "summary": [
                {"feature": name, "mean_abs_shap": abs(value)}
                for name, value in explained["contributions"].items()
            ],
            "force": [
                {"feature": name, "contribution": value, "value": explained["observation"][name]}
                for name, value in explained["contributions"].items()
            ],
        },
    }
    return {
        name: _write_json(
            root,
            f"{name}.json",
            {
                "as_of": stamp.isoformat() if name == "shap" else info["timestamp"].isoformat(),
                "model_id": proof["model_id"],
                "provenance": proof,
                "data": data,
            },
        )
        for name, data in pages.items()
    }


def export_artifacts(package, package_sha, comparison, comparison_sha, output, *, window):
    """Create API preload inputs for one verified run/window without choosing among seeds."""
    if not isinstance(window, str) or not 1 <= len(window) <= 32 or not window.strip():
        raise ValueError("an explicit API window key is required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        with verified_snapshot(package, package_sha) as (snapshot, files):
            bundle = object_json(files["bundle-manifest.json"])
            if bundle["kind"] != "synthetic":
                raise ValueError("this acceptance export requires a synthetic handoff")
            manifest, tables = _comparison(comparison, bundle, files, comparison_sha)
            with prepare_diagnostic_inputs(files["diagnostic.json"], root=snapshot) as arguments:
                return _export(
                    output, files, manifest, tables, arguments, window, package_sha, comparison_sha
                )
    except BaseException:
        shutil.rmtree(output)
        raise


def _export(root, files, manifest, tables, arguments, window, package_sha, comparison_sha):
    bundle = object_json(files["bundle-manifest.json"])
    checkpoint = bundle["checkpoint"]
    proof = ArtifactProvenance(
        kind=bundle["kind"],
        status="unverified",
        training_approved=False,
        policy_status="candidate",
        performance_decision=manifest["performance_decision"],
        model_id=checkpoint["sha256"],
        run_id=bundle["run_id"],
        evaluation={
            **manifest["requested_test"],
            "complete": dict(
                zip(tables["strategy_metrics"].strategy, tables["strategy_metrics"].complete)
            ),
            "anova": "unavailable",
        },
        input_sha256={"handoff_manifest": package_sha, "comparison_manifest": comparison_sha},
        output_sha256=manifest["output_sha256"],
    ).model_dump(mode="json")
    wrapper, observed, info = _ppo_observations(arguments, manifest, tables)
    model_name = "checkpoint.zip"
    (root / model_name).write_bytes(files[checkpoint["path"]])
    descriptor = {
        "path": "observations.parquet",
        **write_frame(observed, root / "observations.parquet"),
    }
    policy = {
        "model_id": proof["model_id"],
        "assets": list(wrapper.layout.assets),
        "lookback": wrapper.layout.lookback,
        "observations": descriptor,
        "provenance": proof,
    }
    tags = [tag for tag in arguments["dataset"].risk_tags if tag.available_at <= observed.index[-1]]
    if tags:
        policy["risk_tags"] = {
            "path": "risk_tags.parquet",
            **write_frame(
                pd.DataFrame([tag.model_dump(mode="json") for tag in tags]),
                root / "risk_tags.parquet",
            ),
        }
    policy_pin = _write_json(root, "policy.json", policy)
    records = _records(tables, window, manifest, proof)
    metrics = _write_json(root, "backtest/metrics.json", {"records": records})
    backtest = _write_json(
        root,
        "backtest/run_manifest.json",
        {"files": {"metrics.json": {**metrics, "path": "metrics.json"}}},
    )
    fit = arguments["training_manifest"]["data_cutoff"]
    bounds = {"train_start": fit["actual_first_index"], "train_end": fit["actual_last_index"]}
    background = collect_training_observations(
        arguments["model"],
        arguments["dataset"],
        arguments["config"],
        layout=wrapper.layout,
        **bounds,
    )
    sample = sample_training_background(
        background,
        pd.Series("synthetic_smoke", index=background.index),
        layout=wrapper.layout,
        n_samples=min(4, len(background)),
        seed=42,
        expected_strata=("synthetic_smoke",),
        **bounds,
    )
    asset = wrapper.layout.assets[0]
    target = observed.iloc[[-1]]
    shap = compute_kernel_shap(
        wrapper, sample, target, model_id=proof["model_id"], outputs=[asset], nsamples=64
    )
    shap.metadata["provenance"] = proof
    save_shap_result(shap, root / "shap")
    stamp = target.index[0]
    curves = pd.read_csv(io.BytesIO(files[f"runs/{proof['run_id']}/curves.csv"]))
    sections = _dashboard(
        root, arguments, tables, records, info, proof, curves, shap.explanation(asset, stamp), stamp
    )
    paths = {
        "summary": save_summary_plot(shap, asset, root / "summary.png"),
        "force": save_force_plot(shap, asset, stamp, root / "force.png"),
    }
    sections["shap"]["plots"] = {
        name: {"path": path.name, "sha256": sha256(path.read_bytes()), "role": name, "title": name}
        for name, path in paths.items()
    }
    dashboard = _write_json(root, "dashboard.json", {"schema_version": 1, "sections": sections})
    action, _ = arguments["model"].predict(target.to_numpy()[0], deterministic=True)
    weights = np.exp(np.asarray(action, dtype=float) - float(np.max(action)))
    weights /= weights.sum()
    return _write_json(
        root,
        "acceptance.json",
        {
            "provenance": proof,
            "window": window,
            "as_of": stamp.isoformat(),
            "asset": asset,
            "direct_weights": dict(zip(wrapper.layout.output_names, weights.tolist())),
            "environment": {
                "API_CHECKPOINT_PATH": model_name,
                "API_CHECKPOINT_SHA256": proof["model_id"],
                "API_POLICY_MANIFEST": policy_pin["path"],
                "API_POLICY_MANIFEST_SHA256": policy_pin["sha256"],
                "API_BACKTEST_DIR": "backtest",
                "API_BACKTEST_MANIFEST_SHA256": backtest["sha256"],
                "API_SHAP_DIR": "shap",
                "API_SHAP_MANIFEST_SHA256": sha256((root / "shap/manifest.json").read_bytes()),
                "API_DASHBOARD_MANIFEST": dashboard["path"],
                "API_DASHBOARD_MANIFEST_SHA256": dashboard["sha256"],
            },
        },
    )
