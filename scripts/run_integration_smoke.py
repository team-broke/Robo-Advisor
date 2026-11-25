"""Produce a complete synthetic CPU acceptance run in a fresh directory."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

import httpx
import numpy as np
import pandas as pd

from robo_advisor.data.benchmark_audit import audit_benchmark_data
from robo_advisor.backtest.comparison import ComparisonPolicy, compare_window, save_comparison
from robo_advisor.diagnostics.inputs import prepare_diagnostic_inputs
from robo_advisor.integration.artifact_export import export_artifacts, verified_snapshot
from robo_advisor.ops.ppo_preflight import check_inputs, json_bytes, object_json, sha256
from robo_advisor.ops.ppo_reproducibility import diagnose_run, pack_run, verify_run

RUN = "synthetic-61"
POLICY = "issue61-smoke-candidate"
WINDOW = "synthetic-smoke"


def pin(root, path):
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256(path.read_bytes())}


def write(root, name, content):
    path = root / name
    path.write_bytes(json_bytes(content))
    return pin(root, path)


def inputs(root):
    data = root / "data"
    data.mkdir()
    dates = pd.bdate_range("2024-01-02", periods=40, tz="UTC") + pd.Timedelta(hours=6.5)
    assets = ["SYNTH_A", "SYNTH_B", "SYNTH_C"]
    rng = np.random.default_rng(61)
    returns = pd.DataFrame(rng.normal(0.001, 0.006, (len(dates), 3)), index=dates, columns=assets)
    values = {
        "returns": returns,
        "rsi": returns * 500 + 50,
        "macd": returns.ewm(span=4, adjust=False).mean(),
        "risk_tags": pd.DataFrame(
            [
                {
                    "event_type": "sharp_move",
                    "asset": asset,
                    "severity": 0.25,
                    "confidence": 0.8,
                    "published_at": dates[day - 1],
                    "available_at": dates[day],
                    "source_url": "https://example.invalid/synthetic-scenario",
                    "trace_id": trace,
                }
                for asset, day, trace in (
                    (assets[0], 27, "synthetic-observed"),
                    (assets[1], 39, "synthetic-final-session"),
                )
            ]
        ),
    }
    descriptors = {}
    for name, frame in values.items():
        path = data / f"{name}.parquet"
        frame.to_parquet(path)
        descriptors[path.name] = pin(root, path)
    cuts = [
        date.isoformat()
        for date in (dates[0], dates[15], dates[24], dates[-1] + pd.Timedelta(days=1))
    ]
    config = write(
        root,
        "training.json",
        {
            "device": "cpu",
            "seed": 42,
            "n_envs": 1,
            "vec_env": "dummy",
            "total_timesteps": 32,
            "ppo": {
                "n_steps": 8,
                "batch_size": 8,
                "n_epochs": 1,
                "net_arch": {"pi": [8], "vf": [8]},
            },
            "env": {"lookback": 3, "min_episode_steps": 2, "max_episode_steps": 4},
            "data": {
                "dataset_dir": "data",
                "train_start": cuts[0],
                "train_end": dates[23].isoformat(),
            },
            "validation": {"start": cuts[1], "eval_freq": 8},
        },
    )
    dataset = write(
        root,
        "dataset.json",
        {
            "schema_version": 1,
            "dataset_kind": "synthetic_test",
            "asset_order": assets,
            "training_approved": False,
            "files": descriptors,
        },
    )
    calendar = write(
        root,
        "calendar.json",
        {
            "schema_version": 1,
            "source": "synthetic weekday schedule",
            "status": "synthetic",
            "start": cuts[0],
            "end": cuts[-1],
            "evaluation_at": dates.map(pd.Timestamp.isoformat).tolist(),
        },
    )
    policy = write(
        root,
        "policy.json",
        {
            "schema_version": 1,
            "version": POLICY,
            "status": "synthetic",
            "calendar_sha256": calendar["sha256"],
        },
    )
    return dates, {
        "schema_version": 1,
        "kind": "synthetic",
        "currency": "KRW",
        "dataset": descriptors,
        "training_config": config,
        "dataset_manifest": dataset,
        "calendar": calendar,
        "evaluation_policy": policy,
        "periods": [
            {"name": name, "start": start, "end": end}
            for name, start, end in zip(("train", "validation", "test"), cuts, cuts[1:])
        ],
    }


def benchmark(root, dates):
    def csv(name, **columns):
        path = root / f"{name}.csv"
        pd.DataFrame(columns).to_csv(path, index=False, lineterminator="\n")
        return pin(root, path)

    us = dates.insert(0, dates[0] - pd.Timedelta(days=1))
    kr_days, us_days = dates.strftime("%Y-%m-%d"), us.strftime("%Y-%m-%d")
    spec = {
        "schema_version": 1,
        "kind": "synthetic",
        "start": kr_days[0],
        "end_exclusive": (dates[-1] + pd.Timedelta(days=1)).date().isoformat(),
        "calendars": {
            "KRX": csv("kr_calendar", session_date=kr_days, close=dates),
            "US": csv(
                "us_calendar",
                session_date=us_days,
                close=(
                    (pd.to_datetime(us_days) + pd.Timedelta(hours=16)).tz_localize(
                        "America/New_York"
                    )
                ),
            ),
        },
        "evaluation": csv("evaluation", session_date=kr_days, evaluation_at=dates),
        "benchmarks": {
            "KOSPI": {
                **csv("kospi", Date=kr_days, Close=3000 + np.arange(len(dates)) * 1.3),
                "index_id": "KOSPI",
                "source": "krx",
                "symbol": "KOSPI",
                "currency": "KRW",
                "return_type": "PR",
            },
            "SP500": {
                **csv("spx", Date=us_days, Close=4000 + np.arange(len(us)) * 2.5),
                "index_id": "SP500",
                "source": "unknown",
                "currency": "USD",
                "return_type": "PR",
            },
        },
        "fx": {
            **csv(
                "fx",
                date=kr_days,
                mar_krw_per_usd=1000.0 + np.arange(len(dates)) * 0.1,
                source_identity="smbs:USD_KRW_MAR:public_period_table",
                actual_available_at=dates - pd.Timedelta(hours=3),
            ),
            "source": "smbs",
            "max_age_days": 7,
        },
    }
    descriptor = write(root, "benchmark.json", spec)
    return descriptor, audit_benchmark_data(spec, root=root)


def run(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    dates, spec = inputs(source)
    report = check_inputs(spec, source).result
    write(output, "preflight.json", report)
    if report["status"] != "ready":
        raise ValueError(report["reasons"])
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "robo_advisor.rl.train",
            "--smoke",
            "--run-id",
            RUN,
            "--config",
            str(source / "training.json"),
            "--dataset-dir",
            str(source / "data"),
        ],
        env={
            **os.environ,
            "MODELS_DIR": str(source / "models"),
            "ARTIFACTS_DIR": str(source / "artifacts"),
        },
        check=True,
    )
    run_dir = source / "artifacts/training" / RUN
    spec.update(
        run_id=RUN,
        training_manifest=pin(source, run_dir / "manifest.json"),
        checkpoint={
            **pin(source, source / f"models/{RUN}/final_model.zip"),
            "run_id": RUN,
            "kind": "final",
        },
        monitor_files=[
            pin(source, path) for path in sorted((run_dir / "monitor").glob("*.monitor.csv"))
        ],
        curves={
            "curves_csv": pin(source, run_dir / "curves.csv"),
            "convergence": pin(source, run_dir / "convergence.json"),
            "plot": pin(source, source / f"artifacts/training/curves/{RUN}.png"),
        },
    )
    write(source, "handoff-spec.json", spec)
    pack_run(spec, source, output / "pending")
    digest = pin(output, output / "pending/bundle-manifest.json")["sha256"]
    write(output, "handoff-pin.json", {"sha256": digest})
    package = output / "relocated" / "handoff"
    package.parent.mkdir()
    shutil.move(output / "pending", package)
    verify_run(package, expected_manifest_sha256=digest)
    diagnose_run(package, output / "diagnostic", expected_manifest_sha256=digest)
    scenario = output / "candidate-benchmark"
    scenario.mkdir()
    benchmark_pin, audit = benchmark(scenario, dates)
    if audit["status"] == "failed":
        raise ValueError(f"synthetic benchmark audit failed: {audit['issues']}")
    with verified_snapshot(package, digest) as (snapshot, files):
        with prepare_diagnostic_inputs(files["diagnostic.json"], root=snapshot) as arguments:
            result = compare_window(
                schedule=dates,
                benchmark_audit=audit,
                policy=ComparisonPolicy(
                    POLICY, "include_first_session", "observe_only", "PR", "smbs", 5
                ),
                **arguments,
            )
    if not result.manifest["comparable"]:
        raise ValueError("synthetic comparison did not cover all four strategies")
    result.manifest["scenario"] = {"kind": "synthetic", "benchmark_input": benchmark_pin}
    comparison = save_comparison(result, output / "comparison")
    return export_artifacts(
        package,
        digest,
        comparison,
        pin(comparison, comparison / "manifest.json")["sha256"],
        output / "api",
        window=WINDOW,
    )


def check_http(client, acceptance):
    stamp = acceptance["as_of"]
    proof = acceptance["provenance"]
    results = {}

    def request(method, path, expected=200, **kwargs):
        response = client.request(method, path, **kwargs)
        key = path + (":" + kwargs["params"]["strategy"] if "params" in kwargs else "")
        results[f"{method} {key}"] = response.status_code
        if response.status_code != expected:
            raise ValueError(f"{method} {key}: expected {expected}, got {response.status_code}")
        return response

    health = request("GET", "/health").json()
    for service in ("optimizer", "explainer", "backtests", "dashboard"):
        if health["services"].get(service) is not True:
            raise ValueError(f"{service} is not ready: {health.get('reasons')}")
    if health["services"].get("research") is not False:
        raise ValueError("synthetic smoke must keep external research disabled")
    optimize = request("POST", "/optimize", json={"as_of": stamp}).json()
    weights = optimize["weights"]
    actual = {item["asset"]: item["weight"] for item in weights["asset_weights"]}
    actual["CASH"] = weights["cash_weight"]
    direct = acceptance["direct_weights"]
    if actual.keys() != direct.keys() or not np.allclose(
        [actual[key] for key in direct], list(direct.values()), rtol=0, atol=1e-7
    ):
        raise ValueError("HTTP predictions differ from direct checkpoint inference")
    prediction_error = max(abs(actual[key] - direct[key]) for key in direct)
    explanation = request(
        "POST", "/explain", json={"as_of": stamp, "asset": acceptance["asset"]}
    ).json()
    artifacts = [optimize, explanation]
    for strategy in ("ppo", "mvo", "equal_weight", "benchmark"):
        artifacts.append(
            request(
                "GET", "/backtest", params={"strategy": strategy, "window": acceptance["window"]}
            ).json()
        )
    for section in ("portfolio", "rl_performance", "shap", "anova", "risk_monitor"):
        artifacts.append(request("GET", f"/dashboard/{section}").json())
    if any(item.get("provenance") != proof for item in artifacts):
        raise ValueError("response provenance was lost or replaced")
    for plot in ("summary", "force"):
        response = request("GET", f"/dashboard/shap/plots/{plot}")
        if not response.headers["content-type"].startswith("image/png") or (
            sha256(response.content) != response.headers["etag"].strip('"')
        ):
            raise ValueError("dashboard image differs from the validated PNG")
    request("POST", "/research", 503, json={"as_of": stamp, "question": "synthetic smoke"})
    request("GET", "/research/jobs/unconfigured", 503)
    return {
        "kind": proof["kind"],
        "status": proof["status"],
        "training_approved": False,
        "http_status": results,
        "max_absolute_weight_error": prediction_error,
        "unavailable": {
            "research": health.get("reasons", {}).get("research"),
            "anova": "single_synthetic_run",
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-url", help="Check the running API against an existing smoke run")
    args = parser.parse_args()
    if args.check_url:
        acceptance = object_json((args.output / "api/acceptance.json").read_bytes())
        with httpx.Client(base_url=args.check_url, timeout=15) as client:
            report = check_http(client, acceptance)
        print(write(args.output, "http-verification.json", report))
    else:
        print(run(args.output))
