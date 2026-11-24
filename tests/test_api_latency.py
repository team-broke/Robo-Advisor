"""Measure the serving contract with offline fixtures and the real research queue."""

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import platform
from time import perf_counter

from fastapi.testclient import TestClient

from robo_advisor.api.app import create_app
from robo_advisor.api.cache import CacheKey, ResearchCache, ResearchJobs
from robo_advisor.api.lifecycle import PreloadSettings
from tests.test_api import AS_OF, STAMP, FakeBacktests, FakeExplainer, FakeOptimizer
from tests.test_api_preload import MODEL, VERSION, runner, wait_job
from tests.test_dashboard_api import SECTIONS, artifact

LIMIT_MS = 5_000
COLD_SAMPLES = 5
WARM_SAMPLES = 25
CACHED_QUESTION = "SPY 위험 요인"
REQUESTS = (
    ("GET", "/health", "/health", {}, 200),
    ("POST", "/optimize", "/optimize", {"json": {"as_of": AS_OF}}, 200),
    ("POST", "/explain", "/explain", {"json": {"asset": "SPY", "as_of": AS_OF}}, 200),
    ("GET", "/backtest", "/backtest", {"params": {"strategy": "mvo", "window": "2024"}}, 200),
    ("GET", "/research/jobs/{job_id}", "/research/jobs/{job_id}", {}, 200),
    *(("GET", f"/dashboard/{name}", f"/dashboard/{name}", {}, 200) for name in SECTIONS),
    (
        "GET",
        "/dashboard/shap/plots/{plot_id}",
        "/dashboard/shap/plots/summary",
        {},
        200,
    ),
    (
        "POST",
        "/research [cache hit]",
        "/research",
        {"json": {"question": CACHED_QUESTION, "as_of": AS_OF}},
        200,
    ),
    ("POST", "/research [job accepted]", "/research", None, 202),
)


@contextmanager
def serving_app(settings, cached_result):
    cache = ResearchCache()
    cache.put(CacheKey(MODEL, VERSION, CACHED_QUESTION, STAMP), cached_result)
    jobs = ResearchJobs(runner, cache, model_id=MODEL, artifact_version=VERSION)
    loaders = {
        "optimizer": lambda _: FakeOptimizer(),
        "explainer": lambda _: FakeExplainer(),
        "backtests": lambda _: FakeBacktests(),
        "research": lambda _: jobs,
    }
    try:
        app = create_app(preload_settings=settings, loaders=loaders)
        started = perf_counter()
        with TestClient(app) as client:
            yield client, jobs, (perf_counter() - started) * 1_000
    finally:
        jobs.close()


def measure(client, jobs, request, sample):
    method, _, path, kwargs, expected = request
    if "{job_id}" in path:
        job_id = jobs.submit(f"status setup {sample}", STAMP)
        wait_job(jobs, job_id)
        path = path.format(job_id=job_id)
    if kwargs is None:
        kwargs = {"json": {"question": f"uncached sample {sample}", "as_of": AS_OF}}

    started = perf_counter()
    response = client.request(method, path, **kwargs)
    elapsed_ms = (perf_counter() - started) * 1_000
    assert response.status_code == expected, response.text
    if expected == 202:
        assert response.json()["status"] == "queued"
        assert response.json()["job_id"]
    return elapsed_ms


def statistics(samples):
    return {
        "count": len(samples),
        "p95_ms": sorted(samples)[math.ceil(0.95 * len(samples)) - 1],
        "max_ms": max(samples),
    }


def test_api_latency(tmp_path):
    manifest, digest, _ = artifact(tmp_path)
    settings = PreloadSettings(
        _env_file=None, dashboard_manifest=manifest, dashboard_manifest_sha256=digest
    )
    cached_result = runner(CACHED_QUESTION, STAMP)
    requests = {}
    startups = []
    for request in REQUESTS:
        method, name, _, _, expected = request
        label = f"{method} {name}"
        cold, warm = [], []
        for sample in range(COLD_SAMPLES):
            with serving_app(settings, cached_result) as (client, jobs, startup_ms):
                startups.append(startup_ms)
                cold.append(measure(client, jobs, request, sample))

        with serving_app(settings, cached_result) as (client, jobs, _):
            measure(client, jobs, request, 0)
            for sample in range(1, WARM_SAMPLES + 1):
                warm.append(measure(client, jobs, request, sample))

        requests[label] = {
            "status_code": expected,
            "cold": statistics(cold),
            "warm": statistics(warm),
        }

    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "transport": "in-process ASGI TestClient",
        "data": "synthetic dashboard, optimizer, SHAP, backtest and offline research fixtures",
        "cold": "first HTTP request per new app lifespan; required state seeded before timing",
        "warm": "one warm-up request, then repeated requests in the same app lifespan",
        "startup": statistics(startups),
        "p95": "nearest rank",
        "limit_ms": LIMIT_MS,
        "research_contract": {
            "200": "cache hit",
            "202": "queue acceptance on a unique cache miss; research processing excluded",
        },
        "endpoints": requests,
    }

    output = Path(os.environ.get("API_LATENCY_OUTPUT", "artifacts/performance/api_latency.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    violations = [
        f"{route} {condition}"
        for route, metrics in requests.items()
        for condition in ("cold", "warm")
        if metrics[condition]["p95_ms"] > LIMIT_MS
    ]
    assert not violations, f"p95 exceeded {LIMIT_MS} ms: {', '.join(violations)}"
