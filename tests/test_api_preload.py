"""네트워크·학습·전체 SHAP 없이 사전 로드와 캐시·job 동작을 검증한다."""

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta, timezone
from hashlib import sha256
import json
import socket
from threading import Event
import time

from fastapi.testclient import TestClient
import pandas as pd
import pytest

from robo_advisor.api.app import create_app
from robo_advisor.api.cache import CacheKey, ResearchCache, ResearchJobs
from robo_advisor.api import lifecycle
from robo_advisor.api.lifecycle import PreloadSettings, DirectoryBacktestLoader, preload_services
from robo_advisor.api.services import Services, ServiceUnavailableError, NotFoundError
from robo_advisor.config import Settings
from robo_advisor.research.graph import run_research
from tests.test_api import AS_OF, STAMP, FakeOptimizer
from tests.test_research_graph import FakeLLM, FakeStore

MODEL = "a" * 64
VERSION = "b" * 64


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)


def runner(question, as_of):
    from dataclasses import replace
    from tests.test_research_graph import hit

    return run_research(
        question,
        as_of,
        llm=FakeLLM(),
        store=FakeStore(
            [
                [
                    replace(
                        hit(),
                        published_at="2024-01-01T00:00:00Z",
                        available_at="2024-01-01T00:00:00Z",
                    ),
                    replace(
                        hit("two"),
                        published_at="2024-01-01T00:00:00Z",
                        available_at="2024-01-01T00:00:00Z",
                    ),
                ]
            ]
        ),
    )


def wait_job(jobs, job_id):
    end = time.monotonic() + 5
    while time.monotonic() < end:
        result = jobs.job(job_id)
        if result.status in {"completed", "failed"}:
            return result
        time.sleep(0.005)
    raise AssertionError("job did not finish")


def test_cache_key_versions_question_and_cutoff():
    key = CacheKey(MODEL, VERSION, "  ＳＰＹ   위험  ", STAMP)
    assert key == CacheKey(MODEL, VERSION, "SPY 위험", STAMP)
    assert (
        key.digest
        == CacheKey(
            MODEL,
            VERSION,
            "SPY 위험",
            STAMP.astimezone(timezone(timedelta(hours=9))),
        ).digest
    )
    assert key.digest != CacheKey("c" * 64, VERSION, key.question, STAMP).digest
    assert key.digest != CacheKey(MODEL, "c" * 64, key.question, STAMP).digest
    assert key.digest != CacheKey(MODEL, VERSION, key.question, STAMP + timedelta(seconds=1)).digest
    with pytest.raises(ValueError):
        CacheKey(MODEL, VERSION, "SPY", STAMP.replace(tzinfo=None))


def test_memory_disk_cache_preload_and_isolation(tmp_path):
    cache = ResearchCache(tmp_path / "cache", max_entries=2)
    key = CacheKey(MODEL, VERSION, "SPY", STAMP)
    cache.put(key, runner("SPY", STAMP))
    reloaded = ResearchCache(tmp_path / "cache", max_entries=2)
    assert reloaded.get(key).status == "completed"
    assert reloaded.get(CacheKey("c" * 64, VERSION, "SPY", STAMP)) is None
    assert reloaded.get(CacheKey(MODEL, "c" * 64, "SPY", STAMP)) is None
    assert reloaded.get(CacheKey(MODEL, VERSION, "SPY", STAMP + timedelta(days=1))) is None
    first = reloaded.get(key)
    assert first is not reloaded.get(key)
    for question in ("IEF", "GLD"):
        reloaded.put(CacheKey(MODEL, VERSION, question, STAMP), runner(question, STAMP))
    assert len(list((tmp_path / "cache").glob("*.json"))) == 2
    assert reloaded.get(key) is None


def test_corrupt_disk_cache_fails_validation(tmp_path):
    cache = ResearchCache(tmp_path)
    key = CacheKey(MODEL, VERSION, "SPY", STAMP)
    cache.put(key, runner("SPY", STAMP))
    path = tmp_path / f"{key.digest}.json"
    path.write_text(path.read_text().replace("SPY", "IEF"))
    with pytest.raises(ValueError, match="integrity"):
        ResearchCache(tmp_path)


def test_result_cutoff_and_question_must_match_cache():
    cache = ResearchCache()
    with pytest.raises(ValueError):
        cache.put(CacheKey(MODEL, VERSION, "SPY", STAMP), runner("IEF", STAMP))
    with pytest.raises(ValueError):
        cache.put(CacheKey(MODEL, VERSION, "SPY", STAMP), runner("SPY", STAMP + timedelta(days=1)))


def test_queue_deduplicates_caps_retention_and_caches():
    started, release = Event(), Event()
    calls = []

    def blocked(question, as_of):
        calls.append(question)
        started.set()
        assert release.wait(5)
        return runner(question, as_of)

    jobs = ResearchJobs(
        blocked,
        ResearchCache(),
        model_id=MODEL,
        artifact_version=VERSION,
        max_workers=1,
        max_jobs=2,
    )
    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            ids = list(executor.map(lambda _: jobs.submit("SPY", STAMP), range(8)))
        assert len(set(ids)) == 1 and started.wait(2)
        first = ids[0]
        assert jobs.job(first).status == "running"
        second = jobs.submit("IEF", STAMP)
        assert jobs.job(second).status == "queued"
        with pytest.raises(ServiceUnavailableError, match="full"):
            jobs.submit("GLD", STAMP)
        release.set()
        assert wait_job(jobs, first).status == "completed"
        assert wait_job(jobs, second).status == "completed"
        assert calls == ["SPY", "IEF"]
        assert jobs.cached("SPY", STAMP).status == "completed"
        third = jobs.submit("GLD", STAMP)
        assert wait_job(jobs, third).status == "completed"
        with pytest.raises(NotFoundError):
            jobs.job(first)
    finally:
        release.set()
        jobs.close()
    with pytest.raises(ServiceUnavailableError, match="closed"):
        jobs.submit("new", STAMP)


@pytest.mark.parametrize("kind", ["exception", "failed_result"])
def test_jobs_report_failures_without_sensitive_error_text(kind):
    def failed(question, as_of):
        if kind == "exception":
            raise RuntimeError("secret@example.com sk-private-api-key thoughts: internal reasoning")
        return run_research(question, as_of, store=FakeStore([[]]), llm=FakeLLM())

    jobs = ResearchJobs(failed, ResearchCache(), model_id=MODEL, artifact_version=VERSION)
    try:
        result = wait_job(jobs, jobs.submit("SPY", STAMP))
        assert result.status == "failed" and result.failure_reason and result.result is None
        assert "secret@example.com" not in result.model_dump_json()
        assert "internal reasoning" not in result.model_dump_json()
        assert jobs.cached("SPY", STAMP) is None
    finally:
        jobs.close()


def test_api_job_lookup_cache_hit_and_404():
    jobs = ResearchJobs(runner, ResearchCache(), model_id=MODEL, artifact_version=VERSION)
    try:
        with TestClient(create_app(Services(research=jobs))) as client:
            response = client.post("/research", json={"question": "SPY", "as_of": AS_OF})
            assert response.status_code == 202
            job_id = response.json()["job_id"]
            wait_job(jobs, job_id)
            assert client.get(f"/research/jobs/{job_id}").json()["status"] == "completed"
            assert client.get("/research/jobs/absent").status_code == 404
            assert (
                client.post("/research", json={"question": "SPY", "as_of": AS_OF}).status_code
                == 200
            )
            schema = client.get("/openapi.json").json()
            assert schema["components"]["schemas"]["ResearchJobStatus"]["examples"]
    finally:
        jobs.close()


def test_lifespan_partial_failure_health_reasons_and_load_once():
    calls = []

    def good(settings):
        calls.append("optimizer")
        return FakeOptimizer()

    def bad(settings):
        calls.append("explainer")
        raise ValueError("/private/path api_key=super-secret")

    app = create_app(
        preload_settings=PreloadSettings(_env_file=None),
        loaders={
            "optimizer": good,
            "explainer": bad,
            "research": lambda _: None,
            "backtests": lambda _: None,
        },
    )
    assert not calls
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["services"]["optimizer"] and not health["services"]["explainer"]
        assert health["reasons"]["explainer"] == "load_failed:ValueError"
        assert "super-secret" not in json.dumps(health)
        for _ in range(2):
            assert client.post("/optimize", json={"as_of": AS_OF}).status_code == 200
        assert client.post("/explain", json={"asset": "SPY", "as_of": AS_OF}).status_code == 503
    assert calls == ["optimizer", "explainer"]


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def policy_artifacts(tmp_path):
    from stable_baselines3 import PPO
    from robo_advisor.data.snapshot import write_frame
    from robo_advisor.explain.policy import ObservationLayout, PPOPolicyWrapper
    from robo_advisor.explain.shap_values import ShapResult, save_shap_result
    from robo_advisor.rl.environment import PortfolioEnv

    layout = ObservationLayout(("SPY", "IEF"), 1)
    index = pd.date_range(STAMP, periods=4)
    returns = pd.DataFrame(0.0, index=index, columns=layout.assets)
    env = PortfolioEnv(returns, returns + 50, returns, lookback=1, n_assets=2)
    model = PPO(
        "MlpPolicy", env, n_steps=2, batch_size=2, policy_kwargs={"net_arch": [4]}, device="cpu"
    )
    model.save(tmp_path / "policy.zip")
    model_id = sha256((tmp_path / "policy.zip").read_bytes()).hexdigest()
    observation = layout.to_frame(env.reset()[0], index=pd.DatetimeIndex([STAMP]))
    wrapper = PPOPolicyWrapper(model, layout)
    prediction = pd.DataFrame(
        wrapper.predict_mean_actions(observation),
        index=observation.index,
        columns=layout.output_names,
    )
    descriptor = write_frame(observation, tmp_path / "observations.parquet")
    manifest_hash = write_json(
        tmp_path / "policy.json",
        {
            "model_id": model_id,
            "assets": list(layout.assets),
            "lookback": 1,
            "observations": {"path": "observations.parquet", **descriptor},
        },
    )
    shap_result = ShapResult(
        {name: observation * 0 for name in layout.output_names},
        {name: float(prediction[name].iloc[0]) for name in layout.output_names},
        prediction,
        observation,
        layout.feature_groups,
        {
            "schema_version": 1,
            "model_id": model_id,
            "assets": list(layout.assets),
            "lookback": 1,
            "outputs": list(layout.output_names),
        },
    )
    save_shap_result(shap_result, tmp_path / "shap")
    env.close()
    return PreloadSettings(
        _env_file=None,
        checkpoint_path=tmp_path / "policy.zip",
        checkpoint_sha256=model_id,
        policy_manifest=tmp_path / "policy.json",
        policy_manifest_sha256=manifest_hash,
        shap_dir=tmp_path / "shap",
        shap_manifest_sha256=sha256((tmp_path / "shap/manifest.json").read_bytes()).hexdigest(),
    )


def test_real_policy_and_shap_preload_only_then_serve_without_training(
    policy_artifacts, monkeypatch
):
    from stable_baselines3 import PPO
    from robo_advisor.explain import shap_values

    def forbidden(*args, **kwargs):
        raise AssertionError("training/full SHAP is forbidden")

    monkeypatch.setattr(PPO, "learn", forbidden)
    monkeypatch.setattr(shap_values, "compute_kernel_shap", forbidden)
    app = create_app(preload_settings=policy_artifacts)
    with TestClient(app) as client:
        assert client.get("/health").json()["services"]["explainer"]
        monkeypatch.setattr(PPO, "load", forbidden)
        monkeypatch.setattr(shap_values, "load_shap_result", forbidden)
        assert (
            client.post("/optimize", json={"as_of": AS_OF}).json()["model_id"]
            == policy_artifacts.checkpoint_sha256
        )
        assert client.post("/explain", json={"asset": "SPY", "as_of": AS_OF}).status_code == 200
        assert client.post("/optimize", json={"as_of": "2024-03-05T06:30:00Z"}).status_code == 404


@pytest.mark.parametrize(
    "field", ["checkpoint_sha256", "policy_manifest_sha256", "shap_manifest_sha256"]
)
def test_bad_pins_disable_only_affected_services(policy_artifacts, field):
    settings = policy_artifacts.model_copy(update={field: "f" * 64})
    loaded = preload_services(settings)
    if field == "shap_manifest_sha256":
        assert loaded.services.optimizer is not None and loaded.services.explainer is None
    elif field == "policy_manifest_sha256":
        assert loaded.services.optimizer is None and loaded.services.explainer is not None
    else:
        assert loaded.services.optimizer is None and loaded.services.explainer is None


def test_directory_metrics_load_and_tamper_detection(tmp_path):
    from tests.test_api import FakeBacktests

    record = FakeBacktests().get("mvo", "2024")
    record["metrics"]["beta"] = None
    digest = write_json(tmp_path / "metrics.json", {"records": [record]})
    pin = write_json(
        tmp_path / "run_manifest.json", {"files": {"metrics.json": {"sha256": digest}}}
    )
    store = DirectoryBacktestLoader().load(tmp_path, pin)
    assert store.get("mvo", "2024")["metrics"]["sharpe"] == 1.2
    with pytest.raises(NotFoundError):
        store.get("ppo", "2024")
    (tmp_path / "metrics.json").write_text("{}")
    with pytest.raises(ValueError, match="SHA256"):
        DirectoryBacktestLoader().load(tmp_path, pin)


def test_retrain_summary_format_rejects_ambiguity_and_requires_pin(tmp_path):
    manifest = {
        "status": "completed",
        "jobs": [{"run_id": "run1", "status": "completed", "dataset": "toy", "window": 1}],
        "windows": {
            "toy": [{"window": 1, "test_start": AS_OF, "test_end": "2025-03-04T06:30:00Z"}]
        },
    }
    pin = write_json(tmp_path / "run_manifest.json", manifest)
    summary = pd.DataFrame(
        [
            {
                "run_id": "run1",
                "dataset": "toy",
                "window": 1,
                "ppo_sharpe": 1.0,
                "equal_weight_sharpe": 0.5,
            }
        ]
    )
    summary.to_csv(tmp_path / "summary.csv", index=False)
    digest = sha256((tmp_path / "summary.csv").read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        DirectoryBacktestLoader().load(tmp_path, pin)
    assert (
        DirectoryBacktestLoader().load(tmp_path, pin, digest).get("ppo", "1")["metrics"]["sharpe"]
        == 1.0
    )
    pd.concat([summary, summary]).to_csv(tmp_path / "summary.csv", index=False)
    with pytest.raises(ValueError, match="ambiguous"):
        DirectoryBacktestLoader().load(
            tmp_path, pin, sha256((tmp_path / "summary.csv").read_bytes()).hexdigest()
        )


def test_research_startup_uses_fake_llm_and_validated_documents(
    policy_artifacts, tmp_path, monkeypatch
):
    from robo_advisor.research import llm
    from robo_advisor.research.ingestion import normalize, save_documents
    from tests.test_research_vector_store import raw

    documents = [
        normalize(
            raw(
                f"https://example.com/{i}",
                "실적 매출 감소",
                "매출 감소 위험",
                STAMP - timedelta(days=1),
            )
        )
        for i in range(2)
    ]
    save_documents(documents, tmp_path / "documents.jsonl")
    digest = sha256((tmp_path / "documents.jsonl").read_bytes()).hexdigest()
    pin = write_json(
        tmp_path / "research.json", {"documents": {"path": "documents.jsonl", "sha256": digest}}
    )
    settings = policy_artifacts.model_copy(
        update={"research_manifest": tmp_path / "research.json", "research_manifest_sha256": pin}
    )
    monkeypatch.setattr(
        lifecycle, "get_settings", lambda: Settings(_env_file=None, llm_api_key="fake-config-key")
    )
    monkeypatch.setattr(llm, "HttpResearchLLM", lambda _: FakeLLM())
    loaded = preload_services(settings)
    assert loaded.services.research is not None, loaded.reasons
    jobs = loaded.services.research
    try:
        result = wait_job(jobs, jobs.submit("실적 매출 감소", STAMP))
        assert result.status == "completed", result.failure_reason
    finally:
        jobs.close()


def test_invalid_preload_environment_keeps_health_available(monkeypatch):
    monkeypatch.setenv("API_CHECKPOINT_SHA256", "bad-pin")
    with TestClient(create_app()) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert not any(response.json()["services"].values())
        assert set(response.json()["reasons"].values()) == {"invalid_preload_configuration"}


def test_lifespan_shuts_down_owned_research():
    jobs = ResearchJobs(runner, ResearchCache(), model_id=MODEL, artifact_version=VERSION)
    app = create_app(
        preload_settings=PreloadSettings(_env_file=None),
        loaders={
            "optimizer": lambda _: None,
            "explainer": lambda _: None,
            "backtests": lambda _: None,
            "research": lambda _: jobs,
        },
    )
    with TestClient(app):
        pass
    with pytest.raises(ServiceUnavailableError, match="closed"):
        jobs.submit("SPY", STAMP)
