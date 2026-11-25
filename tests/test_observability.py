"""요청·background job의 추적 연결과 로그·오류 개인정보 차단 검증."""

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import io
import json
import logging
import socket
from threading import Barrier
from uuid import UUID

from fastapi.testclient import TestClient
import pytest

from robo_advisor.api.app import create_app
from robo_advisor.api.cache import ResearchCache, ResearchJobs
from robo_advisor.api.lifecycle import PolicyOptimizer
from robo_advisor.api.observability import (
    SensitiveLogFilter,
    correlation_context,
    get_correlation_id,
    log_event,
)
from robo_advisor.api.services import Services
from robo_advisor.schemas.risk import RiskTag
from tests.test_api import AS_OF, STAMP, FakeOptimizer, FakeBacktests
from tests.test_api_preload import MODEL, VERSION, runner, wait_job


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)


@pytest.mark.parametrize(
    "supplied",
    [
        None,
        "request-123",
        "",
        "bad value",
        "person@example.com",
        "sk-private-api-key",
        "x" * 65,
        "01012345678",
    ],
)
def test_correlation_header_validation_return_and_reset(supplied):
    headers = {} if supplied is None else {"X-Correlation-ID": supplied}
    with TestClient(create_app(Services())) as client:
        response = client.get("/health", headers=headers)
    correlation = response.headers["X-Correlation-ID"]
    if supplied == "request-123":
        assert correlation == supplied
    else:
        UUID(correlation)
    assert get_correlation_id() is None


def test_duplicate_correlation_headers_are_replaced():
    with TestClient(create_app(Services())) as client:
        response = client.get(
            "/health", headers=[("X-Correlation-ID", "first"), ("X-Correlation-ID", "second")]
        )
    UUID(response.headers["X-Correlation-ID"])


@pytest.mark.parametrize(
    "status,path,body",
    [
        (404, "/absent", None),
        (422, "/optimize", {"as_of": "person@example.com"}),
        (503, "/optimize", {"as_of": AS_OF}),
        (404, "/backtest?strategy=ppo&window=2024", None),
    ],
)
def test_error_response_has_exact_uniform_fields(status, path, body):
    with TestClient(create_app(Services(backtests=FakeBacktests()))) as client:
        response = (
            client.get(path, headers={"X-Correlation-ID": "error-request"})
            if body is None
            else client.post(path, json=body, headers={"X-Correlation-ID": "error-request"})
        )
    assert response.status_code == status
    assert set(response.json()) == {"code", "message", "correlation_id"}
    assert (
        response.json()["correlation_id"] == response.headers["X-Correlation-ID"] == "error-request"
    )
    assert "person@example.com" not in response.text


def test_500_never_exposes_exception_and_returns_header(caplog):
    class Broken(FakeOptimizer):
        def optimize(self, as_of):
            raise RuntimeError("api_key=opaque-value person@example.com reasoning: private thought")

    with (
        caplog.at_level(logging.INFO),
        TestClient(create_app(Services(optimizer=Broken()))) as client,
    ):
        response = client.post(
            "/optimize", json={"as_of": AS_OF}, headers={"X-Correlation-ID": "broken-request"}
        )
    assert response.status_code == 500
    assert response.json() == {
        "code": "internal_error",
        "message": "Internal server error",
        "correlation_id": "broken-request",
    }
    assert response.headers["X-Correlation-ID"] == "broken-request"
    assert "opaque-value" not in caplog.text and "private thought" not in caplog.text
    assert any(
        getattr(record, "audit", {}).get("event") == "request_error" for record in caplog.records
    )
    assert get_correlation_id() is None


def test_log_filter_redacts_args_structures_exceptions_and_private_reasoning():
    filter_ = SensitiveLogFilter(secrets=("opaque-config-key",))
    with correlation_context("log-request"):
        for message, args in (
            (
                "email=%s phone=%s token=%s",
                ("person@example.com", "010-1234-5678", "opaque-config-key"),
            ),
            (
                {
                    "reasoning": "hidden reasoning contents",
                    "prompt": "hidden prompt",
                    "api_key": "secret",
                    "source_ids": ["a" * 64],
                },
                (),
            ),
            ("raw_response: hidden reasoning contents", ()),
            ("https://example.com/?email=person%40example.com&token=short-private", ()),
        ):
            record = logging.LogRecord(
                "test",
                logging.ERROR,
                __file__,
                1,
                message,
                args,
                (ValueError, ValueError("private exception"), None),
            )
            record.extra_payload = {
                "internal_thoughts": "hidden reasoning contents",
                "password": "raw password",
            }
            assert filter_.filter(record)
            visible = record.getMessage() + str(record.extra_payload)
            for secret in (
                "person@example.com",
                "010-1234-5678",
                "opaque-config-key",
                "hidden reasoning contents",
                "hidden prompt",
                "raw password",
                "short-private",
            ):
                assert secret not in visible
            assert record.correlation_id == "log-request" and record.exc_info is None


def test_background_job_trace_risk_tag_and_policy_decision_share_id(caplog):
    jobs = ResearchJobs(runner, ResearchCache(), model_id=MODEL, artifact_version=VERSION)
    optimizer = FakeOptimizer()
    try:
        with (
            caplog.at_level(logging.INFO),
            TestClient(create_app(Services(optimizer=optimizer, research=jobs))) as client,
        ):
            headers = {"X-Correlation-ID": "flow-request"}
            accepted = client.post(
                "/research", json={"question": "SPY", "as_of": AS_OF}, headers=headers
            )
            job_id = accepted.json()["job_id"]
            finished = wait_job(jobs, job_id)
            assert finished.status == "completed" and finished.correlation_id == "flow-request"
            trace_id = finished.result.trace_id
            optimizer.risk_tags = (
                RiskTag(
                    event_type="earnings_shock",
                    asset="SPY",
                    severity=0.8,
                    confidence=0.9,
                    published_at=STAMP - timedelta(days=1),
                    available_at=STAMP,
                    source_url="https://example.com/doc",
                    trace_id=trace_id,
                ),
                RiskTag(
                    event_type="sharp_move",
                    asset="SPY",
                    severity=1,
                    confidence=1,
                    published_at=STAMP,
                    available_at=STAMP + timedelta(seconds=1),
                    source_url="https://example.com/future",
                    trace_id="future-trace",
                ),
            )
            polled = client.get(
                f"/research/jobs/{job_id}", headers={"X-Correlation-ID": "poll-request"}
            )
            assert polled.headers["X-Correlation-ID"] == "poll-request"
            assert polled.json()["correlation_id"] == "flow-request"
            assert (
                client.post("/optimize", json={"as_of": AS_OF}, headers=headers).status_code == 200
            )
        audit = [
            record.audit
            for record in caplog.records
            if getattr(record, "audit", {}).get("correlation_id") == "flow-request"
        ]
        for event in (
            "research_job_queued",
            "research_job_running",
            "research_job_finished",
            "research_result",
            "risk_tag",
            "policy_decision",
        ):
            assert any(item["event"] == event for item in audit), event
        source = next(item for item in audit if item["event"] == "research_result")
        assert source["trace_id"] == trace_id and source["source_ids"] == ["one", "two"]
        tag = next(item for item in audit if item["event"] == "risk_tag")
        decision = next(item for item in audit if item["event"] == "policy_decision")
        assert tag["trace_id"] == trace_id and decision["risk_tag_trace_ids"] == [trace_id]
        assert decision["model_id"] == optimizer.model_id
        assert "future-trace" not in json.dumps(audit)
    finally:
        jobs.close()


def test_parallel_sync_requests_keep_separate_contexts(caplog):
    barrier = Barrier(2)

    class Parallel(FakeOptimizer):
        def optimize(self, as_of):
            current = get_correlation_id()
            barrier.wait(timeout=5)
            log_event("parallel_check", request_id=current)
            return super().optimize(as_of)

    with (
        caplog.at_level(logging.INFO),
        TestClient(create_app(Services(optimizer=Parallel()))) as client,
    ):

        def call(correlation):
            return client.post(
                "/optimize", json={"as_of": AS_OF}, headers={"X-Correlation-ID": correlation}
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            responses = list(executor.map(call, ("parallel-a", "parallel-b")))
    assert [response.headers["X-Correlation-ID"] for response in responses] == [
        "parallel-a",
        "parallel-b",
    ]
    records = [
        record.audit
        for record in caplog.records
        if getattr(record, "audit", {}).get("event") == "parallel_check"
    ]
    assert len(records) == 2 and all(
        item["correlation_id"] == item["request_id"] for item in records
    )
    assert get_correlation_id() is None


def test_uvicorn_access_formatter_keeps_its_contract_and_redacts(monkeypatch):
    from uvicorn.logging import AccessFormatter

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(
        AccessFormatter('%(client_addr)s - "%(request_line)s" %(status_code)s', use_colors=False)
    )
    logger = logging.getLogger("uvicorn.access")
    monkeypatch.setattr(logger, "handlers", [handler])
    monkeypatch.setattr(logger, "level", logging.INFO)
    create_app(Services())
    with correlation_context("access-request"):
        logger.info(
            '%s - "%s %s HTTP/%s" %d',
            "127.0.0.1",
            "GET",
            "/?email=person%40example.com",
            "1.1",
            200,
        )
    assert "access-request" in stream.getvalue() and "person" not in stream.getvalue()


def test_policy_risk_observations_need_matching_tag_provenance():
    import pandas as pd
    from types import SimpleNamespace
    from robo_advisor.explain.policy import ObservationLayout

    layout = ObservationLayout(("SPY",), 1)
    frame = pd.DataFrame(0.0, index=pd.DatetimeIndex([STAMP]), columns=layout.feature_names)
    frame["risk_score__SPY"] = 0.7
    with pytest.raises(ValueError, match="risk observation"):
        PolicyOptimizer(SimpleNamespace(layout=layout), frame, MODEL)


def test_deduplicated_job_keeps_original_request_id(caplog):
    from threading import Event

    started, release = Event(), Event()

    def blocked(question, as_of):
        started.set()
        assert release.wait(5)
        return runner(question, as_of)

    jobs = ResearchJobs(blocked, ResearchCache(), model_id=MODEL, artifact_version=VERSION)
    try:
        with (
            caplog.at_level(logging.INFO),
            TestClient(create_app(Services(research=jobs))) as client,
        ):
            first = client.post(
                "/research",
                json={"question": "SPY", "as_of": AS_OF},
                headers={"X-Correlation-ID": "original-request"},
            )
            assert started.wait(2)
            second = client.post(
                "/research",
                json={"question": "SPY", "as_of": AS_OF},
                headers={"X-Correlation-ID": "duplicate-request"},
            )
            assert first.json()["job_id"] == second.json()["job_id"]
            assert second.headers["X-Correlation-ID"] == "duplicate-request"
            release.set()
            final = wait_job(jobs, first.json()["job_id"])
            assert final.correlation_id == "original-request"
        reused = next(
            record.audit
            for record in caplog.records
            if getattr(record, "audit", {}).get("event") == "research_job_reused"
        )
        assert (
            reused["correlation_id"] == "duplicate-request"
            and reused["origin_correlation_id"] == "original-request"
        )
    finally:
        release.set()
        jobs.close()


def test_known_opaque_key_cannot_be_used_as_correlation_header(monkeypatch):
    from robo_advisor.api import observability
    from robo_advisor.config import Settings

    monkeypatch.setattr(
        observability,
        "get_settings",
        lambda: Settings(_env_file=None, llm_api_key="opaque-test-key"),
    )
    with TestClient(create_app(Services())) as client:
        response = client.get("/health", headers={"X-Correlation-ID": "opaque-test-key"})
    UUID(response.headers["X-Correlation-ID"])


def test_private_failure_reason_is_not_exposed_in_job_status():
    from tests.test_api import research_result

    def private_failure(question, as_of):
        return research_result(question).model_copy(
            update={"reasons": ("reasoning: internal private thoughts",)}
        )

    jobs = ResearchJobs(private_failure, ResearchCache(), model_id=MODEL, artifact_version=VERSION)
    try:
        failed = wait_job(jobs, jobs.submit("SPY", STAMP))
        assert failed.status == "failed"
        assert "internal private thoughts" not in failed.model_dump_json()
    finally:
        jobs.close()


def test_shutdown_cancels_waiting_jobs_without_starting_more_research():
    from threading import Event

    started, release = Event(), Event()
    calls = []

    def blocked(question, as_of):
        calls.append(question)
        started.set()
        assert release.wait(5)
        return runner(question, as_of)

    jobs = ResearchJobs(
        blocked, ResearchCache(), model_id=MODEL, artifact_version=VERSION, max_workers=1
    )
    running = jobs.submit("SPY", STAMP)
    assert started.wait(2)
    waiting = jobs.submit("IEF", STAMP)
    with ThreadPoolExecutor(max_workers=1) as executor:
        shutdown = executor.submit(jobs.close)
        try:
            assert wait_job(jobs, waiting).failure_reason == "service_shutting_down"
        finally:
            release.set()
            shutdown.result(timeout=5)
    assert calls == ["SPY"] and jobs.job(running).status == "completed"


def test_failed_research_trace_is_linked_to_job_and_request(caplog):
    from tests.test_api import research_result

    jobs = ResearchJobs(
        lambda question, as_of: research_result(question),
        ResearchCache(),
        model_id=MODEL,
        artifact_version=VERSION,
    )
    try:
        with caplog.at_level(logging.INFO), correlation_context("failed-flow"):
            job_id = jobs.submit("SPY", STAMP)
            assert wait_job(jobs, job_id).status == "failed"
        result = next(
            record.audit
            for record in caplog.records
            if getattr(record, "audit", {}).get("event") == "research_result"
        )
        assert result["correlation_id"] == "failed-flow"
        assert result["job_id"] == job_id and result["trace_id"] == "trace-1"
        assert result["status"] == "failed"
    finally:
        jobs.close()


def test_completed_job_retains_result_after_concurrent_cache_eviction():
    from robo_advisor.api.cache import CacheKey

    class EvictingCache(ResearchCache):
        def put(self, key, result):
            stored = super().put(key, result)
            other = CacheKey(MODEL, VERSION, "other", key.as_of)
            super().put(other, runner("other", key.as_of))
            return stored

    jobs = ResearchJobs(
        runner, EvictingCache(max_entries=1), model_id=MODEL, artifact_version=VERSION
    )
    try:
        result = wait_job(jobs, jobs.submit("SPY", STAMP))
        assert result.status == "completed" and result.result.question == "SPY"
        assert jobs.cached("SPY", STAMP) is None
    finally:
        jobs.close()
