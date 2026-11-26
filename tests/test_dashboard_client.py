"""HTTP 응답·job 폴링과 대시보드의 모델 접근 금지 경계를 고정한다."""

import ast
from pathlib import Path
import subprocess
import sys

import httpx
import pytest

from dashboard.api_client import APIClient, APIError

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {
    "robo_advisor",
    "torch",
    "stable_baselines3",
    "shap",
    "tensorflow",
    "keras",
    "safetensors",
}


def violations(source):
    found = []
    for node in ast.walk(ast.parse(source)):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        elif isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute)):
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr
            if (
                name in {"__import__", "import_module"}
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                names = [str(node.args[0].value)]
        if any(name.split(".")[0] in FORBIDDEN | {"pickle", "joblib"} for name in names):
            found.append(node.lineno)
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.lower().endswith((".zip", ".pt", ".pth", ".ckpt", ".safetensors"))
        ):
            found.append(node.lineno)
    return found


def test_all_dashboard_sources_obey_http_only_boundary():
    for path in (ROOT / "dashboard").rglob("*.py"):
        assert not violations(path.read_text()), path


@pytest.mark.parametrize(
    "source",
    [
        "import torch",
        "from robo_advisor.api import app",
        "import importlib; importlib.import_module('shap')",
        "open('policy.zip', 'rb')",
        "Path('weights.pt').read_bytes()",
    ],
)
def test_guard_detects_forbidden_access(source):
    assert violations(source)


def test_fresh_runtime_import_does_not_load_model_packages():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import dashboard.api_client; "
            "assert not any(name.split('.')[0] in "
            "{'robo_advisor','torch','stable_baselines3','shap'} for name in sys.modules)",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_base_url_headers_timeout_and_read_only_calls(monkeypatch):
    monkeypatch.setenv("API_BASE_URL", "https://api.example/v1")

    def handle(request):
        assert request.url == "https://api.example/v1/health"
        assert request.headers["X-Correlation-ID"] == "request-123"
        assert request.extensions["timeout"]["read"] == 2
        return httpx.Response(200, json={"status": "ok"}, headers={"X-Correlation-ID": "server-id"})

    with httpx.Client(transport=httpx.MockTransport(handle)) as transport:
        with APIClient(client=transport, timeout=2, correlation_id="request-123") as client:
            assert client.health() == {"status": "ok"} and client.correlation_id == "server-id"
        assert not transport.is_closed


@pytest.mark.parametrize(
    "code,status",
    [
        ("not_found", 404),
        ("validation_error", 422),
        ("service_unavailable", 503),
        ("internal_error", 500),
    ],
)
def test_uniform_api_errors_become_exceptions(code, status):
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                status,
                json={"code": code, "message": "server message", "correlation_id": "request-1"},
            )
        )
    ) as transport:
        with pytest.raises(APIError) as error:
            APIClient(client=transport).health()
        assert (
            error.value.code,
            error.value.message,
            error.value.correlation_id,
            error.value.status_code,
        ) == (code, "server message", "request-1", status)
        assert "server message" not in error.value.user_message


@pytest.mark.parametrize(
    "exception,code", [(httpx.ConnectError, "connection_error"), (httpx.ReadTimeout, "timeout")]
)
def test_network_errors_do_not_expose_original_exception(exception, code):
    def fail(request):
        raise exception("private token-value", request=request)

    with httpx.Client(transport=httpx.MockTransport(fail)) as transport:
        with pytest.raises(APIError) as error:
            APIClient(client=transport).health()
        assert error.value.code == code and "private" not in str(error.value)


@pytest.mark.parametrize("status,data", [(200, []), (503, {"detail": "old response"}), (302, {})])
def test_invalid_responses_are_explicit(status, data):
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(status, json=data))
    ) as transport:
        with pytest.raises(APIError, match="protocol_error"):
            APIClient(client=transport).health()


def test_research_job_polling_keeps_id_and_reports_progress():
    calls, statuses = [], iter(["queued", "running", "completed"])

    def handle(request):
        calls.append((request.method, request.url.path))
        if request.method == "POST":
            return httpx.Response(202, json={"job_id": "job-1", "status": "queued"})
        status = next(statuses)
        return httpx.Response(
            200,
            json={
                "job_id": "job-1",
                "status": status,
                "result": {} if status == "completed" else None,
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as transport:
        client = APIClient(client=transport)
        reply = client.research("SPY", "2024-01-01T00:00:00Z")
        progress = []
        assert reply.status_code == 202
        assert (
            client.poll_job(
                reply.data["job_id"],
                sleep=lambda _: None,
                on_update=lambda data: progress.append(data["status"]),
            )["status"]
            == "completed"
        )
    assert progress == ["queued", "running", "completed"]
    assert calls.count(("POST", "/research")) == 1


def test_poll_timeout_does_not_resubmit_job():
    clock = iter([0, 0, 2])
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"job_id": "job-1", "status": "running"})
        )
    ) as transport:
        with pytest.raises(APIError, match="poll_timeout"):
            APIClient(client=transport).poll_job(
                "job-1", timeout=1, clock=lambda: next(clock), sleep=lambda _: None
            )


def test_completed_cache_response_failed_job_and_png():
    def handle(request):
        if request.url.path.endswith("/research"):
            return httpx.Response(200, json={"status": "completed", "claims": []})
        if "/plots/" in request.url.path:
            return httpx.Response(
                200, content=b"\x89PNG\r\n\x1a\ncontent", headers={"content-type": "image/png"}
            )
        return httpx.Response(
            200,
            json={"job_id": "job-1", "status": "failed", "failure_reason": "insufficient_evidence"},
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as transport:
        client = APIClient(client=transport)
        assert client.research("SPY", "2024-01-01T00:00:00Z").status_code == 200
        assert client.poll_job("job-1")["status"] == "failed"
        assert client.plot("shap", "summary").startswith(b"\x89PNG")
        with pytest.raises(ValueError):
            client.research_job("../escape")


@pytest.mark.parametrize(
    "base_url",
    ["file:///tmp/model", "https://user:pass@example.com", "https://example.com/?token=secret"],
)
def test_invalid_base_urls(base_url):
    with pytest.raises(ValueError):
        APIClient(base_url)
