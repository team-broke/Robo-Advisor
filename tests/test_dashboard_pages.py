"""모의 HTTP 클라이언트로 모든 화면과 비동기 리서치 흐름을 검증한다."""

from copy import deepcopy
from io import BytesIO
from pathlib import Path
import socket
import subprocess
import sys

from PIL import Image
import pytest
from streamlit.testing.v1 import AppTest

from dashboard.api_client import APIError, ResearchReply
from dashboard import ui

ROOT = Path(__file__).resolve().parents[1]
PAGES = [ROOT / "dashboard/app.py", *sorted((ROOT / "dashboard/pages").glob("*.py"))]
STAMP = "2025-01-01T00:00:00Z"
POINTS = [{"timestamp": STAMP, "value": 0.1, "series": "DRL"}]
RESULT = {
    "status": "completed",
    "trace_id": "trace-1",
    "retries": 1,
    "trace": [
        {
            "step": "verify",
            "tool": "llm",
            "query": "SPY",
            "source_ids": ["doc-1"],
            "validation": ["sufficient"],
            "retries": 1,
            "elapsed_ms": 12,
        }
    ],
    "claims": [
        {
            "text": "검증된 결과",
            "citations": [
                {"doc_id": "doc-1", "url": "https://example.com/news", "published_at": STAMP}
            ],
        }
    ],
}
DATA = {
    "portfolio": {
        "weights": {"asset_weights": [{"asset": "SPY", "weight": 0.8}], "cash_weight": 0.2},
        "cumulative_returns": POINTS,
    },
    "rl_performance": {
        "learning_curve": [{"step": 1, "value": 5, "series": "reward"}],
        "metrics": [
            {"strategy": name, "metrics": {key: 0.1 for key in ui.METRICS}}
            for name in ("DRL", "MVO", "동일가중", "벤치마크")
        ],
        "comparison": [
            {**POINTS[0], "series": name} for name in ("DRL", "MVO", "동일가중", "벤치마크")
        ],
    },
    "shap": {
        "summary": [{"feature": "rsi", "mean_abs_shap": 0.2}],
        "force": [
            {"feature": "rsi", "contribution": 0.2, "value": 50},
            {"feature": "macd", "contribution": -0.1},
        ],
        "base_value": 0.1,
        "prediction": 0.2,
        "unit": "raw_policy_mean_logit",
        "output": "SPY",
    },
    "anova": {
        "results": [{"effect": "strategy", "F": 12, "p": 0.001, "eta_squared": 0.3}],
        "tukey": [{"group1": "DRL", "group2": "MVO", "meandiff": 0.1, "reject": True}],
        "diagnostics": [{"test": "shapiro", "p": 0.4, "violated": False}],
        "metric": "sharpe",
        "alpha": 0.05,
    },
    "risk_monitor": {
        "var_95": 0.03,
        "cvar_95": 0.05,
        "var_limit": 0.02,
        "cvar_limit": 0.1,
        "drawdown": POINTS,
        "safeguard_events": [{"timestamp": STAMP, "reason": "한도 초과", "trace_id": "trace-1"}],
        "tags": [
            {
                "asset": "SPY",
                "event_type": "sharp_move",
                "severity": 0.8,
                "confidence": 0.9,
                "published_at": STAMP,
                "available_at": STAMP,
                "trace_id": "trace-1",
                "source_url": "https://example.com/news",
            }
        ],
    },
}


class FakeClient:
    def __init__(self, *, error=None, empty=False, plots=False):
        self.error, self.empty, self.plots = error, empty, plots
        self.submissions = 0
        self.polls = 0
        self.states = ["queued", "running", "completed"]
        self.result = deepcopy(RESULT)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def check(self):
        if self.error:
            raise self.error

    def health(self):
        self.check()
        return {
            "status": "ok",
            "services": {} if self.empty else {"dashboard": True, "research": False},
        }

    def section(self, name):
        self.check()
        plots = []
        if self.plots and name in {"shap", "anova"}:
            plots = [{"id": role, "title": role, "role": role} for role in ("summary", "force")]
        return {
            "section": name,
            "as_of": STAMP,
            "artifact_version": "b" * 64,
            "model_id": "a" * 64,
            "data": {} if self.empty else deepcopy(DATA[name]),
            "plots": plots,
        }

    def plot(self, name, plot_id):
        image = BytesIO()
        Image.new("RGB", (2, 2), "white").save(image, format="PNG")
        return image.getvalue()

    def research(self, question, as_of):
        self.check()
        assert question == "SPY 위험" and as_of == "2025-01-01T00:00:00+00:00"
        self.submissions += 1
        return ResearchReply(202, {"job_id": "job-1", "status": "queued"}, "request-1")

    def research_job(self, job_id):
        self.check()
        assert job_id == "job-1"
        status = self.states[min(self.polls, len(self.states) - 1)]
        self.polls += 1
        return {
            "job_id": job_id,
            "status": status,
            "result": self.result if status == "completed" else None,
            "failure_reason": "secret should never appear",
        }


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket.socket, "connect", denied)


@pytest.fixture
def client(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(ui, "get_client", lambda: client)
    return client


@pytest.mark.parametrize("path", PAGES, ids=lambda path: path.stem)
def test_pages_render_http_values_without_exception(path, client):
    app = AppTest.from_file(str(path), default_timeout=15).run()
    assert not app.exception
    assert app.title
    if path.stem == "2_rl_performance":
        assert app.dataframe[0].value.shape == (12, 4)
    if path.stem == "6_risk_monitor":
        assert len(app.metric) == 2 and len(app.get("progress")) == 2
        assert len(app.get("link_button")) == 1


@pytest.mark.parametrize("path", PAGES, ids=lambda path: path.stem)
@pytest.mark.parametrize("state", ["empty", "connection_error", "service_unavailable"])
def test_unavailable_and_empty_states_are_safe(path, state, client):
    if state == "empty":
        client.empty = True
    else:
        client.error = APIError(state, "sensitive traceback should never appear", "request-1", 503)
    app = AppTest.from_file(str(path), default_timeout=15).run()
    if path.stem == "4_research" and state != "empty":
        app.text_area[0].set_value("SPY 위험")
        app.text_input[0].set_value(STAMP)
        app.button[0].click().run()
    assert not app.exception
    assert app.info or app.warning or path.stem == "4_research"
    assert "sensitive traceback" not in str(app)


@pytest.mark.parametrize("filename", ["3_shap.py", "5_anova.py"])
def test_saved_png_rendering(filename, client):
    client.plots = True
    app = AppTest.from_file(str(ROOT / "dashboard/pages" / filename), default_timeout=15).run()
    assert not app.exception
    assert len(app.get("image")) == 2


def submit_research():
    app = AppTest.from_file(str(ROOT / "dashboard/pages/4_research.py"), default_timeout=15).run()
    app.text_area[0].set_value("SPY 위험")
    app.text_input[0].set_value(STAMP)
    app.button[0].click().run()
    return app


def test_research_polling_trace_citations_and_no_resubmit(client):
    app = submit_research()
    assert not app.exception and "대기 중" in [item.value for item in app.info]
    app.run()
    assert "리서치 실행 중" in [item.value for item in app.info]
    app.run()
    assert not app.exception and client.submissions == 1 and client.polls == 3
    assert len(app.dataframe) == 1 and app.dataframe[0].value.iloc[0]["step"] == "verify"
    assert len(app.get("link_button")) == 1
    assert any(item.value == "검증된 결과" for item in app.markdown)
    app.run()
    assert client.submissions == 1 and client.polls == 3


def test_research_job_failure_does_not_leak_reason(client):
    client.states = ["failed"]
    app = submit_research()
    assert not app.exception and app.warning
    assert "secret should never appear" not in str(app)


def test_research_reconnect_keeps_job_without_resubmit(client):
    app = submit_research()
    client.error = APIError("connection_error", "private", "request-1")
    app.run()
    assert not app.exception and app.warning
    client.error = None
    app.run()
    app.run()
    assert not app.exception and client.submissions == 1
    assert app.dataframe


@pytest.mark.parametrize("stamp", ["2025-01-01", "nonsense"])
def test_research_rejects_naive_or_invalid_cutoff(stamp, client):
    app = AppTest.from_file(str(ROOT / "dashboard/pages/4_research.py")).run()
    app.text_area[0].set_value("SPY 위험")
    app.text_input[0].set_value(stamp)
    app.button[0].click().run()
    assert not app.exception and app.warning and client.submissions == 0


def test_fresh_page_runtime_never_imports_model_packages():
    script = r"""
from pathlib import Path
import sys
from streamlit.testing.v1 import AppTest
from dashboard import ui
class EmptyClient:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def health(self): return {'services': {}}
    def section(self, name):
        return {'section': name, 'as_of': '2025-01-01T00:00:00Z',
                'artifact_version': 'a'*64, 'data': {}, 'plots': []}
ui.get_client = EmptyClient
for path in [Path('dashboard/app.py'), *Path('dashboard/pages').glob('*.py')]:
    app = AppTest.from_file(str(path), default_timeout=15).run()
    assert not app.exception, str(app.exception)
for name in sys.modules:
    assert name.split('.')[0] not in {'robo_advisor', 'torch', 'stable_baselines3', 'shap'}, name
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, text=True, capture_output=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
