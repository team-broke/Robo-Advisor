"""화면 자료의 시작 시 해시 검증과 요청 시 메모리 조회만 허용한다."""

from hashlib import sha256
from io import BytesIO
import json
import socket

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from robo_advisor.api.app import create_app
from robo_advisor.api.dashboard_artifacts import DashboardArtifacts
from robo_advisor.api.lifecycle import PreloadSettings
from robo_advisor.api.services import Services

STAMP = "2025-01-01T00:00:00Z"
SECTIONS = ("portfolio", "rl_performance", "shap", "anova", "risk_monitor")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket.socket, "connect", forbidden)


def artifact(tmp_path, *, section_data=None, png=None):
    """합성 화면 자료와 신뢰 해시를 만든다."""
    buffer = BytesIO()
    Image.new("RGB", (2, 2), "white").save(buffer, format="PNG")
    png = buffer.getvalue() if png is None else png
    (tmp_path / "summary.png").write_bytes(png)
    sections = {}
    for section in SECTIONS:
        raw = {"as_of": STAMP, "model_id": "a" * 64, "data": (section_data or {}).get(section, {})}
        content = json.dumps(raw).encode()
        name = f"{section}.json"
        (tmp_path / name).write_bytes(content)
        sections[section] = {"path": name, "sha256": sha256(content).hexdigest()}
    sections["shap"]["plots"] = {
        "summary": {
            "path": "summary.png",
            "sha256": sha256(png).hexdigest(),
            "title": "Summary",
            "role": "summary",
        }
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": 1, "sections": sections}))
    return path, sha256(path.read_bytes()).hexdigest(), png


def settings(path, digest):
    return PreloadSettings(
        _env_file=None, dashboard_manifest=path, dashboard_manifest_sha256=digest
    )


def app_for(path, digest):
    return create_app(
        preload_settings=settings(path, digest),
        loaders={
            name: lambda _: None for name in ("optimizer", "explainer", "research", "backtests")
        },
    )


def test_startup_load_and_http_read_only(tmp_path, monkeypatch):
    path, digest, png = artifact(tmp_path)
    with TestClient(app_for(path, digest)) as client:

        def forbidden(*args, **kwargs):
            raise AssertionError("request must not read or compute artifacts")

        from robo_advisor.api import lifecycle
        from robo_advisor.explain import shap_values
        from stable_baselines3 import PPO

        monkeypatch.setattr(lifecycle, "_payload", forbidden)
        monkeypatch.setattr(shap_values, "compute_kernel_shap", forbidden)
        monkeypatch.setattr(PPO, "learn", forbidden)
        assert client.get("/health").json()["services"]["dashboard"] is True
        for section in SECTIONS:
            response = client.get(f"/dashboard/{section}", headers={"X-Correlation-ID": "screen-1"})
            assert response.status_code == 200
            assert response.headers["X-Correlation-ID"] == "screen-1"
            body = response.json()
            assert body["section"] == section and body["artifact_version"] == digest
            assert body["as_of"] == STAMP and body["model_id"] == "a" * 64
        response = client.get("/dashboard/shap/plots/summary")
        assert response.content == png and response.headers["content-type"] == "image/png"
        assert response.headers["etag"] == sha256(png).hexdigest()
        assert client.get("/dashboard/shap/plots/absent").status_code == 404
        assert client.get("/dashboard/unknown").status_code == 422


@pytest.mark.parametrize("corruption", ["manifest", "payload", "png"])
def test_hash_failure_disables_service_and_keeps_safe_errors(tmp_path, corruption):
    path, digest, _ = artifact(tmp_path)
    target = {"manifest": path, "payload": tmp_path / "shap.json", "png": tmp_path / "summary.png"}
    with target[corruption].open("ab") as stream:
        stream.write(b"corrupted")
    with TestClient(app_for(path, digest)) as client:
        health = client.get("/health").json()
        assert health["services"]["dashboard"] is False
        assert health["reasons"]["dashboard"] == "load_failed:ValueError"
        response = client.get("/dashboard/shap")
        assert response.status_code == 503
        assert set(response.json()) == {"code", "message", "correlation_id"}
        assert str(tmp_path) not in response.text


@pytest.mark.parametrize(
    "data",
    [
        {
            "portfolio": {
                "cumulative_returns": [{"timestamp": "2026-01-01T00:00:00Z", "value": 0.1}]
            }
        },
        {"portfolio": {"cumulative_returns": [{"timestamp": "2025-01-01", "value": 0.1}]}},
        {"risk_monitor": {"var_95": float("nan")}},
        {"rl_performance": {"learning_curve": [{"step": 1, "value": 2}] * 2}},
        {"portfolio": {"cumulative_returns": [{"timestamp": STAMP, "value": 0.1}] * 2}},
        {"shap": {"summary": [{"feature": "rsi", "mean_abs_shap": -1}]}},
        {"portfolio": {"weights": {"asset_weights": [], "cash_weight": 0.5}}},
        {"anova": {"alpha": 1}},
    ],
)
def test_invalid_or_future_data_rejected(tmp_path, data):
    path, digest, _ = artifact(tmp_path, section_data=data)
    with pytest.raises(ValueError):
        DashboardArtifacts.load(path, digest)


def test_valid_schema_values_and_return_isolation(tmp_path):
    data = {
        "portfolio": {
            "weights": {"asset_weights": [{"asset": "SPY", "weight": 0.8}], "cash_weight": 0.2}
        },
        "anova": {"metric": "sharpe", "results": [{"effect": "strategy", "F": 10, "p": 0.001}]},
        "risk_monitor": {
            "tags": [
                {
                    "asset": "SPY",
                    "event_type": "sharp_move",
                    "severity": 0.8,
                    "confidence": 0.9,
                    "published_at": STAMP,
                    "available_at": STAMP,
                    "source_url": "https://example.com/news",
                    "trace_id": "source-trace",
                }
            ]
        },
    }
    path, digest, _ = artifact(tmp_path, section_data=data)
    store = DashboardArtifacts.load(path, digest)
    assert store.get("anova")["data"]["results"][0]["F"] == 10
    response = store.get("portfolio")
    response["data"]["weights"] = None
    assert store.get("portfolio")["data"]["weights"] is not None


def test_missing_section_and_unconfigured_service(tmp_path):
    path, _, _ = artifact(tmp_path)
    raw = json.loads(path.read_text())
    del raw["sections"]["anova"]
    path.write_text(json.dumps(raw))
    store = DashboardArtifacts.load(path, sha256(path.read_bytes()).hexdigest())
    with TestClient(create_app(Services(dashboard=store))) as client:
        assert client.get("/dashboard/anova").status_code == 404
    with TestClient(create_app(Services())) as client:
        assert client.get("/dashboard/portfolio").status_code == 503


def test_invalid_png_and_unsafe_descriptor_rejected(tmp_path):
    path, digest, _ = artifact(tmp_path, png=b"not a png")
    with pytest.raises(OSError):
        DashboardArtifacts.load(path, digest)
    path, _, _ = artifact(tmp_path)
    raw = json.loads(path.read_text())
    raw["sections"]["portfolio"]["path"] = "../outside.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="unsafe"):
        DashboardArtifacts.load(path, sha256(path.read_bytes()).hexdigest())


def test_blank_optional_preload_variables_do_not_disable_configured_dashboard(
    tmp_path, monkeypatch
):
    path, digest, _ = artifact(tmp_path)
    monkeypatch.setenv("API_CHECKPOINT_SHA256", "")
    monkeypatch.setenv("API_SHAP_MANIFEST_SHA256", "")
    monkeypatch.setenv("API_DASHBOARD_MANIFEST", str(path))
    monkeypatch.setenv("API_DASHBOARD_MANIFEST_SHA256", digest)

    with TestClient(create_app(preload_settings=PreloadSettings(_env_file=None))) as client:
        health = client.get("/health").json()
        assert health["services"]["dashboard"] is True
        assert health["services"]["optimizer"] is False
        assert client.get("/dashboard/portfolio").status_code == 200
