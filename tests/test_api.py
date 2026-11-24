"""FastAPI 5개 엔드포인트의 스키마 검증·상태 코드·Swagger 예제 검증."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from robo_advisor.api.app import create_app
from robo_advisor.api.services import NotFoundError, ServiceUnavailableError, Services
from robo_advisor.research.contracts import ResearchResult
from robo_advisor.schemas.common import AssetWeight, PortfolioWeights

AS_OF = "2024-03-04T06:30:00Z"
STAMP = datetime(2024, 3, 4, 6, 30, tzinfo=timezone.utc)


class FakeOptimizer:
    model_id = "alfred-v2-best"

    def __init__(self):
        self.calls = []

    def optimize(self, as_of):
        self.calls.append(as_of)
        return PortfolioWeights(
            asset_weights=(
                AssetWeight(asset="SPY", weight=0.6),
                AssetWeight(asset="GLD", weight=0.3),
            ),
            cash_weight=0.1,
        )


class FakeExplainer:
    model_id = "alfred-v2-best"

    def explain(self, asset, as_of):
        if asset != "SPY":
            raise NotFoundError(f"unknown explained output: {asset}")
        contributions = pd.Series(
            {"simple_return_lag_0__SPY": 0.30, "weight__CASH": -0.50, "rsi_fraction__SPY": 0.05}
        )
        return {
            "output": asset,
            "timestamp": pd.Timestamp(as_of).isoformat(),
            "base_value": 0.1,
            "prediction": -0.05,
            "contributions": contributions,
            "group_contributions": pd.Series({"simple_return": 0.30, "weight": -0.50, "rsi": 0.05}),
            "observation": pd.Series(np.zeros(3), index=contributions.index),
        }


def research_result(question="SPY 위험 요인"):
    return ResearchResult(
        trace_id="trace-1",
        question=question,
        as_of=STAMP,
        status="failed",
        reasons=("insufficient_evidence",),
        retries=0,
        trace=(),
    )


class FakeResearch:
    def __init__(self, cached=None):
        self.cache = cached
        self.submitted = []

    def cached(self, question, as_of):
        return self.cache

    def submit(self, question, as_of):
        self.submitted.append((question, as_of))
        return "job-123"


class FakeBacktests:
    def get(self, strategy, window):
        if (strategy, window) != ("mvo", "2024"):
            raise NotFoundError("no backtest result")
        return {
            "strategy": "mvo",
            "window": "2024",
            "start": "2024-01-01T00:00:00+09:00",
            "end": "2025-01-01T00:00:00+09:00",
            "metrics": {"sharpe": 1.2, "mdd": 0.05, "beta": float("nan")},
        }


def client(**services):
    return TestClient(create_app(Services(**services)))


def test_health_reports_which_services_are_loaded():
    response = client(optimizer=FakeOptimizer()).get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "services": {"optimizer": True, "explainer": False, "research": False, "backtests": False},
    }


def test_optimize_returns_portfolio_weights_and_model_id():
    optimizer = FakeOptimizer()
    response = client(optimizer=optimizer).post("/optimize", json={"as_of": AS_OF})
    assert response.status_code == 200
    body = response.json()
    assert body["model_id"] == "alfred-v2-best"
    assert body["as_of"] == "2024-03-04T06:30:00Z"
    assert body["weights"]["cash_weight"] == pytest.approx(0.1)
    assert [w["asset"] for w in body["weights"]["asset_weights"]] == ["SPY", "GLD"]
    assert optimizer.calls == [STAMP]


@pytest.mark.parametrize("payload", [{}, {"as_of": "2024-03-04T06:30:00"}, {"as_of": "soon"}])
def test_optimize_rejects_missing_or_naive_timestamp(payload):
    response = client(optimizer=FakeOptimizer()).post("/optimize", json=payload)
    assert response.status_code == 422


def test_unconfigured_service_returns_503():
    response = client().post("/optimize", json={"as_of": AS_OF})
    assert response.status_code == 503
    assert response.json()["detail"] == "optimizer is not loaded"


def test_service_unavailable_error_returns_503():
    class Down(FakeOptimizer):
        def optimize(self, as_of):
            raise ServiceUnavailableError("model warming up")

    response = client(optimizer=Down()).post("/optimize", json={"as_of": AS_OF})
    assert response.status_code == 503
    assert response.json()["detail"] == "model warming up"


def test_explain_returns_top_contributions_by_magnitude():
    response = client(explainer=FakeExplainer()).post(
        "/explain", json={"asset": "SPY", "as_of": AS_OF, "top_k": 2}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["unit"] == "raw_policy_mean_logit"
    assert body["model_id"] == "alfred-v2-best"
    assert body["prediction"] == pytest.approx(-0.05)
    assert [c["feature"] for c in body["contributions"]] == [
        "weight__CASH",
        "simple_return_lag_0__SPY",
    ]
    assert body["group_contributions"] == {"simple_return": 0.3, "weight": -0.5, "rsi": 0.05}


def test_explain_unknown_asset_returns_404():
    response = client(explainer=FakeExplainer()).post(
        "/explain", json={"asset": "XYZ", "as_of": AS_OF}
    )
    assert response.status_code == 404


@pytest.mark.parametrize("top_k", [0, 1000])
def test_explain_rejects_out_of_range_top_k(top_k):
    response = client(explainer=FakeExplainer()).post(
        "/explain", json={"asset": "SPY", "as_of": AS_OF, "top_k": top_k}
    )
    assert response.status_code == 422


def test_research_cache_hit_returns_result_synchronously():
    research = FakeResearch(cached=research_result())
    response = client(research=research).post(
        "/research", json={"question": "SPY 위험 요인", "as_of": AS_OF}
    )
    assert response.status_code == 200
    assert response.json()["trace_id"] == "trace-1"
    assert research.submitted == []


def test_research_cache_miss_returns_202_job():
    research = FakeResearch()
    response = client(research=research).post(
        "/research", json={"question": "SPY 위험 요인", "as_of": AS_OF}
    )
    assert response.status_code == 202
    assert response.json() == {"job_id": "job-123", "status": "queued"}
    assert research.submitted == [("SPY 위험 요인", STAMP)]


def test_research_rejects_blank_question():
    response = client(research=FakeResearch()).post(
        "/research", json={"question": "  ", "as_of": AS_OF}
    )
    assert response.status_code == 422


def test_backtest_returns_metrics_with_undefined_values_as_null():
    response = client(backtests=FakeBacktests()).get(
        "/backtest", params={"strategy": "mvo", "window": "2024"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["metrics"] == {"sharpe": 1.2, "mdd": 0.05, "beta": None}
    assert body["start"] == "2023-12-31T15:00:00Z"


def test_backtest_missing_result_returns_404():
    response = client(backtests=FakeBacktests()).get(
        "/backtest", params={"strategy": "ppo", "window": "2024"}
    )
    assert response.status_code == 404


def test_backtest_rejects_unknown_strategy():
    response = client(backtests=FakeBacktests()).get(
        "/backtest", params={"strategy": "magic", "window": "2024"}
    )
    assert response.status_code == 422


def test_openapi_documents_examples_for_every_endpoint():
    schema = client().get("/openapi.json").json()
    paths = schema["paths"]
    assert set(paths) == {"/health", "/optimize", "/explain", "/research", "/backtest"}
    for path, method in [("/optimize", "post"), ("/explain", "post"), ("/research", "post")]:
        body = paths[path][method]["requestBody"]["content"]["application/json"]
        ref = body["schema"]["$ref"].rsplit("/", 1)[-1]
        assert schema["components"]["schemas"][ref].get("examples"), path
    for name in [
        "OptimizeResponse",
        "ExplainResponse",
        "ResearchJob",
        "BacktestResponse",
        "HealthResponse",
    ]:
        assert schema["components"]["schemas"][name].get("examples"), name
    assert "202" in paths["/research"]["post"]["responses"]


def shap_result():
    from robo_advisor.explain.shap_values import ShapResult

    index = pd.DatetimeIndex([STAMP], name="timestamp")
    features = ["simple_return_lag_0__SPY", "weight__CASH"]
    return ShapResult(
        values={"SPY": pd.DataFrame([[0.2, -0.4]], index=index, columns=features)},
        base_values={"SPY": 0.1},
        predictions=pd.DataFrame({"SPY": [-0.1]}, index=index),
        observations=pd.DataFrame([[0.012, 0.2]], index=index, columns=features),
        feature_groups=("simple_return", "weight"),
        metadata={},
    )


def test_shap_explainer_serves_precomputed_values_through_api():
    from robo_advisor.api.services import ShapExplainer

    explainer = ShapExplainer(shap_result(), model_id="sha256:abc")
    response = client(explainer=explainer).post("/explain", json={"asset": "SPY", "as_of": AS_OF})
    assert response.status_code == 200
    body = response.json()
    assert body["model_id"] == "sha256:abc"
    assert body["prediction"] == pytest.approx(-0.1)
    assert body["contributions"][0] == {
        "feature": "weight__CASH",
        "contribution": -0.4,
        "value": 0.2,
    }
    assert body["group_contributions"] == {"simple_return": 0.2, "weight": -0.4}


@pytest.mark.parametrize(
    "payload",
    [{"asset": "GLD", "as_of": AS_OF}, {"asset": "SPY", "as_of": "2024-03-05T06:30:00Z"}],
)
def test_shap_explainer_missing_output_or_time_returns_404(payload):
    from robo_advisor.api.services import ShapExplainer

    explainer = ShapExplainer(shap_result(), model_id="sha256:abc")
    response = client(explainer=explainer).post("/explain", json=payload)
    assert response.status_code == 404


def test_module_app_serves_swagger_with_no_services_loaded():
    from robo_advisor.api.app import app

    with TestClient(app) as test_client:
        assert test_client.get("/docs").status_code == 200
        assert test_client.get("/health").json()["services"]["optimizer"] is False
