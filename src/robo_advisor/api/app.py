"""FastAPI 앱: /health, /optimize, /explain, /research, /backtest.

요청 경로에서는 학습이나 전체 SHAP 계산을 하지 않는다. 각 엔드포인트는 주입된
서비스(``Services``)에 위임하고, 로드되지 않은 서비스는 503으로 응답한다.
"""

from __future__ import annotations

import math
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse

from robo_advisor.api.schemas import (
    BacktestResponse,
    ExplainRequest,
    ExplainResponse,
    FeatureContribution,
    HealthResponse,
    OptimizeRequest,
    OptimizeResponse,
    ResearchJob,
    ResearchQuery,
    Strategy,
)
from robo_advisor.api.services import NotFoundError, ServiceUnavailableError, Services
from robo_advisor.research.contracts import ResearchResult


def get_services(request: Request) -> Services:
    return request.app.state.services


ServicesDep = Annotated[Services, Depends(get_services)]


def _require(service: Any, name: str) -> Any:
    if service is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"{name} is not loaded")
    return service


def _finite_or_none(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def create_app(services: Services | None = None) -> FastAPI:
    app = FastAPI(
        title="Robo-Advisor API",
        description="PPO 자산배분·SHAP 설명·리서치 근거·백테스트 결과 조회",
        version="0.1.0",
    )
    app.state.services = services or Services()

    @app.exception_handler(NotFoundError)
    def not_found(request: Request, error: NotFoundError) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=status.HTTP_404_NOT_FOUND)

    @app.exception_handler(ServiceUnavailableError)
    def unavailable(request: Request, error: ServiceUnavailableError) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

    @app.get("/health", tags=["system"])
    def health(services: ServicesDep) -> HealthResponse:
        return HealthResponse(status="ok", services=services.loaded())

    @app.post("/optimize", tags=["allocation"])
    def optimize(body: OptimizeRequest, services: ServicesDep) -> OptimizeResponse:
        optimizer = _require(services.optimizer, "optimizer")
        weights = optimizer.optimize(body.as_of)
        return OptimizeResponse(model_id=optimizer.model_id, as_of=body.as_of, weights=weights)

    @app.post("/explain", tags=["explanation"])
    def explain(body: ExplainRequest, services: ServicesDep) -> ExplainResponse:
        explainer = _require(services.explainer, "explainer")
        result = explainer.explain(body.asset, body.as_of)
        contributions = result["contributions"]
        observation = result.get("observation")
        top = contributions.abs().sort_values(ascending=False, kind="stable").index[: body.top_k]
        return ExplainResponse(
            model_id=explainer.model_id,
            output=result["output"],
            timestamp=result["timestamp"],
            base_value=result["base_value"],
            prediction=result["prediction"],
            contributions=[
                FeatureContribution(
                    feature=str(name),
                    contribution=float(contributions[name]),
                    value=None if observation is None else _finite_or_none(observation[name]),
                )
                for name in top
            ],
            group_contributions={
                str(name): float(value) for name, value in result["group_contributions"].items()
            },
        )

    @app.post(
        "/research",
        tags=["research"],
        response_model=ResearchResult,
        responses={
            status.HTTP_202_ACCEPTED: {
                "model": ResearchJob,
                "description": "캐시에 없어 비동기 job으로 등록됨",
            }
        },
    )
    def research(body: ResearchQuery, services: ServicesDep) -> Any:
        service = _require(services.research, "research")
        cached = service.cached(body.question, body.as_of)
        if cached is not None:
            return cached
        job = ResearchJob(job_id=service.submit(body.question, body.as_of), status="queued")
        return JSONResponse(job.model_dump(), status_code=status.HTTP_202_ACCEPTED)

    @app.get("/backtest", tags=["backtest"])
    def backtest(
        strategy: Annotated[Strategy, Query(description="전략 이름")],
        window: Annotated[str, Query(min_length=1, max_length=32, examples=["2024"])],
        services: ServicesDep,
    ) -> BacktestResponse:
        store = _require(services.backtests, "backtests")
        summary = store.get(strategy, window)
        metrics = {name: _finite_or_none(value) for name, value in summary["metrics"].items()}
        return BacktestResponse(**{**summary, "metrics": metrics})

    return app


# `uvicorn robo_advisor.api.app:app` 진입점. 서비스 로드는 #47에서 연결한다.
app = create_app()
