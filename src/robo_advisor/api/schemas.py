"""API 요청·응답 스키마와 Swagger 예제."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from robo_advisor.schemas.common import NonEmptyString, PortfolioWeights, UTCDateTime
from robo_advisor.research.contracts import ResearchResult

AS_OF_EXAMPLE = "2024-03-04T06:30:00Z"
STRATEGIES = ("ppo", "mvo", "equal_weight", "benchmark")
Strategy = Literal["ppo", "mvo", "equal_weight", "benchmark"]


def _examples(*examples: dict) -> ConfigDict:
    return ConfigDict(extra="forbid", json_schema_extra={"examples": list(examples)})


class HealthResponse(BaseModel):
    model_config = _examples(
        {
            "status": "ok",
            "services": {
                "optimizer": True,
                "explainer": True,
                "research": False,
                "backtests": True,
            },
        }
    )

    status: Literal["ok"]
    services: dict[str, bool]
    reasons: dict[str, str] = Field(default_factory=dict)


class OptimizeRequest(BaseModel):
    model_config = _examples({"as_of": AS_OF_EXAMPLE})

    as_of: UTCDateTime = Field(description="이 시각까지 사용 가능했던 관측만 쓴다. 시간대 필수.")


class OptimizeResponse(BaseModel):
    model_config = _examples(
        {
            "model_id": "alfred-v2-best",
            "as_of": AS_OF_EXAMPLE,
            "weights": {
                "asset_weights": [
                    {"asset": "SPY", "weight": 0.35},
                    {"asset": "IEF", "weight": 0.25},
                    {"asset": "GLD", "weight": 0.2},
                ],
                "cash_weight": 0.2,
            },
        }
    )

    model_id: str
    as_of: UTCDateTime
    weights: PortfolioWeights


class ExplainRequest(BaseModel):
    model_config = _examples({"asset": "SPY", "as_of": AS_OF_EXAMPLE, "top_k": 10})

    asset: NonEmptyString = Field(description="설명할 출력 이름. 자산 티커 또는 CASH.")
    as_of: UTCDateTime
    top_k: int = Field(default=10, ge=1, le=289, description="절댓값 기준 상위 기여 피처 수")


class FeatureContribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature: str
    contribution: float
    value: float | None = None


class ExplainResponse(BaseModel):
    model_config = _examples(
        {
            "model_id": "alfred-v2-best",
            "output": "SPY",
            "timestamp": AS_OF_EXAMPLE,
            "unit": "raw_policy_mean_logit",
            "base_value": 0.1,
            "prediction": -0.05,
            "contributions": [
                {"feature": "weight__CASH", "contribution": -0.5, "value": 0.2},
                {"feature": "simple_return_lag_0__SPY", "contribution": 0.3, "value": 0.012},
            ],
            "group_contributions": {"simple_return": 0.3, "weight": -0.5, "rsi": 0.05},
        }
    )

    model_id: str
    output: str
    timestamp: UTCDateTime
    unit: Literal["raw_policy_mean_logit"] = Field(
        default="raw_policy_mean_logit",
        description="softmax 전 정책 평균 로짓 단위. 비중 변화량이 아니다.",
    )
    base_value: float
    prediction: float
    contributions: list[FeatureContribution]
    group_contributions: dict[str, float]


class ResearchQuery(BaseModel):
    model_config = _examples({"question": "SPY 단기 위험 요인", "as_of": AS_OF_EXAMPLE})

    question: NonEmptyString = Field(max_length=1000)
    as_of: UTCDateTime = Field(description="이 시각까지 사용 가능했던 문서만 근거로 쓴다.")


class ResearchJob(BaseModel):
    model_config = _examples({"job_id": "3f2c9a", "status": "queued"})

    job_id: str
    status: Literal["queued"]


class BacktestResponse(BaseModel):
    model_config = _examples(
        {
            "strategy": "mvo",
            "window": "2024",
            "start": "2023-12-31T15:00:00Z",
            "end": "2024-12-31T15:00:00Z",
            "metrics": {"cumulative_return": 0.071, "sharpe": 1.12, "mdd": 0.048, "beta": None},
        }
    )

    strategy: Strategy
    window: str
    start: UTCDateTime
    end: UTCDateTime
    metrics: dict[str, float | None] = Field(description="정의 불가능한 지표(NaN)는 null")


class ResearchJobStatus(BaseModel):
    """보존 중인 비동기 리서치 job의 상태와 완료 결과."""

    model_config = _examples(
        {"job_id": "3f2c9a", "status": "running", "result": None, "failure_reason": None},
        {
            "job_id": "3f2c9a",
            "status": "failed",
            "result": None,
            "failure_reason": "insufficient_evidence",
        },
    )

    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    result: ResearchResult | None = None
    failure_reason: str | None = None
