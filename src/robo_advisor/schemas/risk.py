"""RL Observation과 Agentic RAG에서 사용할 자산별 리스크 이벤트 계약을 정의한다.

이벤트 유형, severity, confidence, 두 값의 곱인 risk_score, 공개·가용 시각, source URL,
trace ID를 담으며 available_at이 published_at보다 빠르지 않도록 검증해 look-ahead bias를 막는다.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from robo_advisor.schemas.common import AssetSymbol, TraceId, UTCDateTime


class RiskEventType(str, Enum):
    """Risk event categories accepted by the observation pipeline."""

    EARNINGS_SHOCK = "earnings_shock"
    REGULATION_CHANGE = "regulation_change"
    SHARP_MOVE = "sharp_move"
    GEOPOLITICAL = "geopolitical"


class RiskTag(BaseModel):
    """A traceable point-in-time risk signal for one asset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_type: RiskEventType
    severity: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    asset: AssetSymbol
    published_at: UTCDateTime
    available_at: UTCDateTime
    source_url: HttpUrl
    trace_id: TraceId

    @model_validator(mode="after")
    def validate_availability(self) -> "RiskTag":
        if self.available_at < self.published_at:
            raise ValueError("available_at must be greater than or equal to published_at")
        return self

    @property
    def risk_score(self) -> float:
        """Return the single observation scalar for this asset's risk tag."""
        return self.severity * self.confidence
