"""RL, RAG, 백테스트, API에서 재사용하는 공통 데이터 계약을 정의한다.

자산 유니버스를 하드코딩하지 않고 자산 티커, 자산·현금 포트폴리오 비중과 합계 검증,
timezone-aware 시각의 UTC 정규화, 외부 출처 정보, end-to-end 추적용 trace ID를 제공한다.
"""

from datetime import datetime, timezone
from typing import Annotated

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    StringConstraints,
    model_validator,
)

WEIGHT_SUM_TOLERANCE = 1e-6


def _to_utc(value: datetime) -> datetime:
    """Reject naive datetimes and normalize aware datetimes to UTC."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must include timezone information")
    return value.astimezone(timezone.utc)


NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
AssetSymbol = NonEmptyString
TraceId = NonEmptyString
UTCDateTime = Annotated[datetime, AfterValidator(_to_utc)]


class AssetWeight(BaseModel):
    """A non-negative allocation for one asset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    asset: AssetSymbol
    weight: float = Field(ge=0.0)


class PortfolioWeights(BaseModel):
    """Asset and cash allocations whose total must be one."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    asset_weights: tuple[AssetWeight, ...]
    cash_weight: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_total_weight(self) -> "PortfolioWeights":
        total = sum(item.weight for item in self.asset_weights) + self.cash_weight
        if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
            raise ValueError(
                "asset and cash weights must sum to 1 " f"within tolerance {WEIGHT_SUM_TOLERANCE}"
            )
        return self


class Source(BaseModel):
    """Traceable origin metadata for externally sourced information."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_url: HttpUrl
    source_name: NonEmptyString | None = None
    source_id: NonEmptyString | None = None
