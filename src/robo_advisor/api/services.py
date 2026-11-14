"""엔드포인트가 호출하는 서비스 계약.

API 계층은 요청·응답 검증만 맡고 모델 로드·SHAP 조회·리서치 실행은 주입한
서비스에 위임한다. 실제 사전 로드·캐시·비동기 job 큐 구현은 #47에서 연결한다.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime
from typing import Any, Protocol

from robo_advisor.research.contracts import ResearchResult
from robo_advisor.schemas.common import PortfolioWeights


class NotFoundError(LookupError):
    """요청한 자산·시각·결과가 사전 계산 산출물에 없다 (404)."""


class ServiceUnavailableError(RuntimeError):
    """서비스가 일시적으로 응답할 수 없다 (503)."""


class Optimizer(Protocol):
    model_id: str

    def optimize(self, as_of: datetime) -> PortfolioWeights:
        """as_of까지 사용 가능한 관측으로 목표 비중을 반환한다."""


class Explainer(Protocol):
    model_id: str

    def explain(self, asset: str, as_of: datetime) -> dict[str, Any]:
        """사전 계산된 설명 하나를 ``ShapResult.explanation`` 형식으로 반환한다."""


class ResearchService(Protocol):
    def cached(self, question: str, as_of: datetime) -> ResearchResult | None:
        """캐시에 있으면 결과를, 없으면 None을 반환한다. 새로 실행하지 않는다."""

    def submit(self, question: str, as_of: datetime) -> str:
        """리서치 실행을 비동기 job으로 등록하고 job_id를 반환한다."""


class BacktestStore(Protocol):
    def get(self, strategy: str, window: str) -> dict[str, Any]:
        """저장된 백테스트 요약(strategy·window·start·end·metrics)을 반환한다."""


@dataclass(frozen=True)
class Services:
    """로드되지 않은 서비스는 None이며 해당 엔드포인트는 503을 반환한다."""

    optimizer: Optimizer | None = None
    explainer: Explainer | None = None
    research: ResearchService | None = None
    backtests: BacktestStore | None = None

    def loaded(self) -> dict[str, bool]:
        return {field.name: getattr(self, field.name) is not None for field in fields(self)}


class ShapExplainer:
    """``ShapResult``를 감싸 사전 계산된 값만 조회한다. SHAP을 다시 계산하지 않는다."""

    def __init__(self, result, model_id: str):
        self._result = result
        self.model_id = model_id

    def explain(self, asset: str, as_of: datetime) -> dict[str, Any]:
        try:
            return self._result.explanation(asset, as_of)
        except ValueError as error:
            raise NotFoundError(str(error)) from error
