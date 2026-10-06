"""현재 NAV의 고점 대비 낙폭으로 최초 조기 종료 시점을 기록한다."""

import logging
from math import isfinite
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from robo_advisor.schemas.common import TraceId


class SafeguardResult(BaseModel):
    """스텝별 낙폭과 최초 종료 신호를 담는 추적 가능한 결과."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    terminated: bool
    reason: Literal["mdd_limit_exceeded"] | None
    step_index: int = Field(ge=0)
    peak_nav: float = Field(gt=0, allow_inf_nan=False)
    nav: float = Field(gt=0, allow_inf_nan=False)
    drawdown: float = Field(ge=0, le=1, allow_inf_nan=False)
    trace_id: TraceId


def _validate_nav(nav: float) -> None:
    """비양수·비유한 NAV를 거부한다."""
    if not isfinite(nav) or nav <= 0:
        raise ValueError("nav must be finite and positive")


class MDDSafeguard:
    """에피소드별 고점과 최초 한도 초과 결과를 추적한다."""

    def __init__(self, initial_nav: float, trace_id: TraceId, limit: float = 0.15):
        """초기 NAV를 고점 기준으로 두며 첫 update의 스텝 인덱스는 0이다."""
        _validate_nav(initial_nav)
        if not isfinite(limit) or not 0 < limit <= 1:
            raise ValueError("drawdown limit must be finite and in (0, 1]")
        self._trace_id = TypeAdapter(TraceId).validate_python(trace_id)
        self._peak_nav = float(initial_nav)
        self._limit = float(limit)
        self._step_index = -1
        self._termination_result: SafeguardResult | None = None

    @property
    def termination_result(self) -> SafeguardResult | None:
        """최초 초과 시점의 결과를 이후에도 변경 없이 반환한다."""
        return self._termination_result

    def update(self, nav: float) -> SafeguardResult:
        """고점을 갱신하고 최초 한도 초과 스텝에서만 terminated=True를 반환한다.

        낙폭은 (peak_nav - nav) / peak_nav이며 누적 최대 낙폭이 아니다.
        nav < peak_nav × (1 - limit)으로 엄격한 초과 여부를 판정한다.
        임계 NAV와 같으면 낙폭을 limit로 표현해 경계의 부동소수점 오차를 피한다.
        이후 호출은 현재 낙폭을 반환하되 종료 신호·로그를 반복하지 않는다.
        유효하지 않은 NAV는 스텝 인덱스나 고점을 변경하지 않는다.
        """
        _validate_nav(nav)
        self._step_index += 1
        self._peak_nav = max(self._peak_nav, float(nav))
        threshold_nav = self._peak_nav * (1 - self._limit)
        drawdown = (self._peak_nav - nav) / self._peak_nav
        if nav == threshold_nav and nav < self._peak_nav:
            drawdown = self._limit
        terminated = self._termination_result is None and nav < threshold_nav
        result = SafeguardResult(
            terminated=terminated,
            reason="mdd_limit_exceeded" if terminated else None,
            step_index=self._step_index,
            peak_nav=self._peak_nav,
            nav=nav,
            drawdown=drawdown,
            trace_id=self._trace_id,
        )
        if terminated:
            self._termination_result = result
            logging.getLogger(__name__).info("MDD limit exceeded", extra=result.model_dump())
        return result
