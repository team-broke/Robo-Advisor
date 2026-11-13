"""월 1회 리밸런싱하는 롤링 MVO 백테스트.

리밸런싱일 d의 목표 비중은 d 미만 수익률로만 추정하고(``rolling_markowitz``),
d의 수익률부터 그 비중으로 받는다. 오늘까지 관측한 뒤 다음 수익률을 받는
PortfolioEnv와 같은 시점 규칙이다. 비용 차감 → 배분 → 수익 반영은 PPO와 같은
``rebalance_step``을 사용하고, 리밸런싱 사이에는 매매 없이 비중이 표류한다.
평가는 현금 100%·NAV 1에서 시작하며 Safe-Guard 조기 종료는 적용하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from robo_advisor.baselines.mvo import rolling_markowitz
from robo_advisor.rl.accounting import rebalance_step

CASH = "CASH"


@dataclass(frozen=True)
class MVOBacktestResult:
    """일별 비용 후 수익률·NAV와 리밸런싱별 목표 비중·추정 결과."""

    returns: pd.Series
    nav: pd.Series
    weights: pd.DataFrame
    rebalances: pd.DataFrame


def _boundary(value: str | pd.Timestamp, tz) -> pd.Timestamp:
    """시간대 없는 경계는 수익률 시간대의 현지 시각으로 해석한다."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(tz) if tz is not None else timestamp
    if tz is None:
        raise ValueError("timezone-aware boundary requires timezone-aware returns")
    return timestamp.tz_convert(tz)


def run_mvo_backtest(
    returns: pd.DataFrame,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    lookback: int = 252,
    max_weight: float = 0.4,
    risk_free_rate: float = 0.0,
    commission_rate: float = 0.00015,
    slippage_rate: float = 0.0005,
    fallback_weights: pd.Series | None = None,
) -> MVOBacktestResult:
    """[start, end) 구간을 평가한다. returns는 비용 전 일별 단순수익률이다.

    첫 평가일과 매월 첫 거래일(평가 구간 안의 날짜 기준)에 리밸런싱한다.
    start 이전 행은 공분산 추정 이력으로만 쓰고 성과에는 포함하지 않는다.
    추정 실패 시 ``rolling_markowitz``의 대체 비중과 사유를 그대로 기록한다.
    """
    if not isinstance(returns, pd.DataFrame) or not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns must be a DataFrame indexed by dates")
    start_ts, end_ts = _boundary(start, returns.index.tz), _boundary(end, returns.index.tz)
    window = returns.loc[(returns.index >= start_ts) & (returns.index < end_ts)]
    if start_ts >= end_ts or window.empty:
        raise ValueError("evaluation window must contain at least one return row")
    if not np.isfinite(window.to_numpy(dtype=float)).all():
        raise ValueError("evaluation returns must be finite")
    history = returns.loc[returns.index < end_ts]

    # 인덱스 시간대의 현지 날짜로 달을 구분한다.
    months = window.index.year * 12 + window.index.month
    is_rebalance = np.r_[True, months[1:] != months[:-1]]

    columns = [*returns.columns, CASH]
    weights = np.r_[np.zeros(len(returns.columns)), 1.0]
    nav = 1.0
    daily, navs, targets, records = [], [], [], []
    for date, row, rebalance in zip(window.index, window.to_numpy(dtype=float), is_rebalance):
        target = weights
        if rebalance:
            estimate = rolling_markowitz(
                history,
                as_of=date,
                lookback=lookback,
                risk_free_rate=risk_free_rate,
                max_weight=max_weight,
                fallback_weights=fallback_weights,
            )
            target = np.r_[estimate.weights.to_numpy(dtype=float), 0.0]
        step = rebalance_step(nav, weights, target, row, commission_rate, slippage_rate)
        if rebalance:
            targets.append(target)
            records.append(
                {
                    "success": estimate.success,
                    "used_fallback": estimate.used_fallback,
                    "reason": estimate.reason,
                    "observation_count": estimate.observation_count,
                    "turnover": step.turnover,
                    "cost": step.cost / nav,
                }
            )
        daily.append(step.nav / nav - 1)
        navs.append(step.nav)
        nav, weights = step.nav, step.weights

    index = window.index.rename("date")
    rebalance_index = index[is_rebalance]
    return MVOBacktestResult(
        returns=pd.Series(daily, index=index, name="mvo"),
        nav=pd.Series(navs, index=index, name="nav"),
        weights=pd.DataFrame(targets, index=rebalance_index, columns=columns),
        rebalances=pd.DataFrame(records, index=rebalance_index),
    )
