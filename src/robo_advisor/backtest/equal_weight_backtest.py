"""현금 0의 월간 동일가중 비교 전략."""

import numpy as np
import pandas as pd

from robo_advisor.backtest.mvo_backtest import MVOBacktestResult
from robo_advisor.rl.accounting import rebalance_step


def run_equal_weight_backtest(returns, *, commission_rate, slippage_rate):
    """평가 구간 첫 세션과 현지 월 첫 세션에 거래하고 나머지는 표류시킨다."""
    dates = returns.index
    if (
        not isinstance(dates, pd.DatetimeIndex)
        or dates.tz is None
        or dates.empty
        or dates.hasnans
        or not dates.is_monotonic_increasing
        or not dates.is_unique
        or not returns.columns.is_unique
        or "CASH" in returns.columns
        or not len(returns.columns)
    ):
        raise ValueError("expected ordered aware returns, unique dates and risky assets")
    month = dates.year * 12 + dates.month
    trade = np.r_[True, month[1:] != month[:-1]]
    count = len(returns.columns)
    equal = np.r_[np.full(count, 1 / count), 0.0]
    weights, nav = np.r_[np.zeros(count), 1.0], 1.0
    daily, navs, records = [], [], []
    for values, rebalance in zip(returns.to_numpy(dtype=float), trade):
        step = rebalance_step(
            nav, weights, equal if rebalance else weights, values, commission_rate, slippage_rate
        )
        daily.append(step.nav / nav - 1)
        navs.append(step.nav)
        if rebalance:
            records.append({"turnover": step.turnover, "cost": step.cost / nav})
        weights, nav = step.weights, step.nav
    return MVOBacktestResult(
        pd.Series(daily, dates, name="equal_weight"),
        pd.Series(navs, dates, name="nav"),
        pd.DataFrame(
            np.tile(equal, (len(records), 1)), dates[trade], columns=[*returns.columns, "CASH"]
        ),
        pd.DataFrame(records, index=dates[trade]),
    )
