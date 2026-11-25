import pandas as pd
import pytest

from robo_advisor.backtest.equal_weight_backtest import run_equal_weight_backtest


def test_monthly_equal_weight_has_cash_zero_trade_cost_and_drift():
    dates = pd.DatetimeIndex(
        ["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02"], tz="Asia/Seoul"
    )
    returns = pd.DataFrame({"A": [0.5, 0.5, 0.0, 0.0], "B": 0.0}, index=dates)
    result = run_equal_weight_backtest(returns, commission_rate=0.01, slippage_rate=0.005)
    assert result.weights.index.equals(dates[[0, 2]])
    assert result.weights.to_numpy().tolist() == [[0.5, 0.5, 0.0]] * 2
    assert result.rebalances.turnover.iloc[0] == pytest.approx(1.0)
    assert result.rebalances.cost.iloc[0] == pytest.approx(0.015)
    assert result.nav.iloc[0] == pytest.approx(0.985 * 1.25)
    assert result.returns.iloc[1] == pytest.approx(0.3)  # A drifted to 60%, no trade.
    assert result.rebalances.turnover.iloc[1] == pytest.approx(2 * (2.25 / 3.25 - 0.5))
    assert result.returns.iloc[2] == pytest.approx(-result.rebalances.cost.iloc[1])


def test_month_is_in_evaluation_timezone():
    dates = pd.DatetimeIndex(["2024-01-31 14:00", "2024-01-31 16:00"], tz="UTC")
    returns = pd.DataFrame({"A": [0.0, 0.0]}, index=dates.tz_convert("Asia/Seoul"))
    result = run_equal_weight_backtest(returns, commission_rate=0.01, slippage_rate=0.0)
    assert len(result.rebalances) == 2
