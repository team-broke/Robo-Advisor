"""월 1회 MVO 백테스트의 리밸런싱 일정, 회계 일치, 미래 참조 검증."""

import numpy as np
import pandas as pd
import pytest

from robo_advisor.backtest.mvo_backtest import run_mvo_backtest
from robo_advisor.rl.accounting import rebalance_step


def sample_returns(periods=330):
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        rng.normal(0.0005, 0.01, (periods, 4)),
        index=pd.bdate_range("2020-01-01", periods=periods, tz="Asia/Seoul"),
        columns=list("ABCD"),
    )


def test_rebalances_on_first_evaluation_day_and_each_month_start():
    returns = sample_returns()
    result = run_mvo_backtest(returns, start="2021-01-15", end="2021-04-01", lookback=20)
    expected = pd.DatetimeIndex(
        ["2021-01-15", "2021-02-01", "2021-03-01"], tz="Asia/Seoul", name="date"
    )
    pd.testing.assert_index_equal(result.rebalances.index, expected)
    pd.testing.assert_index_equal(result.weights.index, expected)
    assert list(result.weights.columns) == ["A", "B", "C", "D", "CASH"]


def test_evaluation_window_is_half_open():
    returns = sample_returns()
    result = run_mvo_backtest(returns, start="2021-01-04", end="2021-02-01", lookback=20)
    assert result.returns.index[0] == pd.Timestamp("2021-01-04", tz="Asia/Seoul")
    assert result.returns.index[-1] == pd.Timestamp("2021-01-29", tz="Asia/Seoul")
    assert result.returns.index.equals(result.nav.index)


def test_accounting_matches_manual_rebalance_steps():
    # 이력이 부족하면 모든 리밸런싱이 동일가중 대체 비중을 사용하므로 직접 재현할 수 있다.
    returns = sample_returns()
    start, end = "2021-01-15", "2021-03-10"
    result = run_mvo_backtest(
        returns, start=start, end=end, lookback=1000, commission_rate=0.001, slippage_rate=0.002
    )
    window = returns.loc[(returns.index >= start) & (returns.index < end)]
    target = np.array([0.25, 0.25, 0.25, 0.25, 0.0])
    nav, weights, expected = 1.0, np.array([0, 0, 0, 0, 1.0]), []
    for date, row in window.iterrows():
        goal = target if date in result.rebalances.index else weights
        step = rebalance_step(nav, weights, goal, row.to_numpy(), 0.001, 0.002)
        expected.append(step.nav / nav - 1)
        nav, weights = step.nav, step.weights
    np.testing.assert_allclose(result.returns.to_numpy(), expected, rtol=0, atol=1e-15)
    assert result.nav.iloc[-1] == pytest.approx(nav, rel=0, abs=1e-15)


def test_first_rebalance_buys_from_cash_with_default_cost():
    returns = sample_returns()
    result = run_mvo_backtest(returns, start="2021-01-15", end="2021-03-10", lookback=1000)
    first = result.rebalances.iloc[0]
    assert first["turnover"] == pytest.approx(1.0)
    assert first["cost"] == pytest.approx(0.00065)


def test_timezone_aware_boundaries_are_converted_to_returns_timezone():
    returns = sample_returns()
    local = run_mvo_backtest(returns, start="2021-01-15", end="2021-02-01", lookback=20)
    # 서울 2021-01-15 00:00 = UTC 2021-01-14 15:00
    utc = run_mvo_backtest(
        returns,
        start=pd.Timestamp("2021-01-14 15:00", tz="UTC"),
        end=pd.Timestamp("2021-01-31 15:00", tz="UTC"),
        lookback=20,
    )
    pd.testing.assert_series_equal(local.returns, utc.returns, check_exact=True)


def test_records_mvo_success_and_fallback_reasons():
    returns = sample_returns()
    short = run_mvo_backtest(returns, start="2021-01-15", end="2021-03-01", lookback=1000)
    assert short.rebalances["used_fallback"].all()
    assert (short.rebalances["reason"] == "insufficient_history").all()
    fitted = run_mvo_backtest(returns, start="2021-01-15", end="2021-03-01", lookback=252)
    assert fitted.rebalances["success"].all()
    assert not fitted.rebalances["used_fallback"].any()
    assert fitted.rebalances["reason"].isna().all()
    assert (fitted.weights["CASH"] == 0).all()
    assert fitted.weights.drop(columns="CASH").le(0.4 + 1e-8).all().all()


@pytest.mark.leakage
def test_rebalance_weights_ignore_same_day_and_future_returns():
    returns = sample_returns()
    base = run_mvo_backtest(returns, start="2021-01-15", end="2021-04-01", lookback=252)
    changed = returns.copy()
    rebalance_day = base.rebalances.index[1]
    changed.loc[changed.index >= rebalance_day] *= -3
    other = run_mvo_backtest(changed, start="2021-01-15", end="2021-04-01", lookback=252)
    pd.testing.assert_frame_equal(base.weights.iloc[:2], other.weights.iloc[:2], check_exact=True)
    before = base.returns.index < rebalance_day
    pd.testing.assert_series_equal(base.returns[before], other.returns[before], check_exact=True)


@pytest.mark.leakage
def test_returns_after_end_do_not_change_result():
    returns = sample_returns()
    base = run_mvo_backtest(returns, start="2021-01-15", end="2021-03-01", lookback=252)
    changed = returns.copy()
    changed.loc[changed.index >= "2021-03-01"] = np.nan
    other = run_mvo_backtest(changed, start="2021-01-15", end="2021-03-01", lookback=252)
    pd.testing.assert_series_equal(base.returns, other.returns, check_exact=True)
    pd.testing.assert_frame_equal(base.weights, other.weights, check_exact=True)


@pytest.mark.parametrize(
    "start, end",
    [("2021-03-01", "2021-03-01"), ("2030-01-01", "2030-02-01"), ("2021-03-01", "2021-01-01")],
)
def test_rejects_empty_or_reversed_window(start, end):
    with pytest.raises(ValueError, match="evaluation window"):
        run_mvo_backtest(sample_returns(), start=start, end=end)


def test_rejects_nonfinite_returns_inside_window():
    returns = sample_returns()
    returns.loc["2021-02-10", "B"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        run_mvo_backtest(returns, start="2021-01-15", end="2021-03-01")
