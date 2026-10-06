"""성과지표의 수계산 기준 및 경계 조건 검증."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from robo_advisor.backtest.metrics import calculate_metrics


def test_reference_series():
    returns = pd.Series([0.1, -0.2, 0.05, -0.1])
    benchmark = returns / 2
    result = calculate_metrics(returns, benchmark, periods_per_year=4)
    wealth = 1.1 * 0.8 * 1.05 * 0.9
    std = np.sqrt(sum((r + 0.0375) ** 2 for r in returns) / 3)
    expected = {
        "cumulative_return": wealth - 1,
        "cagr": wealth - 1,
        "volatility": std * 2,
        "var_95": 0.185,
        "cvar_95": 0.2,
        "mdd": 1 - 0.8 * 1.05 * 0.9,
        "sharpe": -0.0375 * 2 / std,
        "sortino": -0.0375 * 2 / np.sqrt((0.04 + 0.01) / 4),
        "calmar": (wealth - 1) / (1 - 0.8 * 1.05 * 0.9),
        "alpha": 0,
        "beta": 2,
        "information_ratio": -0.0375 * 2 / std,
    }
    assert result == pytest.approx(expected, abs=1e-12)


def test_initial_loss_and_bankruptcy():
    r = pd.Series([-0.2, -1.0, 0.5])
    result = calculate_metrics(r, pd.Series([0.1, 0.2, 0.3]))
    assert result["mdd"] == 1
    assert result["cagr"] == -1


def test_risk_free_and_parametric_tail():
    r = pd.Series([-0.02, 0.01, 0.03])
    result = calculate_metrics(r, r, risk_free_rate=0.1, var_method="parametric")
    daily_rf = 1.1 ** (1 / 252) - 1
    assert result["sharpe"] == pytest.approx((r.mean() - daily_rf) / r.std() * np.sqrt(252))
    assert result["var_95"] == pytest.approx(-r.mean() - r.std() * norm.ppf(0.05))
    assert result["cvar_95"] == pytest.approx(-r.mean() + r.std() * norm.pdf(norm.ppf(0.05)) / 0.05)


def test_undefined_ratios_are_nan():
    result = calculate_metrics(pd.Series([0.0, 0.0]), pd.Series([0.0, 0.0]))
    for key in ("sharpe", "sortino", "calmar", "alpha", "beta", "information_ratio"):
        assert np.isnan(result[key])


def test_fractional_historical_tail():
    r = pd.Series([-0.2, -0.1] + [0.0] * 28)
    assert calculate_metrics(r, r)["cvar_95"] == pytest.approx(0.25 / 1.5)


@pytest.mark.parametrize("values", [[], [0.1], [0.1, np.nan], [0.1, np.inf], [-1.1, 0.1]])
def test_invalid_returns(values):
    r = pd.Series(values, dtype=float)
    with pytest.raises(ValueError):
        calculate_metrics(r, r)


@pytest.mark.parametrize(
    "kwargs", [{"periods_per_year": 0}, {"risk_free_rate": -1}, {"var_method": "unknown"}]
)
def test_invalid_options(kwargs):
    r = pd.Series([0.1, 0.2])
    with pytest.raises(ValueError):
        calculate_metrics(r, r, **kwargs)


def test_misaligned_benchmark_rejected():
    with pytest.raises(ValueError):
        calculate_metrics(pd.Series([0.1, 0.2]), pd.Series([0.1, 0.2], index=[1, 2]))
