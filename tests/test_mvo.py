"""최대 샤프 해, 과거 데이터 범위와 실패 정책 검증."""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import OptimizeResult

from robo_advisor.baselines import mvo


def sample_returns():
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        rng.normal(0.001, 0.01, (300, 4)),
        index=pd.bdate_range("2020-01-01", periods=300),
        columns=list("ABCD"),
    )


def test_constraints_and_no_future_reference():
    returns = sample_returns()
    result = mvo.rolling_markowitz(returns, as_of=returns.index[270])
    assert result.success and not result.used_fallback
    assert result.observation_count == 252
    assert result.weights.sum() == pytest.approx(1)
    assert (result.weights >= 0).all() and (result.weights <= 0.4 + 1e-12).all()
    changed = returns.copy()
    changed.iloc[270:] = np.nan
    changed.iloc[:18] = 1000
    other = mvo.rolling_markowitz(changed, as_of=returns.index[270])
    pd.testing.assert_series_equal(result.weights, other.weights, check_exact=True)


def test_diagonal_covariance_known_maximum_sharpe():
    # 직교 부호열로 동일 분산과 공분산 0인 표본을 구성한다.
    signs = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
    values = np.tile(signs, (63, 1)) * 0.01 + np.array([0.0008, 0.0007, 0.0005])
    returns = pd.DataFrame(values, index=pd.bdate_range("2020", periods=252))
    result = mvo.rolling_markowitz(returns, as_of=returns.index[-1] + pd.Timedelta(days=1))
    assert result.success
    np.testing.assert_allclose(result.weights, [0.4, 0.35, 0.25], atol=2e-5)


@pytest.mark.parametrize(
    "solution, reason",
    [
        (OptimizeResult(success=False, message="iteration limit"), "solver_failure"),
        (OptimizeResult(success=True, x=[0.1] * 4), "weight_sum_violation"),
        (OptimizeResult(success=True, x=[0.7, 0.1, 0.1, 0.1]), "weight_bound_violation"),
        (OptimizeResult(success=True, x=[np.nan] * 4), "nonfinite_or_invalid_weights"),
        (OptimizeResult(success=True, x=[0.5, 0.5]), "nonfinite_or_invalid_weights"),
    ],
)
def test_solver_failures_use_predefined_weights(monkeypatch, caplog, solution, reason):
    returns = sample_returns()
    fallback = pd.Series([0.4, 0.3, 0.2, 0.1], index=returns.columns)
    monkeypatch.setattr(mvo, "minimize", lambda *args, **kwargs: solution)
    result = mvo.rolling_markowitz(returns, as_of=returns.index[-1], fallback_weights=fallback)
    assert not result.success and result.used_fallback
    assert result.reason.startswith(reason)
    assert reason in caplog.text
    pd.testing.assert_series_equal(result.weights, fallback)


def test_solver_exception_is_recorded(monkeypatch):
    def raise_error(*args, **kwargs):
        raise RuntimeError("failed")

    monkeypatch.setattr(mvo, "minimize", raise_error)
    returns = sample_returns()
    result = mvo.rolling_markowitz(returns, as_of=returns.index[-1])
    assert result.reason == "solver_exception: failed"
    np.testing.assert_allclose(result.weights, [0.25] * 4)


def test_insufficient_history_singular_and_missing_data():
    returns = sample_returns()
    assert mvo.rolling_markowitz(returns, as_of=returns.index[10]).reason == "insufficient_history"
    singular = returns.copy()
    singular["B"] = singular["A"]
    assert mvo.rolling_markowitz(singular, as_of=returns.index[-1]).reason == "singular_covariance"
    returns.iloc[100, 0] = np.nan
    assert mvo.rolling_markowitz(returns, as_of=returns.index[-1]).reason == "invalid_returns"


@pytest.mark.parametrize(
    "options",
    [{"max_weight": 0.2}, {"lookback": 1}, {"risk_free_rate": -1}, {"constraint_tolerance": 1}],
)
def test_invalid_settings(options):
    returns = sample_returns()
    with pytest.raises(ValueError):
        mvo.rolling_markowitz(returns, as_of=returns.index[-1], **options)


def test_invalid_fallback_rejected_before_solver(monkeypatch):
    returns = sample_returns()

    def forbidden(*args, **kwargs):
        pytest.fail("invalid fallback must be rejected before optimization")

    monkeypatch.setattr(mvo, "minimize", forbidden)
    with pytest.raises(ValueError, match="fallback"):
        mvo.rolling_markowitz(
            returns,
            as_of=returns.index[-1],
            fallback_weights=pd.Series([1.0, 0, 0, 0], index=returns.columns),
        )


def test_small_solver_residual_is_cleaned(monkeypatch):
    returns = sample_returns()
    monkeypatch.setattr(
        mvo,
        "minimize",
        lambda *a, **k: OptimizeResult(success=True, x=[0.4 + 1e-10, 0.3, 0.3, -1e-10]),
    )
    result = mvo.rolling_markowitz(returns, as_of=returns.index[-1])
    assert result.success
    assert result.weights.sum() == pytest.approx(1, abs=1e-12)
    assert (result.weights >= 0).all() and (result.weights <= 0.4).all()
