"""과거 수익률만 사용하는 롤링 최대 샤프 기준선과 명시적 실패 정책."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MVOResult:
    weights: pd.Series
    success: bool
    used_fallback: bool
    reason: str | None
    observation_count: int


def rolling_markowitz(
    returns: pd.DataFrame,
    *,
    as_of: str | pd.Timestamp,
    lookback: int = 252,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    max_weight: float = 0.4,
    fallback_weights: pd.Series | None = None,
    constraint_tolerance: float = 1e-8,
) -> MVOResult:
    """as_of 미만의 마지막 lookback 관측으로 평균/표본 공분산을 추정한다.

    입력은 비용 적용 전 일별 단순 수익률이다. rf는 연 유효 수익률이다.
    기본 대체 비중은 동일가중이며 사용자 대체 비중은 시작 전에 제약을 검증한다.
    데이터 부족/비유한 값/특이 공분산/최적화 실패/제약 위반을 로그와 결과에
    남긴다. as_of 당일 수익률과 이후 데이터는 추정 및 검증에 사용하지 않는다.
    잘못된 설정과 불가능한 제약은 대체 비중 대신 ValueError로 처리한다.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a DataFrame")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns index must be a DatetimeIndex")
    if (
        returns.index.hasnans
        or not returns.index.is_unique
        or not returns.index.is_monotonic_increasing
        or not returns.columns.is_unique
        or len(returns.columns) == 0
    ):
        raise ValueError("dates and assets must be unique and dates increasing without NaT")
    for name, value, minimum in (
        ("lookback", lookback, 2),
        ("periods_per_year", periods_per_year, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not np.isfinite(risk_free_rate) or risk_free_rate <= -1:
        raise ValueError("risk_free_rate must be finite and greater than -1")
    if not np.isfinite(max_weight) or not 0 < max_weight <= 1:
        raise ValueError("max_weight must be in (0, 1]")
    assets = len(returns.columns)
    if assets * max_weight < 1:
        raise ValueError("max_weight makes full investment infeasible")
    if not np.isfinite(constraint_tolerance) or not 0 < constraint_tolerance <= 1e-4:
        raise ValueError("constraint_tolerance must be in (0, 1e-4]")
    timestamp = pd.Timestamp(as_of)
    if pd.isna(timestamp) or str(timestamp.tzinfo) != str(returns.index.tz):
        raise ValueError("as_of must be valid and have the same timezone as returns")

    def constraint_error(weights: np.ndarray, tolerance: float) -> str | None:
        if weights.shape != (assets,) or not np.isfinite(weights).all():
            return "nonfinite_or_invalid_weights"
        if abs(weights.sum() - 1) > tolerance:
            return "weight_sum_violation"
        if (weights < -tolerance).any() or (weights > max_weight + tolerance).any():
            return "weight_bound_violation"
        return None

    fallback = np.full(assets, 1 / assets)
    if fallback_weights is not None:
        if not isinstance(fallback_weights, pd.Series):
            raise TypeError("fallback_weights must be a Series")
        if not fallback_weights.index.equals(returns.columns):
            raise ValueError("fallback asset order must match returns columns")
        fallback = fallback_weights.to_numpy(dtype=float, copy=True)
    if constraint_error(fallback, 1e-12) is not None:
        raise ValueError("fallback_weights must satisfy full investment and weight bounds")
    if (fallback < 0).any() or (fallback > max_weight).any():
        raise ValueError("fallback_weights must satisfy weight bounds exactly")
    history = returns.loc[returns.index < timestamp].tail(lookback)
    count = len(history)

    def failed(reason: str) -> MVOResult:
        logger.warning("MVO fallback at %s: %s", timestamp, reason)
        return MVOResult(
            pd.Series(fallback.copy(), index=returns.columns), False, True, reason, count
        )

    if count < lookback:
        return failed("insufficient_history")
    try:
        # pandas 블록 배치가 달라도 동일 표본의 부동소수점 집계 순서를 유지한다.
        sample = np.array(history.to_numpy(dtype=float), order="C", copy=True)
    except (TypeError, ValueError):
        return failed("invalid_returns")
    if not np.isfinite(sample).all() or (sample < -1).any():
        return failed("invalid_returns")
    covariance = np.atleast_2d(np.cov(sample, rowvar=False, ddof=1))
    if not np.isfinite(covariance).all():
        return failed("nonfinite_covariance")
    if np.linalg.matrix_rank(covariance) < assets:
        return failed("singular_covariance")
    daily_rf = np.expm1(np.log1p(risk_free_rate) / periods_per_year)
    mean_excess = sample.mean(axis=0) - daily_rf

    def objective(weights: np.ndarray) -> float:
        variance = float(weights @ covariance @ weights)
        if variance <= 0 or not np.isfinite(variance):
            return float("inf")
        return float(-(weights @ mean_excess) / np.sqrt(variance))

    try:
        solution = minimize(
            objective,
            np.full(assets, 1 / assets),
            method="SLSQP",
            bounds=[(0.0, max_weight)] * assets,
            constraints=[{"type": "eq", "fun": lambda weights: weights.sum() - 1}],
            options={"ftol": 1e-10, "maxiter": 1000},
        )
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
        return failed(f"solver_exception: {error}")
    if not solution.success:
        return failed(f"solver_failure: {solution.message}")
    weights = np.asarray(solution.x, dtype=float)
    error = constraint_error(weights, constraint_tolerance)
    if error is not None:
        return failed(error)
    if not np.isfinite(objective(weights)):
        return failed("nonfinite_objective")
    # 허용 오차 이내의 수치 잔차만 정리하고 경계를 다시 확인한다.
    weights = np.clip(weights, 0, max_weight)
    residual = 1 - weights.sum()
    slack = max_weight - weights if residual > 0 else weights
    if residual != 0:
        if slack.sum() <= 0:
            return failed("weight_sum_violation")
        weights += residual * slack / slack.sum()
    error = constraint_error(weights, 1e-12)
    if error is not None:
        return failed(error)
    return MVOResult(pd.Series(weights, index=returns.columns), True, False, None, count)
