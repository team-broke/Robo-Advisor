"""전략 간 비교에 공통으로 사용하는 거래비용·리밸런싱 회계."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class AccountingResult:
    """비용과 수익률 적용이 끝난 NAV·표류 비중을 담는다."""

    nav: float
    weights: NDArray[np.float64]
    turnover: float
    cost: float


def _weights(value: ArrayLike) -> NDArray[np.float64]:
    """마지막 원소를 현금으로 해석하고 비중 계약을 검증한다."""
    weights = np.asarray(value, dtype=np.float64)
    if weights.ndim != 1 or weights.size < 2:
        raise ValueError("weights must contain at least one asset and cash")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("weights must be finite and non-negative")
    if not np.isclose(weights.sum(), 1.0, atol=1e-6, rtol=0):
        raise ValueError("weights must sum to 1 within 1e-6")
    return weights / weights.sum()


def calculate_turnover(current_weights: ArrayLike, target_weights: ArrayLike) -> float:
    """turnover = Σ(i=1..N) |w_target,i - w_drift,i|; 현금은 제외한다.

    current_weights는 리밸런싱 직전 표류 비중이다. 매수·매도 양방향을
    합산하며 1/2을 곱하지 않는다. 마지막 원소는 현금 비중이다.
    """
    current = _weights(current_weights)
    target = _weights(target_weights)
    if current.shape != target.shape:
        raise ValueError("current and target weights must have the same shape")
    return float(np.abs(target[:-1] - current[:-1]).sum())


def log_to_simple_returns(log_returns: ArrayLike) -> NDArray[np.float64]:
    """자산 로그수익률을 exp(r) - 1로 변환하고 유한성을 검증한다."""
    values = np.asarray(log_returns, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("log returns must be a non-empty finite vector")
    with np.errstate(over="ignore"):
        returns = np.expm1(values)
    if not np.all(np.isfinite(returns)):
        raise ValueError("converted simple returns must be finite")
    return returns


def rebalance_step(
    nav: float,
    current_weights: ArrayLike,
    target_weights: ArrayLike,
    asset_returns: ArrayLike,
    commission_rate: float = 0.00015,
    slippage_rate: float = 0.0005,
) -> AccountingResult:
    """비용 차감 → 목표 비중 배분 → 자산 단순수익률 적용 순서로 회계 처리한다.

    비용 = turnover × (commission_rate + slippage_rate) × 직전 NAV.
    비용 차감 후 NAV 전체를 목표 비중으로 배분하고 현금 수익률은 0으로 둔다.
    비중은 자산 N개 + 현금 1개이며 asset_returns는 자산 N개의 단순수익률이다.
    NAV가 소진되는 거래나 전액 손실은 표류 비중을 정의할 수 없어 거부한다.
    """
    if not np.isfinite(nav) or nav <= 0:
        raise ValueError("nav must be finite and positive")
    for rate in (commission_rate, slippage_rate):
        if not np.isfinite(rate) or not 0 <= rate < 1:
            raise ValueError("cost rates must be finite and in [0, 1)")
    turnover = calculate_turnover(current_weights, target_weights)
    target = _weights(target_weights)
    returns = np.asarray(asset_returns, dtype=np.float64)
    if returns.ndim != 1 or returns.size != target.size - 1:
        raise ValueError("returns must have one value per asset, excluding cash")
    if not np.all(np.isfinite(returns)) or np.any(returns < -1):
        raise ValueError("simple returns must be finite and at least -1")

    cost = turnover * (commission_rate + slippage_rate) * nav
    if not np.isfinite(cost) or cost >= nav:
        raise ValueError("cost must leave a positive nav")
    with np.errstate(over="ignore", invalid="ignore"):
        holdings = (nav - cost) * target * np.append(1 + returns, 1.0)
        new_nav = float(holdings.sum())
    if not np.isfinite(new_nav) or new_nav <= 0:
        raise ValueError("resulting nav must be finite and positive")
    return AccountingResult(new_nav, holdings / new_nav, turnover, float(cost))
