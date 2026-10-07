"""일별 단순 수익률의 성과지표. 비용 차감 여부는 입력 수익률에 따른다."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def _returns(series: pd.Series, name: str) -> np.ndarray:
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    values = series.to_numpy(dtype=float)
    if len(values) < 2 or not np.isfinite(values).all() or (values < -1).any():
        raise ValueError(f"{name} requires at least two finite returns >= -1")
    if not series.index.is_unique or not series.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be unique and increasing")
    return values


def calculate_metrics(
    returns: pd.Series,
    benchmark: pd.Series,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    var_method: str = "historical",
) -> dict[str, float]:
    """12개 지표를 계산한다. benchmark는 입력과 동일한 날짜/단위여야 한다.

    rf는 연 유효 수익률이며 일 rf는 기하 변환한다. 변동성은 표본 표준편차,
    Sortino는 전체 표본의 음수 초과수익률 제곱평균을 사용한다. Alpha는 CAPM
    일 절편의 산술 연환산, IR은 일 active return 평균/표준편차의 연환산이다.
    MDD는 초기 자산 1을 포함한 양의 손실률이다. VaR/CVaR도 양의 손실이지만
    전체 꼬리가 수익이면 음수가 될 수 있다. Historical CVaR는 최악 5%의
    확률질량을 사용하여 경계 표본에 분수 가중치를 준다. 정의 불가능한 비율은 NaN.
    """
    values = _returns(returns, "returns")
    reference = _returns(benchmark, "benchmark")
    if not returns.index.equals(benchmark.index):
        raise ValueError("returns and benchmark indices must match exactly")
    if (
        isinstance(periods_per_year, bool)
        or not isinstance(periods_per_year, int)
        or periods_per_year <= 0
    ):
        raise ValueError("periods_per_year must be a positive integer")
    if not np.isfinite(risk_free_rate) or risk_free_rate <= -1:
        raise ValueError("risk_free_rate must be finite and greater than -1")
    if var_method not in {"historical", "parametric"}:
        raise ValueError("var_method must be historical or parametric")

    def ratio(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator > 0 else float("nan")

    scale = np.sqrt(periods_per_year)
    daily_rf = np.expm1(np.log1p(risk_free_rate) / periods_per_year)
    excess = values - daily_rf
    benchmark_excess = reference - daily_rf
    std = values.std(ddof=1)
    wealth = np.concatenate(([1.0], np.cumprod(1.0 + values)))
    mdd = float(np.max(1.0 - wealth / np.maximum.accumulate(wealth)))
    cagr = float(wealth[-1] ** (periods_per_year / len(values)) - 1.0)
    if var_method == "historical":
        var = -float(np.quantile(values, 0.05))
        mass = len(values) * 0.05
        whole = int(np.floor(mass))
        ordered = np.sort(values)
        tail = ordered[:whole].sum()
        if whole < len(values):
            tail += (mass - whole) * ordered[whole]
        cvar = -float(tail / mass)
    else:
        z = norm.ppf(0.05)
        var = float(-values.mean() - std * z)
        cvar = float(-values.mean() + std * norm.pdf(z) / 0.05)
    beta = ratio(
        float(np.cov(excess, benchmark_excess, ddof=1)[0, 1]),
        float(benchmark_excess.var(ddof=1)),
    )
    active = values - reference
    return {
        "cumulative_return": float(wealth[-1] - 1.0),
        "cagr": cagr,
        "volatility": float(std * scale),
        "var_95": var,
        "cvar_95": cvar,
        "mdd": mdd,
        "sharpe": ratio(float(excess.mean() * scale), float(std)),
        "sortino": ratio(
            float(excess.mean() * scale), float(np.sqrt(np.mean(np.minimum(excess, 0) ** 2)))
        ),
        "calmar": ratio(cagr, mdd),
        "alpha": float((excess.mean() - beta * benchmark_excess.mean()) * periods_per_year),
        "beta": beta,
        "information_ratio": ratio(float(active.mean() * scale), float(active.std(ddof=1))),
    }
