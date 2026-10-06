"""같은 포트폴리오 궤적에 적용하는 세 가지 보상 함수."""

from enum import Enum

import numpy as np
from numpy.typing import ArrayLike


class RewardVariant(str, Enum):
    """보상 수식 선택값을 정의한다."""

    RETURN = "return"
    SHARPE = "sharpe"
    MDD_PENALTY = "mdd_penalty"


def calculate_reward(
    portfolio_log_return: float,
    variant: RewardVariant | str = RewardVariant.RETURN,
    *,
    return_history: ArrayLike = (),
    drawdown: float = 0.0,
    penalty_lambda: float = 1.0,
) -> float:
    """r_t, r_t / max(σ_t, 1e-8), r_t - λ × MDD_t 중 하나를 반환한다.

    r_t는 비용 차감 후 포트폴리오 로그수익률이다. return_history에는 현재
    r_t를 제외한 이전 일 단위 로그수익률을 시간순으로 전달한다. σ_t는 직전
    최대 20개 관측의 모집단 표준편차(ddof=0)이며 연율화하지 않는다.
    샤프 변형은 이전 관측이 2개 미만이면 0, 변동성이 0이면 분모 1e-8을 쓴다.
    다른 두 변형은 초기 스텝에도 수식을 적용한다. MDD_t는 누적 최대 낙폭이
    아닌 현재 고점 대비 낙폭이며, 초기 고점에서는 0이다. λ는 음수를 허용하지 않는다.
    """
    selected = RewardVariant(variant)
    if not np.isfinite(portfolio_log_return):
        raise ValueError("portfolio log return must be finite")
    if not np.isfinite(drawdown) or not 0 <= drawdown <= 1:
        raise ValueError("drawdown must be finite and in [0, 1]")
    if not np.isfinite(penalty_lambda) or penalty_lambda < 0:
        raise ValueError("penalty lambda must be finite and non-negative")
    history = np.asarray(return_history, dtype=np.float64)
    if history.ndim != 1 or not np.all(np.isfinite(history)):
        raise ValueError("return history must be a finite vector")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if selected is RewardVariant.SHARPE:
            window = history[-20:]
            if window.size < 2:
                return 0.0
            scale = float(np.max(np.abs(window)))
            sigma = float(np.std(window / scale, ddof=0) * scale) if scale else 0.0
            reward = portfolio_log_return / max(sigma, 1e-8)
        elif selected is RewardVariant.MDD_PENALTY:
            reward = portfolio_log_return - penalty_lambda * drawdown
        else:
            reward = portfolio_log_return
    if not np.isfinite(reward):
        raise ValueError("reward must be finite")
    return float(reward)
