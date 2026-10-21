"""학습된 정책의 결정론적 단일 에피소드 평가."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from robo_advisor.rl.training import EnvironmentConfig, TrainingDataset, _make_env


@dataclass(frozen=True)
class EvaluationResult:
    """수익률은 전이 종료일 기준이며 NAV에는 초기 관측 시점도 포함한다."""

    returns: pd.Series
    nav: pd.Series
    terminated: bool
    truncated: bool
    reason: str | None
    observation_count: int


def evaluate_policy(model, dataset: TrainingDataset, config: EnvironmentConfig) -> EvaluationResult:
    """비용 차감 후 NAV로 단순수익률을 계산하며 조기 종료 뒤는 채우지 않는다."""
    env = _make_env(dataset, config, seed=0)
    dates, returns = [], []
    try:
        observation, info = env.reset(seed=0)
        nav_dates, nav_values = [info["timestamp"]], [float(info["nav"])]
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(action)
            current_nav = float(info["nav"])
            returns.append(current_nav / nav_values[-1] - 1)
            dates.append(info["timestamp"])
            nav_dates.append(info["timestamp"])
            nav_values.append(current_nav)
            if terminated or truncated:
                break
    finally:
        env.close()
    values = pd.Series(returns, index=pd.DatetimeIndex(dates), name="portfolio_return")
    nav = pd.Series(nav_values, index=pd.DatetimeIndex(nav_dates), name="nav")
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("evaluated returns must be finite")
    return EvaluationResult(values, nav, terminated, truncated, info["reason"], len(values))
