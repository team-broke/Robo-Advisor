"""학습 구간 끝의 검증 구간으로 주기적으로 평가하고 최적 모델을 고른다.

검증 구간은 학습 컷오프 안의 마지막 일부다. 학습 환경은 검증 시작 전 행만 쓰고,
검증 롤아웃은 관측 창을 채우는 직전 lookback-1행만 학습 구간에서 빌려 온다.
지표는 비용 차감 후 일별 단순수익률로 계산하며, 정의할 수 없는 지표는 선택에서 뺀다.
점수가 같으면 먼저 얻은 모델을 유지한다.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

from robo_advisor.backtest.metrics import calculate_metrics
from robo_advisor.rl.evaluation import evaluate_policy


def split_validation(dataset, start, lookback: int):
    """학습용과 검증용 데이터셋을 나눈다. 검증용에는 관측 창 문맥 행이 앞에 붙는다."""
    from robo_advisor.rl.training import TrainingDataset

    index = dataset.returns.index
    position = int(index.searchsorted(pd.Timestamp(start)))
    if position <= lookback:
        raise ValueError("validation start leaves too few training rows for the lookback")
    if len(index) - position < 2:
        raise ValueError("validation period needs at least two rows")

    def rows(begin, end):
        return [
            frame.iloc[begin:end].copy() for frame in (dataset.returns, dataset.rsi, dataset.macd)
        ]

    train = TrainingDataset(
        *rows(0, position),
        tuple(tag for tag in dataset.risk_tags if tag.available_at < start),
        dataset.file_sha256,
    )
    validation = TrainingDataset(
        *rows(position - (lookback - 1), len(index)), dataset.risk_tags, dataset.file_sha256
    )
    return train, validation


def score_rollout(model, dataset, env_config, metric: str) -> dict:
    """검증 구간을 처음부터 끝까지 결정론적으로 굴려 지표와 점수를 만든다."""
    rollout = evaluate_policy(model, dataset, env_config)
    metrics = {}
    if len(rollout.returns) >= 2:
        reference = pd.Series(0.0, index=rollout.returns.index)
        metrics = {
            key: float(value) if np.isfinite(value) else None
            for key, value in calculate_metrics(rollout.returns, reference).items()
        }
    return {
        "score": metrics.get(metric),
        "metrics": metrics,
        "observation_count": rollout.observation_count,
        "terminated": rollout.terminated,
        "reason": rollout.reason,
    }


class ValidationCallback(BaseCallback):
    """eval_freq 스텝마다 검증하고 더 나은 점수면 best_path에 저장한다."""

    def __init__(
        self,
        dataset,
        env_config,
        *,
        eval_freq: int,
        n_envs: int,
        metric: str,
        higher_is_better: bool,
        best_path: Path,
    ):
        super().__init__()
        self.dataset = dataset
        self.env_config = env_config
        self.eval_calls = max(1, math.ceil(eval_freq / n_envs))
        self.metric = metric
        self.higher_is_better = higher_is_better
        self.best_path = Path(best_path)
        self.history: list[dict] = []
        self.best: dict | None = None

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_calls == 0:
            self.evaluate()
        return True

    def _better(self, score: float) -> bool:
        if self.best is None:
            return True
        best = self.best["score"]
        return score > best if self.higher_is_better else score < best

    def evaluate(self) -> dict:
        """현재 모델을 평가한다. 같은 스텝을 두 번 평가하지 않는다."""
        timesteps = int(self.model.num_timesteps)
        if self.history and self.history[-1]["timesteps"] == timesteps:
            return self.history[-1]
        record = {
            "timesteps": timesteps,
            **score_rollout(self.model, self.dataset, self.env_config, self.metric),
        }
        self.history.append(record)
        if record["score"] is not None and self._better(record["score"]):
            self.model.save(str(self.best_path))
            self.best = {
                "timesteps": timesteps,
                "score": record["score"],
                "path": str(self.best_path),
            }
        return record
