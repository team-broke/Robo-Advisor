"""학습 진행률과 최근 에피소드 요약을 주기적으로 로그로 남긴다."""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable

from stable_baselines3.common.callbacks import BaseCallback

from robo_advisor.rl.curves import SAFEGUARD_REASON

LOGGER = logging.getLogger("robo_advisor.training")


def format_duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


class ProgressCallback(BaseCallback):
    """interval_seconds마다 진행률을 남기고, 체크포인트 저장 시점도 알린다.

    에피소드 요약은 Monitor가 종료 스텝 info에 넣는 episode(r, l, reason)를 쓴다.
    """

    def __init__(
        self,
        total_timesteps: int,
        *,
        interval_seconds: float = 30.0,
        window: int = 100,
        checkpoint_calls: int | None = None,
        checkpoint_dir=None,
        clock: Callable[[], float] = time.monotonic,
        logger: logging.Logger = LOGGER,
    ):
        super().__init__()
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.total = total_timesteps
        self.interval = interval_seconds
        self.episodes: deque = deque(maxlen=window)
        self.episode_count = 0
        self.checkpoint_calls = checkpoint_calls
        self.checkpoint_dir = checkpoint_dir
        self.clock = clock
        self.log = logger
        self.started = None
        self.last_report = None

    def _on_training_start(self) -> None:
        self.started = self.last_report = self.clock()
        self.log.info("학습 시작: 목표 %s 스텝", f"{self.total:,}")

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", ()):
            episode = info.get("episode")
            if episode is not None:
                self.episodes.append(episode)
                self.episode_count += 1
        if self.checkpoint_calls and self.n_calls % self.checkpoint_calls == 0:
            self.log.info(
                "체크포인트 저장: %s 스텝 (%s)", f"{self.num_timesteps:,}", self.checkpoint_dir
            )
        now = self.clock()
        if now - self.last_report >= self.interval:
            self.last_report = now
            self.log.info(self.summary(now))
        return True

    def _on_training_end(self) -> None:
        self.log.info(self.summary(self.clock()))

    def summary(self, now: float) -> str:
        elapsed = now - self.started
        done = int(self.num_timesteps)
        speed = done / elapsed if elapsed > 0 else 0.0
        remaining = (self.total - done) / speed if speed > 0 and done < self.total else 0.0
        text = (
            f"진행 {min(done / self.total, 1.0):6.1%} | {done:,}/{self.total:,} 스텝 | "
            f"{speed:,.0f} 스텝/초 | 경과 {format_duration(elapsed)} | "
            f"남은 시간 {format_duration(remaining)}"
        )
        if self.episodes:
            count = len(self.episodes)
            reward = sum(float(item["r"]) for item in self.episodes) / count
            length = sum(float(item["l"]) for item in self.episodes) / count
            safeguard = sum(item.get("reason") == SAFEGUARD_REASON for item in self.episodes)
            text += (
                f" | 최근 에피소드 {count}개: 평균 보상 {reward:.4f}, 평균 길이 {length:.0f}, "
                f"Safe-Guard 종료 {safeguard / count:.0%} (누적 {self.episode_count:,}개)"
            )
        return text
