"""학습 진행률과 최근 에피소드 요약을 로그와 실시간 상태 파일로 남긴다.

상태 파일(status.json)은 학습 모니터·대기열이 읽는 계약이다. 형식은 schema_version 1이며
state, 진행률, 속도, 남은 시간, 최근 에피소드, 검증 점수를 담고 원자적으로 교체한다.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from robo_advisor.rl.curves import SAFEGUARD_REASON

LOGGER = logging.getLogger("robo_advisor.training")
STATUS_SCHEMA_VERSION = 1


def format_duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


class ProgressCallback(BaseCallback):
    """interval_seconds마다 진행률을 로그로, status_interval_seconds마다 상태 파일로 남긴다.

    에피소드 요약은 Monitor가 종료 스텝 info에 넣는 episode(r, l, reason)를 쓴다.
    validator를 주면 최근·최적 검증 점수를 상태 파일에 함께 기록한다.
    """

    def __init__(
        self,
        total_timesteps: int,
        *,
        interval_seconds: float = 30.0,
        window: int = 100,
        checkpoint_calls: int | None = None,
        checkpoint_dir=None,
        status_path: Path | None = None,
        status_interval_seconds: float = 5.0,
        run_id: str | None = None,
        validator=None,
        clock: Callable[[], float] = time.monotonic,
        logger: logging.Logger = LOGGER,
    ):
        super().__init__()
        if interval_seconds <= 0 or status_interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.total = total_timesteps
        self.interval = interval_seconds
        self.status_interval = status_interval_seconds
        self.episodes: deque = deque(maxlen=window)
        self.episode_count = 0
        self.checkpoint_calls = checkpoint_calls
        self.checkpoint_dir = checkpoint_dir
        self.status_path = None if status_path is None else Path(status_path)
        self.run_id = run_id
        self.validator = validator
        self.clock = clock
        self.log = logger
        self.started = None
        self.started_at = _utc_now()
        self.last_report = None
        self.last_status = None
        self.last_checkpoint = None

    def _on_training_start(self) -> None:
        self.started = self.last_report = self.last_status = self.clock()
        self.log.info("학습 시작: 목표 %s 스텝", f"{self.total:,}")
        self.write_status()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", ()):
            episode = info.get("episode")
            if episode is not None:
                self.episodes.append(episode)
                self.episode_count += 1
        if self.checkpoint_calls and self.n_calls % self.checkpoint_calls == 0:
            self.last_checkpoint = int(self.num_timesteps)
            self.log.info(
                "체크포인트 저장: %s 스텝 (%s)", f"{self.num_timesteps:,}", self.checkpoint_dir
            )
        now = self.clock()
        if now - self.last_report >= self.interval:
            self.last_report = now
            self.log.info(self.summary(now))
        if now - self.last_status >= self.status_interval:
            self.last_status = now
            self.write_status()
        return True

    def _on_training_end(self) -> None:
        self.log.info(self.summary(self.clock()))
        self.write_status()

    def stats(self, now: float) -> dict:
        elapsed = now - self.started if self.started is not None else 0.0
        done = int(self.num_timesteps)
        speed = done / elapsed if elapsed > 0 else 0.0
        remaining = (self.total - done) / speed if speed > 0 and done < self.total else None
        episodes = {
            "count_total": self.episode_count,
            "recent_count": len(self.episodes),
            "recent_mean_reward": None,
            "recent_mean_length": None,
            "recent_safeguard_rate": None,
        }
        if self.episodes:
            count = len(self.episodes)
            episodes.update(
                recent_mean_reward=sum(float(item["r"]) for item in self.episodes) / count,
                recent_mean_length=sum(float(item["l"]) for item in self.episodes) / count,
                recent_safeguard_rate=sum(
                    item.get("reason") == SAFEGUARD_REASON for item in self.episodes
                )
                / count,
            )
        return {
            "timesteps": done,
            "progress": min(done / self.total, 1.0),
            "steps_per_second": speed,
            "elapsed_seconds": elapsed,
            "eta_seconds": remaining,
            "episodes": episodes,
        }

    def summary(self, now: float) -> str:
        values = self.stats(now)
        text = (
            f"진행 {values['progress']:6.1%} | {values['timesteps']:,}/{self.total:,} 스텝 | "
            f"{values['steps_per_second']:,.0f} 스텝/초 | "
            f"경과 {format_duration(values['elapsed_seconds'])} | "
            f"남은 시간 {format_duration(values['eta_seconds'] or 0)}"
        )
        episodes = values["episodes"]
        if episodes["recent_count"]:
            text += (
                f" | 최근 에피소드 {episodes['recent_count']}개: "
                f"평균 보상 {episodes['recent_mean_reward']:.4f}, "
                f"평균 길이 {episodes['recent_mean_length']:.0f}, "
                f"Safe-Guard 종료 {episodes['recent_safeguard_rate']:.0%} "
                f"(누적 {episodes['count_total']:,}개)"
            )
        return text

    def _validation(self) -> dict | None:
        if self.validator is None:
            return None
        history = self.validator.history
        last = None
        if history:
            last = {"timesteps": history[-1]["timesteps"], "score": history[-1]["score"]}
        best = None
        if self.validator.best is not None:
            best = {
                "timesteps": self.validator.best["timesteps"],
                "score": self.validator.best["score"],
            }
        return {"metric": self.validator.metric, "last": last, "best": best}

    def write_status(self, state: str = "running", message: str | None = None) -> None:
        """상태 파일을 쓴다. 경로가 없으면 아무것도 하지 않는다."""
        if self.status_path is None:
            return
        values = self.stats(self.clock())
        if state != "running":
            values["eta_seconds"] = None
        write_json_atomic(
            self.status_path,
            {
                "schema_version": STATUS_SCHEMA_VERSION,
                "run_id": self.run_id,
                "state": state,
                "started_at": self.started_at,
                "updated_at": _utc_now(),
                "total_timesteps": self.total,
                **values,
                "validation": self._validation(),
                "last_checkpoint_timesteps": self.last_checkpoint,
                "message": message,
            },
        )
