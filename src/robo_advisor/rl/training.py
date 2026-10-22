"""재현 가능한 PPO 학습과 실행별 설정·데이터 이력 기록.

데이터셋은 returns.parquet, rsi.parquet, macd.parquet를 요구한다. 세 파일은
정렬되고 중복 없는 tz-aware DatetimeIndex 및 같은 자산 열 순서를 가져야 한다.
수익률은 일별 단순수익률, RSI는 0~100, MACD는 가격 정규화된 단일 선이다.
선택적 risk_tags.parquet는 RiskTag 필드 열을 사용한다. 가용 시각이 컷오프
이하인 태그만 검증·사용한다. 지표의 인과적 계산·시장 간 시간 정렬·학습 구간
전용 스케일링은 입력 준비 단계의 계약이며 여기서 정규화를 학습하지 않는다.
모든 상대 경로는 실행 시 현재 디렉터리를 기준으로 해석한다.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import pandas as pd
import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from robo_advisor.config import get_settings
from robo_advisor.rl.curves import MONITOR_DIR, write_run_curves
from robo_advisor.rl.environment import PortfolioEnv
from robo_advisor.rl.progress import LOGGER, ProgressCallback, format_duration
from robo_advisor.rl.rewards import RewardVariant
from robo_advisor.schemas.risk import RiskTag

PositiveInteger = Annotated[int, Field(gt=0, strict=True)]


class StrictConfig(BaseModel):
    """오타와 비유한 설정을 거부한다."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        frozen=True,
        validate_default=True,
    )


class PolicyArchitecture(StrictConfig):
    pi: list[PositiveInteger] = Field(default_factory=lambda: [256, 256], min_length=1)
    vf: list[PositiveInteger] = Field(default_factory=lambda: [256, 256], min_length=1)


class PPOConfig(StrictConfig):
    learning_rate: float = Field(default=0.0003, gt=0)
    n_steps: PositiveInteger = 512
    batch_size: int = Field(default=1024, ge=2, strict=True)
    n_epochs: PositiveInteger = 10
    gamma: float = Field(default=0.99, ge=0, le=1)
    gae_lambda: float = Field(default=0.95, ge=0, le=1)
    clip_range: float = Field(default=0.2, gt=0, le=1)
    ent_coef: float = Field(default=0.0, ge=0)
    vf_coef: float = Field(default=0.5, ge=0)
    max_grad_norm: float = Field(default=0.5, gt=0)
    net_arch: PolicyArchitecture = Field(default_factory=PolicyArchitecture)


class EnvironmentConfig(StrictConfig):
    lookback: PositiveInteger = 20
    reward_variant: RewardVariant = RewardVariant.RETURN
    penalty_lambda: float = Field(default=1.0, ge=0)
    mdd_limit: float = Field(
        default_factory=lambda: get_settings().safeguard_mdd_limit,
        gt=0,
        le=1,
    )
    commission_rate: float = Field(
        default_factory=lambda: get_settings().commission_rate,
        ge=0,
        lt=1,
    )
    slippage_rate: float = Field(default_factory=lambda: get_settings().slippage_rate, ge=0, lt=1)


class DataConfig(StrictConfig):
    dataset_dir: Path = Field(default_factory=lambda: get_settings().data_processed_dir)
    train_start: datetime | None = None
    train_end: datetime

    @field_validator("train_start", "train_end")
    @classmethod
    def aware_timestamp(cls, value):
        if value is not None:
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("training dates must include a timezone")
            return value.astimezone(timezone.utc)
        return value

    @model_validator(mode="after")
    def ordered_dates(self):
        if self.train_start is not None and self.train_start > self.train_end:
            raise ValueError("train_start must be <= train_end")
        return self


class CheckpointConfig(StrictConfig):
    save_freq: PositiveInteger = 100000


class ValidationConfig(StrictConfig):
    """학습 컷오프 안의 [start, train_end] 구간을 검증용으로 떼어 낸다."""

    start: datetime
    eval_freq: PositiveInteger = 50000
    metric: Literal[
        "cumulative_return",
        "cagr",
        "volatility",
        "var_95",
        "cvar_95",
        "mdd",
        "sharpe",
        "sortino",
        "calmar",
    ] = "sharpe"
    higher_is_better: bool = Field(default=True, strict=True)

    @field_validator("start")
    @classmethod
    def aware_timestamp(cls, value):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("validation start must include a timezone")
        return value.astimezone(timezone.utc)


class TrainingConfig(StrictConfig):
    seed: int = Field(default=42, ge=0, le=2**32 - 1, strict=True)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    total_timesteps: PositiveInteger = 1000000
    n_envs: PositiveInteger = 8
    vec_env: Literal["dummy", "subproc"] = "subproc"
    ppo: PPOConfig = Field(default_factory=PPOConfig)
    env: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    data: DataConfig
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    validation: ValidationConfig | None = None

    @model_validator(mode="after")
    def training_contract(self, info: ValidationInfo):
        if self.total_timesteps < 100000 and not (info.context or {}).get("smoke", False):
            raise ValueError("total_timesteps must be >= 100000 unless smoke is explicitly enabled")
        if self.ppo.batch_size > self.ppo.n_steps * self.n_envs:
            raise ValueError("batch_size must be <= n_steps * n_envs")
        if self.seed + self.n_envs - 1 > 2**32 - 1:
            raise ValueError("seed + n_envs - 1 exceeds the seed range")
        if self.validation is not None:
            start = self.validation.start
            if start > self.data.train_end or (
                self.data.train_start is not None and start <= self.data.train_start
            ):
                raise ValueError("validation start must fall inside (train_start, train_end]")
        return self


def load_training_config(
    path: str | Path,
    *,
    dataset_dir: str | Path | None = None,
    device: str | None = None,
    total_timesteps: int | None = None,
    smoke: bool = False,
) -> TrainingConfig:
    """CLI의 명시적 덮어쓰기를 적용한 뒤 전체 설정을 검증한다."""
    with Path(path).open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError("training config must be a mapping")
    if dataset_dir is not None:
        data = payload.get("data", {})
        if not isinstance(data, dict):
            raise ValueError("data config must be a mapping")
        payload["data"] = {**data, "dataset_dir": str(dataset_dir)}
    for key, value in (("device", device), ("total_timesteps", total_timesteps)):
        if value is not None:
            payload[key] = value
    return TrainingConfig.model_validate(payload, context={"smoke": smoke})


@dataclass(frozen=True)
class TrainingDataset:
    returns: pd.DataFrame
    rsi: pd.DataFrame
    macd: pd.DataFrame
    risk_tags: tuple[RiskTag, ...]
    file_sha256: dict[str, str]


def load_training_dataset(config: DataConfig) -> TrainingDataset:
    """해시한 바이트에서 읽고 컷오프 밖 데이터를 환경 생성 전에 제외한다."""
    directory = config.dataset_dir.expanduser().resolve()
    hashes = {}

    def read(name):
        content = (directory / name).read_bytes()
        hashes[name] = hashlib.sha256(content).hexdigest()
        return pd.read_parquet(io.BytesIO(content))

    frames = [read(name) for name in ("returns.parquet", "rsi.parquet", "macd.parquet")]
    reference = frames[0]
    for frame in frames:
        index = frame.index
        if (
            not isinstance(index, pd.DatetimeIndex)
            or index.tz is None
            or index.hasnans
            or not index.is_unique
            or not index.is_monotonic_increasing
        ):
            raise ValueError("dataset indices must be ordered, unique and timezone-aware")
        if not index.equals(reference.index) or not frame.columns.equals(reference.columns):
            raise ValueError("dataset frames must have identical indices and asset column order")
    mask = reference.index <= config.train_end
    if config.train_start is not None:
        mask &= reference.index >= config.train_start
    frames = [frame.loc[mask].copy() for frame in frames]
    if frames[0].empty:
        raise ValueError("training cutoff leaves no data")
    tags = []
    if (directory / "risk_tags.parquet").exists():
        raw_tags = read("risk_tags.parquet")
        if not raw_tags.columns.is_unique or set(raw_tags.columns) != set(RiskTag.model_fields):
            raise ValueError("risk_tags columns must match RiskTag fields exactly")
        for record in raw_tags.to_dict(orient="records"):
            available = pd.Timestamp(record["available_at"])
            if pd.isna(available) or available.tzinfo is None:
                raise ValueError("risk tag availability must be timezone-aware")
            if available <= config.train_end:
                tags.append(RiskTag.model_validate(record))
    return TrainingDataset(*frames, tuple(tags), hashes)


def resolve_device(requested: str) -> str:
    """CUDA 불가 시 명시 요청은 실패하고 auto만 CPU로 선택한다."""
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda")
    if requested == "cpu":
        return "cpu"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    available = torch.cuda.is_available()
    if requested == "cuda" and not available:
        raise ValueError("CUDA was requested but is not available")
    return "cuda" if available else "cpu"


def seed_everything(seed: int, device: str) -> None:
    """CPU에서는 고정 스레드 수로, CUDA에서는 결정론 설정으로 실행한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if device == "cuda":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _make_env(
    dataset: TrainingDataset,
    config: EnvironmentConfig,
    seed: int,
    monitor_file: Path | None = None,
):
    env = PortfolioEnv(
        dataset.returns,
        dataset.rsi,
        dataset.macd,
        n_assets=len(dataset.returns.columns),
        risk_tags=dataset.risk_tags,
        **config.model_dump(),
    )
    env.reset(seed=seed)
    # 종료 사유를 함께 남겨 학습곡선에서 Safe-Guard 빈도를 계산한다.
    filename = None if monitor_file is None else str(monitor_file)
    return Monitor(env, filename=filename, info_keywords=("reason",))


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class TrainingResult:
    model: PPO
    final_model_path: Path
    manifest_path: Path
    checkpoint_paths: tuple[Path, ...]


def run_training(
    config: TrainingConfig,
    *,
    run_id: str | None = None,
    smoke: bool = False,
    log_interval: float = 30.0,
) -> TrainingResult:
    """학습·체크포인트 저장 후 manifest를 완성하며 기존 실행을 덮어쓰지 않는다.

    진행 상황은 robo_advisor.training 로거로 남기고 실행 디렉터리의 train.log에도 쓴다.
    log_interval초마다 진행률과 최근 에피소드 요약을 기록한다.

    PPO는 rollout 단위로 학습하므로 실제 스텝은 요청값을 초과할 수 있다.
    save_freq는 n_envs로 나눈 값을 올림해 벡터 호출 간격으로 사용한다.
    warn_only 결정론 설정은 장치·라이브러리 버전 간 완전 동일성을 보장하지 않는다.
    """
    config = TrainingConfig.model_validate(config.model_dump(), context={"smoke": smoke})
    requested_device = config.device
    device = resolve_device(requested_device)
    resolved = config.model_dump(mode="json")
    resolved["device"] = device
    resolved["data"]["dataset_dir"] = str(config.data.dataset_dir.expanduser().resolve())
    config = TrainingConfig.model_validate(resolved, context={"smoke": smoke})
    run_id = (
        run_id if run_id is not None else datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
        raise ValueError("run_id must be a safe directory name of at most 128 characters")
    settings = get_settings()
    model_dir = settings.models_dir.expanduser().resolve() / run_id
    artifact_dir = settings.artifacts_dir.expanduser().resolve() / "training" / run_id
    if model_dir.exists() or artifact_dir.exists():
        raise FileExistsError(f"run_id already exists: {run_id}")
    dataset = load_training_dataset(config.data)
    validation_data = None
    if config.validation is not None:
        from robo_advisor.rl.validation import split_validation

        # 검증 구간 행은 학습 환경에 넣지 않는다.
        dataset, validation_data = split_validation(
            dataset, config.validation.start, config.env.lookback
        )
    # 환경 입력을 출력 디렉터리 생성과 프로세스 시작 전에 검증한다.
    for probe_data in (dataset, validation_data):
        if probe_data is not None:
            _make_env(probe_data, config.env, config.seed).close()
    model_dir.mkdir(parents=True, exist_ok=False)
    artifact_dir.mkdir(parents=True, exist_ok=False)
    monitors = artifact_dir / MONITOR_DIR
    monitors.mkdir()
    checkpoints = model_dir / "checkpoints"
    checkpoints.mkdir()
    manifest_path = artifact_dir / "manifest.json"
    final_path = model_dir / "final_model.zip"
    canonical = json.dumps(resolved, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    save_calls = max(1, math.ceil(config.checkpoint.save_freq / config.n_envs))
    manifest = {
        "run_id": run_id,
        "started_at": _utc_now(),
        "finished_at": None,
        "status": "running",
        "seed": config.seed,
        "env_seeds": [config.seed + rank for rank in range(config.n_envs)],
        "config": resolved,
        "config_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "data_cutoff": {
            "train_start": resolved["data"]["train_start"],
            "train_end": resolved["data"]["train_end"],
            "actual_first_index": dataset.returns.index[0].isoformat(),
            "actual_last_index": dataset.returns.index[-1].isoformat(),
            "rows": len(dataset.returns),
            "assets": list(dataset.returns.columns),
        },
        "data_file_sha256": dataset.file_sha256,
        "package_versions": {
            "python": sys.version.split()[0],
            **{
                name: version(name)
                for name in ("stable-baselines3", "torch", "gymnasium", "numpy", "pandas")
            },
        },
        "torch_cuda_version": torch.version.cuda,
        "requested_device": requested_device,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "torch_num_threads": 1,
        "n_envs": config.n_envs,
        "actual_timesteps": 0,
        "git_commit": _git_commit(),
        "smoke": smoke,
        "checkpoint_interval_timesteps": save_calls * config.n_envs,
        "checkpoint_paths": [],
        "final_model_path": str(final_path),
        "learning_curves": None,
        "validation": None,
    }
    validator = None
    if validation_data is not None:
        from robo_advisor.rl.validation import ValidationCallback

        validator = ValidationCallback(
            validation_data,
            config.env,
            eval_freq=config.validation.eval_freq,
            n_envs=config.n_envs,
            metric=config.validation.metric,
            higher_is_better=config.validation.higher_is_better,
            best_path=model_dir / "best_model.zip",
        )
        manifest["validation"] = {
            "start": resolved["validation"]["start"],
            "end": resolved["data"]["train_end"],
            "metric": config.validation.metric,
            "higher_is_better": config.validation.higher_is_better,
            "eval_interval_timesteps": validator.eval_calls * config.n_envs,
            "first_index": validation_data.returns.index[config.env.lookback - 1].isoformat(),
            "last_index": validation_data.returns.index[-1].isoformat(),
            "rows": len(validation_data.returns) - (config.env.lookback - 1),
            "train_last_index": dataset.returns.index[-1].isoformat(),
            "train_rows": len(dataset.returns),
            "history": [],
            "best": None,
        }

    def write_manifest():
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(manifest_path)

    log_file = logging.FileHandler(artifact_dir / "train.log", encoding="utf-8")
    log_file.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(log_file)
    if LOGGER.level == logging.NOTSET:
        LOGGER.setLevel(logging.INFO)
    started = time.monotonic()
    LOGGER.info(
        "실행 %s: 장치 %s%s, 병렬 환경 %d개(%s), 목표 %s 스텝, seed %d%s",
        run_id,
        device,
        f" ({manifest['gpu_name']})" if manifest["gpu_name"] else "",
        config.n_envs,
        config.vec_env,
        f"{config.total_timesteps:,}",
        config.seed,
        " [smoke]" if smoke else "",
    )
    LOGGER.info(
        "학습 데이터: %d행 %s ~ %s, 자산 %d개, 보상 %s",
        len(dataset.returns),
        dataset.returns.index[0].date(),
        dataset.returns.index[-1].date(),
        len(dataset.returns.columns),
        config.env.reward_variant.value,
    )
    if validator is not None:
        LOGGER.info(
            "검증 구간: %d행 %s ~ %s, %s 스텝마다 %s 평가",
            manifest["validation"]["rows"],
            manifest["validation"]["first_index"][:10],
            manifest["validation"]["last_index"][:10],
            f"{manifest['validation']['eval_interval_timesteps']:,}",
            config.validation.metric,
        )
    seed_everything(config.seed, device)
    write_manifest()
    vec = None
    model = None
    try:
        LOGGER.info("환경 %d개와 PPO 모델 구성 중", config.n_envs)
        factories = [
            partial(
                _make_env,
                dataset,
                config.env,
                config.seed + rank,
                monitors / f"env-{rank:03d}",
            )
            for rank in range(config.n_envs)
        ]
        vec = (
            DummyVecEnv(factories)
            if config.vec_env == "dummy"
            else SubprocVecEnv(factories, start_method="spawn")
        )
        vec.seed(config.seed)
        ppo_options = config.ppo.model_dump(exclude={"net_arch"})
        model = PPO(
            "MlpPolicy",
            vec,
            seed=config.seed,
            device=device,
            policy_kwargs={"net_arch": config.ppo.net_arch.model_dump()},
            **ppo_options,
        )
        callbacks = [
            CheckpointCallback(save_freq=save_calls, save_path=str(checkpoints)),
            ProgressCallback(
                config.total_timesteps,
                interval_seconds=log_interval,
                checkpoint_calls=save_calls,
                checkpoint_dir=checkpoints,
            ),
        ]
        if validator is not None:
            callbacks.append(validator)
        model.learn(total_timesteps=config.total_timesteps, callback=callbacks)
        model.save(str(final_path))
        LOGGER.info("최종 모델 저장: %s", final_path)
        if validator is not None:
            # 마지막 모델도 같은 기준으로 평가해 최적 모델 후보에 넣는다.
            validator.evaluate()
            if validator.best is None:
                LOGGER.warning("검증 점수를 계산할 수 없어 최적 모델을 저장하지 않았습니다")
            else:
                LOGGER.info(
                    "검증 최적 모델: %s 스텝, %s=%.4f (%s)",
                    f"{validator.best['timesteps']:,}",
                    config.validation.metric,
                    validator.best["score"],
                    validator.best["path"],
                )
        # Monitor 파일을 닫아 마지막 에피소드까지 기록한 뒤 곡선을 만든다.
        vec.close()
        vec = None
        manifest["status"] = "completed"
        # 곡선 생성 실패가 저장된 모델을 실패 처리하지 않도록 따로 기록한다.
        # python -m robo_advisor.rl.curves --refresh 로 다시 만들 수 있다.
        try:
            curves = write_run_curves(artifact_dir.parent, run_id)
            manifest["learning_curves"] = {
                key: curves[key] for key in ("status", "episodes", "curves_csv", "plot")
            }
            LOGGER.info("학습곡선: 수렴 판정 %s, %s", curves["status"], curves["plot"])
        except Exception as error:
            manifest["learning_curves"] = {"error": f"{type(error).__name__}: {error}"}
            LOGGER.warning("학습곡선 생성 실패: %s", manifest["learning_curves"]["error"])
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        LOGGER.error("학습 실패: %s", manifest["error"])
        raise
    finally:
        manifest["finished_at"] = _utc_now()
        manifest["actual_timesteps"] = int(model.num_timesteps) if model is not None else 0
        manifest["checkpoint_paths"] = [str(path) for path in sorted(checkpoints.glob("*.zip"))]
        if validator is not None:
            manifest["validation"]["history"] = validator.history
            manifest["validation"]["best"] = validator.best
        try:
            write_manifest()
            LOGGER.info(
                "종료(%s): 소요 %s, manifest %s",
                manifest["status"],
                format_duration(time.monotonic() - started),
                manifest_path,
            )
        finally:
            if vec is not None:
                vec.close()
            LOGGER.removeHandler(log_file)
            log_file.close()
    return TrainingResult(
        model, final_path, manifest_path, tuple(Path(path) for path in manifest["checkpoint_paths"])
    )
