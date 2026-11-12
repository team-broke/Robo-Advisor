"""Window마다 PPO를 처음부터 다시 학습하고 평가 구간에서 백테스트한다.

Window 경계는 walk_forward_split으로 정한다. 각 학습은 새 run_id로 새 PPO를 만들며
이전 Window나 다른 실행의 가중치를 읽지 않는다. 학습 구간의 마지막 일부는 검증용으로
떼어 best_model.zip을 고르고, 평가 구간은 학습이 끝난 뒤에만 읽는다.

입력 데이터셋은 정규화 전 값(returns·rsi·macd)이며 환경이 고정 척도로 관측을 만든다.
학습 구간에서 맞추는 scaler가 없으므로 Window 사이에 공유되는 통계도 없다.

작업은 seed를 바깥 반복으로 두어 중간에 멈춰도 모든 데이터셋·Window·보상 조합이
먼저 한 번씩 끝나게 한다. 실패한 작업은 기록하고 다음 작업으로 넘어가며,
--resume으로 다시 실행하면 끝난 작업을 건너뛴다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from robo_advisor.backtest.metrics import calculate_metrics
from robo_advisor.backtest.walk_forward import WalkForwardWindow, walk_forward_split
from robo_advisor.config import get_settings
from robo_advisor.rl.evaluation import EvaluationResult, evaluate_policy
from robo_advisor.rl.progress import LOGGER, format_duration
from robo_advisor.rl.rewards import RewardVariant
from robo_advisor.rl.training import (
    DataConfig,
    TrainingConfig,
    TrainingDataset,
    load_training_config,
    load_training_dataset,
    resolve_device,
    run_training,
)
from robo_advisor.rl.validation import split_validation

SAFE_NAME = r"[A-Za-z0-9][A-Za-z0-9_.-]{0,47}"
ONE_MICROSECOND = timedelta(microseconds=1)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("boundaries must include a timezone")
    return value


class SplitConfig(StrictModel):
    start_date: datetime
    train_years: Annotated[int, Field(ge=1, strict=True)] = 4
    test_years: Annotated[int, Field(ge=1, strict=True)] = 1
    min_windows: Annotated[int, Field(ge=2, strict=True)] = 2
    # 수집 범위의 시작과 배타적 끝. 휴장일로 관측이 달력 경계에 없을 때 명시한다.
    coverage_start: datetime | None = None
    coverage_end: datetime | None = None

    @field_validator("start_date", "coverage_start", "coverage_end")
    @classmethod
    def timezone_required(cls, value):
        return value if value is None else _aware(value)


class VariantConfig(StrictModel):
    name: Annotated[str, Field(pattern=f"^{SAFE_NAME}$")]
    reward_variant: RewardVariant
    penalty_lambda: float = Field(default=1.0, ge=0)


class RetrainConfig(StrictModel):
    base_training_config: Path
    experiment_id: Annotated[str, Field(pattern=f"^{SAFE_NAME}$")]
    datasets: dict[str, Path] = Field(min_length=1)
    split: SplitConfig
    # 각 Window 학습 구간의 마지막 validation_years년을 검증에 쓴다. 0이면 검증하지 않는다.
    validation_years: Annotated[int, Field(ge=0, strict=True)] = 1
    variants: list[VariantConfig] = Field(min_length=1)
    seeds: list[Annotated[int, Field(ge=0, strict=True)]] = Field(min_length=1)
    total_timesteps: Annotated[int, Field(gt=0, strict=True)] | None = None
    # 평가 모델 선택: 검증 최적 모델이 있으면 best, 없으면 final을 쓴다.
    evaluate_model: Literal["best", "final"] = "best"

    @field_validator("datasets")
    @classmethod
    def safe_dataset_names(cls, value):
        for name in value:
            if not re.fullmatch(SAFE_NAME, name):
                raise ValueError(f"dataset name is not filesystem safe: {name}")
        return value

    @model_validator(mode="after")
    def unique_entries(self):
        names = [variant.name for variant in self.variants]
        if len(set(names)) != len(names):
            raise ValueError("variant names must be unique")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if self.validation_years >= self.split.train_years:
            raise ValueError("validation_years must be shorter than train_years")
        return self


def load_retrain_config(path: str | Path) -> RetrainConfig:
    """상대 경로는 설정 파일 위치를 기준으로 해석한다."""
    path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("retrain config must be a mapping")
    config = RetrainConfig.model_validate(payload)

    def resolve(value: Path) -> Path:
        value = value.expanduser()
        return value if value.is_absolute() else (path.parent / value).resolve()

    return config.model_copy(
        update={
            "base_training_config": resolve(config.base_training_config),
            "datasets": {name: resolve(value) for name, value in config.datasets.items()},
        }
    )


@dataclass(frozen=True)
class Job:
    run_id: str
    dataset: str
    window: int
    variant: VariantConfig
    seed: int
    train_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    validation_start: pd.Timestamp | None


def dataset_windows(dataset_dir: Path, split: SplitConfig) -> list[WalkForwardWindow]:
    """데이터셋 평가시각으로 Window를 나눈다. 경계는 데이터 시간대로 맞춘다."""
    index = pd.read_parquet(dataset_dir / "returns.parquet").index
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise ValueError("dataset index must be timezone-aware")

    def boundary(value):
        return None if value is None else pd.Timestamp(value).tz_convert(index.tz)

    return walk_forward_split(
        index,
        start_date=boundary(split.start_date),
        train_years=split.train_years,
        test_years=split.test_years,
        min_windows=split.min_windows,
        coverage_start=boundary(split.coverage_start),
        coverage_end=boundary(split.coverage_end),
    )


def build_jobs(config: RetrainConfig) -> list[Job]:
    """seed → 데이터셋 → Window → 보상 순서로 작업을 만든다."""
    windows = {name: dataset_windows(path, config.split) for name, path in config.datasets.items()}
    jobs = []
    for seed in config.seeds:
        for name in config.datasets:
            for number, window in enumerate(windows[name], start=1):
                validation_start = None
                if config.validation_years:
                    validation_start = window.test_start - pd.DateOffset(
                        years=config.validation_years
                    )
                for variant in config.variants:
                    jobs.append(
                        Job(
                            run_id=(
                                f"{config.experiment_id}-{name}-w{number}-{variant.name}-s{seed}"
                            ),
                            dataset=name,
                            window=number,
                            variant=variant,
                            seed=seed,
                            train_start=window.train_start,
                            test_start=window.test_start,
                            test_end=window.test_end,
                            validation_start=validation_start,
                        )
                    )
    for job in jobs:
        if len(job.run_id) > 128:
            raise ValueError(f"run_id is too long: {job.run_id}")
    return jobs


def job_training_config(
    base: TrainingConfig, config: RetrainConfig, job: Job, *, smoke: bool
) -> TrainingConfig:
    """평가 구간 시작 직전까지만 학습하도록 컷오프를 바꾼 설정."""
    payload = base.model_dump(mode="json")
    payload["seed"] = job.seed
    if config.total_timesteps is not None:
        payload["total_timesteps"] = config.total_timesteps
    payload["env"]["reward_variant"] = job.variant.reward_variant.value
    payload["env"]["penalty_lambda"] = job.variant.penalty_lambda
    payload["data"]["dataset_dir"] = str(config.datasets[job.dataset])
    payload["data"]["train_start"] = job.train_start.isoformat()
    payload["data"]["train_end"] = (job.test_start - ONE_MICROSECOND).isoformat()
    if job.validation_start is None:
        payload["validation"] = None
    else:
        previous = base.validation.model_dump(mode="json") if base.validation else {}
        payload["validation"] = {**previous, "start": job.validation_start.isoformat()}
    return TrainingConfig.model_validate(payload, context={"smoke": smoke})


def load_test_dataset(training: TrainingConfig, job: Job) -> TrainingDataset:
    """평가 구간 행과 관측 창을 채우는 직전 lookback-1행만 담는다."""
    data = DataConfig(
        dataset_dir=training.data.dataset_dir,
        train_start=job.train_start.to_pydatetime(),
        train_end=(job.test_end - ONE_MICROSECOND).to_pydatetime(),
    )
    full = load_training_dataset(data)
    _, test = split_validation(full, job.test_start, training.env.lookback)
    return test


def equal_weight_policy(n_assets: int):
    """현금 없이 자산에 같은 비중을 두는 기준 정책."""
    logits = np.zeros(n_assets + 1)
    logits[-1] = -20.0

    class Policy:
        def predict(self, observation, deterministic=True):
            return logits.copy(), None

    return Policy()


def _metrics(result: EvaluationResult, benchmark: pd.Series | None) -> dict:
    if len(result.returns) < 2:
        return {}
    reference = (
        pd.Series(0.0, index=result.returns.index)
        if benchmark is None
        else benchmark.reindex(result.returns.index)
    )
    if reference.isna().any():
        reference = pd.Series(0.0, index=result.returns.index)
    return {
        key: float(value) if np.isfinite(value) else None
        for key, value in calculate_metrics(result.returns, reference).items()
    }


def _summary(result: EvaluationResult, benchmark: pd.Series | None) -> dict:
    return {
        "first_index": result.returns.index[0].isoformat() if len(result.returns) else None,
        "last_index": result.returns.index[-1].isoformat() if len(result.returns) else None,
        "observation_count": result.observation_count,
        "terminated": result.terminated,
        "reason": result.reason,
        "final_nav": float(result.nav.iloc[-1]),
        "metrics": _metrics(result, benchmark),
    }


def _write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(value) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class RunLock:
    """같은 실험을 두 프로세스가 동시에 돌리지 못하게 한다. 죽은 프로세스 잠금은 회수한다."""

    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        while True:
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    owner = int(self.path.read_text(encoding="utf-8").strip() or 0)
                except (OSError, ValueError):
                    owner = 0
                if owner and _alive(owner):
                    raise RuntimeError(f"experiment is already running (pid {owner})")
                self.path.unlink(missing_ok=True)
                continue
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(str(os.getpid()))
            return self

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _completed(job_dir: Path, run_id: str, expected_config: dict) -> bool:
    result_path = job_dir / "result.json"
    if not result_path.is_file():
        return False
    result = json.loads(result_path.read_text(encoding="utf-8"))
    return result.get("run_id") == run_id and result.get("training_config") == expected_config


def _archive_stale(paths: list[Path]) -> list[str]:
    """끝나지 않은 이전 시도를 지우지 않고 옆으로 옮긴다."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    moved = []
    for path in paths:
        if path.exists():
            target = path.with_name(f"{path.name}.stale-{stamp}")
            path.rename(target)
            moved.append(str(target))
    return moved


def run_job(
    job: Job,
    training: TrainingConfig,
    config: RetrainConfig,
    job_dir: Path,
    *,
    smoke: bool,
) -> dict:
    """새 PPO를 학습하고 평가 구간에서 PPO와 동일가중을 같은 환경으로 굴린다."""
    from stable_baselines3 import PPO

    settings = get_settings()
    model_dir = settings.models_dir.expanduser().resolve() / job.run_id
    artifact_dir = settings.artifacts_dir.expanduser().resolve() / "training" / job.run_id
    stale = _archive_stale([model_dir, artifact_dir])
    if stale:
        LOGGER.warning("끝나지 않은 이전 시도를 옮김: %s", ", ".join(stale))
    # 새 run_id 디렉터리에서만 저장하므로 이전 Window 가중치를 읽을 경로가 없다.
    result = run_training(training, run_id=job.run_id, smoke=smoke)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    best_path = model_dir / "best_model.zip"
    chosen = "final"
    model_path = result.final_model_path
    if config.evaluate_model == "best" and best_path.is_file():
        chosen, model_path = "best", best_path
    device = resolve_device(training.device)
    model = PPO.load(model_path, device=device)
    test = load_test_dataset(training, job)
    ppo_result = evaluate_policy(model, test, training.env)
    baseline = evaluate_policy(equal_weight_policy(len(test.returns.columns)), test, training.env)
    if (ppo_result.returns.index < job.test_start).any() or (
        ppo_result.returns.index >= job.test_end
    ).any():
        raise ValueError("test rollout left the test window")
    job_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "ppo": ppo_result.returns,
            "equal_weight": baseline.returns.reindex(ppo_result.returns.index),
        }
    )
    frame.to_csv(job_dir / "test_returns.csv", index_label="timestamp")
    record = {
        "run_id": job.run_id,
        "dataset": job.dataset,
        "window": job.window,
        "variant": job.variant.model_dump(mode="json"),
        "seed": job.seed,
        "train_start": job.train_start.isoformat(),
        "train_end": training.data.train_end.isoformat(),
        "validation_start": (
            None if job.validation_start is None else job.validation_start.isoformat()
        ),
        "test_start": job.test_start.isoformat(),
        "test_end": job.test_end.isoformat(),
        "training_config": training.model_dump(mode="json"),
        "training_manifest": str(result.manifest_path),
        "initial_weights": "random",
        "evaluated_model": chosen,
        "evaluated_model_path": str(model_path),
        "actual_timesteps": manifest.get("actual_timesteps"),
        "validation_best": (manifest.get("validation") or {}).get("best"),
        "ppo": _summary(ppo_result, baseline.returns),
        "equal_weight": _summary(baseline, None),
        "finished_at": _utc_now(),
    }
    _write_json(job_dir / "result.json", record)
    return record


SUMMARY_COLUMNS = (
    "cumulative_return",
    "cagr",
    "volatility",
    "sharpe",
    "sortino",
    "mdd",
    "calmar",
)


def write_summary(root: Path, jobs: list[Job]) -> Path:
    rows = []
    for job in jobs:
        path = root / "jobs" / job.run_id / "result.json"
        if not path.is_file():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        row = {
            "run_id": job.run_id,
            "dataset": job.dataset,
            "window": job.window,
            "test_start": record["test_start"][:10],
            "variant": job.variant.name,
            "seed": job.seed,
            "model": record["evaluated_model"],
            "days": record["ppo"]["observation_count"],
            "reason": record["ppo"]["reason"],
        }
        for side in ("ppo", "equal_weight"):
            metrics = record[side]["metrics"]
            for key in SUMMARY_COLUMNS:
                row[f"{side}_{key}"] = metrics.get(key)
        rows.append(row)
    path = root / "summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def run_experiment(
    config: RetrainConfig,
    *,
    smoke: bool = False,
    resume: bool = False,
) -> Path:
    """모든 작업을 순서대로 돌리고 artifacts/backtest/<experiment_id>/에 기록한다."""
    base = load_training_config(config.base_training_config, smoke=smoke)
    jobs = build_jobs(config)
    root = get_settings().artifacts_dir.expanduser().resolve() / "backtest" / config.experiment_id
    if root.exists() and not resume:
        raise FileExistsError(f"experiment already exists, use --resume: {root}")
    (root / "jobs").mkdir(parents=True, exist_ok=True)
    manifest_path = root / "run_manifest.json"
    trainings = {job.run_id: job_training_config(base, config, job, smoke=smoke) for job in jobs}
    manifest = {
        "experiment_id": config.experiment_id,
        "config": config.model_dump(mode="json"),
        "config_sha256": _sha(config.model_dump(mode="json")),
        "base_training_config_sha256": _sha(base.model_dump(mode="json")),
        "smoke": smoke,
        "started_at": _utc_now(),
        "updated_at": _utc_now(),
        "finished_at": None,
        "status": "running",
        "pid": os.getpid(),
        "weight_reuse": "none: every job trains a new PPO under its own run_id",
        "scaler": "none: inputs are unnormalized and the environment uses fixed scales",
        "windows": {},
        "jobs": [],
    }
    for name in config.datasets:
        manifest["windows"][name] = [
            {
                "window": job.window,
                "train_start": job.train_start.isoformat(),
                "validation_start": (
                    None if job.validation_start is None else job.validation_start.isoformat()
                ),
                "test_start": job.test_start.isoformat(),
                "test_end": job.test_end.isoformat(),
            }
            for job in jobs
            if job.dataset == name
            and job.seed == config.seeds[0]
            and job.variant == config.variants[0]
        ]
    for job in jobs:
        manifest["jobs"].append(
            {
                "run_id": job.run_id,
                "dataset": job.dataset,
                "window": job.window,
                "variant": job.variant.name,
                "seed": job.seed,
                "status": "pending",
                "started_at": None,
                "finished_at": None,
                "duration_seconds": None,
                "error": None,
            }
        )
    entries = {entry["run_id"]: entry for entry in manifest["jobs"]}

    def save():
        manifest["updated_at"] = _utc_now()
        _write_json(manifest_path, manifest)

    log_file = logging.FileHandler(root / "retrain.log", encoding="utf-8")
    log_file.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(log_file)
    if LOGGER.level == logging.NOTSET:
        LOGGER.setLevel(logging.INFO)
    started = time.monotonic()
    try:
        with RunLock(root / "run.lock"):
            for job in jobs:
                job_dir = root / "jobs" / job.run_id
                expected = trainings[job.run_id].model_dump(mode="json")
                if resume and _completed(job_dir, job.run_id, expected):
                    entries[job.run_id]["status"] = "skipped_completed"
            save()
            pending = [job for job in jobs if entries[job.run_id]["status"] == "pending"]
            LOGGER.info(
                "실험 %s: 작업 %d개 중 %d개 실행, 작업당 %s 스텝",
                config.experiment_id,
                len(jobs),
                len(pending),
                f"{trainings[jobs[0].run_id].total_timesteps:,}",
            )
            durations = []
            for position, job in enumerate(pending, start=1):
                entry = entries[job.run_id]
                entry["status"] = "running"
                entry["started_at"] = _utc_now()
                entry["error"] = None
                save()
                remaining = ""
                if durations:
                    estimate = sum(durations) / len(durations) * (len(pending) - position + 1)
                    remaining = f", 남은 예상 {format_duration(estimate)}"
                LOGGER.info(
                    "[%d/%d] %s 시작 (학습 %s ~ %s, 평가 %s ~ %s%s)",
                    position,
                    len(pending),
                    job.run_id,
                    job.train_start.date(),
                    (job.test_start - ONE_MICROSECOND).date(),
                    job.test_start.date(),
                    (job.test_end - ONE_MICROSECOND).date(),
                    remaining,
                )
                job_started = time.monotonic()
                try:
                    record = run_job(
                        job,
                        trainings[job.run_id],
                        config,
                        root / "jobs" / job.run_id,
                        smoke=smoke,
                    )
                    entry["status"] = "completed"
                    ppo = record["ppo"]["metrics"]
                    baseline = record["equal_weight"]["metrics"]
                    LOGGER.info(
                        "[%d/%d] %s 완료: PPO 누적 %s / Sharpe %s, 동일가중 누적 %s / Sharpe %s",
                        position,
                        len(pending),
                        job.run_id,
                        _format(ppo.get("cumulative_return")),
                        _format(ppo.get("sharpe")),
                        _format(baseline.get("cumulative_return")),
                        _format(baseline.get("sharpe")),
                    )
                except KeyboardInterrupt:
                    entry["status"] = "interrupted"
                    raise
                except Exception as error:
                    entry["status"] = "failed"
                    entry["error"] = f"{type(error).__name__}: {error}"
                    LOGGER.error("%s 실패: %s", job.run_id, entry["error"])
                    LOGGER.debug(traceback.format_exc())
                finally:
                    elapsed = time.monotonic() - job_started
                    durations.append(elapsed)
                    entry["finished_at"] = _utc_now()
                    entry["duration_seconds"] = round(elapsed, 1)
                    write_summary(root, jobs)
                    save()
            failed = [entry for entry in manifest["jobs"] if entry["status"] == "failed"]
            manifest["status"] = "completed_with_failures" if failed else "completed"
    except BaseException as error:
        if manifest["status"] == "running":
            manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["finished_at"] = _utc_now()
        if manifest_path.parent.exists():
            save()
        LOGGER.info(
            "실험 %s 종료(%s): 소요 %s, %s",
            config.experiment_id,
            manifest["status"],
            format_duration(time.monotonic() - started),
            manifest_path,
        )
        LOGGER.removeHandler(log_file)
        log_file.close()
    return manifest_path


def _format(value) -> str:
    return "-" if value is None else f"{value:.4f}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Window별 PPO 완전 재학습과 평가 구간 백테스트")
    parser.add_argument("--config", default="config/experiments/walk_forward.yaml")
    parser.add_argument("--resume", action="store_true", help="끝난 작업을 건너뛰고 이어서 실행")
    parser.add_argument("--smoke", action="store_true", help="최소 학습 스텝 제한을 우회")
    parser.add_argument("--total-timesteps", type=int, help="작업당 학습 스텝 덮어쓰기")
    parser.add_argument("--dry-run", action="store_true", help="작업 목록과 경계만 출력")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    config = load_retrain_config(args.config)
    if args.total_timesteps is not None:
        config = config.model_copy(update={"total_timesteps": args.total_timesteps})
    if args.dry_run:
        jobs = build_jobs(config)
        base = load_training_config(config.base_training_config, smoke=args.smoke)
        for job in jobs:
            training = job_training_config(base, config, job, smoke=args.smoke)
            print(
                job.run_id,
                f"train={job.train_start.date()}..{training.data.train_end.date()}",
                f"valid={None if job.validation_start is None else job.validation_start.date()}",
                f"test={job.test_start.date()}..{(job.test_end - ONE_MICROSECOND).date()}",
                f"steps={training.total_timesteps:,}",
            )
        print(f"{len(jobs)} jobs")
        return 0
    path = run_experiment(config, smoke=args.smoke, resume=args.resume)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
