"""학습 구간에서만 선별하는 두 단계 보상·lambda 실험.

설정의 base_training_config 및 benchmark_path는 실험 YAML의 위치 기준이다.
실험 상태는 artifacts/experiments/<sweep_id>에 저장한다. Sharpe 등 벤치마크
비의존 지표는 지표 함수의 형식 인자에 현금 수익률 0을 전달한다. Alpha/Beta/IR은
명시적인 단일 열 benchmark parquet를 요구하며 합성 산식을 추정하지 않는다.
조기 종료 표본을 연장하지 않으며 유효한 선별 점수가 두 개 미만이면 실패한다.
기본 설정 해시는 요청 장치를 포함한 해석된 설정의 정렬 JSON을 기준으로 한다.
윈도우별 호출은 서로 다른 sweep_id를 사용하고 실행 중 데이터셋은 고정해야 한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from stable_baselines3 import PPO

from robo_advisor.backtest.metrics import calculate_metrics
from robo_advisor.config import get_settings
from robo_advisor.rl.evaluation import evaluate_policy
from robo_advisor.rl.training import (
    TrainingConfig,
    TrainingDataset,
    load_training_config,
    load_training_dataset,
    resolve_device,
    run_training,
)

LAMBDA_GRID = [0.5, 1.0, 2.0, 3.0, 5.0]
METRICS = Literal[
    "cumulative_return",
    "cagr",
    "volatility",
    "var_95",
    "cvar_95",
    "mdd",
    "sharpe",
    "sortino",
    "calmar",
    "alpha",
    "beta",
    "information_ratio",
]
Seed = Annotated[int, Field(strict=True, ge=0, le=2**32 - 1)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_default=True)


class StageOne(StrictModel):
    variant: Literal["mdd_penalty"] = "mdd_penalty"
    lambdas: list[float] = Field(default_factory=lambda: LAMBDA_GRID.copy())
    seeds: list[Seed] = Field(min_length=1, max_length=1)

    @field_validator("lambdas")
    @classmethod
    def fixed_grid(cls, value):
        if value != LAMBDA_GRID:
            raise ValueError("stage1 lambdas must be [0.5, 1.0, 2.0, 3.0, 5.0]")
        return value


class StageTwo(StrictModel):
    seeds: list[Seed] = Field(min_length=5, max_length=5)
    n_candidates: Literal[2] = 2

    @field_validator("seeds")
    @classmethod
    def unique_seeds(cls, value):
        if len(set(value)) != len(value):
            raise ValueError("stage2 requires five distinct seeds")
        return value


class SelectionRule(StrictModel):
    metric: METRICS = "sharpe"
    higher_is_better: bool = Field(default=True, strict=True)
    benchmark_path: Path | None = None

    @model_validator(mode="after")
    def benchmark_required(self):
        if self.metric in {"alpha", "beta", "information_ratio"} and self.benchmark_path is None:
            raise ValueError("this metric requires an explicit benchmark_path")
        return self


class SweepConfig(StrictModel):
    base_training_config: Path
    sweep_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,59}$")
    stage1: StageOne
    stage2: StageTwo
    selection: SelectionRule = Field(default_factory=SelectionRule)


class SelectionRun(StrictModel):
    lambda_value: float
    run_id: str
    score: float | None
    metrics: dict[str, float | None]
    evaluation: dict


class StageOneSelection(StrictModel):
    sweep_id: str
    base_config_sha256: str
    sweep_config_sha256: str
    data_cutoff: dict
    training_data_sha256: str
    smoke: bool
    selection_rule: dict
    tie_break: Literal["smaller_lambda_first"] = "smaller_lambda_first"
    runs: list[SelectionRun]
    selected_lambdas: list[float] = Field(min_length=2, max_length=2)


def load_sweep_config(path: str | Path) -> SweepConfig:
    """경로를 해석하고 확정된 탐색 수와 알 수 없는 키를 검증한다."""
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        config = SweepConfig.model_validate(yaml.safe_load(stream))
    config.base_training_config = (path.parent / config.base_training_config).resolve()
    if config.selection.benchmark_path is not None:
        config.selection.benchmark_path = (path.parent / config.selection.benchmark_path).resolve()
    return config


def _sha(value) -> str:
    content = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def experiment_run_id(sweep_id: str, stage: int, variant: str, seed: int, penalty=None) -> str:
    suffix = "" if penalty is None else f"-lam{float(penalty):.1f}".replace(".", "p")
    value = f"{sweep_id}-s{stage}-{variant}{suffix}-seed{seed}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError("experiment run_id is not filesystem safe")
    return value


def _job_config(base, variant, seed, penalty, smoke):
    payload = base.model_dump(mode="json")
    payload["seed"] = seed
    payload["env"]["reward_variant"] = variant
    # lambda는 mdd_penalty에만 적용하고 다른 그룹은 기준값을 사용한다.
    payload["env"]["penalty_lambda"] = float(penalty) if penalty is not None else 1.0
    return TrainingConfig.model_validate(payload, context={"smoke": smoke})


def _guard_training_data(dataset: TrainingDataset, base: TrainingConfig) -> None:
    for frame in (dataset.returns, dataset.rsi, dataset.macd):
        if (
            not isinstance(frame.index, pd.DatetimeIndex)
            or frame.index.tz is None
            or frame.index.hasnans
            or not frame.index.is_unique
            or not frame.index.is_monotonic_increasing
            or frame.empty
            or (frame.index > base.data.train_end).any()
            or (base.data.train_start is not None and (frame.index < base.data.train_start).any())
        ):
            raise ValueError("selection data must stay entirely inside the training cutoff")
    if any(tag.available_at > base.data.train_end for tag in dataset.risk_tags):
        raise ValueError("selection risk tags must not exceed the training cutoff")


def _data_sha(dataset: TrainingDataset) -> str:
    digest = hashlib.sha256()
    for frame in (dataset.returns, dataset.rsi, dataset.macd):
        digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
        digest.update(_sha(list(frame.columns)).encode())
    tags = [tag.model_dump(mode="json") for tag in dataset.risk_tags]
    digest.update(_sha(tags).encode())
    return digest.hexdigest()


def _benchmark(config: SweepConfig, base: TrainingConfig):
    path = config.selection.benchmark_path
    if path is None:
        return None
    frame = pd.read_parquet(path)
    if frame.shape[1] != 1:
        raise ValueError("benchmark parquet must have exactly one column")
    series = frame.iloc[:, 0]
    if (
        not isinstance(series.index, pd.DatetimeIndex)
        or series.index.tz is None
        or series.empty
        or series.index.hasnans
        or not series.index.is_unique
        or not series.index.is_monotonic_increasing
        or (series.index > base.data.train_end).any()
        or (base.data.train_start is not None and (series.index < base.data.train_start).any())
    ):
        raise ValueError("selection benchmark must contain training dates only")
    values = series.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < -1).any():
        raise ValueError("benchmark must contain finite simple returns >= -1")
    return series


def evaluate_selection(model, dataset, base, rule, benchmark=None):
    """학습 구간만 평가하고 정의 불가·짧은 표본의 지표는 선별에서 제외한다."""
    _guard_training_data(dataset, base)
    if benchmark is not None and (
        not isinstance(benchmark.index, pd.DatetimeIndex)
        or benchmark.index.tz is None
        or benchmark.index.hasnans
        or not benchmark.index.is_unique
        or not benchmark.index.is_monotonic_increasing
        or (benchmark.index > base.data.train_end).any()
        or (base.data.train_start is not None and (benchmark.index < base.data.train_start).any())
    ):
        raise ValueError("selection benchmark must contain training dates only")
    rollout = evaluate_policy(model, dataset, base.env)
    if (rollout.returns.index > base.data.train_end).any() or (
        base.data.train_start is not None and (rollout.returns.index < base.data.train_start).any()
    ):
        raise ValueError("selection rollout exceeds training cutoff")
    if len(rollout.returns) < 2:
        metrics = {}
        score = None
    else:
        reference = pd.Series(0.0, index=rollout.returns.index)
        if benchmark is not None:
            reference = benchmark.reindex(rollout.returns.index)
        elif rule.metric in {"alpha", "beta", "information_ratio"}:
            raise ValueError("selected metric requires a benchmark")
        calculated = calculate_metrics(rollout.returns, reference)
        metrics = {
            key: float(value) if np.isfinite(value) else None for key, value in calculated.items()
        }
        score = metrics[rule.metric]
    return (
        score,
        metrics,
        {
            "start": rollout.returns.index[0].isoformat() if len(rollout.returns) else None,
            "end": rollout.returns.index[-1].isoformat() if len(rollout.returns) else None,
            "observation_count": rollout.observation_count,
            "terminated": rollout.terminated,
            "truncated": rollout.truncated,
            "reason": rollout.reason,
        },
    )


def _rank(runs, rule):
    eligible = [run for run in runs if run.score is not None]
    ordered = sorted(
        eligible,
        key=lambda run: (-run.score if rule.higher_is_better else run.score, run.lambda_value),
    )
    if len(ordered) < 2:
        raise ValueError("at least two finite selection scores are required")
    return [run.lambda_value for run in ordered[:2]]


def _context(config, smoke):
    # 전달받은 모델도 재검증하여 변경된 설정으로 제약을 우회하지 못하게 한다.
    config = SweepConfig.model_validate(config.model_dump())
    base = load_training_config(config.base_training_config, smoke=smoke)
    payload = base.model_dump(mode="json")
    payload["data"]["dataset_dir"] = str(base.data.dataset_dir.expanduser().resolve())
    base = TrainingConfig.model_validate(payload, context={"smoke": smoke})
    dataset = load_training_dataset(base.data)
    _guard_training_data(dataset, base)
    benchmark = _benchmark(config, base)
    binding = {
        "sweep_id": config.sweep_id,
        "base_config_sha256": _sha(base.model_dump(mode="json")),
        "sweep_config_sha256": _sha(config.model_dump(mode="json")),
        "data_cutoff": base.data.model_dump(mode="json"),
        "training_data_sha256": _sha(
            {
                "dataset": _data_sha(dataset),
                "benchmark": (
                    None
                    if benchmark is None
                    else _sha(
                        {
                            "index": [date.isoformat() for date in benchmark.index],
                            "values": benchmark.tolist(),
                        }
                    )
                ),
            }
        ),
        "smoke": smoke,
    }
    return base, dataset, benchmark, binding


def _execute_job(config, run_id, smoke, resume):
    settings = get_settings()
    artifact_dir = settings.artifacts_dir.expanduser().resolve() / "training" / run_id
    manifest_path = artifact_dir / "manifest.json"
    final_path = settings.models_dir.expanduser().resolve() / run_id / "final_model.zip"
    if resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = config.model_dump(mode="json")
        expected["device"] = resolve_device(config.device)
        if (
            manifest.get("status") != "completed"
            or manifest.get("run_id") != run_id
            or manifest.get("smoke") != smoke
            or manifest.get("requested_device") != config.device
            or manifest.get("config") != expected
            or not final_path.is_file()
        ):
            raise ValueError(f"cannot resume incomplete or incompatible run: {run_id}")
        return PPO.load(final_path, device=expected["device"])
    return run_training(config, run_id=run_id, smoke=smoke).model


def _manifest(directory, binding, stage, jobs, resume, selection_sha=None):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if any(manifest.get(key) != value for key, value in binding.items()):
            raise ValueError("sweep manifest does not match the current configuration or data")
        if str(stage) in manifest["stages"] and not resume:
            raise FileExistsError("this sweep stage already exists; use --resume")
        previous = manifest["stages"].get(str(stage))
        if previous is not None and previous["run_ids"] != [run_id for run_id, _ in jobs]:
            raise ValueError("cannot resume a stage with different run_ids")
        if previous is not None and previous.get("selection_sha256") != selection_sha:
            raise ValueError("cannot resume a stage with a different selection file")
    else:
        manifest = {**binding, "started_at": _now(), "finished_at": None, "stages": {}}
    manifest["stages"][str(stage)] = {
        "run_ids": [run_id for run_id, _ in jobs],
        "status": "running",
        "started_at": _now(),
        "finished_at": None,
    }
    if selection_sha is not None:
        manifest["stages"][str(stage)]["selection_sha256"] = selection_sha
    manifest["status"] = f"stage{stage}_running"
    manifest["finished_at"] = None
    _write_json(path, manifest)
    return path, manifest


def _finish(path, manifest, stage, error=None):
    status = manifest["stages"][str(stage)]
    status.update(status="completed" if error is None else "failed", finished_at=_now())
    manifest["status"] = f"stage{stage}_{status['status']}"
    if error is not None:
        status["error"] = f"{type(error).__name__}: {error}"
    manifest["finished_at"] = _now()
    _write_json(path, manifest)


def run_stage1(config: SweepConfig, *, smoke=False, resume=False) -> Path:
    """lambda 다섯 개를 같은 학습 구간에서 선별하고 두 후보를 파일로 고정한다."""
    base, dataset, benchmark, binding = _context(config, smoke)
    seed = config.stage1.seeds[0]
    jobs = [
        (
            experiment_run_id(config.sweep_id, 1, "mdd_penalty", seed, penalty),
            _job_config(base, "mdd_penalty", seed, penalty, smoke),
        )
        for penalty in config.stage1.lambdas
    ]
    artifact_root = get_settings().artifacts_dir.expanduser().resolve()
    directory = artifact_root / "experiments" / config.sweep_id
    selection_path = directory / "stage1_selection.json"
    path, manifest = _manifest(directory, binding, 1, jobs, resume)
    try:
        records = []
        for (run_id, job), penalty in zip(jobs, config.stage1.lambdas):
            model = _execute_job(job, run_id, smoke, resume)
            score, metrics, evaluation = evaluate_selection(
                model, dataset, job, config.selection, benchmark
            )
            records.append(
                SelectionRun(
                    lambda_value=penalty,
                    run_id=run_id,
                    score=score,
                    metrics=metrics,
                    evaluation=evaluation,
                )
            )
        selected = StageOneSelection(
            **binding,
            selection_rule=config.selection.model_dump(mode="json"),
            runs=records,
            selected_lambdas=_rank(records, config.selection),
        )
        _write_json(selection_path, selected.model_dump(mode="json"))
    except BaseException as error:
        _finish(path, manifest, 1, error)
        raise
    _finish(path, manifest, 1)
    return selection_path


def run_stage2(
    config: SweepConfig, selection_path: str | Path, *, smoke=False, resume=False
) -> Path:
    """검증된 선택 파일로만 네 그룹을 다섯 seed로 학습한다."""
    selection_json = Path(selection_path).read_text(encoding="utf-8")
    selected = StageOneSelection.model_validate_json(selection_json)
    base, _, _, binding = _context(config, smoke)
    document = selected.model_dump(mode="json")
    if any(document[key] != value for key, value in binding.items()):
        raise ValueError("selection file does not match the current sweep, cutoff or base config")
    if selected.selection_rule != config.selection.model_dump(mode="json"):
        raise ValueError("selection rule mismatch")
    expected_runs = {}
    for penalty in config.stage1.lambdas:
        run_id = experiment_run_id(
            config.sweep_id, 1, "mdd_penalty", config.stage1.seeds[0], penalty
        )
        expected_runs[run_id] = penalty
    actual_runs = {run.run_id: run.lambda_value for run in selected.runs}
    if len(selected.runs) != 5 or actual_runs != expected_runs:
        raise ValueError("selection records must contain exactly the five stage1 candidates")
    if any(run.score != run.metrics.get(config.selection.metric) for run in selected.runs):
        raise ValueError("selection scores disagree with the recorded metric")
    if selected.selected_lambdas != _rank(selected.runs, config.selection):
        raise ValueError("selected lambdas disagree with the recorded ranking")
    groups = [("return", None), ("sharpe", None)] + [
        ("mdd_penalty", penalty) for penalty in selected.selected_lambdas
    ]
    jobs = [
        (
            experiment_run_id(config.sweep_id, 2, variant, seed, penalty),
            _job_config(base, variant, seed, penalty, smoke),
        )
        for variant, penalty in groups
        for seed in config.stage2.seeds
    ]
    artifact_root = get_settings().artifacts_dir.expanduser().resolve()
    directory = artifact_root / "experiments" / config.sweep_id
    path, manifest = _manifest(directory, binding, 2, jobs, resume, selection_sha=_sha(document))
    try:
        for run_id, job in jobs:
            _execute_job(job, run_id, smoke, resume)
    except BaseException as error:
        _finish(path, manifest, 2, error)
        raise
    _finish(path, manifest, 2)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="보상·lambda 두 단계 실험")
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for name in ("stage1", "stage2"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--config", default="config/experiments/rewards.yaml")
        subparser.add_argument("--smoke", action="store_true")
        subparser.add_argument("--resume", action="store_true")
        if name == "stage2":
            subparser.add_argument("--selection", required=True)
    args = parser.parse_args(argv)
    config = load_sweep_config(args.config)
    if args.stage == "stage1":
        path = run_stage1(config, smoke=args.smoke, resume=args.resume)
    else:
        path = run_stage2(config, args.selection, smoke=args.smoke, resume=args.resume)
    print(path)


if __name__ == "__main__":
    main()
