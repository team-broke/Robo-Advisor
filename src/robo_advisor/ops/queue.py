"""실험 대기열: YAML 목록의 학습을 순서대로 무인 실행한다.

대기열 파일 예시는 config/experiments/queue.example.yaml에 있다. 실험마다 기준 학습 설정에
overrides를 깊게 병합한 설정을 artifacts/queue/<queue_id>/configs/<run_id>.yaml로 남기고,
별도 프로세스로 `python -m robo_advisor.rl.train`을 실행한다. 출력은
artifacts/queue/<queue_id>/logs/<run_id>.log에 쓴다.

- 이미 완료된 run_id는 건너뛴다(skipped). 완료되지 않은 같은 run_id가 있으면 덮어쓰지 않고
  실패로 기록한다. 지우거나 run_id를 바꾼 뒤 다시 실행한다.
- 한 실험이 실패해도 다음 실험을 계속한다(stop_on_failure로 변경 가능).
- 같은 대기열의 동시 실행은 잠금 파일로 막는다. 잠금을 가진 프로세스가 없으면 잠금을 넘겨받는다.
- 상태는 artifacts/queue/<queue_id>/state.json(schema_version 1)에 기록한다.
- after 명령은 학습 성공 후 셸에서 실행하며 {run_id}, {artifact_dir}를 치환한다. 실패해도
  학습 결과는 완료로 두고 오류만 기록한다.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

SafeId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")]
QUEUE_SCHEMA_VERSION = 1


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class JobDefaults(StrictModel):
    device: Literal["auto", "cpu", "cuda"] | None = None
    log_interval: float = Field(default=30.0, gt=0)
    smoke: bool = False


class Job(StrictModel):
    run_id: SafeId
    config: str
    overrides: dict = Field(default_factory=dict)
    device: Literal["auto", "cpu", "cuda"] | None = None
    log_interval: float | None = Field(default=None, gt=0)
    smoke: bool | None = None
    after: list[str] = Field(default_factory=list)


class QueueConfig(StrictModel):
    queue_id: SafeId
    defaults: JobDefaults = Field(default_factory=JobDefaults)
    stop_on_failure: bool = False
    jobs: list[Job] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_runs(self):
        ids = [job.run_id for job in self.jobs]
        if len(ids) != len(set(ids)):
            raise ValueError("run_id values in a queue must be unique")
        return self


def load_queue(path: str | Path) -> QueueConfig:
    with Path(path).open(encoding="utf-8") as stream:
        return QueueConfig.model_validate(yaml.safe_load(stream))


def deep_merge(base: dict, overrides: dict) -> dict:
    """overrides의 dict 값은 재귀적으로, 그 밖의 값은 통째로 덮어쓴다."""
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class QueueLock:
    """대기열별 잠금 파일. 살아 있는 다른 프로세스가 잡고 있으면 실패한다."""

    def __init__(self, path: Path):
        self.path = path
        self.acquired = False

    def __enter__(self):
        while True:
            try:
                handle = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    owner = int(self.path.read_text().strip() or 0)
                except (OSError, ValueError):
                    owner = 0
                if owner and _pid_alive(owner):
                    raise RuntimeError(f"queue is already running (pid {owner})") from None
                self.path.unlink(missing_ok=True)
                continue
            with os.fdopen(handle, "w") as stream:
                stream.write(str(os.getpid()))
            self.acquired = True
            return self

    def __exit__(self, *args):
        if self.acquired:
            self.path.unlink(missing_ok=True)


def _settings_dirs(artifacts_dir: Path | None) -> Path:
    if artifacts_dir is not None:
        return Path(artifacts_dir).expanduser().resolve()
    from robo_advisor.config import get_settings

    return get_settings().artifacts_dir.expanduser().resolve()


def _previous_status(artifacts_dir: Path, run_id: str) -> str | None:
    manifest = artifacts_dir / "training" / run_id / "manifest.json"
    if not manifest.exists():
        return None
    try:
        return json.loads(manifest.read_text(encoding="utf-8")).get("status", "unknown")
    except (OSError, ValueError):
        return "unknown"


def _tail(path: Path, lines: int = 20) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def run_queue(
    queue_path: str | Path,
    *,
    artifacts_dir: Path | None = None,
    python: str = sys.executable,
    dry_run: bool = False,
    runner: Callable = subprocess.run,
    env: dict | None = None,
) -> dict:
    """대기열을 실행하고 최종 상태를 반환한다."""
    queue_path = Path(queue_path).expanduser().resolve()
    queue = load_queue(queue_path)
    artifacts_dir = _settings_dirs(artifacts_dir)
    root = artifacts_dir / "queue" / queue.queue_id
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    state = {
        "schema_version": QUEUE_SCHEMA_VERSION,
        "queue_id": queue.queue_id,
        "queue_file": str(queue_path),
        "state": "running",
        "pid": os.getpid(),
        "started_at": _utc_now(),
        "updated_at": _utc_now(),
        "finished_at": None,
        "jobs": [
            {
                "run_id": job.run_id,
                "state": "pending",
                "started_at": None,
                "finished_at": None,
                "returncode": None,
                "error": None,
                "config_path": None,
            }
            for job in queue.jobs
        ],
    }

    def save():
        state["updated_at"] = _utc_now()
        _write_json(root / "state.json", state)

    with QueueLock(root / "lock"):
        save()
        current = None
        try:
            for job, record in zip(queue.jobs, state["jobs"]):
                current = record
                previous = _previous_status(artifacts_dir, job.run_id)
                if previous == "completed":
                    record.update(state="skipped", error="already completed")
                    save()
                    continue
                if previous is not None:
                    record.update(
                        state="failed",
                        error=f"run_id exists with status {previous}; remove it or rename",
                    )
                    save()
                    if queue.stop_on_failure:
                        break
                    continue
                base_path = Path(job.config).expanduser()
                if not base_path.is_absolute():
                    base_path = queue_path.parent / base_path
                base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
                config_path = root / "configs" / f"{job.run_id}.yaml"
                config_path.write_text(
                    yaml.safe_dump(
                        deep_merge(base, job.overrides), sort_keys=False, allow_unicode=True
                    ),
                    encoding="utf-8",
                )
                device = job.device or queue.defaults.device
                smoke = queue.defaults.smoke if job.smoke is None else job.smoke
                interval = job.log_interval or queue.defaults.log_interval
                command = [
                    python,
                    "-m",
                    "robo_advisor.rl.train",
                    "--config",
                    str(config_path),
                    "--run-id",
                    job.run_id,
                    "--log-interval",
                    str(interval),
                ]
                if device:
                    command += ["--device", device]
                if smoke:
                    command.append("--smoke")
                record.update(state="running", started_at=_utc_now(), config_path=str(config_path))
                save()
                if dry_run:
                    record.update(state="pending", error="dry run: " + shlex.join(command))
                    save()
                    continue
                log_path = root / "logs" / f"{job.run_id}.log"
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(f"$ {shlex.join(command)}\n")
                    log.flush()
                    completed = runner(
                        command, stdout=log, stderr=subprocess.STDOUT, check=False, env=env
                    )
                    record.update(returncode=completed.returncode, finished_at=_utc_now())
                    if completed.returncode == 0:
                        record["state"] = "completed"
                        artifact_dir = artifacts_dir / "training" / job.run_id
                        for hook in job.after:
                            line = hook.format(run_id=job.run_id, artifact_dir=artifact_dir)
                            log.write(f"$ {line}\n")
                            log.flush()
                            result = runner(
                                line,
                                shell=True,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                check=False,
                                env=env,
                            )
                            if result.returncode != 0:
                                record["error"] = f"after command failed: {line}"
                                break
                if record["state"] != "completed":
                    record.update(state="failed", error=_tail(log_path) or "training failed")
                save()
                if record["state"] == "failed" and queue.stop_on_failure:
                    break
        except BaseException as error:
            if current is not None and current["state"] == "running":
                current.update(state="failed", finished_at=_utc_now(), error=repr(error))
            state["state"] = "interrupted"
            state["finished_at"] = _utc_now()
            save()
            raise
        failed = any(record["state"] == "failed" for record in state["jobs"])
        state["state"] = "failed" if failed else "completed"
        state["finished_at"] = _utc_now()
        save()
    return state


def format_state(state: dict) -> str:
    lines = [f"{state['queue_id']}: {state['state']} (갱신 {state['updated_at']})"]
    for record in state["jobs"]:
        line = f"  {record['state']:9s} {record['run_id']}"
        if record.get("error") and record["state"] in ("failed", "skipped", "pending"):
            line += f"  - {record['error'].splitlines()[-1][:120]}"
        lines.append(line)
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="실험 대기열 실행과 상태 확인")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="대기열을 순서대로 실행")
    run.add_argument("queue")
    run.add_argument("--dry-run", action="store_true", help="설정만 만들고 학습은 실행하지 않음")
    status = sub.add_parser("status", help="대기열 상태 출력")
    status.add_argument("queue")
    for item in (run, status):
        item.add_argument("--artifacts-dir", help="기본값: 설정의 artifacts_dir")
    args = parser.parse_args(argv)
    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else None
    if args.command == "run":
        state = run_queue(args.queue, artifacts_dir=artifacts_dir, dry_run=args.dry_run)
        print(format_state(state))
        if state["state"] != "completed":
            raise SystemExit(1)
        return
    queue = load_queue(args.queue)
    path = _settings_dirs(artifacts_dir) / "queue" / queue.queue_id / "state.json"
    if not path.exists():
        print(f"{queue.queue_id}: 아직 실행 기록 없음")
        return
    print(format_state(json.loads(path.read_text(encoding="utf-8"))))


if __name__ == "__main__":
    main()
