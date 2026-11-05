"""학습 산출물을 변경하지 않고 읽는 모니터 데이터 함수."""

from collections import Counter, deque
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re

import pandas as pd

STALE_SECONDS = 120
STATE_LABELS = {
    "running": "학습 중",
    "completed": "완료",
    "failed": "실패",
    "interrupted": "중단",
    "pending": "대기",
    "skipped": "건너뜀",
    "stale": "응답 없음(stale)",
    "unknown": "확인할 수 없음",
}


def as_dict(value):
    """형식이 다른 중첩 값은 빈 객체로 취급한다."""
    return value if isinstance(value, dict) else {}


def number(value):
    """유한한 숫자만 반환하며 bool과 문자열은 제외한다."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return value if math.isfinite(value) else None
        except OverflowError:
            pass
    return None


def _timestamp(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else None
    except (ValueError, OverflowError):
        return None


def _json(path, warnings, *, required=False):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("JSON 객체가 아닙니다")
        return value
    except FileNotFoundError:
        if required:
            warnings.append(f"{path.name}: 파일이 없습니다.")
    except (OSError, UnicodeError, ValueError, RecursionError) as error:
        warnings.append(f"{path.name}: 읽을 수 없습니다 ({type(error).__name__}).")
    return {}


def _records(value, warnings, name):
    if value is None:
        return []
    if not isinstance(value, list):
        warnings.append(f"{name}: 목록 형식이 아닙니다.")
        return []
    rows = [row for row in value if isinstance(row, dict)]
    if len(rows) != len(value):
        warnings.append(f"{name}: 잘못된 항목을 제외했습니다.")
    return rows


def _directories(root):
    try:
        return sorted(path for path in root.iterdir() if path.is_dir() and not path.is_symlink())
    except OSError:
        return []


def _stale(state, updated_at, now, stale_seconds, warnings):
    if state != "running":
        return False
    updated = _timestamp(updated_at)
    if updated is None:
        warnings.append("갱신 시각이 없어 실행 중인지 확인할 수 없습니다.")
        return True
    return (now - updated).total_seconds() > stale_seconds


def _clock(now, stale_seconds):
    if number(stale_seconds) is None or stale_seconds <= 0:
        raise ValueError("stale_seconds must be positive and finite")
    now = now if now is not None else datetime.now(timezone.utc)
    if not isinstance(now, datetime) or now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return now


def _summary(run_id, manifest, status, convergence, now, stale_seconds, warnings):
    for name, payload, keys in (
        ("manifest.json", manifest, ("run_id", "status", "started_at")),
        ("status.json", status, ("run_id", "state", "updated_at")),
    ):
        if payload and any(key not in payload for key in keys):
            warnings.append(f"{name}: 일부 필수 항목이 없어 가능한 값만 표시합니다.")
        if payload.get("run_id", run_id) != run_id:
            warnings.append(f"{name}: 실행 이름이 폴더 이름과 다릅니다.")
    manifest_state = manifest.get("status")
    live_state = status.get("state")
    terminal = ("completed", "failed")
    state = manifest_state if manifest_state in terminal else live_state
    if state not in (*terminal, "running"):
        state = manifest_state if manifest_state == "running" else "unknown"
    if manifest_state in terminal and live_state in (*terminal, "running"):
        if manifest_state != live_state:
            warnings.append("종료 기록과 실시간 상태가 달라 종료 기록을 우선 표시합니다.")
    if state == "unknown":
        warnings.append("실행 상태가 없거나 지원하지 않는 값입니다.")
    stale = _stale(state, status.get("updated_at"), now, stale_seconds, warnings)
    started_at = manifest.get("started_at") or status.get("started_at")
    finished_at = manifest.get("finished_at")
    started, finished = _timestamp(started_at), _timestamp(finished_at)
    elapsed = number(status.get("elapsed_seconds"))
    if started and finished:
        elapsed = max(0, (finished - started).total_seconds())
    elif elapsed is None and started and state == "running":
        elapsed = max(0, (now - started).total_seconds())
    config = as_dict(manifest.get("config"))
    validation = as_dict(manifest.get("validation"))
    live_validation = as_dict(status.get("validation"))
    best = as_dict(live_validation.get("best")) or as_dict(validation.get("best"))
    if manifest_state in terminal:
        best = as_dict(validation.get("best")) or best
    steps = number(status.get("timesteps"))
    if manifest_state in terminal or steps is None:
        steps = number(manifest.get("actual_timesteps"))
    total = number(status.get("total_timesteps"))
    if total is None:
        total = number(config.get("total_timesteps"))
    progress = number(status.get("progress"))
    if steps is not None and total is not None and total > 0:
        progress = steps / total
    return {
        "run_id": run_id,
        "state": "stale" if stale else state,
        "status_label": STATE_LABELS["stale" if stale else state],
        "manifest_status": manifest_state,
        "live_state": live_state,
        "stale": stale,
        "started_at": started_at,
        "finished_at": finished_at,
        "updated_at": status.get("updated_at"),
        "elapsed_seconds": elapsed,
        "timesteps": steps,
        "total_timesteps": total,
        "progress": min(1.0, max(0.0, progress)) if progress is not None else None,
        "device": manifest.get("device"),
        "gpu_name": manifest.get("gpu_name"),
        "smoke": manifest.get("smoke"),
        "validation_metric": live_validation.get("metric") or validation.get("metric"),
        "best_timesteps": number(best.get("timesteps")),
        "best_score": number(best.get("score")),
        "convergence": convergence.get("status")
        or as_dict(manifest.get("learning_curves")).get("status"),
        "error": manifest.get("error") or status.get("message"),
        "warnings": warnings,
    }


def list_runs(artifacts_dir, *, stale_seconds=STALE_SECONDS, now=None):
    """실행을 최신 시작 순으로 반환한다. 갱신 간격 기본 한도는 120초다."""
    now = _clock(now, stale_seconds)
    runs = []
    for directory in _directories(Path(artifacts_dir) / "training"):
        if directory.name == "curves":
            continue
        warnings = []
        manifest = _json(directory / "manifest.json", warnings, required=True)
        status = _json(directory / "status.json", warnings)
        convergence = _json(directory / "convergence.json", warnings)
        runs.append(
            _summary(directory.name, manifest, status, convergence, now, stale_seconds, warnings)
        )
    floor = datetime.min.replace(tzinfo=timezone.utc)
    return sorted(
        runs, key=lambda row: (_timestamp(row["started_at"]) or floor, row["run_id"]), reverse=True
    )


def load_run(artifacts_dir, models_dir, run_id, *, log_lines=100):
    """한 실행의 선택적 산출물과 로그 끝부분을 안전하게 읽는다."""
    warnings = []
    result = {
        "manifest": {},
        "status": {},
        "convergence": {},
        "curves": pd.DataFrame(),
        "curve_png": None,
        "validation_history": pd.DataFrame(),
        "evaluation": {},
        "log_tail": "",
        "best_model_exists": False,
        "final_model_exists": False,
        "warnings": warnings,
    }
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
        warnings.append("실행 이름이 올바르지 않습니다.")
        return result
    directory = Path(artifacts_dir) / "training" / run_id
    model_dir = Path(models_dir) / run_id
    if directory.is_symlink() or model_dir.is_symlink():
        warnings.append("연결된 디렉터리는 읽지 않습니다.")
        return result
    for key, name in (
        ("manifest", "manifest.json"),
        ("status", "status.json"),
        ("convergence", "convergence.json"),
        ("evaluation", "evaluation.json"),
    ):
        result[key] = _json(directory / name, warnings, required=key == "manifest")
    try:
        result["curves"] = pd.read_csv(directory / "curves.csv")
    except FileNotFoundError:
        pass
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as error:
        warnings.append(f"curves.csv: 읽을 수 없습니다 ({type(error).__name__}).")
    validation = as_dict(result["manifest"].get("validation"))
    result["validation_history"] = pd.DataFrame(
        _records(validation.get("history"), warnings, "검증 이력")
    )
    evaluation = result["evaluation"]
    periods = _records(evaluation.get("periods"), warnings, "평가 기간")
    evaluation["periods"] = [
        {**period, "rows": _records(period.get("rows"), warnings, "평가 전략")}
        for period in periods
    ]
    png = Path(artifacts_dir) / "training" / "curves" / f"{run_id}.png"
    if png.is_file():
        result["curve_png"] = png
    if not isinstance(log_lines, int) or isinstance(log_lines, bool) or log_lines < 0:
        warnings.append("로그 줄 수가 올바르지 않아 기본 100줄을 표시합니다.")
        log_lines = 100
    try:
        with (directory / "train.log").open(encoding="utf-8", errors="replace") as stream:
            result["log_tail"] = "".join(deque(stream, maxlen=log_lines))
    except FileNotFoundError:
        pass
    except OSError:
        warnings.append("train.log: 읽을 수 없습니다.")
    for kind in ("best", "final"):
        result[f"{kind}_model_exists"] = (model_dir / f"{kind}_model.zip").is_file()
    return result


def pid_alive(pid) -> bool:
    """같은 컴퓨터에서 해당 프로세스가 살아 있는지 확인만 한다(신호를 보내지 않음)."""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, OverflowError):
        return False
    except PermissionError:
        return True
    return True


def list_queues(artifacts_dir, *, stale_seconds=STALE_SECONDS, now=None, alive=pid_alive):
    """대기열별 작업 개수와 갱신 상태를 반환하며 PID로 프로세스를 제어하지 않는다.

    대기열 상태 파일은 실험이 시작·끝날 때만 갱신되므로, 실행 중인 대기열은 기록된 PID가
    살아 있는지로 응답 여부를 판단한다. PID가 없으면 갱신 시각 기준으로 판단한다.
    """
    now = _clock(now, stale_seconds)
    queues = []
    for directory in _directories(Path(artifacts_dir) / "queue"):
        warnings = []
        value = _json(directory / "state.json", warnings, required=True)
        if "jobs" not in value:
            warnings.append("대기열 작업 목록이 없습니다.")
        jobs = _records(value.get("jobs"), warnings, "대기열 작업")
        counts = Counter(job.get("state") for job in jobs if isinstance(job.get("state"), str))
        state = value.get("state")
        if state not in ("running", "completed", "failed", "interrupted"):
            state = "unknown"
            warnings.append("대기열 상태가 없거나 지원하지 않는 값입니다.")
        pid = value.get("pid")
        if state == "running" and isinstance(pid, int) and not isinstance(pid, bool):
            stale = not alive(pid)
        else:
            stale = _stale(state, value.get("updated_at"), now, stale_seconds, warnings)
        queues.append(
            {
                "queue_id": directory.name,
                "state": "stale" if stale else state,
                "status_label": STATE_LABELS["stale" if stale else state],
                "stale": stale,
                **{
                    key: counts[key]
                    for key in ("pending", "running", "completed", "failed", "skipped")
                },
                "running_run_ids": [
                    job.get("run_id") for job in jobs if job.get("state") == "running"
                ],
                "pid": value.get("pid"),
                "updated_at": value.get("updated_at"),
                "jobs": jobs,
                "warnings": warnings,
            }
        )
    return queues
