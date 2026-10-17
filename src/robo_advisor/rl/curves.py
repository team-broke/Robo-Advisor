"""학습곡선 생성과 사전 정의 기준의 수렴 판정.

학습 중 환경별 Monitor 파일(artifacts/training/<run_id>/monitor/)에 남은 에피소드
보상·길이·종료 사유를 읽는다. 에피소드는 경과 시각 순으로 합쳐 전역 순서를 만들고
누적 스텝을 x축으로 쓴다. 판정 기준은 결과를 보기 전에 고정한 상수이며 실행마다
판정 결과와 함께 기록한다.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MONITOR_DIR = "monitor"
SAFEGUARD_REASON = "mdd_limit_exceeded"


@dataclass(frozen=True)
class ConvergenceCriteria:
    """마지막 tail_fraction 구간을 둘로 나눠 추세가 남았는지 본다.

    두 절반의 평균 차이와 선형 추세의 구간 전체 변화량이 모두 tail 표준편차의
    tolerance 배 이하이면 수렴으로 판정한다. 에피소드가 min_episodes 미만이거나
    tail이 min_tail_episodes 미만이면 판정하지 않는다.
    """

    tail_fraction: float = 0.2
    min_episodes: int = 100
    min_tail_episodes: int = 20
    tolerance: float = 1.0
    smoothing_window: int = 20


CRITERIA = ConvergenceCriteria()


def load_episodes(run_dir: str | Path) -> pd.DataFrame:
    """환경별 Monitor 파일을 경과 시각 순의 에피소드 표로 합친다."""
    files = sorted((Path(run_dir) / MONITOR_DIR).glob("*.monitor.csv"))
    if not files:
        raise FileNotFoundError(f"no monitor files in {Path(run_dir) / MONITOR_DIR}")
    frames = []
    for path in files:
        frame = pd.read_csv(path, skiprows=1)
        frame["env"] = path.name.removesuffix(".monitor.csv")
        frames.append(frame)
    episodes = pd.concat(frames, ignore_index=True)
    episodes = episodes.sort_values(["t", "env"], kind="stable").reset_index(drop=True)
    episodes = episodes.rename(columns={"r": "reward", "l": "length", "t": "elapsed_seconds"})
    if "reason" not in episodes:
        episodes["reason"] = None
    episodes["episode"] = np.arange(1, len(episodes) + 1)
    episodes["timesteps"] = episodes["length"].cumsum()
    episodes["safeguard"] = (episodes["reason"] == SAFEGUARD_REASON).astype(float)
    return episodes[
        [
            "episode",
            "timesteps",
            "env",
            "reward",
            "length",
            "reason",
            "safeguard",
            "elapsed_seconds",
        ]
    ]


def smooth_curves(episodes: pd.DataFrame, window: int = CRITERIA.smoothing_window) -> pd.DataFrame:
    """원시 값 옆에 이동평균(초기 구간은 가용 관측 평균)을 붙인다."""
    curves = episodes.copy()
    for column in ("reward", "length", "safeguard"):
        curves[f"{column}_smoothed"] = curves[column].rolling(window, min_periods=1).mean()
    return curves


def judge_convergence(
    rewards: pd.Series | np.ndarray, criteria: ConvergenceCriteria = CRITERIA
) -> dict:
    """에피소드 보상 순서열에 사전 정의 기준을 적용한다."""
    values = np.asarray(rewards, dtype=np.float64)
    tail_size = int(np.floor(len(values) * criteria.tail_fraction))
    result = {
        "status": "insufficient_episodes",
        "episodes": int(len(values)),
        "tail_episodes": tail_size,
        "tail_mean": None,
        "tail_std": None,
        "half_mean_change": None,
        "trend_change": None,
    }
    if len(values) < criteria.min_episodes or tail_size < criteria.min_tail_episodes:
        return result
    tail = values[-tail_size:]
    first, second = np.array_split(tail, 2)
    spread = float(np.std(tail, ddof=1))
    slope = float(np.polyfit(np.arange(tail_size), tail, 1)[0])
    half_change = float(second.mean() - first.mean())
    trend_change = slope * (tail_size - 1)
    limit = criteria.tolerance * spread
    converged = abs(half_change) <= limit and abs(trend_change) <= limit
    result.update(
        status="converged" if converged else "not_converged",
        tail_mean=float(tail.mean()),
        tail_std=spread,
        half_mean_change=half_change,
        trend_change=trend_change,
    )
    return result


def plot_curves(curves: pd.DataFrame, path: str | Path, title: str) -> Path:
    """보상·에피소드 길이·Safe-Guard 빈도의 원시/평활 곡선을 저장한다."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    panels = (
        ("reward", "Episode reward"),
        ("length", "Episode length"),
        ("safeguard", "Safe-Guard rate"),
    )
    for axis, (column, label) in zip(axes, panels):
        axis.plot(curves["timesteps"], curves[column], alpha=0.3, linewidth=0.8, label="raw")
        axis.plot(
            curves["timesteps"], curves[f"{column}_smoothed"], linewidth=1.6, label="smoothed"
        )
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
    axes[0].legend(loc="best")
    axes[0].set_title(title)
    axes[-1].set_xlabel("Timesteps")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def write_run_curves(
    training_dir: str | Path, run_id: str, criteria: ConvergenceCriteria = CRITERIA
) -> dict:
    """실행 하나의 곡선 CSV·PNG와 수렴 판정 JSON을 쓰고 요약을 반환한다."""
    training_dir = Path(training_dir)
    run_dir = training_dir / run_id
    curves = smooth_curves(load_episodes(run_dir), criteria.smoothing_window)
    curves_path = run_dir / "curves.csv"
    curves.to_csv(curves_path, index=False)
    plot_path = plot_curves(curves, training_dir / "curves" / f"{run_id}.png", run_id)
    tail_size = int(np.floor(len(curves) * criteria.tail_fraction))
    summary = {
        "run_id": run_id,
        **judge_convergence(curves["reward"], criteria),
        "total_timesteps": int(curves["timesteps"].iloc[-1]) if len(curves) else 0,
        "safeguard_rate": float(curves["safeguard"].mean()) if len(curves) else None,
        "tail_safeguard_rate": (
            float(curves["safeguard"].iloc[-tail_size:].mean()) if tail_size else None
        ),
        "final_smoothed_reward": (
            float(curves["reward_smoothed"].iloc[-1]) if len(curves) else None
        ),
        "criteria": asdict(criteria),
        "curves_csv": str(curves_path),
        "plot": str(plot_path),
    }
    (run_dir / "convergence.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


CONVERGENCE_COLUMNS = (
    "run_id",
    "status",
    "episodes",
    "total_timesteps",
    "tail_episodes",
    "tail_mean",
    "tail_std",
    "half_mean_change",
    "trend_change",
    "safeguard_rate",
    "tail_safeguard_rate",
    "final_smoothed_reward",
    "tail_fraction",
    "min_episodes",
    "min_tail_episodes",
    "tolerance",
    "smoothing_window",
    "plot",
)


def write_convergence_table(training_dir: str | Path, *, refresh: bool = False) -> Path:
    """완료된 모든 실행의 판정을 artifacts/training/convergence.csv로 모은다.

    refresh가 참이거나 convergence.json이 없으면 Monitor 파일에서 다시 만든다.
    """
    training_dir = Path(training_dir)
    rows = []
    for manifest_path in sorted(training_dir.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            continue
        run_dir = manifest_path.parent
        summary_path = run_dir / "convergence.json"
        if refresh or not summary_path.exists():
            summary = write_run_curves(training_dir, run_dir.name)
        else:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append({**summary, **summary["criteria"]})
    path = training_dir / "convergence.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CONVERGENCE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def main(argv=None):
    from robo_advisor.config import get_settings

    parser = argparse.ArgumentParser(description="학습곡선과 수렴 판정표 생성")
    parser.add_argument("--training-dir", help="기본값: <artifacts_dir>/training")
    parser.add_argument("--refresh", action="store_true", help="기존 판정도 Monitor에서 다시 계산")
    args = parser.parse_args(argv)
    training_dir = args.training_dir or get_settings().artifacts_dir / "training"
    print(write_convergence_table(training_dir, refresh=args.refresh))


if __name__ == "__main__":
    main()
