"""합성 산출물로 읽기 전용 모니터의 상태 종합과 누락 처리를 검증한다."""

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from robo_advisor.ops.monitor_data import list_queues, list_runs, load_run

NOW = datetime(2026, 11, 5, 12, tzinfo=timezone.utc)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_run(root, run_id="current", *, started_at="2026-11-05T11:00:00Z", state="running"):
    directory = root / "training" / run_id
    write_json(
        directory / "manifest.json",
        {
            "run_id": run_id,
            "status": state,
            "started_at": started_at,
            "finished_at": "2026-11-05T11:30:00Z" if state == "completed" else None,
            "actual_timesteps": 100000,
            "device": "cuda",
            "gpu_name": "test-device",
            "n_envs": 12,
            "smoke": False,
            "config": {
                "total_timesteps": 1000000,
                "env": {"random_start": True},
                "ppo": {"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
            },
            "validation": {
                "metric": "sharpe",
                "history": [{"timesteps": 100000, "score": 1.2}],
                "best": {"timesteps": 100000, "score": 1.2},
            },
        },
    )
    write_json(
        directory / "status.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "state": "running",
            "updated_at": "2026-11-05T11:59:30Z",
            "timesteps": 123456,
            "total_timesteps": 1000000,
            "elapsed_seconds": 30,
            "steps_per_second": 4000,
            "eta_seconds": 219,
            "episodes": {
                "recent_mean_reward": 0.4,
                "recent_mean_length": 180,
                "recent_safeguard_rate": 0.1,
            },
            "validation": {
                "metric": "sharpe",
                "last": {"timesteps": 150000, "score": 1.1},
                "best": {"timesteps": 100000, "score": 1.2},
            },
        },
    )
    return directory


def test_runs_sort_by_start_and_combine_live_fields(tmp_path):
    write_run(tmp_path, "older", started_at="2026-11-04T11:00:00Z")
    write_run(tmp_path, "newer")
    rows = list_runs(tmp_path, now=NOW)
    assert [row["run_id"] for row in rows] == ["newer", "older"]
    row = rows[0]
    assert row["state"] == "running" and not row["stale"]
    assert row["timesteps"] == 123456
    assert row["progress"] == pytest.approx(0.123456)
    assert row["best_score"] == 1.2 and row["best_timesteps"] == 100000
    assert row["device"] == "cuda" and row["smoke"] is False


@pytest.mark.parametrize(
    "updated_at,expected",
    [
        ("2026-11-05T11:58:00Z", False),
        ("2026-11-05T11:57:59Z", True),
        (None, True),
        ("invalid", True),
        ("2026-11-05T12:00:00", True),
    ],
)
def test_running_stale_boundary_and_unknown_heartbeat(tmp_path, updated_at, expected):
    directory = write_run(tmp_path)
    write_json(directory / "status.json", {"state": "running", "updated_at": updated_at})
    row = list_runs(tmp_path, now=NOW, stale_seconds=120)[0]
    assert row["stale"] is expected
    assert row["state"] == ("stale" if expected else "running")


def test_completed_manifest_wins_over_old_live_status(tmp_path):
    directory = write_run(tmp_path, state="completed")
    write_json(
        directory / "status.json", {"state": "running", "updated_at": "2020-01-01T00:00:00Z"}
    )
    row = list_runs(tmp_path, now=NOW)[0]
    assert row["state"] == "completed" and not row["stale"]
    assert row["timesteps"] == 100000
    assert row["elapsed_seconds"] == 1800
    assert row["warnings"]


def test_terminal_live_status_overrides_running_manifest(tmp_path):
    directory = write_run(tmp_path)
    write_json(directory / "status.json", {"state": "failed", "message": "계산 실패"})
    row = list_runs(tmp_path, now=NOW)[0]
    assert row["state"] == "failed" and not row["stale"]
    assert row["error"] == "계산 실패"


def test_load_run_reads_optional_results_and_log_tail(tmp_path):
    artifacts, models = tmp_path / "artifacts", tmp_path / "models"
    directory = write_run(artifacts, state="completed")
    write_json(directory / "convergence.json", {"status": "converged"})
    write_json(
        directory / "evaluation.json",
        {"periods": [{"label": "테스트", "rows": [{"strategy": "equal_weight", "mdd": 0.05}]}]},
    )
    (directory / "curves.csv").write_text(
        "timesteps,reward,reward_smoothed,length,safeguard\n100,0.1,0.1,100,0\n",
        encoding="utf-8",
    )
    (directory / "train.log").write_text("first\nsecond\nthird\n", encoding="utf-8")
    png = artifacts / "training" / "curves" / "current.png"
    png.parent.mkdir()
    png.write_bytes(b"synthetic-image")
    model_dir = models / "current"
    model_dir.mkdir(parents=True)
    (model_dir / "best_model.zip").write_bytes(b"synthetic-model")
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    detail = load_run(artifacts, models, "current", log_lines=2)
    assert detail["log_tail"] == "second\nthird\n"
    assert detail["curve_png"] == png
    assert detail["curves"].iloc[0]["timesteps"] == 100
    assert detail["validation_history"].iloc[0]["score"] == 1.2
    assert detail["evaluation"]["periods"][0]["rows"][0]["mdd"] == 0.05
    assert detail["best_model_exists"] and not detail["final_model_exists"]
    assert list_runs(artifacts, now=NOW)[0]["convergence"] == "converged"
    after = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert before == after


def test_corrupt_missing_and_wrong_json_shapes_are_safe(tmp_path):
    directory = write_run(tmp_path)
    (directory / "manifest.json").write_text("{broken", encoding="utf-8")
    write_json(directory / "status.json", [])
    write_json(directory / "convergence.json", {"status": "not_converged"})
    write_json(directory / "evaluation.json", {"periods": [None, {"rows": "wrong"}]})
    (directory / "curves.csv").write_text('a,b\n"unfinished', encoding="utf-8")
    rows = list_runs(tmp_path, now=NOW)
    assert rows[0]["state"] == "unknown" and rows[0]["warnings"]
    detail = load_run(tmp_path, tmp_path / "models", "current")
    assert detail["manifest"] == {} and detail["status"] == {}
    assert detail["curves"].empty and detail["validation_history"].empty
    assert detail["evaluation"]["periods"] == [{"rows": []}]
    assert detail["warnings"]
    empty = load_run(tmp_path, tmp_path, "missing")
    assert empty["log_tail"] == "" and empty["warnings"]
    assert list_runs(tmp_path / "missing") == []
    assert list_queues(tmp_path / "missing") == []


def test_partial_nested_fields_and_invalid_ids_do_not_crash(tmp_path):
    directory = write_run(tmp_path)
    write_json(directory / "manifest.json", {"status": "running", "config": [], "validation": []})
    write_json(directory / "status.json", {"validation": {"best": []}, "progress": float("nan")})
    row = list_runs(tmp_path, now=NOW)[0]
    assert row["progress"] is None and row["best_score"] is None
    assert row["stale"]
    assert load_run(tmp_path, tmp_path, "../outside")["warnings"]
    assert load_run(tmp_path, tmp_path, "current", log_lines=-1)["warnings"]


def test_queues_count_jobs_and_detect_stale(tmp_path):
    write_json(
        tmp_path / "queue" / "nightly" / "state.json",
        {
            "state": "running",
            "pid": 1234,
            "updated_at": "2026-11-05T11:57:00Z",
            "jobs": [
                {"run_id": "a", "state": "completed"},
                {"run_id": "b", "state": "running"},
                {"run_id": "c", "state": "pending"},
                {"run_id": "d", "state": "failed"},
                {"run_id": "e", "state": "skipped"},
            ],
        },
    )
    row = list_queues(tmp_path, now=NOW, alive=lambda pid: False)[0]
    assert row["stale"] and row["state"] == "stale"
    assert [row[key] for key in ("pending", "running", "completed", "failed", "skipped")] == [1] * 5
    assert row["running_run_ids"] == ["b"] and row["pid"] == 1234
    # 실험 하나가 오래 돌아 갱신 시각이 오래돼도 프로세스가 살아 있으면 정상이다.
    row = list_queues(tmp_path, now=NOW, alive=lambda pid: pid == 1234)[0]
    assert not row["stale"] and row["state"] == "running"


def test_pid_alive_checks_only_valid_process_ids():
    import os

    from robo_advisor.ops.monitor_data import pid_alive

    assert pid_alive(os.getpid())
    assert not pid_alive(None) and not pid_alive(True) and not pid_alive(-1)
    assert not pid_alive(999999999)


def test_queues_handle_broken_and_partial_records(tmp_path):
    write_json(tmp_path / "queue" / "broken" / "state.json", {"jobs": [None, {"state": []}]})
    row = list_queues(tmp_path, now=NOW)[0]
    assert row["state"] == "unknown" and row["pending"] == 0 and row["warnings"]
    write_json(tmp_path / "queue" / "broken" / "state.json", {"state": "interrupted"})
    assert not list_queues(tmp_path, now=NOW)[0]["stale"]
    (tmp_path / "queue" / "broken" / "state.json").write_text("{", encoding="utf-8")
    assert list_queues(tmp_path, now=NOW)[0]["warnings"]


def test_app_renders_synthetic_paths_without_exception(tmp_path):
    from streamlit.testing.v1 import AppTest

    artifacts, models = tmp_path / "artifacts", tmp_path / "models"
    directory = write_run(artifacts, state="completed")
    write_run(artifacts, "active", started_at="2026-11-04T00:00:00Z")
    write_json(
        artifacts / "queue" / "nightly" / "state.json",
        {"state": "running", "jobs": [{"run_id": "active", "state": "running"}]},
    )
    (directory / "curves.csv").write_text(
        "timesteps,reward,length,safeguard\n100,0.1,100,0\n", encoding="utf-8"
    )
    write_json(
        directory / "evaluation.json",
        {
            "periods": [
                {
                    "label": "테스트",
                    "rows": [
                        {
                            "strategy": "equal_weight",
                            "cumulative_return": 0.06,
                            "mdd": 0.05,
                            "sharpe": 1,
                        },
                        {
                            "strategy": "best",
                            "cumulative_return": 0.08,
                            "mdd": 0.04,
                            "sharpe": 1.2,
                        },
                    ],
                }
            ]
        },
    )
    arguments = ["--artifacts-dir", str(artifacts), "--models-dir", str(models), "--refresh", "10"]
    app = AppTest.from_string(
        f"from robo_advisor.ops.monitor_app import main\nmain({arguments!r})\n"
    ).run(timeout=20)
    assert not app.exception
    assert app.selectbox[0].value == "current"
    assert app.title[0].value == "학습 모니터"
    app.run(timeout=20)
    assert not app.exception
    assert Path(directory / "manifest.json").exists()
