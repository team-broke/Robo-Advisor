"""학습 진행률·단계 로그와 로그 파일 검증."""

import logging

import numpy as np
import pandas as pd
import pytest

from robo_advisor.config import Settings
from robo_advisor.rl import progress, train, training


def write_dataset(directory, periods=24):
    directory.mkdir(parents=True, exist_ok=True)
    index = pd.date_range("2026-01-01", periods=periods, tz="UTC")
    values = np.random.default_rng(5).normal(0, 0.002, (periods, 2))
    returns = pd.DataFrame(values, index=index, columns=["A", "B"])
    for name, frame in (("returns", returns), ("rsi", returns * 0 + 50), ("macd", returns * 0)):
        frame.to_parquet(directory / f"{name}.parquet")


def payload(directory, validation=False):
    raw = {
        "device": "cpu",
        "total_timesteps": 64,
        "n_envs": 2,
        "vec_env": "dummy",
        "ppo": {"n_steps": 16, "batch_size": 16, "n_epochs": 1, "net_arch": {"pi": [8], "vf": [8]}},
        "env": {"lookback": 2, "mdd_limit": 1},
        "data": {"dataset_dir": str(directory), "train_end": "2026-01-24T00:00:00Z"},
        "checkpoint": {"save_freq": 32},
    }
    if validation:
        raw["validation"] = {"start": "2026-01-18T00:00:00Z", "eval_freq": 32}
    return raw


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    write_dataset(tmp_path / "data")
    settings = Settings(
        _env_file=None, models_dir=tmp_path / "models", artifacts_dir=tmp_path / "artifacts"
    )
    monkeypatch.setattr(training, "get_settings", lambda: settings)
    return tmp_path


def config(root, **kwargs):
    return training.TrainingConfig.model_validate(
        payload(root / "data", **kwargs), context={"smoke": True}
    )


def test_format_duration():
    assert progress.format_duration(0) == "0:00:00"
    assert progress.format_duration(3725.9) == "1:02:05"
    assert progress.format_duration(-3) == "0:00:00"


def test_summary_reports_rate_eta_and_recent_episodes():
    callback = progress.ProgressCallback(1000, interval_seconds=5)
    callback.started = 0.0
    callback.num_timesteps = 500
    callback.episodes.extend(
        [
            {"r": 1.0, "l": 10, "reason": "mdd_limit_exceeded"},
            {"r": 3.0, "l": 30, "reason": "data_exhausted"},
        ]
    )
    callback.episode_count = 7
    text = callback.summary(10.0)
    assert "50.0%" in text and "500/1,000" in text and "50 스텝/초" in text
    assert "남은 시간 0:00:10" in text
    assert "평균 보상 2.0000" in text and "평균 길이 20" in text
    assert "Safe-Guard 종료 50%" in text and "누적 7개" in text


def test_interval_must_be_positive():
    with pytest.raises(ValueError):
        progress.ProgressCallback(10, interval_seconds=0)


def test_training_logs_phases_progress_and_file(workspace, caplog):
    caplog.set_level(logging.INFO, logger="robo_advisor.training")
    result = training.run_training(
        config(workspace, validation=True), run_id="logged", smoke=True, log_interval=1e-6
    )
    messages = [record.getMessage() for record in caplog.records]
    joined = "\n".join(messages)
    for expected in (
        "실행 logged: 장치 cpu",
        "학습 데이터:",
        "검증 구간:",
        "환경 2개와 PPO 모델 구성 중",
        "학습 시작: 목표 64 스텝",
        "진행 ",
        "체크포인트 저장: 32 스텝",
        "검증 평가 32 스텝: sharpe=",
        "최종 모델 저장:",
        "학습곡선: 수렴 판정",
        "종료(completed)",
    ):
        assert expected in joined, expected
    assert any("100.0%" in message for message in messages)
    log_text = (result.manifest_path.parent / "train.log").read_text(encoding="utf-8")
    assert "학습 시작: 목표 64 스텝" in log_text and "종료(completed)" in log_text
    assert not any(isinstance(handler, logging.FileHandler) for handler in progress.LOGGER.handlers)


def test_failure_is_logged_and_file_handler_removed(workspace, caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger="robo_advisor.training")

    def fail(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(training.PPO, "learn", fail)
    with pytest.raises(RuntimeError):
        training.run_training(config(workspace), run_id="broken", smoke=True)
    joined = "\n".join(record.getMessage() for record in caplog.records)
    assert "학습 실패: RuntimeError: boom" in joined and "종료(failed)" in joined
    assert not any(isinstance(handler, logging.FileHandler) for handler in progress.LOGGER.handlers)


def test_cli_prints_progress_and_quiet_hides_info(workspace, capsys, tmp_path):
    import yaml

    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(payload(workspace / "data")))
    train.main(["--config", str(path), "--smoke", "--run-id", "cli-log", "--log-interval", "0.5"])
    out = capsys.readouterr().out
    assert "학습 시작: 목표 64 스텝" in out and out.strip().endswith("manifest.json")
    train.main(["--config", str(path), "--smoke", "--run-id", "cli-quiet", "--quiet"])
    quiet = capsys.readouterr().out
    assert "학습 시작" not in quiet and quiet.strip().endswith("manifest.json")
    assert not any(
        isinstance(handler, logging.StreamHandler)
        and handler.stream is not None
        and getattr(handler.stream, "name", "") == "<stdout>"
        for handler in progress.LOGGER.handlers
    )
    with pytest.raises(SystemExit):
        train.main(["--config", str(path), "--log-interval", "0"])
