"""정책 롤아웃의 NAV·수익률 정합성과 조기 종료 검증."""

import numpy as np
import pandas as pd
import pytest

from robo_advisor.rl.evaluation import evaluate_policy
from robo_advisor.rl.training import (
    EnvironmentConfig,
    TrainingConfig,
    TrainingDataset,
    _make_env,
    load_training_dataset,
    run_training,
)
from robo_advisor.config import Settings
from robo_advisor.rl import training


class EqualPolicy:
    def predict(self, observation, deterministic=False):
        assert deterministic
        return np.zeros(3), None


def dataset(loss=False):
    index = pd.date_range("2026-01-01", periods=8, tz="UTC")
    returns = pd.DataFrame(0.01, index=index, columns=["A", "B"])
    if loss:
        returns.iloc[2] = -0.6
    return TrainingDataset(returns, returns * 0 + 50, returns * 0, (), {})


def test_rollout_matches_each_environment_nav_and_net_simple_return():
    data = dataset()
    config = EnvironmentConfig(lookback=2, mdd_limit=1, commission_rate=0.01, slippage_rate=0.02)
    result = evaluate_policy(EqualPolicy(), data, config)
    env = _make_env(data, config, seed=0)
    try:
        _, info = env.reset(seed=0)
        previous = info["nav"]
        for date, value in result.returns.items():
            _, _, _, _, info = env.step(np.zeros(3))
            assert date == info["timestamp"]
            assert result.nav.loc[date] == pytest.approx(info["nav"])
            assert value == pytest.approx(info["nav"] / previous - 1)
            assert np.log1p(value) == pytest.approx(info["portfolio_log_return"])
            previous = info["nav"]
    finally:
        env.close()
    assert result.nav.iloc[0] == 1
    assert len(result.nav) == result.observation_count + 1
    assert result.observation_count == 6
    assert result.returns.index.equals(data.returns.index[2:])
    assert result.truncated and not result.terminated
    assert result.reason == "data_exhausted"
    assert result.returns.iloc[0] < 0


def test_mdd_termination_does_not_fill_remaining_dates():
    result = evaluate_policy(EqualPolicy(), dataset(loss=True), EnvironmentConfig(lookback=2))
    assert result.terminated and not result.truncated
    assert result.reason == "mdd_limit_exceeded"
    assert result.observation_count == 1
    assert len(result.nav) == 2
    assert result.returns.index[-1] == dataset().returns.index[2]


def test_rollout_closes_environment_after_prediction_failure(monkeypatch):
    from robo_advisor.rl import evaluation

    env = _make_env(dataset(), EnvironmentConfig(lookback=2), seed=0)
    closed = []
    original_close = env.close

    def close():
        closed.append(True)
        original_close()

    class BrokenPolicy:
        def predict(self, *args, **kwargs):
            raise RuntimeError("prediction failed")

    monkeypatch.setattr(env, "close", close)
    monkeypatch.setattr(evaluation, "_make_env", lambda *args, **kwargs: env)
    with pytest.raises(RuntimeError, match="prediction failed"):
        evaluate_policy(BrokenPolicy(), dataset(), EnvironmentConfig(lookback=2))
    assert closed == [True]


def test_smoke_trained_policy_rollout(tmp_path, monkeypatch):
    data = dataset()
    directory = tmp_path / "data"
    directory.mkdir()
    for name in ("returns", "rsi", "macd"):
        getattr(data, name).to_parquet(directory / f"{name}.parquet")
    settings = Settings(
        _env_file=None,
        models_dir=tmp_path / "models",
        artifacts_dir=tmp_path / "artifacts",
    )
    monkeypatch.setattr(training, "get_settings", lambda: settings)
    config = TrainingConfig.model_validate(
        {
            "device": "cpu",
            "vec_env": "dummy",
            "n_envs": 2,
            "total_timesteps": 64,
            "ppo": {
                "n_steps": 16,
                "batch_size": 16,
                "n_epochs": 1,
                "net_arch": {"pi": [16], "vf": [16]},
            },
            "env": {"lookback": 2, "mdd_limit": 1},
            "data": {"dataset_dir": directory, "train_end": data.returns.index[-1]},
        },
        context={"smoke": True},
    )
    result = run_training(config, run_id="evaluation-smoke", smoke=True)
    loaded = load_training_dataset(config.data)
    first = evaluate_policy(result.model, loaded, config.env)
    second = evaluate_policy(result.model, loaded, config.env)
    pd.testing.assert_series_equal(first.returns, second.returns, check_exact=True)
    pd.testing.assert_series_equal(first.nav, second.nav, check_exact=True)


def test_evaluation_ignores_random_start_and_episode_limit_without_mutating_config():
    data = dataset()
    fixed = EnvironmentConfig(lookback=2, mdd_limit=1)
    random = EnvironmentConfig(
        lookback=2,
        mdd_limit=1,
        random_start=True,
        min_episode_steps=6,
        max_episode_steps=1,
    )
    before = random.model_dump()
    expected = evaluate_policy(EqualPolicy(), data, fixed)
    actual = evaluate_policy(EqualPolicy(), data, random)
    pd.testing.assert_series_equal(actual.returns, expected.returns, check_exact=True)
    pd.testing.assert_series_equal(actual.nav, expected.nav, check_exact=True)
    assert actual.nav.index[0] == data.returns.index[fixed.lookback - 1]
    assert actual.returns.index[-1] == data.returns.index[-1]
    assert actual.observation_count == 6
    assert actual.truncated and not actual.terminated
    assert actual.reason == "data_exhausted"
    assert random.model_dump() == before


def test_evaluation_resets_training_minimum_even_for_shorter_evaluation_data():
    config = EnvironmentConfig(
        lookback=2, mdd_limit=1, random_start=True, min_episode_steps=60, max_episode_steps=1
    )
    result = evaluate_policy(EqualPolicy(), dataset(), config)
    assert result.observation_count == 6
    assert result.reason == "data_exhausted"


def test_evaluation_preserves_mdd_termination_when_training_options_are_disabled():
    config = EnvironmentConfig(lookback=2, random_start=True, max_episode_steps=1)
    result = evaluate_policy(EqualPolicy(), dataset(loss=True), config)
    assert result.terminated and not result.truncated
    assert result.reason == "mdd_limit_exceeded"
    assert result.observation_count == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("random_start", 1),
        ("random_start", "true"),
        ("min_episode_steps", 0),
        ("min_episode_steps", True),
        ("min_episode_steps", 1.5),
        ("max_episode_steps", 0),
        ("max_episode_steps", True),
        ("max_episode_steps", 1.5),
    ],
)
def test_environment_config_episode_option_validation(field, value):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        EnvironmentConfig(**{field: value})


def test_environment_config_episode_defaults_and_valid_values():
    defaults = EnvironmentConfig()
    assert defaults.random_start is False
    assert defaults.min_episode_steps == 1
    assert defaults.max_episode_steps is None
    configured = EnvironmentConfig(random_start=True, min_episode_steps=60, max_episode_steps=252)
    assert configured.random_start
    assert configured.min_episode_steps == 60
    assert configured.max_episode_steps == 252
