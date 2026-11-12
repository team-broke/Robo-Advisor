"""SHAP의 설명 대상과 실제 환경 입출력 계약을 검증한다."""

import numpy as np
import pandas as pd
import pytest
import torch
from stable_baselines3 import PPO

from robo_advisor.rl.environment import PortfolioEnv


@pytest.fixture
def environment():
    index = pd.date_range("2026-01-01", periods=6, tz="UTC")
    returns = pd.DataFrame([[0.01, 0.02]] * 6, index=index, columns=["A", "B"])
    env = PortfolioEnv(returns, returns * 0 + 50, returns * 0 + 0.03, lookback=2, n_assets=2)
    yield env
    env.close()


@pytest.fixture
def policy(environment):
    # A real Gaussian PPO actor, with hand-controlled output (no training needed).
    model = PPO(
        "MlpPolicy",
        environment,
        seed=17,
        device="cpu",
        n_steps=4,
        batch_size=4,
        policy_kwargs={"net_arch": {"pi": [], "vf": []}},
    )
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.weight[0, 0] = 1
        model.policy.action_net.bias.copy_(torch.tensor([25.0, 0.0, -25.0]))
    return model


def layout():
    from robo_advisor.explain.policy import ObservationLayout

    return ObservationLayout(("A", "B"), lookback=2)


def wrapper(policy):
    from robo_advisor.explain.policy import PPOPolicyWrapper

    return PPOPolicyWrapper(policy, layout())


def test_layout_roundtrip_matches_real_environment(environment):
    observation, info = environment.reset(seed=1)
    schema = layout()
    frame = schema.to_frame(observation, index=pd.DatetimeIndex([info["timestamp"]]))
    assert list(frame.columns) == [
        "simple_return_lag_1__A",
        "simple_return_lag_1__B",
        "simple_return_lag_0__A",
        "simple_return_lag_0__B",
        "weight__A",
        "weight__B",
        "weight__CASH",
        "rsi_fraction__A",
        "rsi_fraction__B",
        "macd_price_normalized__A",
        "macd_price_normalized__B",
        "risk_score__A",
        "risk_score__B",
    ]
    assert schema.feature_groups == (
        ("returns",) * 4 + ("weights",) * 3 + ("rsi",) * 2 + ("macd",) * 2 + ("risk",) * 2
    )
    np.testing.assert_array_equal(schema.as_batch(frame), observation[None, :])
    assert frame.iloc[0, 6] == 1  # initial cash
    assert frame.iloc[0, 7] == 0.5  # environment already divided RSI by 100


def test_raw_mean_is_not_clipped_or_sampled_and_preserves_mode(policy):
    wrapped = wrapper(policy)
    values = np.zeros((2, 13))
    values[1, 0] = 0.5
    policy.policy.set_training_mode(True)
    rng = torch.random.get_rng_state().clone()
    first = wrapped.predict_mean_actions(values)
    np.testing.assert_array_equal(first, [[25, 0, -25], [25.5, 0, -25]])
    np.testing.assert_array_equal(wrapped.predict_mean_actions(values), first)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert policy.policy.training
    np.testing.assert_array_equal(wrapped.for_asset("A")(values), [25, 25.5])
    np.testing.assert_array_equal(wrapped.for_asset("CASH")(values), [-25, -25])
    assert wrapped.output_names == ("A", "B", "CASH")
    with pytest.raises(ValueError, match="asset"):
        wrapped.for_asset("UNKNOWN")


def test_target_weights_match_clipped_deterministic_execution(policy, environment):
    wrapped = wrapper(policy)
    observation, _ = environment.reset(seed=1)
    action, _ = policy.predict(observation, deterministic=True)
    np.testing.assert_array_equal(action, [20, 0, -20])
    _, _, _, _, info = environment.step(action)
    actual = wrapped.predict_target_weights(observation)
    np.testing.assert_allclose(actual[0], info["target_weights"], rtol=1e-12)
    assert actual[0, 0] == pytest.approx(0.9999999979388463)
    assert actual.sum() == pytest.approx(1)


def test_saved_policy_and_frame_batch_preserve_output(policy, tmp_path):
    wrapped = wrapper(policy)
    values = np.zeros((2, 13))
    values[1, 0] = -0.5
    frame = layout().to_frame(values)
    path = tmp_path / "policy.zip"
    policy.save(path)
    restored = PPO.load(path, device="cpu")
    np.testing.assert_array_equal(
        wrapper(restored).predict_mean_actions(frame), wrapped.predict_mean_actions(values)
    )
    np.testing.assert_array_equal(
        wrapped.predict_mean_actions(values[:1]), wrapped.predict_mean_actions(values[0])
    )


@pytest.mark.parametrize(
    "values",
    [
        np.zeros(12),
        np.zeros((1, 14)),
        np.zeros((1, 1, 13)),
        np.empty((0, 13)),
        np.full((1, 13), np.nan),
        np.full((1, 13), np.inf),
        np.ones((1, 13), dtype=complex),
        np.ones((1, 13), dtype=bool),
        np.full((1, 13), "1"),
    ],
)
def test_invalid_observation_is_rejected(policy, values):
    with pytest.raises((TypeError, ValueError)):
        wrapper(policy).predict_mean_actions(values)


def test_reordered_features_and_wrong_model_layout_are_rejected(policy):
    from robo_advisor.explain.policy import ObservationLayout, PPOPolicyWrapper

    frame = layout().to_frame(np.zeros(13))
    with pytest.raises(ValueError, match="columns"):
        wrapper(policy).predict_mean_actions(frame.iloc[:, ::-1])
    with pytest.raises(ValueError, match="space"):
        PPOPolicyWrapper(policy, ObservationLayout(("A",), lookback=2))


@pytest.mark.parametrize(
    "assets,lookback",
    [((), 2), (("A", "A"), 2), (("CASH",), 2), (("A",), 0), (("A",), True), ("AB", 2)],
)
def test_ambiguous_layout_is_rejected(assets, lookback):
    from robo_advisor.explain.policy import ObservationLayout

    with pytest.raises((TypeError, ValueError)):
        ObservationLayout(assets, lookback)


def test_kernel_shap_accepts_scalar_asset_callable(policy):
    import shap

    wrapped = wrapper(policy)
    background = np.zeros((1, 13))
    observation = background.copy()
    observation[0, 0] = 0.5
    explainer = shap.KernelExplainer(wrapped.for_asset("A"), background)
    values = explainer.shap_values(observation, silent=True)
    # Only the first input varies; the hand-controlled actor has slope 1.
    np.testing.assert_allclose(values, [[0.5] + [0] * 12], atol=1e-8)
    assert float(explainer.expected_value) == 25


def test_state_dependent_noise_and_vec_normalization_are_rejected(environment):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from robo_advisor.explain.policy import PPOPolicyWrapper

    sde = PPO("MlpPolicy", environment, use_sde=True, n_steps=4, batch_size=4)
    with pytest.raises(ValueError, match="Gaussian"):
        PPOPolicyWrapper(sde, layout())
    normalized = VecNormalize(DummyVecEnv([lambda: environment]))
    try:
        model = PPO("MlpPolicy", normalized, n_steps=4, batch_size=4)
        with pytest.raises(ValueError, match="VecNormalize"):
            PPOPolicyWrapper(model, layout())
    finally:
        normalized.close()


def test_named_series_checks_feature_order(policy):
    frame = layout().to_frame(np.arange(13, dtype=float))
    np.testing.assert_array_equal(
        wrapper(policy).predict_mean_actions(frame.iloc[0]),
        wrapper(policy).predict_mean_actions(frame),
    )
    with pytest.raises(ValueError, match="columns|index"):
        wrapper(policy).predict_mean_actions(frame.iloc[0].iloc[::-1])


def test_canonical_training_replay_ignores_random_starts_and_episode_length(policy):
    from robo_advisor.explain.background import collect_training_observations
    from robo_advisor.rl.training import EnvironmentConfig, TrainingDataset

    dates = pd.date_range("2026-01-01", periods=10, tz="UTC")
    returns = pd.DataFrame(0.001, index=dates, columns=["A", "B"])
    data = TrainingDataset(returns, returns * 0 + 50, returns * 0, (), {})
    config = EnvironmentConfig(
        lookback=2, random_start=True, min_episode_steps=60, max_episode_steps=252
    )
    kwargs = dict(layout=layout(), train_start=dates[1], train_end=dates[7])
    actual = collect_training_observations(policy, data, config, **kwargs)
    assert actual.index.equals(dates[2:7])
    assert actual.iloc[0]["weight__CASH"] == 1
    assert actual.iloc[1]["weight__A"] > 0.99
    polluted = returns.copy()
    polluted.iloc[8:] = np.nan
    other = TrainingDataset(polluted, data.rsi, data.macd, (), {})
    pd.testing.assert_frame_equal(
        actual, collect_training_observations(policy, other, config, **kwargs)
    )
    pd.testing.assert_frame_equal(
        actual, collect_training_observations(policy, data, config, **kwargs)
    )


def test_canonical_training_replay_rejects_early_guard_and_asset_order(policy):
    from robo_advisor.explain.background import collect_training_observations
    from robo_advisor.rl.training import EnvironmentConfig, TrainingDataset

    dates = pd.date_range("2026-01-01", periods=10, tz="UTC")
    returns = pd.DataFrame(0.001, index=dates, columns=["A", "B"])
    returns.iloc[4] = -0.6
    data = TrainingDataset(returns, returns * 0 + 50, returns * 0, (), {})
    kwargs = dict(layout=layout(), train_start=dates[0], train_end=dates[-1])
    with pytest.raises(ValueError, match="ended early.*mdd"):
        collect_training_observations(policy, data, EnvironmentConfig(lookback=2), **kwargs)
    wrong = TrainingDataset(
        returns.iloc[:, ::-1], data.rsi.iloc[:, ::-1], data.macd.iloc[:, ::-1], (), {}
    )
    with pytest.raises(ValueError, match="asset order"):
        collect_training_observations(policy, wrong, EnvironmentConfig(lookback=2), **kwargs)
