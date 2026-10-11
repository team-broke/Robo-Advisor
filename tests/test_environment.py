"""Environment timing, accounting, termination and observation contracts."""

import gymnasium as gym
import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from robo_advisor.rl.environment import PortfolioEnv
from robo_advisor.schemas.risk import RiskTag


def frames():
    index = pd.date_range("2026-01-01", periods=6, tz="UTC")
    returns = pd.DataFrame(
        [[0.01, -0.01], [0.02, 0.03], [0.1, -0.1], [-0.1, 0.1], [0.05, 0], [0, 0]],
        index=index,
        columns=["A", "B"],
    )
    return returns, returns * 0 + 50, returns * 0 + 0.01


def make_env(**kwargs):
    options = dict(lookback=2, n_assets=2, mdd_limit=1)
    options.update(kwargs)
    return PortfolioEnv(*frames(), **options)


def test_gymnasium_checker_and_observation_layout():
    env = make_env()
    check_env(env, skip_render_check=True)
    obs, info = env.reset(seed=3)
    assert obs.shape == (13,)
    assert obs.dtype == np.float64
    assert env.observation_space.contains(obs)
    np.testing.assert_array_equal(obs[:4], [0.01, -0.01, 0.02, 0.03])
    np.testing.assert_array_equal(obs[4:7], [0, 0, 1])
    np.testing.assert_array_equal(obs[7:9], [0.5, 0.5])
    np.testing.assert_array_equal(obs[9:11], [0.01, 0.01])
    np.testing.assert_array_equal(obs[-2:], [0, 0])
    assert info["timestamp"] == frames()[0].index[1]


@pytest.mark.parametrize("lookback,n_assets", [(1, 1), (3, 4), (20, 12)])
def test_configurable_observation_and_action_dimensions(lookback, n_assets):
    returns = pd.DataFrame(
        np.zeros((lookback + 2, n_assets)),
        index=pd.date_range("2026-01-01", periods=lookback + 2, tz="Asia/Seoul"),
        columns=[f"asset-{i}" for i in range(n_assets)],
    )
    env = PortfolioEnv(returns, returns + 50, returns, lookback=lookback, n_assets=n_assets)
    obs, info = env.reset()
    assert obs.shape == (n_assets * (lookback + 4) + 1,)
    assert env.action_space.shape == (n_assets + 1,)
    assert info["timestamp"] == returns.index[lookback - 1].tz_convert("UTC")
    assert env.observation_space.contains(env.step(np.zeros(n_assets + 1))[0])


@pytest.mark.parametrize("variant", ["return", "sharpe", "mdd_penalty"])
def test_hand_calculated_cost_drift_nav_and_rewards(variant):
    env = make_env(
        initial_nav=100,
        commission_rate=0.01,
        slippage_rate=0.02,
        reward_variant=variant,
        penalty_lambda=2,
    )
    env.reset()
    target = np.array([0.5, 0.25, 0.25])
    previous_weights = np.array([0, 0, 1])
    nav, peak, history = 100.0, 100.0, []
    for row in [2, 3, 4]:
        turnover = np.abs(target[:2] - previous_weights[:2]).sum()
        cost = turnover * 0.03 * nav
        holdings = (nav - cost) * target * np.r_[1 + frames()[0].iloc[row].values, 1]
        next_nav = holdings.sum()
        peak = max(peak, next_nav)
        drawdown = (peak - next_nav) / peak
        value = np.log(next_nav / nav)
        expected = value
        if variant == "sharpe":
            expected = 0 if len(history) < 2 else value / max(np.std(history), 1e-8)
        elif variant == "mdd_penalty":
            expected -= 2 * drawdown
        obs, reward, terminated, truncated, info = env.step(np.log(target))
        assert info["target_weights"] == pytest.approx(target)
        assert info["turnover"] == pytest.approx(turnover)
        assert info["cost"] == pytest.approx(cost)
        assert info["nav"] == pytest.approx(next_nav)
        assert info["portfolio_log_return"] == pytest.approx(value)
        assert info["safeguard"]["nav"] == pytest.approx(next_nav)
        assert info["safeguard"]["drawdown"] == pytest.approx(drawdown)
        assert reward == pytest.approx(expected)
        previous_weights = holdings / next_nav
        np.testing.assert_allclose(info["weights"], previous_weights)
        np.testing.assert_allclose(obs[4:7], previous_weights)
        assert not terminated and not truncated
        nav = next_nav
        history.append(value)


def test_exhaustion_and_step_requires_reset():
    env = make_env()
    with pytest.raises(gym.error.ResetNeeded):
        env.step([0, 0, 0])
    env.reset()
    for step in range(4):
        obs, _, terminated, truncated, info = env.step([0, 0, 0])
        assert not terminated
        assert truncated == (step == 3)
        assert env.observation_space.contains(obs)
    assert info["reason"] == "data_exhausted"
    with pytest.raises(gym.error.ResetNeeded):
        env.step([0, 0, 0])
    env.reset()
    assert not env.step([0, 0, 0])[3]


def test_mdd_termination_including_final_row_and_latched_result():
    returns, rsi, macd = frames()
    returns.iloc[2] = -0.6
    env = PortfolioEnv(returns, rsi, macd, lookback=2, n_assets=2, mdd_limit=0.15)
    env.reset()
    _, _, terminated, truncated, info = env.step([0, 0, 0])
    assert terminated and not truncated
    assert info["reason"] == "mdd_limit_exceeded"
    assert info["termination_result"]["step_index"] == 0
    with pytest.raises(gym.error.ResetNeeded):
        env.step([0, 0, 0])
    # A previously recorded result stays authoritative even when update returns False.
    env.reset()
    env._guard.update(0.5)
    env._guard.update(0.4)
    assert env.step([0, 0, 0])[2]
    env = PortfolioEnv(returns.iloc[:3], rsi.iloc[:3], macd.iloc[:3], lookback=2, n_assets=2)
    env.reset()
    assert env.step([0, 0, 0])[2:4] == (True, False)


def tag(available_at, score=0.8, asset="A"):
    return RiskTag(
        event_type="sharp_move",
        severity=score,
        confidence=0.5,
        asset=asset,
        published_at=available_at,
        available_at=available_at,
        source_url="https://example.com/event",
        trace_id="risk-test",
    )


@pytest.mark.leakage
def test_risk_availability_latest_ties_and_missing_slots():
    index = frames()[0].index
    tags = [tag(index[0]), tag(index[2], 0.2), tag(index[2], 0.6), tag(index[5], 1)]
    env = make_env(risk_tags=tags)
    obs, _ = env.reset()
    np.testing.assert_allclose(obs[-2:], [0.4, 0])
    obs = env.step([0, 0, 0])[0]
    np.testing.assert_allclose(obs[-2:], [0.3, 0])
    np.testing.assert_allclose(env.step([0, 0, 0])[0][-2:], [0.3, 0])


@pytest.mark.leakage
def test_published_tag_is_hidden_until_available():
    index = frames()[0].index
    delayed = tag(index[2]).model_copy(update={"published_at": index[0].to_pydatetime()})
    env = make_env(risk_tags=[delayed])
    assert env.reset()[0][-2] == 0
    assert env.step([0, 0, 0])[0][-2] == pytest.approx(0.4)


@pytest.mark.leakage
def test_future_changes_do_not_change_observation_or_current_transition():
    original = frames()
    changed = tuple(frame.copy() for frame in original)
    changed[0].iloc[3:] = 0.9
    changed[1].iloc[3:] = 99
    changed[2].iloc[3:] = 10
    first = PortfolioEnv(*original, lookback=2, n_assets=2)
    second = PortfolioEnv(*changed, lookback=2, n_assets=2, risk_tags=[tag(original[0].index[3])])
    np.testing.assert_array_equal(first.reset()[0], second.reset()[0])
    left, right = first.step([0, 0, 0]), second.step([0, 0, 0])
    np.testing.assert_array_equal(left[0], right[0])
    assert left[1:4] == right[1:4]
    assert left[4]["nav"] == right[4]["nav"]


def test_seed_reproducibility_reset_and_input_copies():
    env = make_env()
    trajectories = []
    for _ in range(2):
        obs, _ = env.reset(seed=42)
        env.action_space.seed(42)
        trajectory = [obs]
        for _ in range(4):
            trajectory.append(env.step(env.action_space.sample())[:4])
        trajectories.append(trajectory)
    np.testing.assert_array_equal(trajectories[0][0], trajectories[1][0])
    for left, right in zip(trajectories[0][1:], trajectories[1][1:]):
        np.testing.assert_array_equal(left[0], right[0])
        assert left[1:] == right[1:]
    inputs = frames()
    env = PortfolioEnv(*inputs, lookback=2, n_assets=2)
    expected, info = env.reset()
    inputs[0].iloc[:] = 0
    info["weights"][:] = 0
    np.testing.assert_array_equal(env.reset()[0], expected)


@pytest.mark.parametrize(
    "overrides",
    [
        {"lookback": 0},
        {"lookback": True},
        {"lookback": 6},
        {"lookback": 1.5},
        {"n_assets": 0},
        {"n_assets": 3},
        {"initial_nav": 0},
        {"initial_weights": [0.5, 0.5]},
        {"initial_weights": [1, 1, 0]},
        {"initial_weights": [-1, 1, 1]},
        {"commission_rate": -1},
        {"slippage_rate": np.inf},
        {"penalty_lambda": -1},
        {"mdd_limit": 0},
        {"reward_variant": "invalid"},
        {"risk_tags": [1]},
        {"risk_tags": [tag(frames()[0].index[0], asset="unknown")]},
    ],
)
def test_invalid_configuration(overrides):
    with pytest.raises(ValueError):
        make_env(**overrides)


@pytest.mark.parametrize(
    "case", ["nan", "loss", "rsi", "macd", "alignment", "columns", "naive", "order", "duplicate"]
)
def test_invalid_data(case):
    returns, rsi, macd = frames()
    if case == "nan":
        returns.iloc[0, 0] = np.nan
    elif case == "loss":
        returns.iloc[0, 0] = -1.1
    elif case == "rsi":
        rsi.iloc[1, 0] = 101
    elif case == "macd":
        macd.iloc[1, 0] = np.inf
    elif case == "alignment":
        rsi.index = rsi.index + pd.Timedelta(days=1)
    elif case == "columns":
        macd = macd.iloc[:, ::-1]
    else:
        for frame in (returns, rsi, macd):
            if case == "naive":
                frame.index = frame.index.tz_localize(None)
            elif case == "order":
                frame.index = frame.index[::-1]
            else:
                frame.index = pd.DatetimeIndex([frame.index[0]] * len(frame))
    with pytest.raises(ValueError):
        PortfolioEnv(returns, rsi, macd, lookback=2, n_assets=2)


@pytest.mark.leakage
def test_indicator_warmup_is_not_backfilled():
    returns, rsi, macd = frames()
    rsi.iloc[0] = np.nan
    macd.iloc[0] = np.nan
    env = PortfolioEnv(returns, rsi, macd, lookback=2, n_assets=2)
    assert np.isfinite(env.reset()[0]).all()
    rsi.iloc[1] = np.nan
    with pytest.raises(ValueError):
        PortfolioEnv(returns, rsi, macd, lookback=2, n_assets=2)


@pytest.mark.parametrize(
    "action", [[0, 0], [[0, 0, 0]], [0, np.nan, 0], [0, np.inf, 0], [21, 0, 0]]
)
def test_invalid_action_does_not_advance_state(action):
    env = make_env()
    env.reset()
    with pytest.raises(ValueError):
        env.step(action)
    assert env.step([0, 0, 0])[4]["timestamp"] == frames()[0].index[2]


def test_extreme_logits_and_custom_initial_weights():
    env = make_env(initial_weights=[0.2, 0.3, 0.5])
    np.testing.assert_allclose(env.reset()[0][4:7], [0.2, 0.3, 0.5])
    info = env.step([20, -20, -20])[4]
    assert info["target_weights"].sum() == pytest.approx(1)
    assert np.all(info["target_weights"] >= 0)
