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
    assert info["start_index"] == env.lookback - 1


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


def naive_scores(tags, assets, at):
    scores = np.zeros(len(assets))
    for i, asset in enumerate(assets):
        available = [t for t in tags if t.asset == asset and t.available_at <= at]
        if available:
            latest = max(t.available_at for t in available)
            scores[i] = max(t.risk_score for t in available if t.available_at == latest)
    return scores


@pytest.mark.leakage
def test_precomputed_risk_scores_match_scanning_every_tag():
    rng = np.random.default_rng(7)
    index = frames()[0].index
    offsets = pd.to_timedelta(rng.integers(-36, 7 * 24, size=40), unit="h")
    tags = [
        tag((index[0] + offset).to_pydatetime(), score=round(float(score), 1), asset=asset)
        for offset, score, asset in zip(
            offsets, rng.uniform(0, 1, size=40), rng.choice(["A", "B"], size=40)
        )
    ]
    tags += [tag(index[2], 0.1), tag(index[2], 0.9)]
    env = make_env(risk_tags=tags)
    obs, info = env.reset()
    observed = [(obs, info["timestamp"])]
    truncated = False
    while not truncated:
        obs, _, _, truncated, info = env.step([0, 0, 0])
        observed.append((obs, info["timestamp"]))
    for obs, at in observed:
        np.testing.assert_allclose(obs[-2:], naive_scores(tags, ["A", "B"], at))


def test_random_start_seed_reproducibility_diversity_and_bounds():
    env = make_env(random_start=True, min_episode_steps=2)
    starts = set()
    for seed in range(30):
        observation, info = env.reset(seed=seed)
        repeated, repeated_info = env.reset(seed=seed)
        assert info["start_index"] == repeated_info["start_index"]
        np.testing.assert_array_equal(observation, repeated)
        cursor = info["start_index"]
        assert env.lookback - 1 <= cursor <= len(frames()[0]) - 1 - 2
        assert info["timestamp"] == frames()[0].index[cursor]
        assert env.observation_space.contains(observation)
        starts.add(cursor)
    assert len(starts) >= 2


def test_random_start_sequence_is_reproducible_without_reseeding_each_episode():
    first = make_env(random_start=True)
    second = make_env(random_start=True)
    first.reset(seed=17)
    second.reset(seed=17)
    left = [first.reset()[1]["start_index"] for _ in range(20)]
    right = [second.reset()[1]["start_index"] for _ in range(20)]
    assert left == right
    assert len(set(left)) >= 2


def test_random_start_reserves_available_minimum_steps():
    env = make_env(random_start=True, min_episode_steps=3)
    for seed in range(10):
        _, info = env.reset(seed=seed)
        remaining = len(frames()[0]) - 1 - info["start_index"]
        count = 0
        truncated = False
        while not truncated:
            _, _, terminated, truncated, info = env.step([0, 0, 0])
            count += 1
            assert not terminated
        assert count == remaining
        assert count >= 3
        assert info["reason"] == "data_exhausted"
    boundary = make_env(random_start=True, min_episode_steps=4)
    assert boundary.reset(seed=7)[1]["start_index"] == boundary.lookback - 1


@pytest.mark.parametrize("random_start", [False, True])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, 5])
def test_invalid_minimum_episode_steps(random_start, value):
    with pytest.raises(ValueError):
        make_env(random_start=random_start, min_episode_steps=value)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_maximum_episode_steps(value):
    with pytest.raises(ValueError):
        make_env(max_episode_steps=value)


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_random_start_requires_boolean(value):
    with pytest.raises(ValueError):
        make_env(random_start=value)


def test_episode_limit_truncates_and_reset_clears_step_count():
    env = make_env(max_episode_steps=2)
    env.reset()
    _, _, terminated, truncated, info = env.step([0, 0, 0])
    assert not terminated and not truncated
    assert info["reason"] is None
    assert info["episode_step"] == 1
    _, _, terminated, truncated, info = env.step([0, 0, 0])
    assert not terminated and truncated
    assert info["reason"] == "episode_limit"
    assert info["episode_step"] == 2
    with pytest.raises(gym.error.ResetNeeded):
        env.step([0, 0, 0])
    env.reset()
    assert env.step([0, 0, 0])[4]["episode_step"] == 1


@pytest.mark.parametrize("limit", [4, 10])
def test_data_exhaustion_precedes_equal_or_longer_episode_limit(limit):
    env = make_env(max_episode_steps=limit)
    env.reset()
    for _ in range(4):
        _, _, terminated, truncated, info = env.step([0, 0, 0])
    assert not terminated and truncated
    assert info["reason"] == "data_exhausted"
    assert info["episode_step"] == 4


@pytest.mark.parametrize("final_row", [False, True])
def test_mdd_precedes_episode_limit_and_data_exhaustion(final_row):
    returns, rsi, macd = frames()
    returns.iloc[2] = -0.6
    if final_row:
        returns, rsi, macd = (frame.iloc[:3] for frame in (returns, rsi, macd))
    env = PortfolioEnv(
        returns, rsi, macd, lookback=2, n_assets=2, max_episode_steps=1, mdd_limit=0.15
    )
    env.reset()
    _, _, terminated, truncated, info = env.step([0, 0, 0])
    assert terminated and not truncated
    assert info["reason"] == "mdd_limit_exceeded"
    assert info["episode_step"] == 1


@pytest.mark.leakage
def test_random_reset_observation_does_not_use_rows_after_start():
    original = frames()
    first = PortfolioEnv(*original, lookback=2, n_assets=2, random_start=True, min_episode_steps=2)
    expected, info = first.reset(seed=11)
    cursor = info["start_index"]
    changed = tuple(frame.copy() for frame in original)
    changed[0].iloc[cursor + 1 :] = 0.9
    changed[1].iloc[cursor + 1 :] = 99
    changed[2].iloc[cursor + 1 :] = 10
    future_tag = tag(original[0].index[cursor + 1])
    second = PortfolioEnv(
        *changed,
        lookback=2,
        n_assets=2,
        random_start=True,
        min_episode_steps=2,
        risk_tags=[future_tag],
    )
    actual, changed_info = second.reset(seed=11)
    assert changed_info["start_index"] == cursor
    np.testing.assert_array_equal(actual, expected)
    expected_history = original[0].iloc[cursor - 1 : cursor + 1].values.ravel()
    np.testing.assert_array_equal(expected[:4], expected_history)


def test_random_start_checker():
    check_env(make_env(random_start=True, min_episode_steps=2), skip_render_check=True)


def test_random_reset_clears_accounting_safeguard_and_reward_history():
    options = dict(
        random_start=True,
        min_episode_steps=3,
        max_episode_steps=2,
        initial_nav=100,
        initial_weights=[0.2, 0.3, 0.5],
        reward_variant="sharpe",
    )
    env = make_env(**options)
    fresh = make_env(**options)
    env.reset(seed=4)
    env.step([1, 0, -1])
    env.step([-1, 1, 0])
    actual, info = env.reset(seed=4)
    expected, fresh_info = fresh.reset(seed=4)
    np.testing.assert_array_equal(actual, expected)
    assert info["nav"] == fresh_info["nav"] == 100
    np.testing.assert_array_equal(info["weights"], [0.2, 0.3, 0.5])
    left, right = env.step([1, 0, -1]), fresh.step([1, 0, -1])
    np.testing.assert_array_equal(left[0], right[0])
    assert left[1:4] == right[1:4]
    assert left[1] == 0
    assert left[4]["safeguard"] == right[4]["safeguard"]
    assert left[4]["safeguard"]["step_index"] == 0
    assert left[4]["termination_result"] is None
