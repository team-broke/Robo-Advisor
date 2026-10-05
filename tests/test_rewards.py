"""동일 궤적의 손계산 수식과 초기·경계 구간 보상을 확인한다."""

import numpy as np
import pytest

from robo_advisor.rl.rewards import RewardVariant, calculate_reward


def test_three_formulas_on_the_same_trajectory():
    # NAV 100 → 110 → 99 → 108.9의 로그수익률과 현재 고점 대비 낙폭.
    up = np.log(1.1)
    down = np.log(0.9)
    returns = [up, down, up]
    drawdowns = [0, 0.1, 0.01]
    sigma = (up - down) / 2

    for step, (value, drawdown) in enumerate(zip(returns, drawdowns)):
        inputs = dict(return_history=returns[:step], drawdown=drawdown)
        assert calculate_reward(value, "return", **inputs) == pytest.approx(value)
        expected_sharpe = 0 if step < 2 else value / sigma
        assert calculate_reward(value, "sharpe", **inputs) == pytest.approx(expected_sharpe)
        assert calculate_reward(value, "mdd_penalty", **inputs) == pytest.approx(value - drawdown)


def test_custom_lambda_and_zero_lambda():
    assert calculate_reward(0.02, "mdd_penalty", drawdown=0.1, penalty_lambda=2) == pytest.approx(
        -0.18
    )
    assert calculate_reward(0.02, "mdd_penalty", drawdown=0.1, penalty_lambda=0) == 0.02


@pytest.mark.parametrize("history", [[], [0.1]])
def test_initial_sharpe_reward_is_zero(history):
    assert calculate_reward(0.02, "sharpe", return_history=history) == 0


@pytest.mark.parametrize("value", [0.0, 0.02, -0.02])
def test_zero_volatility_uses_epsilon(value):
    reward = calculate_reward(value, "sharpe", return_history=[0.01] * 20)

    assert reward == pytest.approx(value / 1e-8)
    assert np.isfinite(reward)


def test_volatility_below_epsilon_uses_floor():
    assert calculate_reward(1e-9, "sharpe", return_history=[0, 1e-10]) == pytest.approx(0.1)


def test_only_previous_twenty_returns_are_used_without_annualization():
    history = [100.0] + [-0.01, 0.03] * 10

    assert calculate_reward(0.01, "sharpe", return_history=history) == pytest.approx(0.5)


def test_initial_other_variants_follow_their_formulas():
    assert calculate_reward(-0.01) == -0.01
    assert calculate_reward(-0.01, RewardVariant.MDD_PENALTY) == -0.01


@pytest.mark.parametrize("variant", list(RewardVariant))
def test_valid_rewards_are_finite_and_history_is_not_changed(variant):
    history = np.array([-0.02, 0.01, 0.03])
    original = history.copy()

    reward = calculate_reward(-0.005, variant, return_history=history, drawdown=0.15)

    assert np.isfinite(reward)
    np.testing.assert_array_equal(history, original)


def test_large_finite_history_does_not_overflow_volatility():
    assert calculate_reward(1, "sharpe", return_history=[-1e308, 1e308]) == pytest.approx(
        1e-308, abs=0
    )


@pytest.mark.parametrize("variant", list(RewardVariant))
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_returns_rejected(variant, value):
    with pytest.raises(ValueError):
        calculate_reward(value, variant)


@pytest.mark.parametrize(
    "overrides",
    [
        {"variant": "unknown"},
        {"drawdown": -0.1},
        {"drawdown": 1.01},
        {"drawdown": np.nan},
        {"drawdown": np.inf},
        {"penalty_lambda": -1},
        {"penalty_lambda": np.nan},
        {"penalty_lambda": np.inf},
        {"return_history": [np.nan]},
        {"return_history": [np.inf]},
        {"return_history": [[0, 1]]},
    ],
)
def test_invalid_inputs_rejected(overrides):
    with pytest.raises(ValueError):
        calculate_reward(0.01, **overrides)


def test_nonfinite_reward_rejected():
    with pytest.raises(ValueError):
        calculate_reward(1e308, "sharpe", return_history=[0, 0])
