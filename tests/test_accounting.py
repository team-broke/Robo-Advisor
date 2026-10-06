"""손계산 기준 사례로 거래비용·리밸런싱 순서를 확인한다."""

import numpy as np
import pytest

from robo_advisor.rl.accounting import (
    calculate_turnover,
    log_to_simple_returns,
    rebalance_step,
)


@pytest.mark.parametrize(
    "current,target,returns,turnover,cost,nav,holdings",
    [
        ([0.6, 0.3, 0.1], [0.6, 0.3, 0.1], [0.1, -0.1], 0, 0, 103, [66, 27, 10]),
        ([1, 0, 0], [0, 1, 0], [0, 0.1], 2, 0.13, 109.857, [0, 109.857, 0]),
        (
            [0.5, 0.3, 0.2],
            [0.4, 0.4, 0.2],
            [0.1, -0.05],
            0.2,
            0.013,
            101.98674,
            [43.99428, 37.99506, 19.9974],
        ),
    ],
)
def test_hand_calculated_steps(current, target, returns, turnover, cost, nav, holdings):
    result = rebalance_step(100, current, target, returns)

    assert result.turnover == pytest.approx(turnover)
    assert result.cost == pytest.approx(cost)
    assert result.nav == pytest.approx(nav)
    np.testing.assert_allclose(result.weights, np.array(holdings) / nav)
    assert result.weights.sum() == pytest.approx(1)
    assert np.all(result.weights >= 0)


def test_turnover_uses_drifted_weights_and_excludes_cash():
    first = rebalance_step(100, [0.5, 0.5], [0.5, 0.5], [1])
    second = rebalance_step(first.nav, first.weights, [0.5, 0.5], [0])

    assert first.weights == pytest.approx([2 / 3, 1 / 3])
    assert second.turnover == pytest.approx(1 / 6)
    assert second.cost == pytest.approx(150 * (1 / 6) * 0.00065)


def test_custom_rates_cash_and_input_preservation():
    current = np.array([0.0, 1.0])
    target = np.array([1.0, 0.0])
    returns = np.array([0.1])
    result = rebalance_step(100, current, target, returns, 0.01, 0.02)

    assert result.cost == pytest.approx(3)
    assert result.nav == pytest.approx(106.7)
    np.testing.assert_array_equal(current, [0, 1])
    np.testing.assert_array_equal(target, [1, 0])
    np.testing.assert_array_equal(returns, [0.1])
    cash = rebalance_step(100, [0, 1], [0, 1], [-1])
    assert cash.nav == 100
    np.testing.assert_array_equal(cash.weights, [0, 1])


@pytest.mark.parametrize(
    "weights",
    [[1], [], [[0.5, 0.5]], [0.4, 0.4], [-0.1, 1.1], [np.nan, 0], [np.inf, 0]],
)
def test_invalid_weights(weights):
    with pytest.raises(ValueError):
        calculate_turnover(weights, [0.5, 0.5])
    with pytest.raises(ValueError):
        calculate_turnover([0.5, 0.5], weights)


def test_different_weight_shapes():
    with pytest.raises(ValueError):
        calculate_turnover([0.5, 0.5], [0.3, 0.3, 0.4])


@pytest.mark.parametrize(
    "overrides",
    [
        {"nav": 0},
        {"nav": -1},
        {"nav": np.nan},
        {"nav": np.inf},
        {"commission_rate": -0.1},
        {"commission_rate": 1},
        {"slippage_rate": np.inf},
        {"slippage_rate": np.nan},
        {"asset_returns": [-1.01]},
        {"asset_returns": [np.nan]},
        {"asset_returns": [np.inf]},
        {"asset_returns": []},
        {"asset_returns": [0, 0]},
        {"asset_returns": [[0]]},
    ],
)
def test_invalid_step_inputs(overrides):
    inputs = dict(nav=100, current_weights=[0.5, 0.5], target_weights=[0.5, 0.5])
    inputs["asset_returns"] = [0]
    inputs.update(overrides)

    with pytest.raises(ValueError):
        rebalance_step(**inputs)


def test_exhausted_nav_rejected():
    with pytest.raises(ValueError):
        rebalance_step(100, [1, 0, 0], [0, 1, 0], [0, 0], 0.3, 0.3)
    with pytest.raises(ValueError):
        rebalance_step(100, [1, 0], [1, 0], [-1])


def test_asset_total_loss_preserves_remaining_cash():
    result = rebalance_step(100, [0.5, 0.5], [0.5, 0.5], [-1])

    assert result.nav == 50
    np.testing.assert_array_equal(result.weights, [0, 1])


def test_nav_overflow_rejected():
    with pytest.raises(ValueError):
        rebalance_step(100, [1, 0], [1, 0], [1e308])


def test_log_return_conversion():
    np.testing.assert_allclose(log_to_simple_returns(np.log1p([0.1, -0.2, 0])), [0.1, -0.2, 0])


@pytest.mark.parametrize("values", [[], [[0]], [np.nan], [np.inf], [1000]])
def test_invalid_log_returns(values):
    with pytest.raises(ValueError):
        log_to_simple_returns(values)
