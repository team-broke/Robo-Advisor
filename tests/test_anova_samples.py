"""#42 ANOVA 표본: 검정 1 seed×윈도우, 검정 2·3 21거래일 블록과 국면 라벨."""

import numpy as np
import pandas as pd
import pytest

from robo_advisor.backtest.metrics import calculate_metrics
from robo_advisor.experiments.anova import strategy_anova, strategy_regime_anova
from robo_advisor.experiments.anova_samples import (
    average_seed_returns,
    block_samples,
    regime_block_samples,
    reward_samples,
)

STRATEGIES = ["ppo", "mvo", "equal_weight"]


def noisy(dates, seed, mean=0.0005):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, 0.01, len(dates)), index=dates)


def sharpe(series):
    return calculate_metrics(series, series)["sharpe"]


def strategy_frame(dates, seed=0):
    return pd.DataFrame(
        {name: noisy(dates, seed + offset) for offset, name in enumerate(STRATEGIES)}
    )


# 검정 1 ---------------------------------------------------------------------


def run_returns(rewards=("return", "sharpe", "mdd_penalty"), seeds=(42, 43), windows=("W1", "W2")):
    dates = pd.bdate_range("2024-01-02", periods=30)
    runs = {}
    for r, reward in enumerate(rewards):
        for seed in seeds:
            for w, window in enumerate(windows):
                runs[(reward, seed, window)] = noisy(dates, 100 * r + 10 * seed + w)
    return runs


def test_reward_samples_one_row_per_reward_seed_window_with_metrics_sharpe():
    runs = run_returns()
    samples = reward_samples(runs)
    assert list(samples.columns) == ["reward", "seed", "window", "sharpe"]
    assert len(samples) == 3 * 2 * 2
    assert not samples.duplicated(["reward", "seed", "window"]).any()
    row = samples.set_index(["reward", "seed", "window"]).loc[("sharpe", 43, "W2"), "sharpe"]
    assert row == pytest.approx(sharpe(runs[("sharpe", 43, "W2")]), rel=1e-12)


def test_reward_samples_rejects_unbalanced_design():
    runs = run_returns()
    del runs[("return", 42, "W1")]
    with pytest.raises(ValueError, match="same seed"):
        reward_samples(runs)


def test_reward_samples_requires_two_rewards():
    with pytest.raises(ValueError, match="two reward"):
        reward_samples(run_returns(rewards=("return",)))


# 검정 2 ---------------------------------------------------------------------


def test_blocks_start_at_test_start_and_drop_incomplete_tail():
    dates = pd.bdate_range("2024-01-02", periods=21 * 3 + 5)
    samples = block_samples({"W1": strategy_frame(dates)})
    assert list(samples.columns) == [
        "window",
        "block_id",
        "block_start",
        "block_end",
        "strategy",
        "sharpe",
    ]
    assert len(samples) == 3 * len(STRATEGIES)
    starts = sorted(samples.block_start.unique())
    assert starts == [dates[0], dates[21], dates[42]]
    assert samples.block_end.max() == dates[62]


def test_block_sharpe_uses_only_that_blocks_daily_returns():
    dates = pd.bdate_range("2024-01-02", periods=42)
    frame = strategy_frame(dates)
    samples = block_samples({"W1": frame}).set_index(["block_id", "strategy"])
    assert samples.loc[(1, "mvo"), "sharpe"] == pytest.approx(
        sharpe(frame["mvo"].iloc[21:42]), rel=1e-12
    )


def test_each_block_has_one_row_per_strategy_not_daily_rows():
    dates = pd.bdate_range("2024-01-02", periods=63)
    samples = block_samples({"W1": strategy_frame(dates), "W2": strategy_frame(dates, 9)})
    counts = samples.groupby(["window", "block_id"]).strategy.agg(list)
    assert all(sorted(value) == sorted(STRATEGIES) for value in counts)
    assert len(samples) == 2 * 3 * 3


@pytest.mark.leakage
def test_returns_outside_complete_blocks_do_not_change_samples():
    dates = pd.bdate_range("2024-01-02", periods=21 * 2 + 7)
    frame = strategy_frame(dates)
    changed = frame.copy()
    changed.iloc[42:] = 0.5
    pd.testing.assert_frame_equal(block_samples({"W1": frame}), block_samples({"W1": changed}))


def test_block_samples_reject_missing_returns_and_too_short_window():
    dates = pd.bdate_range("2024-01-02", periods=42)
    frame = strategy_frame(dates)
    frame.iloc[3, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        block_samples({"W1": frame})
    with pytest.raises(ValueError, match="complete block"):
        block_samples({"W1": strategy_frame(dates[:20])})


def test_undefined_block_sharpe_fails_instead_of_dropping_the_block():
    dates = pd.bdate_range("2024-01-02", periods=21)
    frame = strategy_frame(dates)
    frame["equal_weight"] = 0.0
    with pytest.raises(ValueError, match="W1.*block 0.*equal_weight"):
        block_samples({"W1": frame})


def test_average_seed_returns_is_daily_mean_across_seeds():
    dates = pd.bdate_range("2024-01-02", periods=3)
    seeds = pd.DataFrame({42: [0.01, 0.02, -0.01], 43: [0.03, 0.0, 0.01]}, index=dates)
    expected = pd.Series([0.02, 0.01, 0.0], index=dates, name="ppo")
    pd.testing.assert_series_equal(average_seed_returns(seeds), expected)
    seeds.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        average_seed_returns(seeds)


def test_block_samples_feed_strategy_anova():
    dates = pd.bdate_range("2024-01-02", periods=21 * 4)
    samples = block_samples({"W1": strategy_frame(dates), "W2": strategy_frame(dates, 5)})
    result = strategy_anova(samples, "sharpe")
    assert result.group_counts.iloc[:, -1].tolist() == [8, 8, 8]


# 검정 3 ---------------------------------------------------------------------


def regime_setup(history_mean, test_blocks=4, block_size=21, lookback=60):
    """앞 lookback일 이력 + 테스트 구간. 블록마다 다른 추세를 만든다."""
    calendar = pd.bdate_range("2023-09-01", periods=lookback + block_size * test_blocks)
    benchmark = pd.Series(history_mean, index=calendar, dtype=float)
    test_dates = calendar[lookback:]
    return benchmark, test_dates


def test_regime_labels_come_from_sixty_sessions_before_each_block():
    benchmark, test_dates = regime_setup(0.002)
    benchmark.loc[test_dates[:21]] = -0.004  # 블록 0 수익률이 블록 1 국면을 결정
    blocks = block_samples({"W1": strategy_frame(test_dates)})
    result = regime_block_samples(blocks, {"W1": benchmark})
    labels = result.samples.drop_duplicates("block_id").set_index("block_id").regime
    assert labels[0] == "up"  # 직전 60일 0.2%/일 → 상승
    assert labels[1] == "sideways"  # 39일 상승 + 21일 하락 ≈ -0.6%
    assert set(result.samples.columns) >= {"window", "block_id", "strategy", "regime", "sharpe"}


@pytest.mark.leakage
def test_regime_labels_ignore_block_and_future_benchmark_returns():
    benchmark, test_dates = regime_setup(0.002)
    blocks = block_samples({"W1": strategy_frame(test_dates)})
    base = regime_block_samples(blocks, {"W1": benchmark}).samples
    changed = benchmark.copy()
    changed.loc[test_dates[63] :] = -0.2  # 마지막 블록 내부와 그 이후
    after = regime_block_samples(blocks, {"W1": changed}).samples
    pd.testing.assert_frame_equal(base, after)


def test_unknown_blocks_are_excluded_and_counted():
    benchmark, test_dates = regime_setup(0.002)
    short_history = benchmark.iloc[10:]  # 첫 블록 직전 이력 50일뿐
    blocks = block_samples({"W1": strategy_frame(test_dates)})
    result = regime_block_samples(blocks, {"W1": short_history})
    assert result.excluded_unknown_blocks == 1
    assert 0 not in set(result.samples.block_id)
    assert len(result.samples) == 3 * len(STRATEGIES)


def test_cells_below_two_blocks_mark_test_three_infeasible_without_retuning():
    benchmark, test_dates = regime_setup(0.002)
    benchmark.loc[test_dates[:21]] = -0.004
    blocks = block_samples({"W1": strategy_frame(test_dates)})
    result = regime_block_samples(blocks, {"W1": benchmark})
    assert not result.feasible
    assert ("ppo", "up") in result.insufficient_cells
    assert result.cell_counts.loc[("ppo", "up")] == 1
    assert result.cell_counts.loc[("ppo", "sideways")] == 3
    assert result.rules["block_size"] == 21 and result.rules["threshold"] == 0.05


def test_feasible_regime_samples_feed_two_way_anova():
    lookback, size = 60, 21
    calendar = pd.bdate_range("2023-09-01", periods=lookback + size * 8)
    pattern = [0.002] * (lookback + size * 3) + [-0.01] * (size * 5)
    benchmark = pd.Series(pattern, index=calendar)
    test_dates = calendar[lookback:]
    blocks = block_samples({"W1": strategy_frame(test_dates)})
    result = regime_block_samples(blocks, {"W1": benchmark})
    assert result.feasible, result.cell_counts
    anova = strategy_regime_anova(result.samples, "sharpe")
    assert result.cell_counts.sum() == len(result.samples) == 8 * len(STRATEGIES)
    assert not anova.effects.empty


def test_regime_rejects_benchmark_calendar_that_disagrees_with_blocks():
    benchmark, test_dates = regime_setup(0.002)
    blocks = block_samples({"W1": strategy_frame(test_dates)})
    with pytest.raises(ValueError, match="benchmark"):
        regime_block_samples(blocks, {"W1": benchmark.drop(test_dates[30])})
    with pytest.raises(ValueError, match="W1"):
        regime_block_samples(blocks, {"W2": benchmark})
