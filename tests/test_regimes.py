"""Pre-block compounding, stable session boundaries and coverage contracts."""

from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from robo_advisor.experiments.regimes import label_market_regimes


def calendar(size=15):
    return pd.bdate_range("2026-01-01", periods=size)


def run(values, **overrides):
    dates = calendar(max(15, len(values)))
    inputs = dict(
        trading_calendar=dates,
        block_anchor=dates[2],
        block_size=3,
        min_blocks_per_regime=2,
        lookback=2,
    )
    inputs.update(overrides)
    return label_market_regimes(pd.Series(values, index=dates[: len(values)]), **inputs)


@pytest.mark.parametrize(
    "history,expected,cumulative",
    [
        ([0.03, 0.03], "up", 0.0609),
        ([-0.03, -0.03], "down", -0.0591),
        ([0.1, -0.1], "sideways", -0.01),
        ([0.05, 0], "sideways", 0.05),
        ([-0.05, 0], "sideways", -0.05),
        ([0.05000001, 0], "up", 0.05000001),
        ([-0.05000001, 0], "down", -0.05000001),
        ([-1, 0.5], "down", -1),
    ],
)
def test_hand_compounding_labels_and_equal_boundaries(history, expected, cumulative):
    result = run(history + [0, 0, 0])
    assert result.labels.tolist() == [expected] * 3
    block = result.blocks[0]
    assert block.cumulative_return == pytest.approx(cumulative)
    assert block.history_count == 2
    assert block.reason is None
    assert block.complete


def test_default_sixty_day_compounding_excludes_current_day():
    dates = calendar(63)
    values = pd.Series([0.001] * 60 + [-0.9] * 3, index=dates)
    result = label_market_regimes(
        values,
        trading_calendar=dates,
        block_anchor=dates[60],
        block_size=3,
        min_blocks_per_regime=1,
    )
    assert result.blocks[0].cumulative_return == pytest.approx(1.001**60 - 1)
    assert result.labels.tolist() == ["up"] * 3
    assert result.rules.lookback == 60
    assert result.rules.threshold == 0.05


def test_only_immediately_previous_window_is_used():
    result = run([0.9, 0.9, 0, 0, 0, 0, 0, 0])
    assert [block.label for block in result.blocks] == ["up", "sideways"]
    assert result.blocks[1].cumulative_return == 0


def test_warmup_unknown_and_never_reclassified_with_future_history():
    dates = calendar()
    first = run([0] * 5, block_anchor=dates[0], lookback=4)
    extended = run([0] * 12, block_anchor=dates[0], lookback=4)
    assert [block.label for block in extended.blocks] == [
        "unknown",
        "unknown",
        "sideways",
        "sideways",
    ]
    assert [block.history_count for block in extended.blocks] == [0, 3, 4, 4]
    assert extended.blocks[0].reason == "insufficient_history"
    assert extended.blocks[0].cumulative_return is None
    assert extended.unknown_observation_count == 6
    assert extended.unknown_block_count == 2
    pd.testing.assert_series_equal(first.labels, extended.labels.loc[first.labels.index])


def test_current_block_changes_cannot_change_its_label():
    original = [0.03, 0.03, 0, 0, 0, 0, 0, 0]
    altered = [0.03, 0.03, -0.9, 0.8, -0.8, 0.9, -0.5, 0.7]
    first, second = run(original), run(altered)
    pd.testing.assert_series_equal(first.labels.iloc[:3], second.labels.iloc[:3])
    assert first.blocks[0] == second.blocks[0]
    # The next block may reflect the now-past changed block.
    assert first.blocks[1].label != second.blocks[1].label


def test_future_changes_and_appends_keep_existing_decisions():
    first = run([0.03, 0.03, 0, 0])
    second = run([0.03, 0.03, 0, 0, -0.8, 0.9, 0.9, 0.9])
    pd.testing.assert_series_equal(first.labels, second.labels.loc[first.labels.index])
    assert first.blocks[0].start == second.blocks[0].start
    assert first.blocks[0].label == second.blocks[0].label
    assert not first.blocks[0].complete
    assert second.blocks[0].complete
    # Extending the fixed calendar itself also preserves prior boundaries.
    dates = calendar(30)
    third = run([0.03, 0.03, 0, 0], trading_calendar=dates, block_anchor=dates[2])
    pd.testing.assert_series_equal(first.labels, third.labels)
    assert first.blocks == third.blocks


def test_sliced_evaluation_retains_fixed_boundaries_and_history():
    dates = calendar()
    values = [0.03, 0.03, 0.9, -0.3, -0.3, 0, 0, 0, 0, 0]
    full = run(values)
    sliced = run(values, evaluation_start=dates[3], evaluation_end=dates[7])
    pd.testing.assert_series_equal(sliced.labels, full.labels.loc[dates[3:7]])
    assert [block.start for block in sliced.blocks] == [dates[2], dates[5]]
    assert [block.block_id for block in sliced.blocks] == [0, 1]
    assert [block.label for block in sliced.blocks] == ["up", "down"]
    assert not any(block.complete for block in sliced.blocks)
    assert [block.observation_count for block in sliced.blocks] == [2, 2]


def test_trimming_unneeded_input_history_does_not_shift_blocks():
    dates = calendar()
    values = pd.Series([0] * 3 + [0.03, 0.03] + [0] * 7, index=dates[:12])
    options = dict(
        trading_calendar=dates,
        block_anchor=dates[5],
        block_size=3,
        min_blocks_per_regime=1,
        lookback=2,
    )
    full = label_market_regimes(values, **options)
    trimmed = label_market_regimes(values.iloc[3:], **options)
    pd.testing.assert_series_equal(full.labels, trimmed.labels)
    assert full.blocks == trimmed.blocks
    missing = label_market_regimes(values.iloc[6:], **options)
    assert missing.blocks[0].start == dates[5]
    assert missing.blocks[0].label == "unknown"
    assert missing.blocks[0].reason == "insufficient_history"


def test_coverage_separates_days_blocks_and_eligible_minimum():
    result = run([0.03, 0.03, 0, -0.03, -0.03, 0, 0, 0, 0])
    assert [block.label for block in result.blocks] == ["up", "down", "sideways"]
    assert result.coverage["up"].observation_count == 3
    assert result.coverage["up"].block_count == 1
    assert result.coverage["up"].complete_block_count == 1
    assert result.coverage["up"].shortfall == 1
    assert not result.coverage["up"].sufficient
    assert result.coverage["sideways"].observation_count == 1
    assert result.coverage["sideways"].block_count == 1
    assert result.coverage["sideways"].complete_block_count == 0
    assert result.coverage["sideways"].shortfall == 2
    assert result.insufficient_regimes == ("up", "down", "sideways")
    sufficient = run([0.03, 0.03, 0, -0.03, -0.03, 0, 0, 0, 0], min_blocks_per_regime=1)
    assert sufficient.coverage["up"].sufficient
    assert sufficient.coverage["down"].sufficient
    assert sufficient.insufficient_regimes == ("sideways",)


def test_empty_regime_counts_and_calendar_tail():
    dates = calendar(4)
    result = run([0.03, 0.03, 0, 0], trading_calendar=dates, block_anchor=dates[2])
    assert result.blocks[0].last_scheduled_date is None
    assert not result.blocks[0].complete
    assert result.coverage["down"].observation_count == 0
    assert result.coverage["down"].block_count == 0
    assert result.coverage["down"].shortfall == 2


def test_applied_rules_reproduce_results_and_inputs_are_unchanged():
    dates = calendar()
    values = pd.Series([0.03, 0.03] + [0] * 7, index=dates[:9])
    original = values.copy()
    result = label_market_regimes(
        values,
        trading_calendar=dates,
        block_anchor=dates[2],
        block_size=3,
        min_blocks_per_regime=1,
        lookback=2,
        threshold=0.06,
        boundary_tolerance=0,
    )
    rules = asdict(result.rules)
    for key in ("boundary_policy", "incomplete_policy", "sample_unit"):
        rules.pop(key)
    rules["trading_calendar"] = pd.DatetimeIndex(rules["trading_calendar"])
    repeated = label_market_regimes(values, **rules)
    assert result.blocks == repeated.blocks
    assert result.coverage == repeated.coverage
    assert result.rules == repeated.rules
    pd.testing.assert_series_equal(values, original)
    pd.testing.assert_series_equal(result.labels, repeated.labels)


@pytest.mark.parametrize(
    "overrides",
    [
        {"lookback": 0},
        {"lookback": True},
        {"lookback": 1.5},
        {"block_size": 0},
        {"block_size": -1},
        {"block_size": False},
        {"min_blocks_per_regime": 0},
        {"min_blocks_per_regime": 1.5},
        {"threshold": 0},
        {"threshold": 1},
        {"threshold": -0.05},
        {"threshold": np.nan},
        {"threshold": np.inf},
        {"threshold": None},
        {"threshold": True},
        {"boundary_tolerance": -1},
        {"boundary_tolerance": 0.05},
        {"boundary_tolerance": np.nan},
        {"boundary_tolerance": None},
        {"block_anchor": "2025-12-01"},
        {"block_anchor": pd.NaT},
        {"block_anchor": "invalid"},
        {"block_anchor": "2026-01-05T00:00:00Z"},
        {"evaluation_start": "2025-12-01"},
        {"evaluation_start": calendar()[3], "evaluation_end": calendar()[3]},
        {"evaluation_start": calendar()[-1]},
        {"trading_calendar": calendar()[::-1]},
        {"trading_calendar": pd.DatetimeIndex([])},
    ],
)
def test_invalid_configuration(overrides):
    with pytest.raises(ValueError):
        run([0] * 8, **overrides)


@pytest.mark.parametrize(
    "case",
    [
        "nan",
        "inf",
        "loss",
        "order",
        "duplicate",
        "nat",
        "gap",
        "outside",
        "text",
        "complex",
        "bool",
        "type",
    ],
)
def test_invalid_benchmark(case):
    dates = calendar()
    values = pd.Series([0.0] * 8, index=dates[:8])
    if case == "nan":
        values.iloc[0] = np.nan
    elif case == "inf":
        values.iloc[-1] = np.inf
    elif case == "loss":
        values.iloc[0] = -1.01
    elif case == "order":
        values = values.iloc[::-1]
    elif case == "duplicate":
        values.index = pd.DatetimeIndex([dates[0]] * 8)
    elif case == "nat":
        values.index = pd.DatetimeIndex([pd.NaT] + list(dates[1:8]))
    elif case == "gap":
        values = values.drop(dates[3])
    elif case == "outside":
        values.index = values.index + pd.Timedelta(hours=1)
    elif case == "text":
        values = values.astype(str)
    elif case == "complex":
        values = values.astype(complex)
    elif case == "bool":
        values = values.astype(bool)
    else:
        values = values.to_frame()
    with pytest.raises(ValueError):
        label_market_regimes(
            values,
            trading_calendar=dates,
            block_anchor=dates[2],
            block_size=3,
            min_blocks_per_regime=1,
            lookback=2,
        )


def test_timezone_aware_calendar_and_compounding_overflow():
    dates = calendar().tz_localize("Asia/Seoul")
    values = pd.Series([0.03, 0.03, 0, 0, 0], index=dates[:5])
    result = label_market_regimes(
        values,
        trading_calendar=dates,
        block_anchor=dates[2].tz_convert("UTC"),
        block_size=3,
        min_blocks_per_regime=1,
        lookback=2,
    )
    assert result.labels.tolist() == ["up"] * 3
    with pytest.raises(ValueError, match="timezone"):
        label_market_regimes(
            values,
            trading_calendar=calendar(),
            block_anchor=calendar()[2],
            block_size=3,
            min_blocks_per_regime=1,
        )
    with pytest.raises(ValueError, match="compounded"):
        run([1e308, 1e308, 0, 0, 0])


def test_intraday_and_duplicate_calendars_are_rejected():
    dates = calendar()
    duplicate = pd.DatetimeIndex([dates[0]] + list(dates[:14]))
    intraday = pd.DatetimeIndex([dates[0], dates[0] + pd.Timedelta(hours=1)] + list(dates[2:]))
    for invalid in (duplicate, intraday):
        with pytest.raises(ValueError):
            run([0] * 8, trading_calendar=invalid)


def test_custom_threshold_and_rounding_tolerance():
    assert run([0.03, 0.03, 0, 0, 0], threshold=0.1).blocks[0].label == "sideways"
    assert run([0.05 + 5e-13, 0, 0, 0, 0]).blocks[0].label == "sideways"
    assert run([0.05 + 5e-13, 0, 0, 0, 0], boundary_tolerance=0).blocks[0].label == "up"
