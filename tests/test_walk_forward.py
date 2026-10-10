"""Walk-Forward 경계, 워밍업 및 미래 참조 검증."""

import numpy as np
import pandas as pd
import pytest

from robo_advisor.backtest.walk_forward import walk_forward_split


@pytest.mark.leakage
def test_reference_calendar_and_warmup():
    dates = pd.date_range("2020-01-01", "2025-12-31", freq="D")
    windows = walk_forward_split(dates, start_date="2020-01-01", warmup_periods=20)
    assert len(windows) == 2
    for year, window in zip((2024, 2025), windows):
        assert dates[window.train[0]] == pd.Timestamp(year=year - 4, month=1, day=1)
        assert dates[window.test[0]] == pd.Timestamp(year=year, month=1, day=1)
        assert dates[window.test[-1]] == pd.Timestamp(year=year, month=12, day=31)
        assert not np.intersect1d(window.train, window.test).size
        assert (dates[window.train] < window.test_start).all()
        assert (dates[window.warmup] < window.test_start).all()
        assert len(window.warmup) == 20
    assert not np.intersect1d(windows[0].test, windows[1].test).size


@pytest.mark.leakage
def test_future_dates_do_not_change_existing_windows():
    dates = pd.date_range("2020-01-01", "2026-12-31")
    shorter = walk_forward_split(dates[dates < "2026"], start_date="2020-01-01")
    longer = walk_forward_split(dates, start_date="2020-01-01")
    for original, extended in zip(shorter, longer):
        np.testing.assert_array_equal(original.train, extended.train)
        np.testing.assert_array_equal(original.test, extended.test)


def test_trading_calendar_explicit_coverage_end():
    dates = pd.bdate_range("2019-01-01", "2023-12-29")
    windows = walk_forward_split(
        dates, start_date="2019-01-01", train_years=3, coverage_end="2024-01-01"
    )
    assert len(windows) == 2


def test_partial_year_is_not_returned():
    dates = pd.date_range("2020-01-01", "2026-06-01")
    assert len(walk_forward_split(dates, start_date="2020-01-01")) == 2


@pytest.mark.parametrize(
    "option", [{"train_years": 0}, {"test_years": 0}, {"min_windows": 1}, {"warmup_periods": -1}]
)
def test_invalid_options(option):
    with pytest.raises(ValueError):
        walk_forward_split(pd.date_range("2020", "2026"), start_date="2020", **option)


def test_insufficient_history_and_unsorted_dates():
    dates = pd.date_range("2020", "2025")
    with pytest.raises(ValueError, match="complete windows"):
        walk_forward_split(dates, start_date="2020")
    with pytest.raises(ValueError, match="increasing"):
        walk_forward_split(dates[::-1], start_date="2020")
    with pytest.raises(ValueError, match="covered"):
        walk_forward_split(dates, start_date="2019")


def test_insufficient_warmup_and_timezone_mismatch():
    dates = pd.date_range("2020", "2026")
    with pytest.raises(ValueError, match="warmup"):
        walk_forward_split(dates, start_date="2020", warmup_periods=5000)
    with pytest.raises(ValueError, match="timezone"):
        walk_forward_split(dates.tz_localize("UTC"), start_date="2020")


def test_holiday_at_first_boundary():
    dates = pd.bdate_range("2020-01-02", "2025-12-31")
    windows = walk_forward_split(dates, start_date="2020-01-01", coverage_start="2020-01-01")
    assert len(windows) == 2
    assert windows[0].train_start == pd.Timestamp("2020-01-01")
    assert dates[windows[0].train[0]] == pd.Timestamp("2020-01-02")


def test_leap_day_boundaries_have_no_gap():
    dates = pd.date_range("2020-02-29", "2030-03-01")
    windows = walk_forward_split(dates, start_date="2020-02-29")
    for previous, following in zip(windows, windows[1:]):
        assert previous.test_end == following.test_start
        assert previous.test[-1] + 1 == following.test[0]
