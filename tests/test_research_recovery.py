"""Causal price recovery, history carry and warmup sensitivity."""

import numpy as np
import pandas as pd
import pytest

from robo_advisor.features.dataset import TEAM12_ASSETS
from robo_advisor.features.research_recovery import (
    OHLC,
    OHLCV,
    crosscheck_missing_ohlcv,
    replay_observed_prices,
)


@pytest.fixture
def price_evidence():
    dates = pd.bdate_range("2022-01-03", periods=3)
    quote = pd.DataFrame([[100, 110, 95, 104, 300]] * 3, index=dates, columns=OHLCV)
    naver = quote.copy().astype(float)
    naver[OHLC] *= 0.9
    yahoo = quote.copy().astype(float)
    yahoo["Adj Close"] = naver["Close"]
    archive = yahoo.drop(index=dates[1])
    yahoo.loc[dates[1]] = np.nan
    return archive, yahoo, naver, quote.loc[[dates[1]]]


def test_recovers_observed_ohlcv_without_using_adjusted_comparison_values(price_evidence):
    archive, yahoo, naver, krx = price_evidence
    proof = crosscheck_missing_ohlcv(archive, yahoo, naver, krx, [], adjusted=True)
    assert proof["basis_compatible"].tolist() == [True]
    assert proof["diagnostic_only_adjustment_factor"].iloc[0] == pytest.approx(0.9)
    assert krx["Close"].iloc[0] == 104


@pytest.mark.parametrize("case", ["corporate_action", "wrong_basis", "yahoo_conflict"])
def test_unresolved_basis_cannot_be_patched(price_evidence, case):
    archive, yahoo, naver, krx = price_evidence
    day = krx.index[0]
    events = [day] if case == "corporate_action" else []
    if case == "wrong_basis":
        naver.loc[day, "Close"] = 104
    if case == "yahoo_conflict":
        yahoo.loc[day, "Close"] = 105
    proof = crosscheck_missing_ohlcv(archive, yahoo, naver, krx, events, adjusted=True)
    assert not proof.loc[day, "basis_compatible"]
    assert proof.loc[day, "unresolved_reason"] is not None


def test_cannot_replace_archived_observation(price_evidence):
    archive, yahoo, naver, _ = price_evidence
    with pytest.raises(ValueError, match="replace an archived"):
        crosscheck_missing_ohlcv(archive, yahoo, naver, archive[OHLCV], [], adjusted=True)


@pytest.fixture
def prices():
    days = pd.bdate_range("2020-01-01", periods=105)
    values = 100.0 + np.sin(np.arange(len(days)) / 3.0) + np.arange(len(days)) / 15
    frame = pd.DataFrame({asset: values for asset in TEAM12_ASSETS}, index=days)
    times = pd.Series(days.tz_localize("Asia/Seoul") + pd.Timedelta(hours=16), index=days)
    return frame, times


@pytest.mark.leakage
def test_per_asset_keeps_unbroken_history_and_cannot_bridge_a_missing_return(prices):
    complete, times = prices
    frame = complete.copy()
    frame.iloc[42, -1] = np.nan
    history = replay_observed_prices(frame, times, asset_history=True)
    shared = replay_observed_prices(frame, times)
    clean = replay_observed_prices(complete, times)
    assert history.asset_positions.iloc[43, 0] == 42
    assert history.asset_positions.iloc[43, -1] == -1
    assert history.asset_positions.iloc[44, -1] == 0
    assert history.rewards.iloc[41:44].isna().all().all()
    assert history.rows["feature_ready"].equals(shared.rows["feature_ready"])
    column = "SPY__macd"
    assert history.features[column].iloc[77] == clean.features[column].iloc[77]
    assert history.features[column].iloc[77] != shared.features[column].iloc[77]


@pytest.mark.leakage
def test_earlier_features_ignore_future_prices_and_warmup_only_masks_rows(prices):
    frame, times = prices
    baseline = replay_observed_prices(frame, times)
    twenty = replay_observed_prices(frame, times, warmup_rows=20)
    shared = baseline.rows["feature_ready"]
    pd.testing.assert_frame_equal(baseline.features.loc[shared], twenty.features.loc[shared])
    assert (twenty.rows["feature_ready"] & ~shared).sum() == 13
    changed = frame.copy()
    changed.iloc[70:] *= 2.0
    after = replay_observed_prices(changed, times, warmup_rows=20)
    pd.testing.assert_frame_equal(after.features.iloc[:70], twenty.features.iloc[:70])
    pd.testing.assert_frame_equal(after.rewards.iloc[:69], twenty.rewards.iloc[:69])
