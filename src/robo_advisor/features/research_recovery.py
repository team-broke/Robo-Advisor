"""Research-only price recovery and causal feature replay for archived Team12 data."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from robo_advisor.data.alignment import calculate_log_returns
from robo_advisor.data.validation import validate_price_data

from .dataset import TEAM12_ASSETS, WARMUP_ROWS, _FEATURES, _aware, _index
from .indicators import calculate_macd, calculate_rsi

OHLC = ["Open", "High", "Low", "Close"]
OHLCV = [*OHLC, "Volume"]


def crosscheck_missing_ohlcv(archive, yahoo, naver, krx, yahoo_event_dates, *, adjusted):
    """Check KRX as-exported quotes with same-basis anchors and an independent feed.

    The existing Yahoo Close is the price basis. Naver's adjusted series is
    corroboration only. No comparison factor enters the recovered prices.
    """
    for name, frame in (("archive", archive), ("Yahoo", yahoo), ("Naver", naver), ("KRX", krx)):
        _index(frame.index, name)
    if len(krx.index.intersection(archive.index)):
        raise ValueError("Recovery may never replace an archived observation")

    anchor_fields = ["Open", "High", "Close", "Volume"]
    stable = archive[anchor_fields].eq(yahoo.reindex(archive.index)[anchor_fields]).all(axis=1)
    factors = archive["Adj Close"].div(archive["Close"])
    if adjusted:
        # Each feed rounds adjusted quotes. Two KRW is a corroboration tolerance,
        # never a correction amount or a permitted error in the actual market Close.
        rounding_bound = 2.0
        adjusted_anchor = yahoo["Adj Close"].div(yahoo["Close"]).reindex(archive.index)
        stable &= factors.sub(adjusted_anchor).abs().le(1e-6)
    else:
        rounding_bound = 0.0
        stable &= factors.eq(1) & yahoo["Close"].eq(yahoo["Adj Close"]).reindex(archive.index)

    matched = archive[OHLC].mul(factors, axis=0).sub(naver[OHLC]).abs().max(axis=1, skipna=False)
    stable &= matched.reindex(archive.index).le(rounding_bound)
    result = []
    for day, row in krx.iterrows():
        pos = archive.index.searchsorted(day)
        reason = None
        before = after = factor = residual = None
        if pos == 0 or pos == len(archive) or day not in naver.index:
            reason = "missing_adjacent_evidence"
        elif not stable.iloc[pos - 1 : pos + 1].all():
            reason = "price_basis_anchor_disagreement"
        else:
            before, after = archive.index[pos - 1 : pos + 1]
            factor = float(factors.loc[before])
            observed = yahoo.reindex([day]).iloc[0][OHLCV]
            naver_row = naver.loc[day, OHLC]
            residual = float(naver_row.sub(row[OHLC] * factor).abs().max(skipna=False))
            price_check = validate_price_data(krx.loc[[day]], price_columns=OHLC)
            if (
                price_check["status"] != "passed"
                or not np.isfinite(row["Volume"])
                or row["Volume"] <= 0
            ):
                reason = "invalid_krx_observation"
            elif (observed.notna() & observed.ne(row)).any():
                reason = "conflicting_yahoo_market_quote"
            elif abs(factor - factors.loc[after]) > 1e-6 or any(
                before < event <= after for event in yahoo_event_dates
            ):
                reason = "corporate_action_or_factor_change_between_anchors"
            elif not np.isfinite(residual) or residual > rounding_bound:
                reason = "independent_price_disagreement"
        result.append(
            {
                "date": day,
                "basis_compatible": reason is None,
                "unresolved_reason": reason,
                "previous_anchor": before,
                "following_anchor": after,
                "diagnostic_only_adjustment_factor": factor,
                "max_naver_ohlc_residual_krw": residual,
            }
        )
    return pd.DataFrame.from_records(result).set_index("date")


@dataclass(frozen=True)
class ResearchReplay:
    features: pd.DataFrame
    rewards: pd.DataFrame
    rows: pd.DataFrame
    asset_positions: pd.DataFrame


def replay_observed_prices(prices, evaluation_at, *, warmup_rows=WARMUP_ROWS, asset_history=False):
    """Use the complete evaluation grid, including every unobservable row.

    Feature history may be per asset. A missing endpoint always blocks that
    asset's one-session return and prevents a cross-gap portfolio reward.
    """
    _index(prices.index, "research valuation dates")
    if tuple(prices.columns) != TEAM12_ASSETS or not evaluation_at.index.equals(prices.index):
        raise ValueError("Expected Team12 asset order and one cutoff per evaluation date")
    if warmup_rows not in {20, WARMUP_ROWS}:
        raise ValueError(
            "Use the standard 33 observations or the 20-observation research candidate"
        )
    evaluation = _aware(evaluation_at, "research evaluation_at")
    values = prices.to_numpy(dtype=float)
    if not evaluation.is_monotonic_increasing or not evaluation.is_unique:
        raise ValueError("Evaluation times must be strictly increasing")
    local_dates = evaluation.tz_convert("Asia/Seoul").tz_localize(None).normalize()
    if not local_dates.equals(prices.index):
        raise ValueError("Evaluation time must belong to its Seoul date label")
    if not np.isfinite(values[~np.isnan(values)]).all() or (values <= 0).any():
        raise ValueError("Observed prices must be positive and finite")
    prices = prices.astype(float)
    returns = calculate_log_returns(prices)
    complete = prices.notna().all(axis=1) & returns.notna().all(axis=1)
    segments = (~complete).cumsum()
    positions = complete.groupby(segments).cumsum().astype("int64").sub(1)
    feature_columns = [f"{asset}__{kind}" for asset in TEAM12_ASSETS for kind in _FEATURES]
    features = pd.DataFrame(np.nan, index=prices.index, columns=feature_columns, dtype=float)
    asset_positions = pd.DataFrame(-1, index=prices.index, columns=prices.columns, dtype="int64")
    for asset in TEAM12_ASSETS:
        valid = (prices[asset].notna() & returns[asset].notna()) if asset_history else complete
        asset_segments = (~valid).cumsum() if asset_history else segments
        for _, close in prices.loc[valid, asset].groupby(asset_segments[valid], sort=False):
            asset_positions.loc[close.index, asset] = np.arange(len(close))
            trailing = returns.loc[close.index, asset]
            macd = calculate_macd(close)
            block = pd.DataFrame(
                {
                    f"{asset}__log_return": trailing,
                    f"{asset}__rsi14": calculate_rsi(close),
                    f"{asset}__macd": macd["macd"],
                    f"{asset}__macd_signal": macd["macd_signal"],
                    f"{asset}__macd_histogram": macd["macd"] - macd["macd_signal"],
                    f"{asset}__volatility20": trailing.rolling(20, min_periods=20).std(ddof=0),
                }
            ).iloc[warmup_rows:]
            features.loc[block.index, block.columns] = block

    warmup = complete & asset_positions.lt(warmup_rows).any(axis=1)
    ready = complete & np.isfinite(features.to_numpy()).all(axis=1)
    features.loc[~ready] = np.nan
    known = complete & complete.shift(-1, fill_value=False)
    rewards = returns.shift(-1).where(known, axis=0)
    rows = pd.DataFrame(
        {
            "evaluation_at": evaluation,
            "price_complete": prices.notna().all(axis=1),
            "alignment_eligible": complete,
            "segment_id": segments,
            "warmup_position": positions,
            "warmup": warmup,
            "feature_ready": ready,
            "next_date": pd.Series(prices.index, index=prices.index).shift(-1),
            "return_interval_contiguous": complete & complete.shift(1, fill_value=False),
            "reward_known": known,
        },
        index=prices.index,
    )
    return ResearchReplay(features, rewards, rows, asset_positions)
