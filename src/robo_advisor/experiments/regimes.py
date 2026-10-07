"""Causal market regime labels and descriptive block coverage reports."""

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Literal

import numpy as np
import pandas as pd

Regime = Literal["up", "down", "sideways", "unknown"]
KNOWN_REGIMES = ("up", "down", "sideways")


@dataclass(frozen=True)
class RegimeRules:
    """Applied rules, including the fixed calendar, for reproduction."""

    lookback: int
    threshold: float
    boundary_tolerance: float
    block_size: int
    block_anchor: pd.Timestamp
    min_blocks_per_regime: int
    trading_calendar: tuple[pd.Timestamp, ...]
    evaluation_start: pd.Timestamp
    evaluation_end: pd.Timestamp | None
    boundary_policy: str = "sideways_including_equal_boundaries"
    incomplete_policy: str = "label_but_exclude_from_minimum"
    sample_unit: str = "complete_labeled_block"


@dataclass(frozen=True)
class RegimeBlock:
    """One fixed-calendar block, possibly only partly observed in evaluation."""

    block_id: int
    start: pd.Timestamp
    last_scheduled_date: pd.Timestamp | None
    observed_start: pd.Timestamp
    observed_end: pd.Timestamp
    observation_count: int
    complete: bool
    label: Regime
    history_count: int
    cumulative_return: float | None
    reason: str | None


@dataclass(frozen=True)
class RegimeCoverage:
    """Observation days are descriptive; only complete blocks meet the minimum."""

    observation_count: int
    block_count: int
    complete_block_count: int
    required_blocks: int
    shortfall: int
    sufficient: bool


@dataclass(frozen=True)
class RegimeResult:
    """Daily labels, block decisions and structured sample shortfalls.

    Counts describe one benchmark trajectory. Even disjoint complete blocks
    are not asserted to be independent ANOVA samples: overlapping labeling
    windows and temporal dependence remain. Seed/window performance samples
    and the statistical evaluation protocol belong to the downstream analysis.
    """

    labels: pd.Series
    blocks: tuple[RegimeBlock, ...]
    coverage: dict[str, RegimeCoverage]
    insufficient_regimes: tuple[str, ...]
    unknown_observation_count: int
    unknown_block_count: int
    rules: RegimeRules


def _calendar(value: pd.DatetimeIndex, name: str) -> pd.DatetimeIndex:
    if (
        not isinstance(value, pd.DatetimeIndex)
        or value.empty
        or value.hasnans
        or not value.is_unique
        or not value.normalize().is_unique
        or not value.is_monotonic_increasing
    ):
        raise ValueError(
            f"{name} must be a nonempty ordered DatetimeIndex with one session per day"
        )
    return value.copy()


def _timestamp(value, calendar: pd.DatetimeIndex, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a timestamp") from error
    if pd.isna(timestamp) or (timestamp.tzinfo is None) != (calendar.tz is None):
        raise ValueError(f"{name} must match calendar timezone awareness")
    return timestamp.tz_convert(calendar.tz) if calendar.tz is not None else timestamp


def label_market_regimes(
    benchmark_returns: pd.Series,
    *,
    trading_calendar: pd.DatetimeIndex,
    block_anchor,
    block_size: int,
    min_blocks_per_regime: int,
    lookback: int = 60,
    threshold: float = 0.05,
    boundary_tolerance: float = 1e-12,
    evaluation_start=None,
    evaluation_end=None,
) -> RegimeResult:
    """Label fixed trading-session blocks using only pre-block simple returns.

    Supply an already aligned synthetic benchmark; no index synthesis or
    strategy performance optimization is performed here. block_size, anchor
    and minimum block count have no project defaults. Calendar must retain
    the same sessions through each evaluated block across calls. Blocks start
    at calendar[anchor_position + k*block_size], k >= 0; pre-anchor returns
    serve only as history. Calendar is required because a holiday calendar
    cannot be inferred invariantly from a sliced series.

    benchmark_returns must be a contiguous subset of that calendar. Labels use
    exactly the lookback sessions immediately BEFORE the scheduled block start,
    even if evaluation starts inside a block. The compounded return is
    product(1+r)-1, computed with log1p/expm1; a -100% day gives -1. A value
    above threshold+tolerance is up, below -threshold-tolerance is down;
    equal boundaries (and absolute rounding tolerance) are sideways. Missing
    pre-block history yields unknown, never sideways. No bfill is used.

    evaluation_start is inclusive, evaluation_end exclusive. Start defaults to
    the later of anchor and first input date; end defaults to all supplied data.
    To slice evaluation without changing established labels, keep the pre-block
    history in benchmark_returns and change these bounds. Removing required
    history necessarily makes that block unknown; it cannot be reconstructed
    from current or future block returns. Appending future observations does
    not change existing block boundaries or labels, though it can complete the
    last block and update coverage counts.

    Partial first/last blocks keep their pre-block label. A complete block has
    all block_size sessions inside evaluation and input; an incomplete calendar
    tail is also incomplete. Reports separate days, all observed blocks, and
    complete known blocks. The minimum is checked only against the latter;
    it is a coverage requirement, not a test of statistical independence.
    """
    for name, value in (
        ("lookback", lookback),
        ("block_size", block_size),
        ("min_blocks_per_regime", min_blocks_per_regime),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, Real)
        or not isfinite(threshold)
        or not 0 < threshold < 1
    ):
        raise ValueError("threshold must be finite and in (0, 1)")
    if (
        isinstance(boundary_tolerance, bool)
        or not isinstance(boundary_tolerance, Real)
        or not isfinite(boundary_tolerance)
        or not 0 <= boundary_tolerance < threshold
    ):
        raise ValueError("boundary_tolerance must be finite and in [0, threshold)")
    calendar = _calendar(trading_calendar, "trading_calendar")
    if not isinstance(benchmark_returns, pd.Series):
        raise ValueError("benchmark_returns must be a Series")
    index = _calendar(benchmark_returns.index, "benchmark_returns index")
    if str(index.tz) != str(calendar.tz):
        raise ValueError("benchmark index timezone must match calendar")
    positions = calendar.get_indexer(index)
    if np.any(positions < 0) or np.any(np.diff(positions) != 1):
        raise ValueError("benchmark_returns must cover consecutive calendar sessions")
    if (
        not pd.api.types.is_numeric_dtype(benchmark_returns.dtype)
        or pd.api.types.is_complex_dtype(benchmark_returns.dtype)
        or pd.api.types.is_bool_dtype(benchmark_returns.dtype)
    ):
        raise ValueError("benchmark returns must be real numeric values")
    try:
        values = benchmark_returns.to_numpy(dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError("benchmark returns must be numeric") from error
    if not np.isfinite(values).all() or np.any(values < -1):
        raise ValueError("simple returns must be finite and at least -1")
    anchor = _timestamp(block_anchor, calendar, "block_anchor")
    anchor_position = calendar.get_indexer([anchor])[0]
    if anchor_position < 0:
        raise ValueError("block_anchor must be a calendar session")
    start = (
        max(anchor, index[0])
        if evaluation_start is None
        else _timestamp(evaluation_start, calendar, "evaluation_start")
    )
    end = None if evaluation_end is None else _timestamp(evaluation_end, calendar, "evaluation_end")
    if start < anchor or (end is not None and end <= start):
        raise ValueError("evaluation must start at or after anchor and end after start")
    mask = index >= start
    if end is not None:
        mask &= index < end
    selected = positions[mask]
    if selected.size == 0:
        raise ValueError("evaluation must contain at least one benchmark observation")
    labels = pd.Series("unknown", index=index[mask], name="regime", dtype=object)
    block_ids = (selected - anchor_position) // block_size
    blocks = []
    for block_id in np.unique(block_ids):
        first = anchor_position + int(block_id) * block_size
        stop = first + block_size
        block_mask = block_ids == block_id
        observed = selected[block_mask]
        history_start = max(0, first - lookback, int(positions[0]))
        history_stop = min(first, int(positions[-1]) + 1)
        count = max(0, history_stop - history_start)
        label, cumulative, reason = "unknown", None, "insufficient_history"
        if count == lookback:
            window = values[history_start - positions[0] : history_stop - positions[0]]
            if np.any(window == -1):
                cumulative = -1.0
            else:
                with np.errstate(over="ignore"):
                    cumulative = float(np.expm1(np.log1p(window).sum()))
                if not np.isfinite(cumulative):
                    raise ValueError("compounded historical return must be finite")
            label = (
                "up"
                if cumulative > threshold + boundary_tolerance
                else "down" if cumulative < -threshold - boundary_tolerance else "sideways"
            )
            reason = None
        labels.iloc[np.flatnonzero(block_mask)] = label
        blocks.append(
            RegimeBlock(
                block_id=int(block_id),
                start=calendar[first],
                last_scheduled_date=calendar[stop - 1] if stop <= len(calendar) else None,
                observed_start=calendar[observed[0]],
                observed_end=calendar[observed[-1]],
                observation_count=len(observed),
                complete=len(observed) == block_size,
                label=label,
                history_count=count,
                cumulative_return=cumulative,
                reason=reason,
            )
        )
    coverage = {}
    for regime in KNOWN_REGIMES:
        matching = [block for block in blocks if block.label == regime]
        complete_count = sum(block.complete for block in matching)
        coverage[regime] = RegimeCoverage(
            observation_count=sum(block.observation_count for block in matching),
            block_count=len(matching),
            complete_block_count=complete_count,
            required_blocks=min_blocks_per_regime,
            shortfall=max(0, min_blocks_per_regime - complete_count),
            sufficient=complete_count >= min_blocks_per_regime,
        )
    return RegimeResult(
        labels=labels,
        blocks=tuple(blocks),
        coverage=coverage,
        insufficient_regimes=tuple(key for key, value in coverage.items() if not value.sufficient),
        unknown_observation_count=int((labels == "unknown").sum()),
        unknown_block_count=sum(block.label == "unknown" for block in blocks),
        rules=RegimeRules(
            lookback=lookback,
            threshold=float(threshold),
            boundary_tolerance=float(boundary_tolerance),
            block_size=block_size,
            block_anchor=anchor,
            min_blocks_per_regime=min_blocks_per_regime,
            trading_calendar=tuple(calendar),
            evaluation_start=start,
            evaluation_end=end,
        ),
    )
