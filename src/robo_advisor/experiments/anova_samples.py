"""#42 ANOVA 표본 생성: 검정 1 seed×윈도우, 검정 2·3 21거래일 블록.

평가 프로토콜(#1) 7·8절을 따른다. 일별 수익률을 표본으로 쓰지 않는다.

- 검정 1: 보상·seed·테스트 윈도우마다 테스트 구간 샤프 한 행.
- 검정 2: 윈도우의 테스트 첫날부터 겹치지 않는 ``block_size`` 거래일 블록마다
  전략별 블록 샤프 한 행. 끝의 불완전 블록은 제외한다. 세 전략이 같은 블록을
  공유하므로 블록 간 반복측정 구조가 남는다(리포트에 한계로 명시).
- 검정 3: 검정 2의 블록에 ``regimes.label_market_regimes``로 블록 직전
  ``lookback`` 거래일 벤치마크 수익률 기반 국면을 붙인다. unknown 블록은 제외하고
  개수를 보고한다. 셀 표본이 부족하면 블록 크기·임계값을 바꾸지 않고 수행 불가로
  표시한다.

샤프는 ``backtest.metrics.calculate_metrics``와 같은 정의(연 rf의 기하 일 변환,
표본 표준편차, √252 연환산)다. 정의 불가능한 샤프는 표본을 버리지 않고 실패한다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from robo_advisor.experiments.regimes import label_market_regimes

PERIODS_PER_YEAR = 252
REWARD_COLUMNS = ["reward", "seed", "window", "sharpe"]
BLOCK_COLUMNS = ["window", "block_id", "block_start", "block_end", "strategy", "sharpe"]


def _finite_returns(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all() or (array < -1).any():
        raise ValueError(f"{name} returns must be finite and at least -1")
    return array


def _sharpe(values: np.ndarray, risk_free_rate: float) -> float:
    daily_rf = np.expm1(np.log1p(risk_free_rate) / PERIODS_PER_YEAR)
    std = values.std(ddof=1)
    if not std > 0:
        return float("nan")
    return float((values - daily_rf).mean() * np.sqrt(PERIODS_PER_YEAR) / std)


def _ordered_index(index: pd.Index, name: str) -> None:
    if (
        not isinstance(index, pd.DatetimeIndex)
        or not index.is_unique
        or not index.is_monotonic_increasing
    ):
        raise ValueError(f"{name} must have a unique increasing DatetimeIndex")


def reward_samples(
    run_returns: Mapping[tuple[str, int, str], pd.Series], *, risk_free_rate: float = 0.0
) -> pd.DataFrame:
    """검정 1 표본. 키는 (보상, seed, 윈도우), 값은 그 실행의 테스트 일별 수익률.

    모든 보상이 같은 seed×윈도우 조합을 가져야 한다(균형 설계).
    """
    if not run_returns:
        raise ValueError("run_returns must not be empty")
    designs: dict[str, set[tuple[int, str]]] = {}
    rows = []
    for key, series in run_returns.items():
        reward, seed, window = key
        if not isinstance(series, pd.Series) or len(series) < 2:
            raise ValueError(f"{key} requires a Series of at least two returns")
        _ordered_index(series.index, str(key))
        value = _sharpe(_finite_returns(series, str(key)), risk_free_rate)
        if not np.isfinite(value):
            raise ValueError(f"{key} has an undefined Sharpe ratio")
        designs.setdefault(reward, set()).add((seed, window))
        rows.append((reward, seed, window, value))
    if len(designs) < 2:
        raise ValueError("test 1 requires at least two reward functions")
    if len({frozenset(cells) for cells in designs.values()}) != 1:
        raise ValueError("every reward must have the same seed x window runs")
    return (
        pd.DataFrame(rows, columns=REWARD_COLUMNS)
        .sort_values(["reward", "window", "seed"], kind="stable")
        .reset_index(drop=True)
    )


def average_seed_returns(seed_returns: pd.DataFrame, *, name: str = "ppo") -> pd.Series:
    """PPO seed들의 일별 수익률 평균. 블록 샤프는 이 평균 수익률로 계산한다."""
    if not isinstance(seed_returns, pd.DataFrame) or seed_returns.shape[1] == 0:
        raise ValueError("seed_returns must be a DataFrame with one column per seed")
    _finite_returns(seed_returns.to_numpy(dtype=float), "seed")
    return seed_returns.mean(axis=1).rename(name)


def block_samples(
    window_returns: Mapping[str, pd.DataFrame],
    *,
    block_size: int = 21,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """검정 2 표본. 값은 윈도우의 테스트 일별 수익률(열 = 전략, 같은 날짜)."""
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 2:
        raise ValueError("block_size must be an integer of at least 2")
    if not window_returns:
        raise ValueError("window_returns must not be empty")
    rows = []
    for window, frame in window_returns.items():
        if not isinstance(frame, pd.DataFrame) or frame.shape[1] < 2:
            raise ValueError(f"{window} requires a DataFrame with two or more strategies")
        _ordered_index(frame.index, str(window))
        count = len(frame) // block_size
        if count == 0:
            raise ValueError(f"{window} has no complete block of {block_size} sessions")
        values = _finite_returns(frame.iloc[: count * block_size].to_numpy(), str(window))
        for block_id in range(count):
            span = slice(block_id * block_size, (block_id + 1) * block_size)
            start, end = frame.index[span][[0, -1]]
            for column, strategy in enumerate(frame.columns):
                value = _sharpe(values[span, column], risk_free_rate)
                if not np.isfinite(value):
                    raise ValueError(
                        f"{window} block {block_id} {strategy} has an undefined Sharpe ratio"
                    )
                rows.append((window, block_id, start, end, str(strategy), value))
    return pd.DataFrame(rows, columns=BLOCK_COLUMNS)


@dataclass(frozen=True)
class RegimeSamples:
    """검정 3 표본과 제외·셀 부족 보고."""

    samples: pd.DataFrame
    excluded_unknown_blocks: int
    cell_counts: pd.Series
    insufficient_cells: tuple[tuple[str, str], ...]
    feasible: bool
    rules: dict


def regime_block_samples(
    blocks: pd.DataFrame,
    benchmark_returns: Mapping[str, pd.Series],
    *,
    block_size: int = 21,
    lookback: int = 60,
    threshold: float = 0.05,
    min_blocks_per_cell: int = 2,
) -> RegimeSamples:
    """검정 3 표본. ``blocks``는 ``block_samples`` 결과다.

    ``benchmark_returns[window]``는 테스트 첫날 이전 ``lookback`` 거래일 이력을 포함한
    연속 일별 수익률이며, 그 인덱스를 거래일 달력으로 쓴다. 블록 경계가 이 달력과
    어긋나면 실패한다.
    """
    if list(blocks.columns) != BLOCK_COLUMNS:
        raise ValueError("blocks must come from block_samples")
    labelled = []
    excluded = 0
    for window, group in blocks.groupby("window", sort=False):
        if window not in benchmark_returns:
            raise ValueError(f"{window} has no benchmark returns")
        benchmark = benchmark_returns[window]
        calendar = benchmark.index
        starts = group.drop_duplicates("block_id").set_index("block_id")
        test_start = starts.block_start.iloc[0]
        last = calendar.get_indexer([starts.block_end.iloc[-1]])[0]
        if last < 0:
            raise ValueError(f"{window} block dates are missing from the benchmark calendar")
        result = label_market_regimes(
            benchmark,
            trading_calendar=calendar,
            block_anchor=test_start,
            block_size=block_size,
            min_blocks_per_regime=min_blocks_per_cell,
            lookback=lookback,
            threshold=threshold,
            evaluation_start=test_start,
            evaluation_end=calendar[last + 1] if last + 1 < len(calendar) else None,
        )
        labels = {}
        for block in result.blocks:
            if block.block_id in starts.index:
                expected = starts.loc[block.block_id]
                if (
                    block.start != expected.block_start
                    or block.last_scheduled_date != expected.block_end
                ):
                    raise ValueError(f"{window} benchmark calendar disagrees with block dates")
                labels[block.block_id] = block.label
        if set(labels) != set(starts.index):
            raise ValueError(f"{window} benchmark calendar disagrees with block dates")
        excluded += sum(label == "unknown" for label in labels.values())
        regimes = group.block_id.map(labels)
        labelled.append(group.assign(regime=regimes)[regimes != "unknown"])
    samples = pd.concat(labelled, ignore_index=True)
    strategies = sorted(blocks.strategy.unique())
    regimes = sorted(samples.regime.unique())
    index = pd.MultiIndex.from_product([strategies, regimes], names=["strategy", "regime"])
    counts = samples.groupby(["strategy", "regime"]).size().reindex(index, fill_value=0)
    insufficient = tuple(cell for cell, count in counts.items() if count < min_blocks_per_cell)
    return RegimeSamples(
        samples=samples,
        excluded_unknown_blocks=excluded,
        cell_counts=counts,
        insufficient_cells=insufficient,
        feasible=len(regimes) >= 2 and not insufficient,
        rules={
            "block_size": block_size,
            "lookback": lookback,
            "threshold": threshold,
            "min_blocks_per_cell": min_blocks_per_cell,
            "sample_unit": "complete_block_sharpe",
        },
    )
