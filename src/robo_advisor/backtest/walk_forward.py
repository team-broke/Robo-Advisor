"""고정 시작일을 사용하는 달력 연도 기반 rolling 분할기."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardWindow:
    """iloc 위치와 반개구간 경계. warmup은 test 이전 문맥이며 평가 대상이 아니다."""

    train_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train: np.ndarray
    test: np.ndarray
    warmup: np.ndarray


def walk_forward_split(
    dates: pd.DatetimeIndex,
    *,
    start_date: str | pd.Timestamp,
    train_years: int = 4,
    test_years: int = 1,
    warmup_periods: int = 0,
    min_windows: int = 2,
    coverage_start: str | pd.Timestamp | None = None,
    coverage_end: str | pd.Timestamp | None = None,
) -> list[WalkForwardWindow]:
    """학습/평가를 [start, end)로 나누고 test_years씩 이동한다.

    start_date는 첫 학습 시작일로 사전에 고정한다. 워밍업은 테스트 직전의
    관측 문맥이며 학습 데이터와 겹칠 수 있다. scaler는 train에만 fit해야 한다.
    coverage_start/end는 수집 데이터의 시작일/배타적 종료일로, 휴장일 때문에
    관측이 달력 경계에 없을 때 호출자가 명시한다. 기본값은 첫 관측/마지막 관측 다음 날.
    미완료 테스트 구간은 반환하지 않으며 최소 윈도우 부족은 오류로 처리한다.
    """
    if not isinstance(dates, pd.DatetimeIndex):
        raise TypeError("dates must be a DatetimeIndex")
    if dates.empty or dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("dates must be nonempty, unique, increasing and without NaT")
    for name, value, minimum in (
        ("train_years", train_years, 1),
        ("test_years", test_years, 1),
        ("warmup_periods", warmup_periods, 0),
        ("min_windows", min_windows, 2),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    start = pd.Timestamp(start_date)
    available_start = pd.Timestamp(coverage_start) if coverage_start is not None else dates[0]
    end = (
        pd.Timestamp(coverage_end) if coverage_end is not None else dates[-1] + pd.Timedelta(days=1)
    )
    if pd.isna(start) or pd.isna(end) or pd.isna(available_start):
        raise ValueError("date boundaries must not be NaT")
    if any(str(boundary.tzinfo) != str(dates.tz) for boundary in (start, available_start, end)):
        raise ValueError("date boundaries must have the same timezone as dates")
    if available_start > dates[0] or start < available_start or end <= dates[-1]:
        raise ValueError("start must be covered and coverage_end must follow the final observation")
    windows = []
    anchor = start
    while True:
        test_start = start + pd.DateOffset(years=train_years + len(windows) * test_years)
        test_end = start + pd.DateOffset(years=train_years + (len(windows) + 1) * test_years)
        if test_end > end:
            break
        train = np.flatnonzero((dates >= anchor) & (dates < test_start))
        test = np.flatnonzero((dates >= test_start) & (dates < test_end))
        if not train.size or not test.size:
            raise ValueError("each window requires nonempty train and test data")
        boundary = int(dates.searchsorted(test_start))
        if boundary < warmup_periods:
            raise ValueError("insufficient past observations for warmup")
        warmup = np.arange(boundary - warmup_periods, boundary)
        for positions in (train, test, warmup):
            positions.setflags(write=False)
        windows.append(WalkForwardWindow(anchor, test_start, test_end, train, test, warmup))
        # 원래 시작일을 기준으로 계산하여 윤일 반올림이 누적되지 않게 한다.
        anchor = start + pd.DateOffset(years=len(windows) * test_years)
    if len(windows) < min_windows:
        raise ValueError(f"at least {min_windows} complete windows are required")
    return windows
