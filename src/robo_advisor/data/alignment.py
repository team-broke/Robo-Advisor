"""KRX/US 공통 거래일의 시점 정합, 원화 가격 및 후행 로그수익률.

명시적인 시장 캘린더와 환율 이용 가능 시각이 필요하다. 관측 행으로 휴장일을
추정하지 않는다. #10 검증기의 날짜/가격/결측 계약을 재사용하며 입력은 수정하지
않는다. 처리 순서와 반환 계약은 docs/trading-calendar-alignment.md 참조.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from robo_advisor.data.validation import (
    _calendar,
    _session_dates,
    validate_missing_intervals,
    validate_price_data,
)

KR_TIMEZONE = "Asia/Seoul"
US_TIMEZONE = "America/New_York"


class AlignmentError(ValueError):
    """검증 실패. reports에 #10 원천 검증 결과 또는 시점 결함을 보존한다."""

    def __init__(self, message, reports):
        super().__init__(message)
        self.reports = reports


@dataclass
class LaggedPrices:
    """전체 미국 거래일 축의 1세션 래그 결과. 최초 세션은 NaN/NaT다."""

    prices: pd.DataFrame
    source_dates: pd.DataFrame
    available_at: pd.DataFrame
    forward_filled: pd.DataFrame
    diagnostics: dict


@dataclass
class AlignmentResult:
    """동일한 index/자산 순서의 가격·수익률 및 사용한 원천 관측 계보.

    prices_krw/log_returns에는 NaN/Inf가 없다. source_dates, available_at,
    forward_filled는 자산별로 대응한다. fx에는 USD/KRW rate, observation_date,
    available_at, forward_filled가 있다. 시각은 UTC aware, index/원천 날짜는
    timezone-naive 현지 거래일 라벨이다. 빈 결과도 진단과 함께 반환한다.
    """

    prices_krw: pd.DataFrame
    log_returns: pd.DataFrame
    source_dates: pd.DataFrame
    available_at: pd.DataFrame
    forward_filled: pd.DataFrame
    fx: pd.DataFrame
    evaluation_at: pd.Series
    return_start_dates: pd.Series
    diagnostics: dict


def _prices(data, name, timezone=None, *, allow_missing=True):
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    report = validate_price_data(data, price_columns=tuple(data.columns), market_timezone=timezone)
    defects = [
        issue
        for issue in report["issues"]
        if not (allow_missing and issue["code"] == "MISSING_PRICE")
    ]
    if defects:
        raise AlignmentError(f"{name}: invalid dates or prices", {name: report})
    dates, _ = _session_dates(data.index, timezone)
    values = data.copy().astype(float)
    values.index = dates
    return values, report


def _clock(dates, timezone, hour, minute=0):
    local = dates + pd.Timedelta(hours=hour, minutes=minute)
    return pd.Series(local.tz_localize(timezone).tz_convert("UTC"), index=dates)


def _availability(series, dates, name, timezone):
    """이용 가능 시각은 명시적인 aware 시각이며 관측일 시작보다 이르지 않아야 한다."""
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series of timezone-aware timestamps")
    # 날짜 검증은 #10과 같은 정규화/순서 계약을 사용한다.
    labels = _calendar(series.index, timezone)
    if not labels.equals(dates):
        raise ValueError(f"{name} index must match observation dates in the same order")
    timestamps = []
    for value in series:
        if not isinstance(value, (str, datetime, pd.Timestamp)):
            raise ValueError(f"{name} must contain timezone-aware timestamps without NaT")
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp) or timestamp.tzinfo is None:
            raise ValueError(f"{name} must contain timezone-aware timestamps without NaT")
        timestamps.append(timestamp.tz_convert("UTC"))
    values = pd.Series(pd.DatetimeIndex(timestamps), index=dates, name=name)
    starts = _clock(dates, timezone, 0)
    if (values < starts).any():
        raise AlignmentError(
            f"{name} precedes its observation date", {name: "INVALID_AVAILABILITY"}
        )
    return values


def align_trading_calendars(
    kr_calendar: pd.DatetimeIndex, us_calendar: pd.DatetimeIndex
) -> pd.DatetimeIndex:
    """실제 시장 거래일 목록의 교집합. 중복/역순/NaT는 거부하고 정렬하지 않는다.

    캘린더 출처와 기간의 완전성은 호출자가 보장한다. KRX는 서울, US는 뉴욕
    현지 날짜로 #10 규칙에 따라 정규화한다. 공통 날짜는 동일 시각을 뜻하지 않는다.
    """
    kr_dates = _calendar(kr_calendar, KR_TIMEZONE)
    us_dates = _calendar(us_calendar, US_TIMEZONE)
    return kr_dates[kr_dates.isin(us_dates)].copy()


def _fill_market(values, calendar, available, limit, name):
    report = validate_missing_intervals(
        values,
        columns=tuple(values.columns),
        trading_calendar=calendar,
        max_missing_ratio=1.0,
        max_consecutive_missing=limit,
    )
    if report["status"] != "passed":
        raise AlignmentError(
            f"{name}: calendar or missing interval validation failed", {name: report}
        )
    raw = values.reindex(calendar)
    dates = pd.DataFrame({column: calendar for column in raw.columns}, index=calendar)
    times = pd.DataFrame({column: available.reindex(calendar) for column in raw.columns})
    dates = dates.where(raw.notna())
    times = times.where(raw.notna())
    if limit:
        filled = raw.ffill(limit=limit)
        dates = dates.ffill(limit=limit)
        times = times.ffill(limit=limit)
    else:
        filled = raw.copy()
    flags = raw.isna() & filled.notna()
    after = validate_missing_intervals(
        filled,
        columns=tuple(filled.columns),
        trading_calendar=calendar,
        max_missing_ratio=1.0,
        max_consecutive_missing=limit,
    )
    diagnostics = {
        "missing_before_fill": report,
        "missing_after_fill": after,
        "forward_fill_counts": {column: int(flags[column].sum()) for column in raw.columns},
    }
    return filled, dates, times, flags, diagnostics


def lag_overseas_prices(
    prices_usd: pd.DataFrame,
    us_calendar: pd.DatetimeIndex,
    *,
    available_at: pd.Series | None = None,
    max_forward_fill: int = 5,
) -> LaggedPrices:
    """전체 US 캘린더에서 결측 정책을 적용하고 1개 해외 거래 세션만큼 shift(1).

    누락 행도 세션으로 센다. 교집합으로 자른 뒤 shift하지 않는다. 기본 이용 가능
    시각은 뉴욕 16:00(조기 폐장에도 보수적인 시각)이며 늦은 공급자 공개 시각은
    available_at으로 전달한다. 실제 이용 가능 여부는 한국 평가 시각에서 검사한다.
    """
    calendar = _calendar(us_calendar, US_TIMEZONE)
    values, validation = _prices(prices_usd, "overseas", US_TIMEZONE)
    available = (
        _clock(values.index, US_TIMEZONE, 16)
        if available_at is None
        else _availability(available_at, values.index, "overseas_available_at", US_TIMEZONE)
    )
    filled, dates, times, flags, diagnostics = _fill_market(
        values, calendar, available, max_forward_fill, "overseas"
    )
    diagnostics["raw_validation"] = validation
    return LaggedPrices(
        filled.shift(1),
        dates.shift(1),
        times.shift(1),
        flags.shift(1, fill_value=False),
        diagnostics,
    )


def convert_prices_to_krw(
    prices_usd: pd.DataFrame,
    fx_rates: pd.Series,
    *,
    fx_direction: str,
) -> pd.DataFrame:
    """같은 평가일로 시점 정합된 해외 가격만 KRW로 환산한다.

    USD/KRW는 1 USD당 KRW로 곱하고, KRW/USD는 1 KRW당 USD로 나눈다.
    방향은 필수이며 수치 크기로 추측하지 않는다. 시점 선택은 preprocess의 책임이다.
    결측은 유지하고 비양수/비유한 값과 환산 overflow/underflow는 거부한다.
    """
    if fx_direction not in ("USD/KRW", "KRW/USD"):
        raise ValueError("fx_direction must be 'USD/KRW' or 'KRW/USD'")
    prices, _ = _prices(prices_usd, "prices_usd")
    if not isinstance(fx_rates, pd.Series):
        raise TypeError("fx_rates must be a pandas Series")
    fx, _ = _prices(fx_rates.to_frame("rate"), "fx_rates")
    if not fx.index.equals(prices.index):
        raise ValueError("fx_rates must match price dates in the same order")
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        converted = (
            prices.mul(fx["rate"], axis=0)
            if fx_direction == "USD/KRW"
            else prices.div(fx["rate"], axis=0)
        )
    converted, _ = _prices(converted, "converted_prices")
    return converted


def calculate_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """직전 평가 행으로부터의 후행 log return. 첫 행/미보정 결측 전후는 NaN이다.

    log(price).diff()는 log(price_t/price_t_minus_1)와 같으며 비율의 수치적
    overflow를 피한다. 결측 행을 먼저 drop하여 기간을 건너뛰지 않는다.
    """
    values, _ = _prices(prices, "log_return_prices")
    return np.log(values).diff()


def _select_fx(fx_rates, available_at, cutoffs, direction, max_age, fill_limit):
    if direction not in ("USD/KRW", "KRW/USD"):
        raise ValueError("fx_direction must be 'USD/KRW' or 'KRW/USD'")
    if not isinstance(fx_rates, pd.Series):
        raise TypeError("fx_rates must be a pandas Series")
    rates, validation = _prices(fx_rates.to_frame("rate"), "fx", KR_TIMEZONE)
    if available_at is None:
        raise ValueError("fx_available_at is required; observation dates do not prove availability")
    available = _availability(available_at, rates.index, "fx_available_at", KR_TIMEZONE)
    if not available.is_monotonic_increasing:
        raise ValueError("fx_available_at must be increasing; do not reorder observations silently")
    if not isinstance(max_age, pd.Timedelta) or pd.isna(max_age) or max_age <= pd.Timedelta(0):
        raise ValueError("max_fx_age must be a positive pandas Timedelta")
    missing = validate_missing_intervals(
        rates,
        columns=("rate",),
        trading_calendar=rates.index,
        max_missing_ratio=1.0,
        max_consecutive_missing=fill_limit,
    )
    if missing["status"] != "passed":
        raise AlignmentError("fx: missing interval validation failed", {"fx": missing})
    valid = rates["rate"].notna()
    released = pd.DatetimeIndex(available[valid])
    positions = released.searchsorted(pd.DatetimeIndex(cutoffs), side="right") - 1
    output = pd.DataFrame(index=cutoffs.index)
    output["rate"] = np.nan
    output["observation_date"] = pd.NaT
    output["available_at"] = pd.Series(pd.NaT, index=cutoffs.index, dtype="datetime64[ns, UTC]")
    found = positions >= 0
    if found.any():
        selected = positions[found]
        rate_values = rates.loc[valid, "rate"].to_numpy()[selected]
        if direction == "KRW/USD":
            with np.errstate(over="ignore", divide="ignore"):
                rate_values = 1.0 / rate_values
        if not np.isfinite(rate_values).all() or (rate_values <= 0).any():
            raise AlignmentError("fx: inverted rate is not finite and positive", {"fx": validation})
        output.loc[found, "rate"] = rate_values
        output.loc[found, "observation_date"] = rates.index[valid][selected]
        output.loc[found, "available_at"] = released[selected]
    # 관측일/공개일 모두의 나이를 제한하여 지연 공개된 오래된 관측의 재사용을 막는다.
    observation_start = output["observation_date"].dt.tz_localize(KR_TIMEZONE).dt.tz_convert("UTC")
    stale = (cutoffs - output["available_at"] > max_age) | (cutoffs - observation_start > max_age)
    output["forward_filled"] = found & (output["observation_date"] != output.index)
    output.loc[stale, "rate"] = np.nan
    return output, stale, {"raw_validation": validation, "missing_before_fill": missing}


def preprocess_market_data(
    domestic_prices_krw: pd.DataFrame,
    overseas_prices_usd: pd.DataFrame,
    fx_rates: pd.Series,
    *,
    kr_calendar: pd.DatetimeIndex,
    us_calendar: pd.DatetimeIndex,
    fx_available_at: pd.Series,
    fx_direction: str,
    domestic_available_at: pd.Series | None = None,
    overseas_available_at: pd.Series | None = None,
    max_forward_fill: int = 5,
    max_fx_age: pd.Timedelta = pd.Timedelta(days=7),
) -> AlignmentResult:
    """KRX/US 공통 평가일 서울 15:30의 가격과 후행 수익률을 구성한다.

    시장별 제한적 forward fill → 전체 US 세션 lag → 공통 평가 시각의 공개된
    FX 선택/환산 → 공통 날짜 축 log return → 불완전 행 제외. 가격/FX 공개 지연은
    평가 시각에서 mask하며 이후 fill하지 않는다. 벤치마크 USD 가격도
    overseas_prices_usd에 열로 전달하면 같은 통화/lag 규칙이 적용된다.
    """
    kr_dates = _calendar(kr_calendar, KR_TIMEZONE)
    us_dates = _calendar(us_calendar, US_TIMEZONE)
    common = align_trading_calendars(kr_calendar, us_calendar)
    if common.empty:
        raise AlignmentError("calendars have no common trading sessions", {"calendar": "NO_COMMON"})
    domestic, validation = _prices(domestic_prices_krw, "domestic", KR_TIMEZONE)
    if not isinstance(overseas_prices_usd, pd.DataFrame):
        raise TypeError("overseas_prices_usd must be a pandas DataFrame")
    if domestic.columns.intersection(overseas_prices_usd.columns).size:
        raise ValueError("domestic and overseas asset names must be disjoint")
    domestic_times = (
        _clock(domestic.index, KR_TIMEZONE, 15, 30)
        if domestic_available_at is None
        else _availability(
            domestic_available_at, domestic.index, "domestic_available_at", KR_TIMEZONE
        )
    )
    kr_values, kr_sources, kr_times, kr_flags, kr_diagnostics = _fill_market(
        domestic, kr_dates, domestic_times, max_forward_fill, "domestic"
    )
    kr_diagnostics["raw_validation"] = validation
    us = lag_overseas_prices(
        overseas_prices_usd,
        us_calendar,
        available_at=overseas_available_at,
        max_forward_fill=max_forward_fill,
    )
    cutoffs = _clock(common, KR_TIMEZONE, 15, 30)
    sources = pd.concat([kr_sources.reindex(common), us.source_dates.reindex(common)], axis=1)
    times = pd.concat([kr_times.reindex(common), us.available_at.reindex(common)], axis=1)
    flags = pd.concat([kr_flags.reindex(common), us.forward_filled.reindex(common)], axis=1)
    values = pd.concat([kr_values.reindex(common), us.prices.reindex(common)], axis=1)
    unavailable = times.gt(cutoffs, axis=0)
    values = values.mask(unavailable)
    fx, stale, fx_diagnostics = _select_fx(
        fx_rates, fx_available_at, cutoffs, fx_direction, max_fx_age, max_forward_fill
    )
    overseas = convert_prices_to_krw(values[us.prices.columns], fx["rate"], fx_direction="USD/KRW")
    prices = pd.concat([values[domestic.columns], overseas], axis=1)
    returns = calculate_log_returns(prices)
    complete = prices.notna().all(axis=1) & returns.notna().all(axis=1)
    excluded = []
    # Both calendars are already increasing; union is used for diagnostics, never to repair inputs.
    for day in kr_dates.union(us_dates):
        if day not in common:
            missing_market = "KRX" if day not in kr_dates else "US"
            missing_calendar = kr_dates if missing_market == "KRX" else us_dates
            reason = (
                f"OUTSIDE_{missing_market}_CALENDAR_RANGE"
                if day < missing_calendar[0] or day > missing_calendar[-1]
                else f"{missing_market}_CLOSED"
            )
            excluded.append({"date": day.date().isoformat(), "reasons": [reason]})
    for position, day in enumerate(common):
        if complete.iloc[position]:
            continue
        reasons = []
        if unavailable.loc[day].any():
            reasons.append("PRICE_NOT_AVAILABLE")
        if values.loc[day].isna().any():
            reasons.append("MISSING_ALIGNED_PRICE")
        if pd.isna(fx.loc[day, "rate"]):
            reasons.append("FX_STALE" if stale.loc[day] else "FX_NOT_AVAILABLE")
        if position == 0:
            reasons.append("RETURN_WARMUP")
        elif prices.iloc[position - 1].isna().any():
            reasons.append("MISSING_PREVIOUS_PRICE")
        excluded.append({"date": day.date().isoformat(), "reasons": reasons})
    output_prices = prices.loc[complete].copy()
    output_returns = returns.loc[complete].copy()
    finite = (
        np.isfinite(output_prices.to_numpy()).all() and np.isfinite(output_returns.to_numpy()).all()
    )
    if not finite:
        raise AlignmentError("final prices or returns are not finite", {"finite": False})
    diagnostics = {
        "policy": {
            "calendar": "KRX_US_INTERSECTION",
            "overseas_lag_sessions": 1,
            "evaluation_time": "15:30 Asia/Seoul",
            "domestic_availability": (
                "15:30 Asia/Seoul" if domestic_available_at is None else "explicit"
            ),
            "overseas_availability": (
                "16:00 America/New_York" if overseas_available_at is None else "explicit"
            ),
            "fx_input_direction": fx_direction,
            "output_currency": "KRW",
            "max_forward_fill": max_forward_fill,
            "max_fx_age": str(max_fx_age),
            "returns": "previous_common_session; no gap bridging",
        },
        "domestic": kr_diagnostics,
        "overseas": us.diagnostics,
        "fx": fx_diagnostics,
        "excluded_dates": excluded,
        "common_row_count": len(common),
        "dropped_common_row_count": int((~complete).sum()),
        "final_row_count": len(output_prices),
        "aligned_missing_count": int(prices.isna().sum().sum()),
        "forward_fill_counts": {column: int(flags[column].sum()) for column in flags.columns},
        "fx_forward_fill_count": int((fx["forward_filled"] & fx["rate"].notna()).sum()),
        "finite_check": {"passed": bool(finite), "nan_count": 0, "inf_count": 0},
    }
    return_starts = pd.Series(common, index=common, name="return_start_date").shift(1)
    return AlignmentResult(
        output_prices,
        output_returns,
        sources.loc[complete].copy(),
        times.loc[complete].copy(),
        flags.loc[complete].copy(),
        fx.loc[complete].copy(),
        cutoffs.loc[complete].copy(),
        return_starts.loc[complete].copy(),
        diagnostics,
    )
