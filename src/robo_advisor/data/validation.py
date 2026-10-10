"""원천 일별 시계열의 결함과 검증 미완료 사유를 기록한다.

입력은 한 자산의 DataFrame/Series이며 원본을 정렬, 삭제, 보간하지 않는다.
수집기(#9)는 날짜/가격 검증과 필요한 개별 검증을 호출하고, 스냅샷(#11)은
반환 dict를 출처/기간 메타데이터와 함께 기록할 수 있다. 전처리(#12) 전에
호출자가 failed/unverified 결과의 오류 처리 또는 제외 정책을 결정해야 한다.
valid는 모든 수행 항목이 확인된 passed 상태에서만 True이다.

일별 날짜는 market_timezone이 있으면 해당 시장 시간대로 변환한 현지 날짜,
없으면 입력 시간대의 현지 날짜를 사용한다. UTC 시각 스키마와 달리 거래일
라벨을 보존한다. naive/aware 혼합은 거부하며 시장 시간대가 없으면 aware
입력도 같은 시간대여야 한다. 달력과 비교 자료 역시 같은 시장 날짜를 써야 한다.
"""

from __future__ import annotations

from datetime import date, datetime
from numbers import Real

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype


def _issue(code, message, *, severity="error", **details):
    return {"code": code, "message": message, "severity": severity, **details}


def _result(issues, **metrics):
    severities = {issue["severity"] for issue in issues}
    status = (
        "failed"
        if "error" in severities
        else "unverified" if "unverified" in severities else "passed"
    )
    return {"valid": status == "passed", "status": status, "issues": issues, "metrics": metrics}


def _date_label(value):
    return value.date().isoformat()


def _session_dates(values, market_timezone=None):
    """거래일 순서를 보존하고 해석할 수 없는 날짜를 위치와 함께 기록한다."""
    timestamps, issues = [], []
    for position, value in enumerate(values):
        try:
            if not isinstance(value, (str, date, datetime, np.datetime64, pd.Timestamp)):
                raise ValueError("date must be a datetime or a date string")
            timestamp = pd.Timestamp(value)
            if pd.isna(timestamp):
                raise ValueError("date must not be NaT")
            timestamps.append(timestamp)
        except (TypeError, ValueError, OverflowError):
            issues.append(
                _issue("INVALID_DATE", "Invalid trading date", position=position, value=str(value))
            )
    if issues:
        return None, issues
    aware = [timestamp.tzinfo is not None for timestamp in timestamps]
    if any(aware) and not all(aware):
        return None, [_issue("INCONSISTENT_TIMEZONE", "Dates mix naive and aware timestamps")]
    if market_timezone is None and len({str(timestamp.tz) for timestamp in timestamps}) > 1:
        return None, [_issue("INCONSISTENT_TIMEZONE", "Dates must use the same timezone")]
    if market_timezone is not None:
        timestamps = [
            (
                timestamp.tz_convert(market_timezone)
                if timestamp.tzinfo is not None
                else timestamp.tz_localize(market_timezone)
            )
            for timestamp in timestamps
        ]
    dates = pd.DatetimeIndex(timestamps).normalize()
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    return dates, []


def _date_issues(dates):
    issues = []
    for position in np.flatnonzero(dates.duplicated(keep=False)):
        issues.append(
            _issue(
                "DUPLICATE_DATE",
                "Duplicate trading date",
                date=_date_label(dates[position]),
                position=int(position),
            )
        )
    for position in np.flatnonzero(dates[1:] < dates[:-1]) + 1:
        issues.append(
            _issue(
                "OUT_OF_ORDER_DATE",
                "Trading dates are out of order",
                position=int(position),
                date=_date_label(dates[position]),
                previous_date=_date_label(dates[position - 1]),
            )
        )
    return issues


def _date_context_issues(groups, market_timezone):
    """시장 시간대가 없으면 별도 입력들도 같은 시간대의 거래일을 요구한다."""
    if market_timezone is not None:
        return []
    representatives = [group[0] for group in groups if len(group)]
    _, issues = _session_dates(representatives)
    return issues


def _frame_dates(data, date_column, market_timezone):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if data.empty:
        return None, [_issue("EMPTY_DATA", "Data must not be empty")]
    if not data.columns.is_unique:
        return None, [
            _issue(
                "DUPLICATE_COLUMN",
                "Data columns must be unique",
                columns=[
                    str(column) for column in data.columns[data.columns.duplicated(keep=False)]
                ],
            )
        ]
    if date_column is not None:
        if date_column not in data.columns:
            return None, [_issue("MISSING_COLUMN", "Date column is missing", column=date_column)]
        values = data[date_column]
    else:
        if not isinstance(data.index, pd.DatetimeIndex):
            return None, [_issue("INVALID_DATE", "Use a DatetimeIndex or specify date_column")]
        values = data.index
    dates, issues = _session_dates(values, market_timezone)
    if dates is not None:
        issues.extend(_date_issues(dates))
    return dates, issues


def _columns(columns):
    if isinstance(columns, str):
        raise ValueError("columns must be a nonempty sequence of unique column names")
    columns = tuple(columns)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("columns must be a nonempty sequence of unique column names")
    return columns


def validate_price_data(
    data: pd.DataFrame,
    *,
    price_columns=("Close",),
    date_column=None,
    market_timezone=None,
) -> dict:
    """중복 거래일, 역순, 비양수, NaN/Inf, 비수치 가격을 모두 기록한다.

    price_columns에는 OHLC 등 가격 컬럼만 명시한다. Volume의 0은 가격 결함이 아니다.
    문자열 숫자도 타입 결함으로 기록하며 암묵적인 변환으로 수집 오류를 숨기지 않는다.
    """
    columns = _columns(price_columns)
    dates, issues = _frame_dates(data, date_column, market_timezone)
    raw_dates = None
    if data.columns.is_unique and date_column is not None and date_column in data.columns:
        raw_dates = list(data[date_column])
    elif date_column is None and isinstance(data.index, pd.DatetimeIndex):
        raw_dates = data.index
    for column in columns:
        if column not in data.columns:
            issues.append(_issue("MISSING_COLUMN", "Price column is missing", column=column))
            continue
        # Duplicate labels cannot be interpreted as one price series.
        if not data.columns.is_unique:
            continue
        series = data[column]
        numeric = is_numeric_dtype(series.dtype) and not (
            is_bool_dtype(series.dtype) or is_complex_dtype(series.dtype)
        )
        if not numeric:
            issues.append(
                _issue(
                    "INVALID_PRICE_TYPE",
                    "Price column must have a real numeric dtype",
                    column=column,
                )
            )
        for position, value in enumerate(series):
            details = {"column": column, "position": position}
            if dates is not None:
                details["date"] = _date_label(dates[position])
            elif raw_dates is not None:
                row_dates, _ = _session_dates([raw_dates[position]], market_timezone)
                if row_dates is not None:
                    details["date"] = _date_label(row_dates[0])
                else:
                    details["source_date"] = str(raw_dates[position])
            if value is None or value is pd.NA or (np.isscalar(value) and pd.isna(value)):
                issues.append(_issue("MISSING_PRICE", "Price is missing", **details))
            elif isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                issues.append(
                    _issue("INVALID_PRICE_TYPE", "Price must be a real number", **details)
                )
            elif not np.isfinite(value):
                issues.append(_issue("NON_FINITE_PRICE", "Price must be finite", **details))
            elif value <= 0:
                issues.append(
                    _issue("NON_POSITIVE_PRICE", "Price must be greater than zero", **details)
                )
    return _result(issues, row_count=len(data), price_columns=list(columns))


def _calendar(trading_calendar, market_timezone):
    if not isinstance(trading_calendar, pd.DatetimeIndex) or trading_calendar.empty:
        raise ValueError("trading_calendar must be a nonempty DatetimeIndex")
    dates, issues = _session_dates(trading_calendar, market_timezone)
    if issues or _date_issues(dates):
        raise ValueError("trading_calendar must contain unique, increasing dates without NaT")
    return dates


def _missing_statistics(series, dates):
    missing = series.isna().to_numpy()
    boundaries = np.diff(np.r_[False, missing, False].astype(int))
    starts, ends = np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1) - 1
    intervals = [
        {
            "start_position": int(start),
            "end_position": int(end),
            "start_date": _date_label(dates[start]),
            "end_date": _date_label(dates[end]),
            "length": int(end - start + 1),
        }
        for start, end in zip(starts, ends)
    ]
    count = int(missing.sum())
    return {
        "missing_count": count,
        "total_count": len(series),
        "missing_ratio": count / len(series),
        "max_consecutive_missing": max((interval["length"] for interval in intervals), default=0),
        "intervals": intervals,
    }


def validate_missing_intervals(
    data: pd.DataFrame,
    *,
    columns=("Close",),
    trading_calendar=None,
    max_missing_ratio=0.05,
    max_consecutive_missing=5,
    date_column=None,
    market_timezone=None,
) -> dict:
    """결측 비율과 연속 길이를 검증한다. 보간하지 않으며 임계값과 같으면 허용한다.

    제공한 calendar의 전체 거래일을 분모로 하며, 누락 행과 NaN을 모두 센다.
    calendar가 없으면 관측 행만 평가하고 존재하지 않는 날짜의 판단은 unverified다.
    구간 위치는 검사 축(제공 calendar 또는 원본 행 순서)에서 0부터 시작하며 양 끝을 포함한다.
    가격 타입과 Inf 등의 결함은 validate_price_data로 별도로 확인해야 한다.
    """
    columns = _columns(columns)
    if (
        isinstance(max_missing_ratio, bool)
        or not isinstance(max_missing_ratio, Real)
        or not (np.isfinite(max_missing_ratio) and 0 <= max_missing_ratio <= 1)
    ):
        raise ValueError("max_missing_ratio must be finite and between 0 and 1")
    if (
        isinstance(max_consecutive_missing, bool)
        or not isinstance(max_consecutive_missing, int)
        or max_consecutive_missing < 0
    ):
        raise ValueError("max_consecutive_missing must be a nonnegative integer")
    max_missing_ratio = float(max_missing_ratio)
    dates, issues = _frame_dates(data, date_column, market_timezone)
    if issues:
        return _result(issues, columns={}, calendar_status="unverified")
    missing_columns = [column for column in columns if column not in data.columns]
    if missing_columns:
        issues.extend(
            _issue("MISSING_COLUMN", "Missing-data column is absent", column=column)
            for column in missing_columns
        )
        return _result(issues, columns={}, calendar_status="unverified")
    values = data.loc[:, list(columns)].copy()
    values.index = dates
    calendar_status = "unverified"
    if trading_calendar is None:
        issues.append(
            _issue(
                "CALENDAR_UNVERIFIED",
                "Absent trading sessions cannot be checked without a calendar",
                severity="unverified",
            )
        )
    else:
        sessions = _calendar(trading_calendar, market_timezone)
        original_dates = data[date_column].to_numpy() if date_column is not None else data.index
        context_issues = _date_context_issues([original_dates, trading_calendar], market_timezone)
        if context_issues:
            return _result(context_issues, columns={}, calendar_status="unverified")
        calendar_status = "verified"
        for position in np.flatnonzero(~dates.isin(sessions)):
            issues.append(
                _issue(
                    "OUTSIDE_CALENDAR",
                    "Observation is outside the supplied calendar",
                    position=int(position),
                    date=_date_label(dates[position]),
                )
            )
        values = values.reindex(sessions)
        dates = sessions
    statistics = {}
    for column in columns:
        stats = _missing_statistics(values[column], dates)
        statistics[column] = stats
        if stats["missing_ratio"] > max_missing_ratio:
            issues.append(
                _issue(
                    "EXCESSIVE_MISSING_RATIO",
                    "Missing ratio exceeds the configured limit",
                    column=column,
                    missing_ratio=stats["missing_ratio"],
                    limit=max_missing_ratio,
                )
            )
        for interval in stats["intervals"]:
            if interval["length"] > max_consecutive_missing:
                issues.append(
                    _issue(
                        "EXCESSIVE_MISSING_INTERVAL",
                        "Consecutive missing sessions exceed the configured limit",
                        column=column,
                        limit=max_consecutive_missing,
                        **interval,
                    )
                )
    return _result(issues, columns=statistics, calendar_status=calendar_status)


def validate_listing_date(
    collection_start,
    listing_date=None,
    *,
    require_full_period=False,
    market_timezone=None,
) -> dict:
    """실제 상장일과 수집 시작일을 비교한다. 첫 관측일로 상장일을 추정하지 않는다.

    늦은 상장은 기본적으로 정보로 기록한다. 전체 기간이 필요한 호출자만
    require_full_period=True로 INSUFFICIENT_HISTORY를 제외 사유로 사용할 수 있다.
    """
    if not isinstance(require_full_period, bool):
        raise ValueError("require_full_period must be a bool")
    start, issues = _session_dates([collection_start], market_timezone)
    if issues:
        raise ValueError("collection_start must be a valid date")
    metrics = {"collection_start": _date_label(start[0]), "listing_date": None}
    if listing_date is None:
        return _result(
            [
                _issue(
                    "LISTING_DATE_UNKNOWN",
                    "Actual listing date was not supplied",
                    severity="unverified",
                )
            ],
            **metrics,
        )
    combined, issues = _session_dates([collection_start, listing_date], market_timezone)
    if issues:
        raise ValueError(
            "collection_start and listing_date must be valid dates in a consistent timezone"
        )
    metrics["listing_date"] = _date_label(combined[1])
    issues = []
    if combined[1] > combined[0]:
        issues.append(
            _issue(
                "LATE_LISTING",
                "Pre-listing data is unavailable because the asset was listed after collection_start",
                severity="info",
                date=metrics["listing_date"],
                collection_start=metrics["collection_start"],
            )
        )
        if require_full_period:
            issues.append(
                _issue(
                    "INSUFFICIENT_HISTORY",
                    "Listing date cannot satisfy the required full period",
                    date=metrics["listing_date"],
                    collection_start=metrics["collection_start"],
                )
            )
    return _result(issues, **metrics)


def _dividend_series(series, name, market_timezone, *, allow_zero=False):
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    if allow_zero and series.empty and isinstance(series.index, pd.DatetimeIndex):
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]), name=name), []
    frame = series.to_frame(name="value")
    report = validate_price_data(frame, price_columns=("value",), market_timezone=market_timezone)
    issues = report["issues"]
    if allow_zero:
        # No dividend on ordinary sessions is represented by zero, not a cash event.
        issues = [
            issue
            for issue in issues
            if not (issue["code"] == "NON_POSITIVE_PRICE" and series.iloc[issue["position"]] == 0)
        ]
    for issue in issues:
        issue["column"] = name
    dates, _ = _frame_dates(frame, None, market_timezone)
    if issues:
        return None, issues
    return pd.Series(series.to_numpy(dtype=float), index=dates, name=name), []


def check_dividend_adjustment(
    prices: pd.Series,
    *,
    raw_close=None,
    reinvested_close=None,
    dividends=None,
    trading_calendar=None,
    comparison_basis=None,
    split_free=None,
    sources=None,
    event_date_kind="ex_date",
    rtol=1e-5,
    atol=1e-8,
    market_timezone=None,
) -> dict:
    """분배금 재투자와의 일관성을 명시적인 근거와 분배락일로 검증한다.

    prices는 검사 대상, raw_close는 미조정 종가, reinvested_close는 같은 종목과
    통화의 gross total return 가격 계열, dividends는 1주당 현금 분배금이다.
    sources에는 위 4계열의 출처 ID를 지정하며 split_free=True로 검사 이벤트
    전후에 분할이 없음을 명시한다. 전체 기간이나 미래 분배금까지 추정하지 않는다.
    지급일만 있는 자료, net return, 임의 수정주가의 조정 공식은 지원하지 않는다.

    직전 거래일 p와 분배락일 d의 기대 수익률은 (raw[d]+D)/raw[p]-1이다.
    gross 재투자 비교 자료가 이 값과 일치할 때만 검사 대상 수익률과 대조한다.
    비교 자료와의 차이만으로 미반영을 확정하지 않는다. 제공한 이벤트가 모두
    확인되면 passed, 확인된 근거와 모순되면 failed, 근거가 부족하면 unverified다.
    rtol/atol은 가격 수준이 아닌 수익률의 허용오차다.
    정의: S&P Index Mathematics Methodology, Total Return Calculations
    https://www.spglobal.com/spdji/en/documents/methodologies/methodology-index-math.pdf
    """
    for name, tolerance in (("rtol", rtol), ("atol", atol)):
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, Real)
            or not np.isfinite(tolerance)
            or tolerance < 0
        ):
            raise ValueError(f"{name} must be finite and nonnegative")
    if sources is not None and not isinstance(sources, dict):
        raise TypeError("sources must be a dict of source identifiers")
    metrics = {
        "sources": dict(sources or {}),
        "comparison_basis": comparison_basis,
        "event_date_kind": event_date_kind,
        "split_free": split_free,
        "rtol": float(rtol),
        "atol": float(atol),
        "events": [],
    }
    issues, series = [], {}
    for name, value in (
        ("prices", prices),
        ("raw_close", raw_close),
        ("reinvested_close", reinvested_close),
        ("dividends", dividends),
    ):
        if value is None:
            issues.append(
                _issue(
                    "MISSING_DIVIDEND_EVIDENCE",
                    "Required dividend comparison data is absent",
                    severity="unverified",
                    column=name,
                )
            )
            continue
        normalized, defects = _dividend_series(
            value, name, market_timezone, allow_zero=name == "dividends"
        )
        issues.extend(defects)
        if normalized is not None:
            series[name] = normalized
    events = series.get("dividends", pd.Series(dtype=float))
    events = events[events > 0]
    metrics["events"] = [
        {"date": _date_label(day), "dividend": float(dividend), "status": "unverified"}
        for day, dividend in events.items()
    ]
    if comparison_basis != "gross_total_return":
        issues.append(
            _issue(
                "UNSUPPORTED_COMPARISON_BASIS",
                "An explicit same-asset, same-currency gross_total_return reference is required",
                severity="unverified",
            )
        )
    if split_free is not True:
        issues.append(
            _issue(
                "SPLIT_STATUS_UNVERIFIED",
                "Split-free event windows have not been confirmed",
                severity="unverified",
            )
        )
    if event_date_kind != "ex_date":
        issues.append(
            _issue(
                "EX_DATE_UNVERIFIED",
                "Cash payment dates cannot be used as ex-dates",
                severity="unverified",
            )
        )
    for name in ("prices", "raw_close", "reinvested_close", "dividends"):
        source = metrics["sources"].get(name)
        if not isinstance(source, str) or not source.strip():
            issues.append(
                _issue(
                    "DIVIDEND_SOURCE_UNVERIFIED",
                    "Source identifier is required",
                    severity="unverified",
                    column=name,
                )
            )
    if trading_calendar is None:
        issues.append(
            _issue(
                "CALENDAR_UNVERIFIED",
                "Previous trading session requires an explicit calendar",
                severity="unverified",
            )
        )
    sessions = (
        _calendar(trading_calendar, market_timezone) if trading_calendar is not None else None
    )
    if issues:
        return _result(issues, **metrics)
    context_issues = _date_context_issues(
        [value.index for value in (prices, raw_close, reinvested_close, dividends)]
        + [trading_calendar],
        market_timezone,
    )
    if context_issues:
        return _result(context_issues, **metrics)
    if events.empty:
        return _result(
            [
                _issue(
                    "NO_DIVIDEND_EVENTS",
                    "No cash dividend event is available to verify",
                    severity="unverified",
                )
            ],
            **metrics,
        )
    for (day, dividend), event in zip(events.items(), metrics["events"]):
        position = sessions.get_indexer([day])[0]
        if position <= 0:
            issues.append(
                _issue(
                    "DIVIDEND_SESSION_UNVERIFIED",
                    "Ex-date or its previous session is not covered",
                    severity="unverified",
                    date=_date_label(day),
                )
            )
            continue
        previous = sessions[position - 1]
        event["previous_date"] = _date_label(previous)
        if any(
            previous not in series[name].index or day not in series[name].index
            for name in ("prices", "raw_close", "reinvested_close")
        ):
            issues.append(
                _issue(
                    "DIVIDEND_SESSION_UNVERIFIED",
                    "Comparison prices do not cover both adjacent sessions",
                    severity="unverified",
                    date=_date_label(day),
                    previous_date=_date_label(previous),
                )
            )
            continue
        raw_previous, raw_current = float(series["raw_close"][previous]), float(
            series["raw_close"][day]
        )
        price_previous, price_current = float(series["prices"][previous]), float(
            series["prices"][day]
        )
        reference_previous, reference_current = float(series["reinvested_close"][previous]), float(
            series["reinvested_close"][day]
        )
        expected_return = (raw_current + dividend) / raw_previous - 1
        raw_return = raw_current / raw_previous - 1
        reference_return = reference_current / reference_previous - 1
        price_return = price_current / price_previous - 1
        event.update(
            {
                "raw_previous": raw_previous,
                "raw_current": raw_current,
                "price_previous": price_previous,
                "price_current": price_current,
                "reference_previous": reference_previous,
                "reference_current": reference_current,
                "expected_return": expected_return,
                "raw_return": raw_return,
                "reference_return": reference_return,
                "price_return": price_return,
            }
        )
        if not np.isfinite([expected_return, raw_return, reference_return, price_return]).all():
            event["status"] = "failed"
            issues.append(
                _issue(
                    "NON_FINITE_DIVIDEND_RETURN",
                    "Dividend comparison arithmetic is not finite",
                    date=_date_label(day),
                )
            )
        elif np.isclose(raw_return, expected_return, rtol=rtol, atol=atol):
            issues.append(
                _issue(
                    "DIVIDEND_SIGNAL_UNVERIFIED",
                    "Cash dividend effect is indistinguishable within tolerance",
                    severity="unverified",
                    date=_date_label(day),
                )
            )
        elif not np.isclose(reference_return, expected_return, rtol=rtol, atol=atol):
            issues.append(
                _issue(
                    "DIVIDEND_REFERENCE_UNVERIFIED",
                    "Reference is inconsistent with supplied raw prices and cash dividend",
                    severity="unverified",
                    date=_date_label(day),
                )
            )
        elif np.isclose(reference_return, raw_return, rtol=rtol, atol=atol) or (
            np.isclose(price_return, raw_return, rtol=rtol, atol=atol)
            and np.isclose(price_return, expected_return, rtol=rtol, atol=atol)
        ):
            issues.append(
                _issue(
                    "DIVIDEND_SIGNAL_UNVERIFIED",
                    "Reference or checked return cannot distinguish raw and reinvested prices",
                    severity="unverified",
                    date=_date_label(day),
                )
            )
        elif np.isclose(price_return, reference_return, rtol=rtol, atol=atol) and np.isclose(
            price_return,
            expected_return,
            rtol=rtol,
            atol=atol,
        ):
            event["status"] = "passed"
        else:
            event["status"] = "failed"
            code = (
                "DIVIDEND_NOT_REFLECTED"
                if np.isclose(price_return, raw_return, rtol=rtol, atol=atol)
                else "DIVIDEND_COMPARISON_MISMATCH"
            )
            issues.append(
                _issue(
                    code,
                    "Checked price return does not match the verified dividend-reinvestment reference",
                    date=_date_label(day),
                    previous_date=_date_label(previous),
                )
            )
    return _result(issues, **metrics)
