"""Issue #9 원천 수집 어댑터. 원본 보존, 검증과 정책 결정을 분리한다.

실데이터 호출은 기본 비활성화다. 오프라인 호출자는 download/http_get을 주입한다.
거래일 정렬·보간, 수익률 계산, 스냅샷 저장 및 가짜 available_at 생성은 하지 않는다.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from io import StringIO
from numbers import Real
from pathlib import Path
import re
from urllib.parse import quote

import numpy as np
import pandas as pd

from .validation import (
    _session_dates,
    check_dividend_adjustment,
    validate_listing_date,
    validate_missing_intervals,
    validate_price_data,
)

WARMUP_START = "2019-09-01"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
OHLC = ("Open", "High", "Low", "Close")
OHLCV = (*OHLC, "Volume")


@dataclass(frozen=True)
class DatasetSpec:
    """수집 대상과 팀 선정 상태. confirmed는 수집 품질 검증을 뜻하지 않는다."""

    dataset_id: str
    symbol: str
    source: str
    role: str
    market: str
    currency: str
    timezone: str
    asset_class: str
    rationale: str
    selection_status: str
    unit: str

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field.name} must be a nonempty string")
        if self.source not in {"yahoo", "fred"}:
            raise ValueError("source must be yahoo or fred")
        if self.role not in {"etf", "benchmark", "fx", "risk_free"}:
            raise ValueError("Unsupported dataset role")
        if self.selection_status not in {"candidate", "confirmed"}:
            raise ValueError("selection_status must be candidate or confirmed")
        try:
            pd.Timestamp("2000-01-01").tz_localize(self.timezone)
        except Exception as exc:
            raise ValueError("Invalid market timezone") from exc
        if self.source == "fred":
            if self.role != "risk_free" or self.unit != "percent_per_annum":
                raise ValueError("FRED adapter requires an annual percentage risk-free rate")
            if self.market != "US_RATE" or self.currency != "USD":
                raise ValueError("FRED rate market/currency must be US_RATE/USD")
        elif self.role == "risk_free":
            raise ValueError("Use the FRED adapter for risk-free rate candidates")
        elif self.role == "fx":
            if (self.symbol, self.market, self.currency, self.unit) != (
                "KRW=X",
                "FX",
                "KRW",
                "KRW_per_USD",
            ):
                raise ValueError("FX contract requires KRW=X quoted in KRW per USD")
            if self.timezone != "Asia/Seoul":
                raise ValueError("FX observation days must use the #12 Asia/Seoul contract")
        elif self.market == "KRX":
            if self.currency != "KRW":
                raise ValueError("KRX prices require KRW currency")
            if self.role == "etf" and not re.fullmatch(r"\d{6}\.KS", self.symbol):
                raise ValueError("Domestic ETF Yahoo symbols must include .KS")
        elif self.market == "US":
            if self.currency != "USD" or self.symbol.endswith(".KS"):
                raise ValueError("US prices require a US symbol and USD currency")
        else:
            raise ValueError("Yahoo price market must be KRX or US")


@dataclass
class CollectionResult:
    data: pd.DataFrame
    raw_data: pd.DataFrame | None
    metadata: dict
    validation: dict
    raw_text: str | None = None

    @property
    def status(self):
        return self.metadata["status"]

    @property
    def error(self):
        return next(
            (issue for issue in self.metadata["issues"] if issue["severity"] == "error"), None
        )


def _issue(code, message, *, severity="error", **details):
    return {"code": code, "message": message, "severity": severity, **details}


def _report(issues, **metrics):
    severities = {issue["severity"] for issue in issues}
    status = (
        "failed"
        if "error" in severities
        else ("unverified" if "unverified" in severities else "passed")
    )
    return {"valid": status == "passed", "status": status, "issues": issues, "metrics": metrics}


def _bounds(start, end):
    dates = []
    for name, value in (("start", start), ("end", end)):
        try:
            if isinstance(value, (bool, int, float)):
                raise ValueError("date required")
            day = pd.Timestamp(value)
            if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
                raise ValueError("naive daily date required")
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f"{name} must be a valid timezone-naive daily date") from exc
        dates.append(day)
    if dates[0] >= dates[1]:
        raise ValueError("start must precede the exclusive end")
    return dates


def load_asset_config(path) -> dict:
    """YAML 설정을 읽되 후보를 자동으로 최종 유니버스/Rf 정책으로 승격하지 않는다."""
    import yaml

    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Expected asset config schema_version 1")
    collection = config.get("collection", {})
    if collection.get("end_exclusive") is not True:
        raise ValueError("Collection end must be exclusive")
    _bounds(collection.get("start_date"), "2260-01-01")
    entries = config.get("datasets")
    if not isinstance(entries, list) or not entries:
        raise ValueError("datasets must be a nonempty list")
    try:
        specs = [DatasetSpec(**entry) for entry in entries]
    except TypeError as exc:
        raise ValueError("Invalid dataset specification") from exc
    by_id = {spec.dataset_id: spec for spec in specs}
    if len(by_id) != len(specs) or len({(s.source, s.symbol) for s in specs}) != len(specs):
        raise ValueError("Dataset IDs and source/symbol pairs must be unique")
    universe = config.get("universe", {})
    selected = universe.get("selected_etfs")
    if universe.get("target_etf_count") != 12 or not isinstance(selected, list):
        raise ValueError("Expected an explicit 12 ETF target and selection list")
    if universe.get("status") == "unconfirmed":
        if selected:
            raise ValueError("Unconfirmed universe must not select ETFs")
    elif universe.get("status") == "confirmed":
        if (
            len(selected) != 12
            or len(set(selected)) != 12
            or any(
                item not in by_id
                or by_id[item].role != "etf"
                or by_id[item].selection_status != "confirmed"
                for item in selected
            )
        ):
            raise ValueError("Confirmed universe requires 12 distinct confirmed ETF IDs")
    else:
        raise ValueError("Invalid universe status")
    policy = config.get("risk_free_policy", {})
    policy_keys = ("selected_dataset", "currency", "daily_conversion", "day_count")
    if policy.get("status") == "unconfirmed":
        if any(policy.get(key) is not None for key in policy_keys):
            raise ValueError("Unconfirmed risk-free policy must leave decisions unset")
    elif policy.get("status") == "confirmed":
        chosen = by_id.get(policy.get("selected_dataset"))
        if (
            chosen is None
            or chosen.role != "risk_free"
            or chosen.selection_status != "confirmed"
            or any(not policy.get(key) for key in policy_keys)
            or policy["currency"] != chosen.currency
        ):
            raise ValueError("Confirmed risk-free policy requires complete explicit decisions")
    else:
        raise ValueError("Invalid risk-free policy status")
    config["datasets"] = specs
    return config


def _new_result(spec, start, end):
    metadata = {
        "dataset_id": spec.dataset_id,
        "symbol": spec.symbol,
        "source": spec.source,
        "role": spec.role,
        "market": spec.market,
        "currency": spec.currency,
        "timezone": spec.timezone,
        "unit": spec.unit,
        "asset_class": spec.asset_class,
        "selection_status": spec.selection_status,
        "rationale": spec.rationale,
        "requested_start": start.date().isoformat(),
        "requested_end_exclusive": end.date().isoformat(),
        "actual_start": None,
        "actual_end": None,
        "row_count": 0,
        "collected_at": None,
        "auto_adjust": True if spec.source == "yahoo" else None,
        "price_basis": (
            "Yahoo OHLC adjusted by yfinance auto_adjust=True using the provider adjustment ratio; "
            "independent gross total-return/dividend verification remains required"
            if spec.source == "yahoo"
            else "FRED source annual percentage rate; no daily conversion"
        ),
        "availability_status": "unverified",
        "available_at": None,
        "issues": [],
        "source_url": (
            f"https://finance.yahoo.com/quote/{quote(spec.symbol, safe='')}/"
            if spec.source == "yahoo"
            else FRED_CSV_URL
        ),
    }
    if spec.role == "fx":
        metadata.update(fx_direction="USD/KRW", fx_direction_status="unverified")
        metadata["price_basis"] = (
            "Provider USD/KRW quote in KRW per USD; auto_adjust=True requested"
        )
    elif spec.role == "benchmark":
        metadata["price_basis"] = (
            "Provider price-index OHLC in index points; auto_adjust=True requested; "
            "not an independently verified dividend-reinvestment index"
        )
    return CollectionResult(pd.DataFrame(), None, metadata, {})


def _finish(result, issues):
    all_issues = list(issues)
    for name, report in result.validation.items():
        all_issues.extend({**issue, "check": name} for issue in report["issues"])
    summary = _report(all_issues)
    result.metadata.update(status=summary["status"], issues=all_issues)
    return result


def _record_source(result, raw):
    result.raw_data = raw.copy(deep=True)
    result.data = raw.copy(deep=True)
    result.metadata["row_count"] = len(raw)
    if result.metadata["collected_at"] is None:
        result.metadata["collected_at"] = datetime.now(timezone.utc).isoformat()


def _normalize_dates(result, spec, start, end):
    dates, issues = _session_dates(result.data.index, spec.timezone)
    if dates is None:
        return None, issues
    dates.name = result.data.index.name
    result.data.index = dates
    if len(dates):
        result.metadata.update(
            actual_start=dates.min().date().isoformat(), actual_end=dates.max().date().isoformat()
        )
    for position in np.flatnonzero((dates < start) | (dates >= end)):
        issues.append(
            _issue(
                "OUTSIDE_REQUESTED_PERIOD",
                "Source observation is outside [start, end)",
                position=int(position),
                date=dates[position].date().isoformat(),
            )
        )
    return dates, issues


def _availability(result, spec, dates, available_at):
    if available_at is None:
        return [
            _issue(
                "AVAILABILITY_UNVERIFIED",
                "Historical available_at was not supplied",
                severity="unverified",
            )
        ]
    if not isinstance(available_at, pd.Series):
        result.metadata["availability_status"] = "failed"
        return [_issue("INVALID_AVAILABLE_AT", "available_at must be a date-indexed Series")]
    # Preserve supplied evidence even when it is invalid. Collection time is never substituted.
    result.metadata["available_at"] = available_at.copy(deep=True)
    labels, problems = _session_dates(available_at.index, spec.timezone)
    if dates is None or labels is None or not labels.equals(dates):
        result.metadata["availability_status"] = "failed"
        return [_issue("INVALID_AVAILABLE_AT", "available_at labels must match all source rows")]
    normalized = []
    for position, (day, value) in enumerate(zip(dates, available_at)):
        try:
            if not isinstance(value, (str, datetime, pd.Timestamp)):
                raise ValueError("Timestamp required")
            instant = pd.Timestamp(value)
            if pd.isna(instant) or instant.tzinfo is None:
                raise ValueError("Timezone-aware timestamp required")
            instant = instant.tz_convert("UTC")
            if instant < day.tz_localize(spec.timezone).tz_convert("UTC"):
                raise ValueError("Publication precedes observation day")
            if normalized and instant < normalized[-1]:
                raise ValueError("Publication instants must be nondecreasing")
            normalized.append(instant)
        except (ValueError, TypeError, OverflowError):
            problems.append(
                _issue(
                    "INVALID_AVAILABLE_AT",
                    "Invalid historical publication instant",
                    position=position,
                )
            )
    if not problems:
        result.metadata["available_at"] = pd.Series(normalized, index=dates, name=available_at.name)
        result.metadata["availability_status"] = "verified"
    else:
        result.metadata["availability_status"] = "failed"
    return problems


def _policy_issues(spec):
    issues = []
    if spec.selection_status == "candidate":
        issues.append(
            _issue(
                "SELECTION_UNCONFIRMED",
                "Candidate is not a final team selection",
                severity="unverified",
            )
        )
    if spec.role == "risk_free":
        issues.append(
            _issue(
                "RISK_FREE_POLICY_UNCONFIRMED",
                "Rf currency, daily conversion and day count are unset",
                severity="unverified",
            )
        )
    return issues


def _flatten_yahoo(data, symbol):
    if not isinstance(data.columns, pd.MultiIndex):
        return []
    if data.columns.nlevels != 2:
        return [_issue("INVALID_YAHOO_SCHEMA", "Expected two price/ticker column levels")]
    price_levels = [
        level for level in (0, 1) if set(OHLCV).issubset(set(data.columns.get_level_values(level)))
    ]
    if len(price_levels) != 1:
        return [_issue("INVALID_YAHOO_SCHEMA", "Cannot identify the OHLCV column level")]
    level = price_levels[0]
    tickers = set(data.columns.get_level_values(1 - level))
    if tickers != {symbol}:
        return [
            _issue(
                "YAHOO_TICKER_MISMATCH",
                "Response must contain exactly the requested ticker",
                expected=symbol,
                received=sorted(map(str, tickers)),
            )
        ]
    data.columns = pd.Index(data.columns.get_level_values(level), name=None)
    return []


def _numeric_report(data, column, *, allow_negative):
    issues = []
    if column not in data.columns or not data.columns.is_unique:
        return _report([_issue("MISSING_COLUMN", "Numeric source column is absent", column=column)])
    for position, value in enumerate(data[column]):
        details = {"column": column, "position": position}
        if value is None or value is pd.NA or (np.isscalar(value) and pd.isna(value)):
            issues.append(_issue("MISSING_VALUE", "Source value is missing", **details))
        elif isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            issues.append(
                _issue("INVALID_VALUE_TYPE", "Source value must be real numeric", **details)
            )
        elif not np.isfinite(value):
            issues.append(_issue("NON_FINITE_VALUE", "Source value must be finite", **details))
        elif not allow_negative and value < 0:
            issues.append(_issue("NEGATIVE_VOLUME", "Volume must be nonnegative", **details))
    return _report(issues)


def _ohlc_report(data):
    issues = []
    if not data.columns.is_unique or not set(OHLC).issubset(data.columns):
        return _report(issues)
    for position, row in enumerate(data.loc[:, list(OHLC)].itertuples(index=False, name=None)):
        if not all(
            isinstance(value, Real)
            and not isinstance(value, (bool, np.bool_))
            and np.isfinite(value)
            for value in row
        ):
            continue  # Type/NaN/Inf evidence is already retained by #10 price validation.
        opening, high, low, close = row
        if low > high or low > min(opening, close) or high < max(opening, close):
            issues.append(
                _issue(
                    "INVALID_OHLC_RANGE",
                    "OHLC values contradict the low/high bounds",
                    position=position,
                )
            )
    return _report(issues)


def collect_yahoo(
    spec: DatasetSpec,
    *,
    end,
    start=WARMUP_START,
    download=None,
    trading_calendar=None,
    available_at=None,
    quote_metadata=None,
    allow_network=False,
) -> CollectionResult:
    """국내/해외 ETF, 지수, USD/KRW 원천 일별 OHLCV를 가져온다.

    quote_metadata는 호출자가 확보한 {symbol, base_currency, quote_currency} 근거다.
    값 크기나 티커 문자열만으로 환율 방향을 verified로 판정하지 않는다.
    """
    if not isinstance(spec, DatasetSpec) or spec.source != "yahoo":
        raise ValueError("collect_yahoo requires a Yahoo DatasetSpec")
    first, last = _bounds(start, end)
    result = _new_result(spec, first, last)
    issues = _policy_issues(spec)
    if download is None:
        if not allow_network:
            return _finish(
                result,
                issues
                + [
                    _issue(
                        "NETWORK_NOT_AUTHORIZED",
                        "Supply a mock download or explicitly authorize network",
                    )
                ],
            )
        try:
            import yfinance as yf
        except ImportError:
            return _finish(
                result, issues + [_issue("CLIENT_UNAVAILABLE", "yfinance is not installed")]
            )
        download = yf.download
    try:
        raw = download(
            tickers=spec.symbol,
            start=first.date().isoformat(),
            end=last.date().isoformat(),
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        return _finish(
            result,
            issues
            + [
                _issue(
                    "COLLECTION_FAILED", "Yahoo request failed", exception_type=type(exc).__name__
                )
            ],
        )
    if not isinstance(raw, pd.DataFrame):
        return _finish(result, issues + [_issue("INVALID_RESPONSE", "Expected a Yahoo DataFrame")])
    _record_source(result, raw)
    if raw.empty:
        return _finish(result, issues + [_issue("EMPTY_DATA", "Yahoo response is empty")])
    schema_issues = _flatten_yahoo(result.data, spec.symbol)
    issues.extend(schema_issues)
    dates, date_issues = _normalize_dates(result, spec, first, last)
    issues.extend(date_issues)
    if schema_issues:
        issues.extend(_availability(result, spec, dates, available_at))
        return _finish(result, issues)
    result.validation["price"] = validate_price_data(
        result.data, price_columns=OHLC, market_timezone=spec.timezone
    )
    result.validation["volume"] = _numeric_report(result.data, "Volume", allow_negative=False)
    result.validation["ohlc"] = _ohlc_report(result.data)
    result.validation["missing"] = validate_missing_intervals(
        result.data, trading_calendar=trading_calendar, market_timezone=spec.timezone
    )
    if spec.role == "etf":
        result.validation["listing"] = validate_listing_date(first, market_timezone=spec.timezone)
        if "Close" in result.data.columns and result.data.columns.is_unique:
            result.validation["dividend"] = check_dividend_adjustment(
                result.data["Close"], market_timezone=spec.timezone
            )
    issues.extend(_availability(result, spec, dates, available_at))
    if spec.role == "fx":
        if quote_metadata is None:
            issues.append(
                _issue(
                    "FX_DIRECTION_UNVERIFIED",
                    "Provider quote direction evidence was not supplied",
                    severity="unverified",
                )
            )
        elif not isinstance(quote_metadata, dict) or any(
            quote_metadata.get(key) != value
            for key, value in {
                "symbol": "KRW=X",
                "base_currency": "USD",
                "quote_currency": "KRW",
            }.items()
        ):
            result.metadata["fx_direction_status"] = "failed"
            issues.append(_issue("FX_DIRECTION_MISMATCH", "Quote evidence does not match USD/KRW"))
        else:
            result.metadata["fx_direction_status"] = "verified"
            result.metadata["fx_quote_evidence"] = dict(quote_metadata)
    return _finish(result, issues)


def collect_fred(
    spec: DatasetSpec,
    *,
    end,
    start=WARMUP_START,
    http_get=None,
    trading_calendar=None,
    available_at=None,
    allow_network=False,
) -> CollectionResult:
    """FRED 연율 % 관측치를 수집한다. 종료일은 배타적이며 일별 Rf로 변환하지 않는다."""
    if not isinstance(spec, DatasetSpec) or spec.source != "fred":
        raise ValueError("collect_fred requires a FRED DatasetSpec")
    first, last = _bounds(start, end)
    result = _new_result(spec, first, last)
    issues = _policy_issues(spec)
    result.metadata["source_url"] = FRED_CSV_URL
    if http_get is None:
        if not allow_network:
            return _finish(
                result,
                issues
                + [
                    _issue(
                        "NETWORK_NOT_AUTHORIZED",
                        "Supply a mock http_get or explicitly authorize network",
                    )
                ],
            )
        try:
            import requests
        except ImportError:
            return _finish(
                result, issues + [_issue("CLIENT_UNAVAILABLE", "requests is not installed")]
            )
        http_get = requests.get
    try:
        response = http_get(
            FRED_CSV_URL,
            params={
                "id": spec.symbol,
                "cosd": first.date().isoformat(),
                "coed": (last - pd.Timedelta(days=1)).date().isoformat(),
            },
            timeout=60,
        )
        response.raise_for_status()
        result.raw_text = response.text
        result.metadata["collected_at"] = datetime.now(timezone.utc).isoformat()
    except Exception as exc:
        return _finish(
            result,
            issues
            + [
                _issue(
                    "COLLECTION_FAILED", "FRED request failed", exception_type=type(exc).__name__
                )
            ],
        )
    try:
        rows = list(csv.reader(StringIO(result.raw_text), strict=True))
        if not rows or not rows[0]:
            raise ValueError("Empty CSV")
        header, body = rows[0], rows[1:]
        raw = pd.DataFrame(body, columns=header)
    except (csv.Error, TypeError, ValueError):
        return _finish(
            result,
            issues
            + [
                _issue(
                    "INVALID_FRED_CSV",
                    "Cannot parse FRED CSV; original response retained in raw_text",
                )
            ],
        )
    _record_source(result, raw)
    if raw.empty:
        return _finish(result, issues + [_issue("EMPTY_DATA", "FRED response is empty")])
    if (
        len(header) != 2
        or header[0] not in {"DATE", "observation_date"}
        or header[1] != spec.symbol
        or not raw.columns.is_unique
    ):
        return _finish(
            result,
            issues
            + [
                _issue("INVALID_FRED_SCHEMA", "Expected date and exactly the requested FRED series")
            ],
        )
    values = []
    for position, value in enumerate(raw[spec.symbol]):
        if value == ".":
            values.append(np.nan)
        else:
            try:
                values.append(float(value))
            except (TypeError, ValueError, OverflowError):
                values.append(value)
                issues.append(
                    _issue(
                        "INVALID_RATE_VALUE",
                        "FRED rate is not a numeric value or '.' missing marker",
                        position=position,
                        value=value,
                    )
                )
    result.data = pd.DataFrame({"Value": values}, index=pd.Index(raw[header[0]], name="Date"))
    dates, date_issues = _normalize_dates(result, spec, first, last)
    issues.extend(date_issues)
    result.validation["rate"] = _numeric_report(result.data, "Value", allow_negative=True)
    result.validation["missing"] = validate_missing_intervals(
        result.data,
        columns=("Value",),
        trading_calendar=trading_calendar,
        market_timezone=spec.timezone,
    )
    issues.extend(_availability(result, spec, dates, available_at))
    return _finish(result, issues)
