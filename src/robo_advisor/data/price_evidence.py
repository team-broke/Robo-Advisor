"""Read-only price-definition and source-version evidence for archived team12 data.

Snapshot objects must first be verified against external manifest hashes by the
caller. Reference calendars, Close/Adj Close ratios and a matching correction
candidate never establish historical publication or training approval.
"""

from __future__ import annotations

from numbers import Real

import numpy as np
import pandas as pd

from .collectors import OHLC, _ohlc_report
from .importers import TEAM12_ASSETS
from .snapshot import Snapshot
from .validation import check_dividend_adjustment, validate_missing_intervals, validate_price_data

_EVENT = pd.Timestamp("2020-03-10")
_RTOL, _ATOL = 1e-6, 1e-10


def _issue(code, message, *, severity="error", **details):
    return {"code": code, "severity": severity, "message": message, **details}


def _report(issues, **metrics):
    severity = {item["severity"] for item in issues}
    return {
        "status": (
            "failed"
            if "error" in severity
            else "unverified" if "unverified" in severity else "passed"
        ),
        "training_approved": False,
        "issues": issues,
        "metrics": metrics,
    }


def _equal(left, right):
    if not isinstance(left, pd.DataFrame) or not isinstance(right, pd.DataFrame):
        return False
    try:
        pd.testing.assert_frame_equal(left, right, check_exact=True)
    except AssertionError:
        return False
    return True


def _daily(index):
    return (
        isinstance(index, pd.DatetimeIndex)
        and not index.empty
        and index.tz is None
        and not index.hasnans
        and index.is_unique
        and index.is_monotonic_increasing
        and index.equals(index.normalize())
    )


def _bound(value):
    if not isinstance(value, str):
        raise ValueError("Requested coverage requires explicit ISO daily dates")
    day = pd.Timestamp(value)
    if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
        raise ValueError("Requested coverage requires naive daily dates")
    return day


def _ratio(data):
    result = {
        "status": "unverified",
        "rtol": _RTOL,
        "atol": _ATOL,
        "interpretation": "Ratios are diagnostic; they do not identify corporate actions.",
    }
    if "Adj Close" not in data:
        return {**result, "reason": "adjusted_close_absent"}
    close, adjusted = (pd.to_numeric(data[c], errors="coerce") for c in ("Close", "Adj Close"))
    valid = np.isfinite(close) & np.isfinite(adjusted) & close.gt(0) & adjusted.gt(0)
    ratio = pd.Series(np.nan, index=data.index, dtype=float)
    with np.errstate(over="ignore", under="ignore"):
        ratio.loc[valid] = adjusted[valid] / close[valid]
    if not np.isfinite(ratio.loc[valid]).all() or ratio.loc[valid].le(0).any():
        raise ValueError("Adjusted/Close ratio arithmetic is not finite and positive")
    values = ratio.dropna()
    same = np.isclose(values, 1.0, rtol=_RTOL, atol=_ATOL)
    changed = (
        ratio.notna()
        & ratio.shift(1).notna()
        & ~np.isclose(ratio, ratio.shift(1), rtol=_RTOL, atol=_ATOL, equal_nan=True)
    )
    return {
        **result,
        "valid_rows": len(values),
        "unavailable_rows": int((~valid).sum()),
        "minimum": float(values.min()) if len(values) else None,
        "maximum": float(values.max()) if len(values) else None,
        "equal_close_rows": int(same.sum()),
        "different_close_rows": int((~same).sum()),
        "ratio_change_dates": [str(day.date()) for day in data.index[changed]],
    }


def _asset(result, asset, calendar, selected):
    domestic = asset in TEAM12_ASSETS[-2:]
    expected = {
        "dataset_id": asset,
        "symbol": {"KODEX_200TR": "278530.KS", "KTB10Y": "148070.KS"}.get(asset, asset),
        "source": (
            ("yahoo_krx_low_correction" if asset == "KTB10Y" else "yahoo_archived_close")
            if selected
            else "yahoo"
        ),
        "role": "etf",
        "market": "KRX" if domestic else "US",
        "currency": "KRW" if domestic else "USD",
        "timezone": "Asia/Seoul" if domestic else "America/New_York",
    }
    metadata, data = result.metadata, result.data
    issues = [
        _issue(
            "PRICE_SOURCE_IDENTITY_MISMATCH",
            "Archived source identity differs.",
            field=k,
            expected=v,
            actual=metadata.get(k),
        )
        for k, v in expected.items()
        if metadata.get(k) != v
    ]
    if selected and result.status == "passed":
        issues.append(
            _issue(
                "SOURCE_STATUS_PROMOTION",
                "This interface has no evidence to approve selected research sources.",
            )
        )
    if not data.columns.is_unique or "Close" not in data or not _daily(data.index):
        return _report(
            issues
            + [
                _issue(
                    "PRICE_SCHEMA_INVALID",
                    "Ordered unique naive daily prices including Close are required.",
                )
            ]
        )
    columns = tuple(c for c in (*OHLC, "Adj Close") if c in data)
    numeric = validate_price_data(data, price_columns=columns, market_timezone=expected["timezone"])
    issues.extend(numeric["issues"])
    issues.extend(_ohlc_report(data)["issues"])
    begin, end = _bound(metadata.get("requested_start")), _bound(
        metadata.get("requested_end_exclusive")
    )
    if begin >= end or ((data.index < begin) | (data.index >= end)).any():
        issues.append(
            _issue(
                "PRICE_SCOPE_MISMATCH", "Prices contradict the declared half-open request period."
            )
        )
    sessions, absent, missing = None, [], {}
    if calendar is not None:
        if not _daily(calendar):
            raise ValueError("Reference calendar must contain ordered unique naive daily dates")
        sessions = calendar[(calendar >= begin) & (calendar < end)]
        if len(sessions):
            missing = validate_missing_intervals(data, columns=columns, trading_calendar=sessions)
            issues.extend(missing["issues"])
            absent = [str(day.date()) for day in sessions.difference(data.index)]
        else:
            issues.append(
                _issue(
                    "CALENDAR_COVERAGE_UNVERIFIED",
                    "Reference calendar has no sessions in this period.",
                    severity="unverified",
                )
            )
    issues.append(
        _issue(
            "CALENDAR_AUTHORITY_UNVERIFIED",
            "Reference sessions do not prove official coverage or missing-data causes.",
            severity="unverified",
        )
    )
    dividend = check_dividend_adjustment(data["Close"], market_timezone=expected["timezone"])
    issues.append(
        _issue(
            "DIVIDEND_SPLIT_EVIDENCE_UNVERIFIED",
            "Request options and price ratios do not prove total-return or split evidence.",
            severity="unverified",
        )
    )
    if result.status == "failed":
        issues.append(
            _issue("SOURCE_DECLARED_FAILED", "Preserve the original failed source state.")
        )
    elif result.status != "passed":
        issues.append(
            _issue(
                "SOURCE_DECLARED_UNVERIFIED",
                "Preserve the original unverified source state.",
                severity="unverified",
            )
        )
    return _report(
        issues,
        rows=len(data),
        declared_status=result.status,
        identity={
            key: metadata.get(key)
            for key in (
                "source",
                "symbol",
                "currency",
                "market",
                "timezone",
                "price_basis",
                "auto_adjust",
            )
        },
        observed_nan={c: int(data[c].isna().sum()) for c in columns},
        absent_days=absent,
        reference_session_count=len(sessions) if sessions is not None else None,
        calendar_authority="reference_only" if calendar is not None else "not_supplied",
        requested_start=metadata.get("requested_start"),
        requested_end_exclusive=metadata.get("requested_end_exclusive"),
        missing=missing.get("metrics", {}),
        close_adjusted_ratio=_ratio(data),
        dividend_check=dividend,
    )


def _cell(value):
    if pd.isna(value) or (isinstance(value, Real) and not np.isfinite(value)):
        return None
    return value.item() if isinstance(value, np.generic) else value


def _ktb(parent_snapshot, selected_snapshot, candidate_snapshot):
    parent, selected = (s.datasets.get("KTB10Y") for s in (parent_snapshot, selected_snapshot))
    candidate = candidate_snapshot.datasets.get("KTB10Y_KRX_CORRECTION")
    if parent is None or selected is None or candidate is None:
        return _report(
            [
                _issue(
                    "KTB_SOURCE_MISSING",
                    "Parent, selected and correction-candidate KTB sources are required.",
                )
            ]
        )
    left, right = parent.data, selected.data
    if (
        not left.index.equals(right.index)
        or not left.columns.equals(right.columns)
        or not _daily(left.index)
        or not left.columns.is_unique
    ):
        return _report(
            [
                _issue(
                    "KTB_SOURCE_AXES_MISMATCH",
                    "KTB source axes differ; no sorting or reindexing is allowed.",
                )
            ]
        )
    same = left.eq(right) | (left.isna() & right.isna())
    changes = [
        {
            "date": str(left.index[r].date()),
            "column": str(left.columns[c]),
            "old": _cell(left.iat[r, c]),
            "new": _cell(right.iat[r, c]),
        }
        for r, c in np.argwhere(~same.to_numpy())
    ]
    raw_preserved = _equal(selected.raw_data, left)
    validation_preserved = selected.validation.get("parent_validation") == parent.validation
    issues = []
    if not raw_preserved:
        issues.append(
            _issue(
                "KTB_PARENT_RAW_NOT_PRESERVED",
                "Selected raw_data must retain the complete original parent prices.",
            )
        )
    if not validation_preserved:
        issues.append(
            _issue(
                "KTB_PARENT_VALIDATION_NOT_PRESERVED",
                "Selected parent_validation differs from the original validation.",
            )
        )
    if parent.status != "failed" or selected.status != "unverified":
        issues.append(
            _issue(
                "KTB_SOURCE_STATUS_MISMATCH",
                "Parent failed and selected unverified states must remain explicit.",
            )
        )
    if not (
        len(changes) == 1
        and changes[0]["date"] == str(_EVENT.date())
        and changes[0]["column"] == "Low"
    ):
        issues.append(
            _issue(
                "KTB_UNEXPECTED_SOURCE_DELTA",
                "Only the reviewed 2020-03-10 Low cell may differ in this source version.",
            )
        )
    expected = {
        "dataset_id": "KTB10Y_KRX_CORRECTION",
        "source": "krx",
        "symbol": "148070",
        "role": "etf",
        "market": "KRX",
        "currency": "KRW",
        "timezone": "Asia/Seoul",
    }
    for key, value in expected.items():
        if candidate.metadata.get(key) != value:
            issues.append(
                _issue(
                    "KTB_CANDIDATE_IDENTITY_MISMATCH",
                    "Correction-candidate identity differs.",
                    field=key,
                )
            )
    metrics = {
        "changed_cells": changes,
        "allowed_delta": {"date": str(_EVENT.date()), "column": "Low"},
        "parent_raw_preserved": raw_preserved,
        "parent_validation_preserved": validation_preserved,
        "parent_status": parent.status,
        "selected_status": selected.status,
        "candidate_status": candidate.status,
        "interpretation": "Matching values do not verify authority or corporate-action basis.",
    }
    if (
        not _daily(candidate.data.index)
        or not candidate.data.columns.is_unique
        or _EVENT not in candidate.data.index
        or _EVENT not in right.index
        or not set(OHLC) <= set(candidate.data)
        or not set(OHLC) <= set(right)
    ):
        return _report(
            issues
            + [
                _issue(
                    "KTB_CORRECTION_SCOPE_INVALID",
                    "Candidate and selected event prices need unique OHLC.",
                )
            ],
            **metrics,
        )
    candidate_values, selected_values = {}, {}
    for column in OHLC:
        original, adopted = candidate.data.loc[_EVENT, column], right.loc[_EVENT, column]
        candidate_values[column], selected_values[column] = _cell(original), _cell(adopted)
        finite = all(
            isinstance(v, Real) and not isinstance(v, (bool, np.bool_)) and np.isfinite(v) and v > 0
            for v in (original, adopted)
        )
        if not finite or original != adopted:
            issues.append(
                _issue(
                    "KTB_CORRECTION_CANDIDATE_MISMATCH",
                    "Selected OHLC differs from the preserved correction candidate.",
                    date=str(_EVENT.date()),
                    column=column,
                )
            )
    issues.extend(_ohlc_report(candidate.data)["issues"])
    return _report(
        issues, **metrics, candidate_ohlc=candidate_values, selected_ohlc=selected_values
    )


def audit_price_evidence(
    parent_snapshot, selected_snapshot, candidate_snapshot, *, calendars
) -> dict:
    """Inspect preverified snapshots without changing original data or status.

    Calendars are reference evidence. Complete action ledgers and independent
    gross total-return data are not inputs, so those checks remain unverified.
    """
    issues = [
        _issue(
            "PRICE_BASIS_UNVERIFIED",
            "Archived Close/Adj Close do not prove historical corporate-action definitions.",
            severity="unverified",
        )
    ]
    assets, comparison = {}, {}
    try:
        if not all(
            isinstance(s, Snapshot)
            for s in (parent_snapshot, selected_snapshot, candidate_snapshot)
        ):
            raise ValueError("Expected three Snapshot objects verified by the caller")
        if not isinstance(calendars, dict):
            raise ValueError("calendars must map market names to reference session indexes")
        for asset in TEAM12_ASSETS:
            views = {}
            for name, snapshot in (("parent", parent_snapshot), ("selected", selected_snapshot)):
                result = snapshot.datasets.get(asset)
                market = "KRX" if asset in TEAM12_ASSETS[-2:] else "US"
                views[name] = (
                    _asset(result, asset, calendars.get(market), name == "selected")
                    if result is not None
                    else _report(
                        [_issue("PRICE_SOURCE_MISSING", "Required archived asset is missing.")]
                    )
                )
                if views[name]["status"] == "failed":
                    issues.append(
                        _issue(
                            "ASSET_PRICE_EVIDENCE_FAILED",
                            "Inspect detailed source evidence.",
                            asset=asset,
                            snapshot=name,
                        )
                    )
            assets[asset] = views
            if (
                asset != "KTB10Y"
                and asset in parent_snapshot.datasets
                and asset in selected_snapshot.datasets
                and not _equal(
                    parent_snapshot.datasets[asset].data, selected_snapshot.datasets[asset].data
                )
            ):
                issues.append(
                    _issue(
                        "UNEXPECTED_SELECTED_PRICE_DELTA",
                        "Other selected asset prices must remain identical to the archived parent.",
                        asset=asset,
                    )
                )
        comparison = _ktb(parent_snapshot, selected_snapshot, candidate_snapshot)
        if comparison["status"] == "failed":
            issues.extend(comparison["issues"])
    except (ValueError, TypeError, OverflowError) as exc:
        issues.append(_issue("PRICE_EVIDENCE_INPUT_INVALID", str(exc)))
    return _report(issues, assets=assets, ktb_comparison=comparison)
