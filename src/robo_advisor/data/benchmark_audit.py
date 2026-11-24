"""Read-only benchmark source and point-in-time arithmetic audit for Issue #27."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .alignment import align_trading_calendars, convert_prices_to_krw
from .audit import _anchored_file, _aware, _daily, _fx_input, _issue, _report, audit_fx_data
from .validation import validate_missing_intervals, validate_price_data

_IDENTITIES = {
    "KOSPI": {("yahoo", "^KS11", "PR"), ("krx", "KOSPI", "PR"), ("krx", "KOSPI TR", "TR")},
    "SP500": {
        ("yahoo", "^GSPC", "PR"),
        ("fred", "SP500", "PR"),
        ("sp_dji", "SPX", "PR"),
        ("sp_dji", "SPTR", "TR"),
        ("unknown", None, "PR"),
    },
}
_MARKET = {"KOSPI": ("KRX", "KRW"), "SP500": ("US", "USD")}
_TIMEZONE = {"KRX": "Asia/Seoul", "US": "America/New_York"}
_DATE_FIELDS = ("return_start_date", "sp500_source_date", "fx_observation_date")
_VALUE_FIELDS = (
    "fx_krw_per_usd",
    "sp500_krw",
    "kospi_simple_return",
    "sp500_krw_simple_return",
)


def _date(value):
    return None if pd.isna(value) else value.date().isoformat()


def _dates(index):
    return [_date(date) for date in index]


def _csv(root, descriptor):
    path = _anchored_file(root, descriptor)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != descriptor["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {descriptor['path']}")
    with io.StringIO(payload.decode("utf-8-sig"), newline="") as handle:
        header = next(csv.reader(handle), [])
        if not header or len(set(header)) != len(header):
            raise ValueError(f"Expected unique CSV columns: {descriptor['path']}")
        handle.seek(0)
        return pd.read_csv(handle)


def _indexed(raw, field):
    frame = raw.set_index(field)
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    _daily(frame.index)
    return frame


def _timestamps(values, *, required=False):
    # Parse each representation independently before the shared timezone check.
    parsed = values.map(lambda value: pd.Timestamp(value) if isinstance(value, str) else value)
    return _aware(parsed, required=required)


def _local_dates(timestamps, market):
    return pd.DatetimeIndex(timestamps).tz_convert(_TIMEZONE[market]).tz_localize(None).normalize()


def _calendar(root, descriptor, market):
    frame = _indexed(_csv(root, descriptor), "session_date")
    close_fields = {"close", "scheduled_close_utc"}.intersection(frame)
    if len(close_fields) != 1:
        raise ValueError(f"{market} calendar requires exactly one supported close column")
    close = _timestamps(frame[next(iter(close_fields))], required=True)
    if not _local_dates(close, market).equals(frame.index):
        raise ValueError(f"{market} close instants must match their local session dates")
    return close


def _bounds(config):
    if config.get("schema_version") != 1:
        raise ValueError("Expected benchmark audit schema_version 1")
    bounds = pd.DatetimeIndex([config["start"], config["end_exclusive"]])
    _daily(bounds)
    return bounds[0], bounds[1]


def _scope(index, start, end):
    return (index >= start) & (index < end)


def _price(root, asset, descriptor, close, start, end, seed, issues):
    market, currency = _MARKET[asset]
    identity = (descriptor["source"], descriptor.get("symbol"), descriptor["return_type"])
    if (
        descriptor["index_id"] != asset
        or descriptor["currency"] != currency
        or identity not in _IDENTITIES[asset]
    ):
        raise ValueError(f"{asset} index identity, return type or currency is inconsistent")
    frame = _indexed(_csv(root, descriptor), "Date")
    if "Ticker" in frame and not frame["Ticker"].eq(descriptor.get("symbol")).all():
        raise ValueError(f"{asset} file ticker differs from declared symbol")
    if "Source" in frame and (
        descriptor["source"] != "yahoo" or not frame["Source"].eq("Yahoo Finance").all()
    ):
        raise ValueError(f"{asset} file source differs from declared provider")
    expected = close.index[_scope(close.index, start, end)].union(seed)
    frame = frame.loc[_scope(frame.index, start, end) | frame.index.isin(seed)]
    # A missing quote is an audit finding. Invalid prices and off-calendar rows are errors.
    price = validate_price_data(frame, price_columns=("Close",))
    issues.extend(
        {**item, "index_id": asset} for item in price["issues"] if item["code"] != "MISSING_PRICE"
    )
    missing = validate_missing_intervals(
        frame,
        trading_calendar=expected,
        max_missing_ratio=1.0,
        max_consecutive_missing=len(expected),
    )
    issues.extend({**item, "index_id": asset} for item in missing["issues"])
    absent = expected.difference(frame.index)
    null = frame.index[frame["Close"].isna()]
    if len(absent) or len(null):
        issues.append(
            _issue(
                "INDEX_OBSERVATION_MISSING",
                "Reference sessions lack a usable index close",
                severity="unverified",
                index_id=asset,
                absent_dates=_dates(absent),
                null_dates=_dates(null),
            )
        )
    reported = (
        _timestamps(frame["available_at"])
        if "available_at" in frame
        else close.iloc[:0].reindex(frame.index)
    )
    session_close = close.reindex(frame.index)
    if reported.lt(session_close).any():
        raise ValueError(f"{asset} close availability cannot precede the session close")
    issues.append(
        _issue(
            "INDEX_SOURCE_UNVERIFIED",
            "Hash and identity checks do not authenticate the original request, index definition "
            "or historical release times",
            severity="unverified",
            index_id=asset,
        )
    )
    stats = {
        "source": identity[0],
        "symbol": identity[1],
        "return_type": identity[2],
        "currency": currency,
        "sha256": descriptor["sha256"],
        "source_status": "unverified",
        "identity_columns_checked": [key for key in ("Ticker", "Source") if key in frame],
        "observed_rows": len(frame),
        "reference_sessions": len(expected),
        "absent_dates": _dates(absent),
        "null_dates": _dates(null),
        "close_availability_assumed_rows": int(reported.isna().sum()),
    }
    return frame["Close"], reported.fillna(session_close), reported.notna(), stats


def validate_comparison_periods(strategy_starts, benchmark_starts):
    """Raise unless two series identify identical start/end pairs for usable returns."""
    for starts in (strategy_starts, benchmark_starts):
        if not isinstance(starts, pd.Series):
            raise ValueError("Return start dates must be a Series indexed by return end dates")
        _daily(starts.index)
        labels = pd.DatetimeIndex(pd.to_datetime(starts))
        _daily(labels)
        if (labels >= starts.index).any():
            raise ValueError("Return starts must precede their end dates")
    if not strategy_starts.index.equals(benchmark_starts.index) or not pd.DatetimeIndex(
        pd.to_datetime(strategy_starts)
    ).equals(pd.DatetimeIndex(pd.to_datetime(benchmark_starts))):
        raise ValueError("Strategy and benchmark return intervals must match exactly")


def _check_claims(root, descriptor, expected, issues):
    claims = _indexed(_csv(root, descriptor), "session_date")
    if not claims.index.equals(expected.index):
        raise ValueError("Claimed benchmark rows must match the full evaluation axis")
    for column in (*_DATE_FIELDS, *_VALUE_FIELDS):
        left, right = claims[column], expected[column]
        if column in _DATE_FIELDS:
            left = pd.to_datetime(left)
            same = left.eq(right) | (left.isna() & right.isna())
        else:
            same = np.isclose(
                pd.to_numeric(left).to_numpy(dtype=float),
                right.to_numpy(dtype=float),
                atol=1e-10,
                rtol=1e-12,
                equal_nan=True,
            )
        if not np.all(same):
            issues.append(
                _issue(
                    "BENCHMARK_REPLAY_MISMATCH",
                    "Claimed observations or values differ from the causal replay",
                    column=column,
                    dates=_dates(expected.index[~np.asarray(same)]),
                )
            )


def _align(prices, dates, cutoffs, fx_rows, fx_observations, kr_dates, us_dates):
    values = pd.DataFrame(index=dates)
    values["evaluation_at"] = cutoffs
    flags = {}
    for asset, mapping in (("KOSPI", kr_dates), ("SP500", us_dates)):
        close, available, reported = prices[asset]
        quote = close.reindex(mapping).to_numpy(dtype=float)
        time = pd.Series(available.reindex(mapping).array, index=dates)
        known = pd.Series(reported.reindex(mapping, fill_value=False).array, index=dates)
        late = time.gt(cutoffs)
        values[f"{asset.lower()}_available_at"] = time
        values[f"{asset.lower()}_timing_reported"] = known
        values[f"{asset.lower()}_close"] = np.where(late, np.nan, quote)
        flags[f"{asset}_CLOSE_MISSING"] = np.isnan(quote)
        flags[f"{asset}_AFTER_CUTOFF"] = late
    values["sp500_source_date"] = us_dates
    flags["US_PREVIOUS_SESSION_UNAVAILABLE"] = us_dates.isna()
    values["fx_krw_per_usd"] = [row["rate"] for row in fx_rows]
    values["fx_krw_per_usd"] = values["fx_krw_per_usd"].astype(float)
    values["fx_observation_age_hours"] = [row["observation_age_hours"] for row in fx_rows]
    fx_positions = [row["source_row_position"] for row in fx_rows]
    values["fx_source_row_position"] = fx_positions
    values["fx_observation_date"] = [
        fx_observations.index[p] if p >= 0 else pd.NaT for p in fx_positions
    ]
    for field in ("reference_at", "actual_available_at", "assumed_usable_at"):
        values[f"fx_{field}"] = [
            fx_observations[field].iloc[p] if p >= 0 else pd.NaT for p in fx_positions
        ]
    flags["FX_NOT_AVAILABLE"] = [
        r["selection_status"] == "no_released_observation" for r in fx_rows
    ]
    flags["FX_STALE"] = [r["selection_status"] == "observation_too_old" for r in fx_rows]
    converted = convert_prices_to_krw(
        values[["sp500_close"]], values["fx_krw_per_usd"], fx_direction="USD/KRW"
    )
    values["sp500_krw"] = converted["sp500_close"]
    levels = values[["kospi_close", "sp500_krw"]]
    complete = levels.notna().all(axis=1)
    adjacent = complete & complete.shift(1, fill_value=False)
    with np.errstate(over="ignore", invalid="ignore"):
        returns = levels.div(levels.shift(1)).sub(1).where(adjacent, axis=0)
    if not np.isfinite(returns.loc[adjacent].to_numpy()).all():
        raise ValueError("Benchmark return arithmetic is not finite")
    values["kospi_simple_return"] = returns["kospi_close"]
    values["sp500_krw_simple_return"] = returns["sp500_krw"]
    values["return_start_date"] = pd.Series(dates, index=dates).shift(1)
    reasons = pd.DataFrame(flags, index=dates)
    reasons["RETURN_WARMUP"] = values["return_start_date"].isna()
    reasons["PREVIOUS_EVALUATION_INCOMPLETE"] = ~complete.shift(1, fill_value=True)
    values["exclusion_reasons"] = reasons.apply(
        lambda row: list(reasons.columns[row.to_numpy(dtype=bool)]), axis=1
    )
    return values


def audit_benchmark_data(config, *, root):
    """Replay two standalone index legs from pinned local evidence, with no price fill."""
    issues = []
    try:
        root = Path(root)
        start, end = _bounds(config)
        kr = _calendar(root, config["calendars"]["KRX"], "KRX")
        us = _calendar(root, config["calendars"]["US"], "US")
        common = align_trading_calendars(kr.index, us.index)
        dates = common[_scope(common, start, end)]
        _daily(dates)
        evaluation = _indexed(_csv(root, config["evaluation"]), "session_date")
        evaluation = evaluation.loc[_scope(evaluation.index, start, end)]
        if not evaluation.index.equals(dates):
            raise ValueError("Evaluation dates must match every common session in the audit range")
        cutoffs = _timestamps(evaluation["evaluation_at"], required=True)
        us_dates = pd.DatetimeIndex(pd.Series(us.index, index=us.index).shift(1).loc[dates])
        us_seed = us_dates[us_dates.notna() & (us_dates < start)].unique()
        specs = config["benchmarks"]
        if specs["KOSPI"]["return_type"] != specs["SP500"]["return_type"]:
            raise ValueError("KOSPI and S&P 500 return types must match")
        results = {
            "KOSPI": _price(root, "KOSPI", specs["KOSPI"], kr, start, end, dates[:0], issues),
            "SP500": _price(root, "SP500", specs["SP500"], us, start, end, us_seed, issues),
        }
        sources = {name: result[3] for name, result in results.items()}
        if any(item["severity"] == "error" for item in issues):
            return _report(issues, sources=sources)
        fx_spec = config["fx"]
        fx_raw = _csv(root, fx_spec)
        fx_raw["actual_available_at"] = _timestamps(fx_raw["actual_available_at"])
        if fx_spec["source"] == "alfred":
            fx_raw["assumed_usable_at"] = _timestamps(fx_raw["assumed_usable_at"], required=True)
        observations = _fx_input(fx_raw, fx_spec["source"])
        fx = audit_fx_data(
            observations,
            cutoffs,
            source=fx_spec["source"],
            direction="USD/KRW",
            unit="KRW_per_USD",
            max_age_days=fx_spec["max_age_days"],
        )
        issues.extend(fx["issues"])
        if fx["status"] == "failed":
            return _report(issues, sources=sources, fx=fx)
        rows = _align(
            {name: result[:3] for name, result in results.items()},
            dates,
            cutoffs,
            fx["metrics"]["selected"],
            observations,
            dates,
            us_dates,
        )
        issues.append(
            _issue(
                "CALENDAR_EVIDENCE_UNVERIFIED",
                "Supplied schedules and evaluation instants are not independently authenticated",
                severity="unverified",
            )
        )
        comparison = "not_supplied"
        if "claims" in config:
            _check_claims(root, config["claims"], rows, issues)
            comparison = (
                "failed"
                if any(i["code"] == "BENCHMARK_REPLAY_MISMATCH" for i in issues)
                else "passed"
            )
        valid = rows["kospi_simple_return"].notna()
        periods_check = "not_supplied"
        if "comparison" in config:
            strategy = _indexed(_csv(root, config["comparison"]), "session_date")
            try:
                validate_comparison_periods(
                    strategy["return_start_date"], rows.loc[valid, "return_start_date"]
                )
                periods_check = "passed"
            except ValueError as exc:
                issues.append(_issue("RETURN_INTERVAL_MISMATCH", str(exc)))
                periods_check = "failed"
        serialized = rows.reset_index(names="session_date")
        for field in ("session_date", *_DATE_FIELDS):
            serialized[field] = serialized[field].map(_date)
        return _report(
            issues,
            sources=sources,
            fx={
                "source": fx_spec["source"],
                "sha256": fx_spec["sha256"],
                "max_age_days": fx_spec["max_age_days"],
            },
            calendars={name: config["calendars"][name]["sha256"] for name in ("KRX", "US")},
            evaluation_sha256=config["evaluation"]["sha256"],
            common_sessions=len(dates),
            usable_return_sessions=int(valid.sum()),
            affected_return_dates=_dates(dates[~valid]),
            claims_check=comparison,
            periods_check=periods_check,
            rows=json.loads(
                serialized.to_json(
                    orient="records", date_format="iso", date_unit="ns", double_precision=15
                )
            ),
        )
    except (OSError, ValueError, TypeError, KeyError, OverflowError, AttributeError) as exc:
        code = (
            "EVIDENCE_HASH_MISMATCH"
            if str(exc).startswith("SHA-256 mismatch:")
            else "INPUT_INVALID"
        )
        return _report(issues + [_issue(code, str(exc))])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text(encoding="utf-8-sig"))
        report = audit_benchmark_data(config, root=args.root)
    except (OSError, ValueError) as exc:
        report = _report([_issue("INPUT_INVALID", str(exc))])
    print(json.dumps(report, indent=2, allow_nan=False))
    return {"passed": 0, "failed": 1, "unverified": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
