"""Offline audits of observed team12 sources and separate FX research scenarios.

Audits never repair prices, create observations, grant training approval or call
providers. Actual releases override research assumptions. Hash agreement proves
preservation against supplied anchors, not a provider's historical truthfulness.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

from robo_advisor.data.collectors import OHLC, _ohlc_report
from robo_advisor.data.snapshot import read_frame, read_snapshot
from robo_advisor.data.validation import validate_missing_intervals, validate_price_data
from robo_advisor.features.dataset import TEAM12_ASSETS, assess_dataset_readiness


def _issue(code, message, *, severity="error", **details):
    return {"code": code, "severity": severity, "message": message, **details}


def _report(issues, **metrics):
    status = "passed"
    if any(item["severity"] == "error" for item in issues):
        status = "failed"
    elif any(item["severity"] == "unverified" for item in issues):
        status = "unverified"
    return {"status": status, "training_approved": False, "issues": issues, "metrics": metrics}


def _daily(index):
    if (
        not isinstance(index, pd.DatetimeIndex)
        or index.empty
        or index.tz is not None
        or index.hasnans
        or not index.is_unique
        or not index.is_monotonic_increasing
        or not index.equals(index.normalize())
    ):
        raise ValueError("Expected ordered unique timezone-naive daily labels")


def _aware(values, *, required=False):
    for value in values:
        if pd.isna(value):
            if required:
                raise ValueError("Missing required timestamp")
        elif isinstance(value, (bool, int, float)) or pd.Timestamp(value).tzinfo is None:
            raise ValueError("Timestamp must explicitly include its timezone")
    return pd.Series(pd.to_datetime(values, utc=True), index=values.index)


def _fx_replay(observations, evaluation_at, direction, max_age_days):
    reference = _aware(observations["reference_at"], required=True)
    actual = _aware(observations["actual_available_at"])
    assumed = _aware(observations["assumed_usable_at"])
    effective = actual.where(actual.notna(), assumed)
    if (effective < reference).any():
        raise ValueError("A release/assumption cannot precede the observation reference")
    cutoffs = _aware(evaluation_at, required=True)
    rates = observations["rate"].to_numpy(dtype=float)
    records = []
    for date, cutoff in cutoffs.items():
        eligible = np.flatnonzero(
            np.isfinite(rates) & effective.le(cutoff).to_numpy() & reference.le(cutoff).to_numpy()
        )
        row = {
            "rate": np.nan,
            "observation_date": pd.NaT,
            "reference_at": pd.NaT,
            "actual_available_at": pd.NaT,
            "assumed_usable_at": pd.NaT,
            "source_row_position": -1,
            "observation_age_hours": np.nan,
            "selection_status": "no_released_observation",
        }
        if len(eligible):
            position = int(eligible[-1])
            age = (cutoff - reference.iloc[position]) / pd.Timedelta(hours=1)
            release_age = (cutoff - effective.iloc[position]) / pd.Timedelta(hours=1)
            rate = rates[position] if direction == "USD/KRW" else 1.0 / rates[position]
            if not np.isfinite(rate) or rate <= 0:
                raise ValueError("FX reciprocal must be finite and positive")
            stale = max(age, release_age) > max_age_days * 24
            row.update(
                rate=np.nan if stale else float(rate),
                observation_date=observations.index[position],
                reference_at=reference.iloc[position],
                actual_available_at=actual.iloc[position],
                assumed_usable_at=assumed.iloc[position],
                source_row_position=position,
                observation_age_hours=float(age),
                selection_status="observation_too_old" if stale else "observed_asof_quote",
            )
        records.append(row)
    return pd.DataFrame(records, index=evaluation_at.index)


def audit_fx_data(
    observations, evaluation_at, *, source, direction, unit, max_age_days, selected=None
):
    """Replay as-of pointers to real FX rows under an explicit research age policy.

    ``reference_at`` defines the fixing/effective-date age origin. Actual releases
    are never replaced by earlier assumptions. NaN source quotes remain counted;
    neither holidays nor missing observation dates are invented. Selected rates
    are normalized to KRW per USD. The source definition still needs independent
    evidence, regardless of a successful arithmetic audit.
    """
    issues = []
    try:
        if source not in {"smbs", "alfred"}:
            raise ValueError("Source must be a separate smbs or alfred scenario")
        if (direction, unit) not in {("USD/KRW", "KRW_per_USD"), ("KRW/USD", "USD_per_KRW")}:
            raise ValueError("FX direction and unit must be an explicit matching pair")
        if type(max_age_days) is not int or not 1 <= max_age_days <= 14:
            raise ValueError("max_age_days must be an explicit integer from 1 to 14")
        if not isinstance(observations, pd.DataFrame) or not observations.columns.is_unique:
            raise ValueError("Observed FX must be a DataFrame with unique columns")
        required = {"rate", "reference_at", "actual_available_at", "assumed_usable_at"}
        if not required.issubset(observations):
            raise ValueError("Observed FX requires reference, actual and assumed timestamps")
        if not isinstance(evaluation_at, pd.Series):
            raise ValueError("evaluation_at must be a Series")
        _daily(observations.index)
        _daily(evaluation_at.index)
        cutoffs = _aware(evaluation_at, required=True)
        local_dates = (
            pd.DatetimeIndex(cutoffs).tz_convert("Asia/Seoul").tz_localize(None).normalize()
        )
        if (
            not cutoffs.is_monotonic_increasing
            or not cutoffs.is_unique
            or not local_dates.equals(evaluation_at.index)
        ):
            raise ValueError(
                "Evaluation timestamps must increase and match their Korean session labels"
            )
        reference = pd.DatetimeIndex(_aware(observations["reference_at"], required=True))
        timezone = "Asia/Seoul" if source == "smbs" else "America/New_York"
        if (
            not reference.tz_convert(timezone)
            .tz_localize(None)
            .normalize()
            .equals(observations.index)
        ):
            raise ValueError("FX reference timestamps differ from their source observation dates")
        dtype = observations["rate"].dtype
        if not is_numeric_dtype(dtype) or is_bool_dtype(dtype) or is_complex_dtype(dtype):
            raise ValueError("FX rates require a real numeric dtype")
        rates = observations["rate"].to_numpy(dtype=float)
        if np.isinf(rates).any() or (rates[np.isfinite(rates)] <= 0).any():
            raise ValueError("Observed FX must be positive and finite, or explicitly missing")
        replay = _fx_replay(observations, evaluation_at, direction, max_age_days)
        if selected is not None:
            if not isinstance(selected, pd.DataFrame) or not selected.index.equals(replay.index):
                raise ValueError("Stored FX selection must match the full evaluation axis")
            if not selected.columns.is_unique or not set(replay).issubset(selected):
                raise ValueError("Stored FX selection is missing pointer/evidence columns")
            for column in replay:
                left, right = selected[column], replay[column]
                if column in {"rate", "observation_age_hours", "source_row_position"}:
                    match = np.isclose(
                        left.to_numpy(dtype=float),
                        right.to_numpy(dtype=float),
                        rtol=1e-12,
                        atol=1e-10,
                        equal_nan=True,
                    )
                elif column == "observation_date":
                    left = pd.to_datetime(left)
                    match = left.eq(right) | (left.isna() & right.isna())
                elif column.endswith("_at"):
                    left, right = _aware(left), _aware(right)
                    match = left.eq(right) | (left.isna() & right.isna())
                else:
                    match = left.eq(right)
                for date in replay.index[~np.asarray(match)]:
                    issues.append(
                        _issue(
                            "FX_SELECTION_MISMATCH",
                            "Stored FX differs from causal replay",
                            date=date.date().isoformat(),
                            column=column,
                        )
                    )
        missing_actual = int(observations["actual_available_at"].isna().sum())
        if missing_actual:
            issues.append(
                _issue(
                    "ACTUAL_RELEASE_UNVERIFIED",
                    "Research assumptions are not actual releases",
                    severity="unverified",
                    rows=missing_actual,
                )
            )
        issues.append(
            _issue(
                "FX_DEFINITION_EVIDENCE_REQUIRED",
                "Declared units do not independently authenticate a provider",
                severity="unverified",
            )
        )
        excluded = replay[replay["selection_status"] != "observed_asof_quote"]
        used = replay[replay["selection_status"] == "observed_asof_quote"]
        pointers = []
        for date, row in replay.iterrows():
            pointers.append(
                {
                    "session_date": date.date().isoformat(),
                    "source_row_position": int(row["source_row_position"]),
                    "rate": None if pd.isna(row["rate"]) else float(row["rate"]),
                    "observation_age_hours": (
                        None
                        if pd.isna(row["observation_age_hours"])
                        else float(row["observation_age_hours"])
                    ),
                    "selection_status": row["selection_status"],
                }
            )
        return _report(
            issues,
            source=source,
            direction=direction,
            unit=unit,
            output_unit="KRW_per_USD",
            max_age_days=max_age_days,
            raw_rows=len(observations),
            missing_rate_count=int(np.isnan(rates).sum()),
            actual_available_count=len(observations) - missing_actual,
            used_rows=len(used),
            max_used_age_hours=None if used.empty else float(used["observation_age_hours"].max()),
            excluded_counts=dict(Counter(excluded["selection_status"])),
            excluded_dates=[
                {"date": d.date().isoformat(), "reason": r}
                for d, r in excluded["selection_status"].items()
            ],
            selected=pointers,
            selection_check=(
                "failed" if issues and any(i["severity"] == "error" for i in issues) else "passed"
            ),
        )
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        return _report([_issue("INPUT_INVALID", str(exc))])


def _source_report(snapshot, calendars):
    reports = {}
    for name, result in snapshot.datasets.items():
        data, metadata = result.data, result.metadata
        columns = tuple(c for c in (*OHLC, "Adj Close", "rate") if c in data)
        issues = list(result.validation.get("issues", []))
        if columns:
            numeric = validate_price_data(data, price_columns=columns)
            # Reuse the collector's OHLC bounds rule; enrich its row pointers.
            for issue in numeric["issues"]:
                if metadata.get("role") == "fx" and issue["code"] == "MISSING_PRICE":
                    issue = {**issue, "severity": "unverified"}
                issues.append(issue)
            for issue in _ohlc_report(data)["issues"]:
                position = issue["position"]
                issues.append({**issue, "date": str(data.index[position].date())})
            calendar = calendars.get(metadata.get("market"))
            if calendar is not None:
                # Audit only the requested source coverage, not a guessed listing lifetime.
                begin = pd.Timestamp(metadata["requested_start"])
                end = pd.Timestamp(metadata["requested_end_exclusive"])
                calendar = calendar[(calendar >= begin) & (calendar < end)]
            missing = validate_missing_intervals(data, columns=columns, trading_calendar=calendar)
            issues.extend(missing["issues"])
        else:
            numeric, missing = {}, {}
            issues.append(
                _issue(
                    "PRICE_SCHEMA_UNVERIFIED",
                    "No supported price/rate column",
                    severity="unverified",
                )
            )
        if result.status != "passed":
            issues.append(
                _issue(
                    "SOURCE_DECLARED_STATUS",
                    "Preserve source verification status",
                    severity="error" if result.status == "failed" else "unverified",
                    source_status=result.status,
                )
            )
        if metadata.get("availability_status") != "verified":
            issues.append(
                _issue(
                    "SOURCE_RELEASE_UNVERIFIED",
                    "Historical publication evidence is absent",
                    severity="unverified",
                )
            )
        identity = {
            key: metadata.get(key)
            for key in (
                "source",
                "symbol",
                "market",
                "currency",
                "unit",
                "price_basis",
                "auto_adjust",
                "availability_status",
                "collected_at",
                "source_identity",
                "source_file_sha256",
            )
        }
        reports[name] = _report(
            issues,
            identity=identity,
            declared_status=result.status,
            rows=len(data),
            start=metadata.get("actual_start"),
            end=metadata.get("actual_end"),
            numeric_status=numeric.get("status"),
            missing=missing.get("metrics", {}),
            calendar_evidence=(
                "reference_only" if metadata.get("market") in calendars else "not_supplied"
            ),
        )
    issues = [
        _issue(
            "SOURCE_CHECK",
            "Source requires attention",
            severity="error" if r["status"] == "failed" else "unverified",
            dataset_id=n,
        )
        for n, r in reports.items()
        if r["status"] != "passed"
    ]
    for asset in TEAM12_ASSETS:
        if asset not in reports:
            issues.append(
                _issue("SOURCE_MISSING", "Required investment asset absent", dataset_id=asset)
            )
    issues.append(
        _issue(
            "CALENDAR_AUTHORITY_UNVERIFIED",
            "Reference session CSVs are not authenticated official calendars",
            severity="unverified",
        )
    )
    return _report(
        issues,
        integrity="passed",
        manifest_sha256=snapshot.manifest_sha256,
        source_counts=dict(Counter(r["status"] for r in reports.values())),
        sources=reports,
    )


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path(root, relative):
    if not isinstance(relative, str) or "\\" in relative or ":" in relative:
        raise ValueError("Paths must be portable relative paths")
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts or not value.parts:
        raise ValueError("Paths must remain within the supplied evidence root")
    candidate = root / value
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError("Evidence path escapes its root")
    return candidate


def _anchored_file(root, descriptor):
    if not isinstance(descriptor, dict):
        raise ValueError("Evidence file descriptor must be an object")
    relative = descriptor.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError("Evidence file descriptor requires a nonempty relative path")
    expected = descriptor.get("sha256")
    if not isinstance(expected, str) or re.fullmatch("[0-9a-f]{64}", expected) is None:
        raise ValueError("An explicit external SHA-256 anchor is required")
    path = _path(root, relative)
    if _sha(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {relative}")
    return path


def _fx_input(raw, source):
    field = "date" if source == "smbs" else "observation_date"
    dates = pd.DatetimeIndex(pd.to_datetime(raw[field]))
    _daily(dates)
    if source == "smbs":
        if not raw["source_identity"].eq("smbs:USD_KRW_MAR:public_period_table").all():
            raise ValueError("SMBS source identity differs from the declared scenario")
        rate = raw["mar_krw_per_usd"]
        reference = dates.tz_localize("Asia/Seoul").tz_convert("UTC")
        assumed = (
            (dates + pd.Timedelta(days=1, hours=10)).tz_localize("Asia/Seoul").tz_convert("UTC")
        )
    elif source == "alfred":
        rate = raw["DEXKOUS"]
        reference = (
            (dates + pd.Timedelta(hours=12)).tz_localize("America/New_York").tz_convert("UTC")
        )
        vintage = pd.DatetimeIndex(pd.to_datetime(raw["vintage_date"]))
        if vintage.hasnans or vintage.tz is not None or not vintage.equals(vintage.normalize()):
            raise ValueError("ALFRED vintage dates must be complete naive daily labels")
        if (vintage < dates).any():
            raise ValueError("ALFRED vintage precedes its observation")
        assumed = _aware(raw["assumed_usable_at"], required=True)
        floor = (vintage + pd.Timedelta(days=1)).tz_localize("America/New_York").tz_convert("UTC")
        if (pd.DatetimeIndex(assumed) < floor).any():
            raise ValueError("ALFRED use precedes the declared vintage-day research cutoff")
    else:
        raise ValueError("Unknown FX scenario")
    actual = _aware(raw["actual_available_at"])
    return pd.DataFrame(
        {
            "rate": rate.to_numpy(),
            "reference_at": list(reference),
            "actual_available_at": list(actual),
            "assumed_usable_at": list(assumed),
        },
        index=dates,
    )


def _excluded_report(eligibility, start, end):
    model = eligibility.loc[(eligibility.index >= start) & (eligibility.index < end)]
    if model["feature_eligible"].dtype != bool:
        raise ValueError("Feature eligibility must be boolean")
    excluded = model.loc[~model["feature_eligible"]]
    reasons = excluded["exclusion_reasons"].fillna("WARMUP")
    return {
        "evaluation_rows": len(model),
        "feature_rows": int(model["feature_eligible"].sum()),
        "excluded_rows": len(excluded),
        "excluded_counts": dict(Counter(reasons)),
        "excluded_dates": [{"date": d.date().isoformat(), "reason": r} for d, r in reasons.items()],
    }


def _check_row_provenance(rows, selection, evaluation, calendars):
    """Cross-check repeated values through their full pointers, not rates alone."""
    row_evaluation = _aware(rows["evaluation_at"], required=True)
    if not row_evaluation.equals(evaluation.reindex(rows.index)):
        raise ValueError("Row evaluation timestamps differ from the anchored date audit")
    for column in (
        "rate",
        "observation_date",
        "reference_at",
        "actual_available_at",
        "assumed_usable_at",
        "source_row_position",
        "observation_age_hours",
        "selection_status",
    ):
        left = rows[f"FX__{column}"]
        right = selection.reindex(rows.index)[column]
        if column in {"rate", "source_row_position", "observation_age_hours"}:
            match = np.isclose(
                left.to_numpy(dtype=float),
                right.to_numpy(dtype=float),
                rtol=1e-12,
                atol=1e-10,
                equal_nan=True,
            )
        else:
            if column.endswith("_at"):
                left, right = _aware(left), _aware(right)
            elif column == "observation_date":
                left, right = pd.to_datetime(left), pd.to_datetime(right)
            match = left.eq(right) | (left.isna() & right.isna())
        if not np.asarray(match).all():
            raise ValueError(f"Row FX lineage disagrees with causal selection: {column}")
    us_dates = calendars["US"]
    positions = us_dates.get_indexer(rows.index)
    if (positions <= 0).any() or not rows.index.isin(calendars["KRX"]).all():
        raise ValueError("Evaluation rows require both markets and a preceding US session")
    previous_us = us_dates[positions - 1]
    for asset in TEAM12_ASSETS:
        expected = previous_us if asset in TEAM12_ASSETS[:10] else rows.index
        actual = pd.DatetimeIndex(pd.to_datetime(rows[f"{asset}__source_dates"]))
        if not actual.equals(expected):
            raise ValueError(f"Source dates violate full-market-session lag: {asset}")


def audit_research_package(config, *, root):
    """Read a portable audit specification; return failures without repairing evidence."""
    root = Path(root)
    reports, issues, snapshots = {}, [], {}
    try:
        if (
            not isinstance(config, dict)
            or type(config.get("schema_version")) is not int
            or config["schema_version"] != 1
        ):
            raise ValueError("Unsupported audit specification")
        for field, required in (
            ("snapshots", {"parent", "selected"}),
            ("scenarios", {"smbs", "alfred"}),
            ("calendars", {"KRX", "US"}),
        ):
            if not isinstance(config.get(field), dict) or set(config[field]) != required:
                raise ValueError(
                    f"Audit specification requires exactly {sorted(required)} in {field}"
                )
        calendars = {}
        for market, descriptor in config["calendars"].items():
            calendar = pd.read_csv(_anchored_file(root, descriptor))
            dates = pd.DatetimeIndex(pd.to_datetime(calendar["session_date"]))
            _daily(dates)
            calendars[market] = dates
        for name, descriptor in config["snapshots"].items():
            try:
                if not isinstance(descriptor, dict) or not isinstance(descriptor.get("path"), str):
                    raise ValueError("Snapshot descriptor requires a relative path")
                _anchored_file(
                    root,
                    {
                        "path": descriptor["path"] + "/manifest.json",
                        "sha256": descriptor.get("sha256"),
                    },
                )
                snapshot = read_snapshot(
                    _path(root, descriptor["path"]), expected_manifest_sha256=descriptor["sha256"]
                )
                snapshots[name] = snapshot
                reports[name] = _source_report(snapshot, calendars)
            except (ValueError, OSError, ImportError) as exc:
                reports[name] = _report([_issue("SNAPSHOT_INVALID", str(exc))], integrity="failed")
        from robo_advisor.features.audit import audit_feature_artifact

        for name, spec in config["scenarios"].items():
            try:
                folder = _path(root, spec["dataset"]["path"])
                lineage_path = _anchored_file(
                    root,
                    {
                        "path": spec["dataset"]["path"] + "/lineage.json",
                        "sha256": spec["dataset"]["sha256"],
                    },
                )
                lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
                if lineage.get("schema_version") != "experimental-real-team12-v1":
                    raise ValueError(
                        "This package replay requires the explicitly experimental schema"
                    )
                rows_descriptor = lineage["files"]["rows"]
                if rows_descriptor["path"] != "rows.parquet":
                    raise ValueError("Unexpected rows frame location")
                rows = read_frame(folder / "rows.parquet", rows_descriptor)
                raw = pd.read_csv(_anchored_file(root, spec["fx"]))
                observations = _fx_input(raw, name)
                fx_selection = pd.read_csv(
                    _anchored_file(root, spec["selected_fx"]),
                    index_col="session_date",
                    parse_dates=["session_date"],
                )
                eligibility = pd.read_csv(
                    _anchored_file(root, spec["eligibility"]),
                    index_col="session_date",
                    parse_dates=["session_date"],
                )
                _daily(eligibility.index)
                evaluation = _aware(eligibility["evaluation_at"], required=True)
                if not rows.index.isin(eligibility.index).all():
                    raise ValueError("Dataset rows fall outside the audited evaluation dates")
                fx_report = audit_fx_data(
                    observations,
                    evaluation,
                    source=name,
                    direction=spec["direction"],
                    unit=spec["unit"],
                    max_age_days=spec["max_age_days"],
                    selected=fx_selection,
                )
                _check_row_provenance(rows, fx_selection, evaluation, calendars)
                if eligibility["feature_eligible"].dtype != bool or not np.array_equal(
                    eligibility.loc[rows.index, "feature_eligible"], rows["feature_eligible"]
                ):
                    raise ValueError("Stored row eligibility differs from the anchored date audit")
                eligible_dates = eligibility.index[eligibility["feature_eligible"]]
                if not eligible_dates.equals(rows.index[rows["feature_eligible"]]):
                    raise ValueError(
                        "Eligible dates differ between the artifact and exclusion audit"
                    )
                snapshot = snapshots[spec["snapshot"]]
                recorded = lineage["provenance"]["source_identity"]
                if any(
                    recorded[a]["version_manifest_sha256"] != snapshot.manifest_sha256
                    for a in TEAM12_ASSETS
                ):
                    raise ValueError("Feature lineage points at a different source snapshot")
                fx_source = snapshot.datasets[spec["fx_dataset_id"]]
                if not observations.index.equals(fx_source.data.index) or not np.allclose(
                    observations["rate"], fx_source.data["rate"], equal_nan=True, rtol=0, atol=0
                ):
                    raise ValueError("FX CSV differs from the preserved source snapshot")
                prices = pd.DataFrame(index=rows.index, columns=list(TEAM12_ASSETS), dtype=float)
                for asset in TEAM12_ASSETS:
                    source = snapshot.datasets[asset]
                    expected_currency = "KRW" if asset in TEAM12_ASSETS[-2:] else "USD"
                    if source.metadata.get("currency") != expected_currency:
                        raise ValueError(f"Unexpected source currency: {asset}")
                    dates = pd.DatetimeIndex(pd.to_datetime(rows[f"{asset}__source_dates"]))
                    if not dates.isin(source.data.index).all():
                        raise ValueError(f"Unknown source observation: {asset}")
                    if not dates.equals(
                        pd.DatetimeIndex(
                            pd.to_datetime(eligibility.loc[rows.index, f"{asset}__source_date"])
                        )
                    ):
                        raise ValueError(f"Source dates disagree with the exclusion audit: {asset}")
                    prices[asset] = source.data["Close"].reindex(dates).to_numpy()
                    if expected_currency == "USD":
                        prices[asset] *= pd.to_numeric(rows["FX__rate"])
                selected_rows = fx_selection.reindex(rows.index)
                if not np.allclose(
                    rows["FX__rate"], selected_rows["rate"], rtol=0, atol=0, equal_nan=True
                ):
                    raise ValueError("Dataset FX rate differs from the replayed source selection")
                feature_report = audit_feature_artifact(
                    folder, expected_lineage_sha256=spec["dataset"]["sha256"], prices_krw=prices
                )
                excluded = _excluded_report(
                    eligibility,
                    pd.Timestamp(lineage["model_start"]),
                    pd.Timestamp(lineage["model_end_exclusive"]),
                )
                if excluded["feature_rows"] != lineage["feature_rows"]:
                    raise ValueError(
                        "Excluded-date accounting differs from the stored feature rows"
                    )
                reports[name] = _report(
                    [
                        _issue(
                            "RESEARCH_ONLY",
                            "Actual releases and price adjustment history remain unverified",
                            severity="unverified",
                        )
                    ],
                    fx=fx_report,
                    features=feature_report,
                    exclusions=excluded,
                )
                if fx_report["status"] == "failed" or feature_report["status"] == "failed":
                    reports[name]["status"] = "failed"
            except (ValueError, TypeError, KeyError, OSError, ImportError) as exc:
                reports[name] = _report([_issue("SCENARIO_INVALID", str(exc))])
        if "selected" in snapshots:
            readiness = assess_dataset_readiness(
                snapshots["selected"],
                expected_manifest_sha256=snapshots["selected"].manifest_sha256,
            )
        else:
            readiness = {
                "status": "blocked",
                "training_approved": False,
                "reasons": [{"code": "SOURCE_SNAPSHOT_UNREADABLE"}],
            }
        for name, report in reports.items():
            if report["status"] != "passed":
                issues.append(
                    _issue(
                        "AUDIT_COMPONENT",
                        "Component requires attention",
                        severity="error" if report["status"] == "failed" else "unverified",
                        component=name,
                    )
                )
        return _report(
            issues,
            components=reports,
            strict_readiness=readiness,
            ppo={
                "status": "blocked",
                "reason": (
                    "Issue #94 adapter, source approval and gap/execution " "policies are pending"
                ),
            },
        )
    except (ValueError, TypeError, KeyError, OSError, ImportError) as exc:
        return _report([_issue("AUDIT_INPUT_INVALID", str(exc))], components=reports)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--output", type=Path, help="New JSON report; existing files are never overwritten"
    )
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        report = audit_research_package(config, root=args.root)
        rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output is not None:
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(rendered + "\n")
        print(rendered)
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps(_report([_issue("AUDIT_INPUT_INVALID", str(exc))]), ensure_ascii=False))
        return 1
    return {"passed": 0, "failed": 1, "unverified": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
