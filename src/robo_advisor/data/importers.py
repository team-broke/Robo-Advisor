"""Read-only legacy market imports; storage eligibility is not training approval.

CSV raw_data contains the original decoded field strings. XLSX raw_data is an
explicit XML lexical projection, with per-cell type/address/style evidence.
Neither representation repairs, sorts or fills source observations.
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import hashlib
from io import BytesIO, StringIO
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd

from .collectors import CollectionResult, DatasetSpec, load_asset_config
from .validation import (
    check_dividend_adjustment,
    validate_listing_date,
    validate_missing_intervals,
    validate_price_data,
)

TEAM12_ASSETS = (
    "SPY",
    "IWM",
    "EFA",
    "EEM",
    "IEF",
    "TIP",
    "LQD",
    "GLD",
    "DBC",
    "VNQ",
    "KODEX_200TR",
    "KTB10Y",
)
_OHLC = ("Open", "High", "Low", "Close")
_NUMERIC = (*_OHLC, "Adj Close", "Volume")
_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


class LegacyImportError(ValueError):
    """The source cannot be identified or parsed under the declared import contract."""


def _issue(code, message, *, severity="error", **details):
    return {"code": code, "message": message, "severity": severity, **details}


def _report(issues, **metrics):
    status = (
        "failed"
        if any(item["severity"] == "error" for item in issues)
        else (
            "unverified" if any(item["severity"] == "unverified" for item in issues) else "passed"
        )
    )
    return {"valid": status == "passed", "status": status, "issues": issues, "metrics": metrics}


def _bounds(start, end):
    days = []
    for value in (start, end):
        try:
            if isinstance(value, (bool, int, float)):
                raise ValueError("A daily date is required")
            day = pd.Timestamp(value)
            if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
                raise ValueError("A naive daily date is required")
            days.append(day)
        except (ValueError, TypeError, OverflowError) as exc:
            raise LegacyImportError("Import bounds must be timezone-naive daily dates") from exc
    if days[0] >= days[1]:
        raise LegacyImportError("Import end must be exclusive and after start")
    return days


def _source(path, expected_sha256):
    path = Path(path)
    try:
        blob = path.read_bytes()
    except OSError as exc:
        raise LegacyImportError("Legacy source is inaccessible") from exc
    digest = hashlib.sha256(blob).hexdigest()
    if expected_sha256 is not None:
        if not isinstance(expected_sha256, str) or not re.fullmatch(
            "[0-9a-f]{64}", expected_sha256
        ):
            raise LegacyImportError("Expected source SHA-256 must be 64 lowercase hex digits")
        if digest != expected_sha256:
            raise LegacyImportError("Legacy source SHA-256 mismatch; review the changed source")
    return blob, {"path": str(path.resolve()), "sha256": digest, "bytes": len(blob)}


def _headers(header):
    if not header or any(not isinstance(item, str) or not item for item in header):
        raise LegacyImportError("Source headers must be nonempty strings")
    if len(set(header)) != len(header) or any(item.startswith("__excel_") for item in header):
        raise LegacyImportError("Duplicate or reserved source headers")


def _csv_source(path, expected_sha256):
    blob, provenance = _source(path, expected_sha256)
    try:
        raw_text = blob.decode("utf-8")
        rows = list(csv.reader(StringIO(raw_text.removeprefix("\ufeff"), newline=""), strict=True))
    except (UnicodeError, csv.Error) as exc:
        raise LegacyImportError("Expected a valid UTF-8 CSV source") from exc
    if not rows:
        raise LegacyImportError("CSV is empty")
    _headers(rows[0])
    if any(len(row) != len(rows[0]) for row in rows[1:]):
        raise LegacyImportError("CSV row width differs from its header; no rows were discarded")
    raw = pd.DataFrame(rows[1:], columns=rows[0], dtype=object)
    provenance.update(
        format="csv",
        raw_schema="csv-decoded-fields-v1",
        encoding="utf-8",
        bom_present=raw_text.startswith("\ufeff"),
        source_rows=list(range(2, len(rows) + 1)),
    )
    return raw, raw_text, provenance


def _sheet_rows(zipped, target, shared):
    root = ET.fromstring(zipped.read(target))
    rows = []
    for row in root.findall("m:sheetData/m:row", _NS):
        row_label = row.get("r", "")
        if not re.fullmatch(r"[1-9][0-9]*", row_label):
            raise LegacyImportError("Missing or invalid Excel row address")
        cells, evidence = {}, {}
        for cell in row:
            reference = cell.get("r", "")
            if not re.fullmatch(r"[A-Z]+[1-9][0-9]*", reference):
                raise LegacyImportError("Missing or invalid Excel cell address")
            if re.search(r"[0-9]+$", reference).group() != row_label:
                raise LegacyImportError("Excel cell and enclosing row addresses disagree")
            column = re.sub(r"\d", "", reference)
            if column in cells or cell.find("m:f", _NS) is not None:
                raise LegacyImportError("Duplicate or formula source cells are unsupported")
            kind = cell.get("t", "n")
            value = cell.find("m:v", _NS)
            lexical = value.text if value is not None else None
            if kind == "s":
                try:
                    index = int(lexical)
                    if not 0 <= index < len(shared):
                        raise ValueError("Shared-string reference is outside its table")
                    decoded = shared[index]
                except (ValueError, TypeError, IndexError) as exc:
                    raise LegacyImportError("Invalid Excel shared-string reference") from exc
            elif kind == "inlineStr":
                decoded = "".join(item.text or "" for item in cell.findall(".//m:t", _NS))
            elif kind == "n":
                decoded = lexical
            else:
                raise LegacyImportError("Unsupported source Excel cell type")
            cells[column] = decoded
            evidence[column] = {
                "ref": reference,
                "type": kind,
                "style": cell.get("s"),
                "xml_value": lexical,
            }
        if any(value not in (None, "") for value in cells.values()):
            rows.append((int(row.attrib["r"]), cells, evidence))
    return rows


def _workbook_source(path, expected_sha256):
    blob, provenance = _source(path, expected_sha256)
    try:
        with ZipFile(BytesIO(blob)) as zipped:
            shared = []
            if "xl/sharedStrings.xml" in zipped.namelist():
                shared = [
                    "".join(part.text or "" for part in item.findall(".//m:t", _NS))
                    for item in ET.fromstring(zipped.read("xl/sharedStrings.xml"))
                ]
            workbook = ET.fromstring(zipped.read("xl/workbook.xml"))
            properties = workbook.find("m:workbookPr", _NS)
            if properties is not None and properties.get("date1904") not in (None, "0", "false"):
                raise LegacyImportError("Only the audited 1900 Excel date system is supported")
            relationships = {
                item.attrib["Id"]: item.attrib["Target"]
                for item in ET.fromstring(zipped.read("xl/_rels/workbook.xml.rels"))
            }
            sheets = {}
            for sheet in workbook.find("m:sheets", _NS):
                if sheet.get("name") not in {"가격원자료", "이용안내"}:
                    continue
                target = relationships[sheet.attrib[f"{{{_REL}}}id"]]
                target = target.lstrip("/") if target.startswith("/") else "xl/" + target
                if ".." in Path(target).parts:
                    raise LegacyImportError("Unsupported Excel relationship path")
                sheets[sheet.get("name")] = _sheet_rows(zipped, target, shared)
    except (BadZipFile, KeyError, ET.ParseError, ValueError) as exc:
        if isinstance(exc, LegacyImportError):
            raise
        raise LegacyImportError("Invalid or unsupported Excel workbook") from exc
    if "가격원자료" not in sheets or not sheets["가격원자료"]:
        raise LegacyImportError("Expected the 가격원자료 source sheet")
    header_row, headers, _ = sheets["가격원자료"][0]
    _headers(list(headers.values()))
    required = {"Date", "Ticker", *_OHLC, "Volume"}
    if not required.issubset(headers.values()):
        raise LegacyImportError("Expected market OHLCV source columns in 가격원자료")
    records, source_rows = [], []
    for source_row, cells, evidence in sheets["가격원자료"][1:]:
        if set(cells) - set(headers):
            raise LegacyImportError("Excel source cells have no header")
        records.append(
            {
                **{label: cells.get(column) for column, label in headers.items()},
                "__excel_row": source_row,
                "__excel_cells": json.dumps(evidence, ensure_ascii=True, sort_keys=True),
            }
        )
        source_rows.append(source_row)
    raw = pd.DataFrame(records, columns=[*headers.values(), "__excel_row", "__excel_cells"])
    claimed_time = None
    guide_cells = {
        column + str(row): value
        for row, cells, _ in sheets.get("이용안내", [])
        for column, value in cells.items()
    }
    if guide_cells.get("A3") == "수집 시각 UTC" and guide_cells.get("B3") is not None:
        try:
            serial = float(guide_cells["B3"])
            if np.isfinite(serial):
                claimed_time = (
                    datetime(1899, 12, 30, tzinfo=timezone.utc) + timedelta(days=serial)
                ).isoformat()
        except (ValueError, OverflowError):
            pass
    provenance.update(
        format="xlsx",
        raw_schema="xlsx-xml-lexical-cells-v1",
        sheet="가격원자료",
        header_row=header_row,
        date_system="1900",
        recorded_collection_time=claimed_time,
        recorded_collection_time_cell="이용안내!B3",
        recorded_collection_time_verified=False,
    )
    return raw, provenance


def _dates(values, *, excel):
    dates = []
    for value in values:
        try:
            if excel:
                value = datetime(1899, 12, 30) + timedelta(days=float(value))
            day = pd.Timestamp(value)
            if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
                day = pd.NaT
        except (ValueError, TypeError, OverflowError):
            day = pd.NaT
        dates.append(day)
    return pd.DatetimeIndex(dates, name="Date")


def _build(spec, raw, raw_text, provenance, start, end):
    if (
        not isinstance(spec, DatasetSpec)
        or spec.source != "yahoo"
        or spec.role not in {"etf", "benchmark"}
    ):
        raise LegacyImportError("Legacy OHLCV import requires a Yahoo ETF/benchmark DatasetSpec")
    first, last = _bounds(start, end)
    if not {"Date", *_OHLC, "Volume"}.issubset(raw.columns):
        raise LegacyImportError(
            "Expected source Date and full OHLCV; derived prices are unsupported"
        )
    excel = provenance["format"] == "xlsx"
    source_rows = raw["__excel_row"].tolist() if excel else provenance["source_rows"]
    dates = _dates(raw["Date"], excel=excel)
    selected = ((dates >= first) & (dates < last)) | dates.isna()
    positions = np.flatnonzero(selected).tolist()
    chosen = raw.iloc[positions]
    data = pd.DataFrame(index=dates[selected])
    parse_issues = []
    for column in _NUMERIC:
        if column not in raw:
            continue
        values = pd.to_numeric(chosen[column], errors="coerce")
        for position in np.flatnonzero(
            values.isna() & chosen[column].notna() & chosen[column].ne("")
        ):
            parse_issues.append(
                _issue(
                    "SOURCE_NON_NUMERIC",
                    "Non-numeric source value retained in raw_data",
                    column=column,
                    position=int(position),
                )
            )
        data[column] = values.to_numpy()
    for column in ("Ticker", "Source"):
        if column in raw:
            expected = spec.symbol if column == "Ticker" else "Yahoo Finance"
            if any(value != expected for value in raw[column]):
                raise LegacyImportError(f"Source ticker/symbol identity differs: {column}")
    if "Source URL" in raw:
        mismatches = [
            value
            for value in raw["Source URL"].dropna().unique()
            if unquote(urlsplit(value).path.rsplit("/", 1)[-1]) != spec.symbol
        ]
        if mismatches:
            raise LegacyImportError("Workbook source URL ticker differs from specification")
    if "Unit" in raw and any(value != spec.unit for value in raw["Unit"]):
        parse_issues.append(
            _issue("SOURCE_UNIT_MISMATCH", "Source unit differs from specification")
        )
    policy = [
        _issue(
            "HISTORICAL_AVAILABILITY_UNVERIFIED",
            "Historical available_at was not supplied",
            severity="unverified",
        ),
        _issue(
            "LEGACY_ADJUSTMENT_UNVERIFIED",
            "Provider Close/Adj Close preserved; auto_adjust execution and "
            "independent adjustment unverified",
            severity="unverified",
        ),
        _issue(
            "LEGACY_SOURCE_LINEAGE",
            "Legacy source is an archived representation, not a fresh API response",
            severity="unverified",
        ),
    ]
    if excel and provenance.get("recorded_collection_time"):
        policy.append(
            _issue(
                "RECORDED_COLLECTION_TIME_UNVERIFIED",
                "Workbook collection clock is a recorded claim, not an execution log",
                severity="unverified",
            )
        )
    elif not excel:
        policy.append(
            _issue(
                "SOURCE_COLLECTION_TIME_UNKNOWN",
                "Original collection clock is absent; import time is separate",
                severity="unverified",
            )
        )
    provenance = {
        **provenance,
        "raw_row_count": len(raw),
        "selected_source_positions": positions,
        "selected_source_rows": [int(source_rows[index]) for index in positions],
        "excluded_by_explicit_scope": len(raw) - len(data),
        "scope_definition": "local import range [start,end), not the original API request",
        "original_request_verified": False,
        "quality_validation_scope": "Selected data; out-of-range raw history is preserved, "
        "not price-validated",
        "imported_at": datetime.now(timezone.utc).isoformat(),
        "importer_version": "legacy-market-import-v1",
    }
    validation = {
        "legacy_import": _report([*parse_issues, *policy], provenance=provenance),
        "price": validate_price_data(
            data,
            price_columns=[c for c in (*_OHLC, "Adj Close") if c in data],
            market_timezone=spec.timezone,
        ),
        "missing": validate_missing_intervals(
            data,
            columns=[c for c in (*_OHLC, "Adj Close") if c in data],
            trading_calendar=None,
            market_timezone=spec.timezone,
        ),
        "listing": validate_listing_date(first, None, market_timezone=spec.timezone),
        "adjustment": check_dividend_adjustment(
            data.get("Adj Close", data["Close"]),
            raw_close=data["Close"],
            market_timezone=spec.timezone,
        ),
    }
    quality = []
    for position, row in enumerate(data.loc[:, list(_OHLC)].itertuples(index=False, name=None)):
        if all(pd.notna(value) and np.isfinite(value) for value in row):
            opening, high, low, close = row
            if low > high or low > min(opening, close) or high < max(opening, close):
                quality.append(
                    _issue(
                        "INVALID_OHLC_RANGE",
                        "OHLC contradicts its low/high bounds",
                        position=position,
                    )
                )
    for position, value in enumerate(data["Volume"]):
        if pd.isna(value) or not np.isfinite(value) or value < 0:
            quality.append(
                _issue("INVALID_VOLUME", "Volume must be finite and nonnegative", position=position)
            )
    validation["ohlcv"] = _report(quality)
    for report in validation.values():
        for issue in report["issues"]:
            position = issue.get("position")
            if isinstance(position, int) and 0 <= position < len(positions):
                issue["raw_position"] = positions[position]
                issue["source_row"] = int(source_rows[positions[position]])
    issues = [issue.copy() for report in validation.values() for issue in report["issues"]]
    actual = data.index if not data.empty and not data.index.hasnans else None
    urls = raw["Source URL"].dropna().unique().tolist() if "Source URL" in raw else []
    metadata = {
        **{key: getattr(spec, key) for key in spec.__dataclass_fields__},
        "requested_start": first.date().isoformat(),
        "requested_end_exclusive": last.date().isoformat(),
        "actual_start": actual.min().date().isoformat() if actual is not None else None,
        "actual_end": actual.max().date().isoformat() if actual is not None else None,
        "row_count": len(data),
        "collected_at": provenance.get("recorded_collection_time"),
        "auto_adjust": None,
        "price_basis": "Legacy market OHLC, provider Close and Adj Close kept separately; "
        "auto_adjust execution and independent dividend adjustment unverified",
        "available_at": None,
        "availability_status": "unverified",
        "issues": issues,
        "status": _report(issues)["status"],
        "source_url": urls[0] if len(urls) == 1 else None,
    }
    return CollectionResult(data, raw.copy(deep=True), metadata, validation, raw_text)


def import_legacy_csv(spec, path, *, start="2019-09-01", end, expected_sha256=None):
    """Read UTF-8 OHLCV CSV without sorting, filling or changing the source file."""
    raw, text, provenance = _csv_source(path, expected_sha256)
    return _build(spec, raw, text, provenance, start, end)


def _from_workbook(spec, workbook, start, end):
    raw, provenance = workbook
    chosen = raw.loc[raw["Ticker"] == spec.symbol].copy(deep=True)
    if chosen.empty:
        raise LegacyImportError("Requested ticker is absent from 가격원자료")
    return _build(spec, chosen, None, provenance, start, end)


def import_legacy_workbook(spec, path, *, start="2019-09-01", end, expected_sha256=None):
    """Read only the audited source sheet; never execute or use derived worksheet formulas."""
    return _from_workbook(spec, _workbook_source(path, expected_sha256), start, end)


def _relative_source(root, filename):
    if (
        not isinstance(filename, str)
        or Path(filename).is_absolute()
        or ".." in Path(filename).parts
    ):
        raise LegacyImportError("Legacy CSV filename must stay within csv_root")
    root = Path(root).resolve()
    candidate = (root / filename).resolve()
    if not candidate.is_relative_to(root):
        raise LegacyImportError("Legacy CSV filename escapes csv_root")
    return candidate


def import_team12_sources(config_path, *, workbook_path, csv_root, start="2019-09-01", end):
    """Import exactly the confirmed team12; auxiliary data never enters this mapping."""
    _bounds(start, end)
    config = load_asset_config(config_path)
    if tuple(config["universe"]["selected_etfs"]) != TEAM12_ASSETS:
        raise LegacyImportError("Expected the explicit ordered confirmed team12 universe")
    settings = config.get("legacy_import", {})
    sources = settings.get("sources", {})
    if settings.get("schema_version") != 1 or set(sources) != set(TEAM12_ASSETS):
        raise LegacyImportError("Expected one explicit legacy source for each team12 ETF")
    specs = {spec.dataset_id: spec for spec in config["datasets"]}
    results, workbook = {}, None
    for dataset_id in TEAM12_ASSETS:
        source = sources[dataset_id]
        if not isinstance(source, dict) or not source.get("sha256"):
            raise LegacyImportError("Each legacy source requires its audited SHA-256")
        if source.get("kind") == "workbook":
            if workbook is None:
                workbook = _workbook_source(workbook_path, source["sha256"])
            elif source["sha256"] != workbook[1]["sha256"]:
                raise LegacyImportError("Workbook sources must pin the same source digest")
            result = _from_workbook(specs[dataset_id], workbook, start, end)
        elif source.get("kind") == "csv":
            path = _relative_source(csv_root, source.get("filename"))
            result = import_legacy_csv(
                specs[dataset_id], path, start=start, end=end, expected_sha256=source["sha256"]
            )
        else:
            raise LegacyImportError("Unsupported legacy source kind")
        results[dataset_id] = result
    reference = settings.get("reference_kospi")
    if not isinstance(reference, dict) or not reference.get("sha256"):
        raise LegacyImportError("Expected the explicit KOSPI peer-observation reference")
    reference_path = _relative_source(csv_root, reference.get("filename"))
    peer, _, peer_provenance = _csv_source(reference_path, reference.get("sha256"))
    if not {"Date", "Ticker", "Source"}.issubset(peer):
        raise LegacyImportError("KOSPI peer observations require Date, Ticker and Source")
    if set(peer["Ticker"]) != {"^KS11"} or set(peer["Source"]) != {"Yahoo Finance"}:
        raise LegacyImportError("KOSPI peer ticker/source identity is not verified")
    peer_dates = _dates(peer["Date"], excel=False)
    if peer_dates.hasnans or peer_dates.has_duplicates:
        raise LegacyImportError("Invalid peer observation dates; not an official calendar")
    first, last = _bounds(start, end)
    scope_peer = set(peer_dates[(peer_dates >= first) & (peer_dates < last)])
    for dataset_id in ("KODEX_200TR", "KTB10Y"):
        result = results[dataset_id]
        difference = sorted(scope_peer - set(result.data.index.dropna()))
        all_dates = _dates(result.raw_data["Date"], excel=False)
        issues = [
            _issue(
                "OBSERVED_DATE_DIFFERENCE",
                "Peer KOSPI observations are not an official trading calendar; no rows filled",
                severity="unverified",
                dates=[day.date().isoformat() for day in difference],
                count=len(difference),
                all_history_count=len(set(peer_dates) - set(all_dates.dropna())),
            )
        ]
        result.validation["peer_observations"] = _report(
            issues, reference=peer_provenance, official_calendar=False
        )
        result.metadata["issues"].extend(issue.copy() for issue in issues)
        result.metadata["status"] = _report(result.metadata["issues"])["status"]
    return results
