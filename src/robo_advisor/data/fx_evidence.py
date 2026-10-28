"""Read-only preservation and observed-value checks of archived FX evidence.

Receipts prove what was collected, not when it was historically published. A
matching SHA or vintage date never creates actual_available_at or approval.
Only the archived SMBS table template is decoded, without executing JavaScript.
"""

from __future__ import annotations

from collections import Counter
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import parse_qs, unquote, urlsplit

import numpy as np
import pandas as pd

from .audit import _anchored_file, _issue, _report, _sha


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate evidence JSON key")
        result[key] = value
    return result


def _json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique)
    if not isinstance(value, dict):
        raise ValueError("Evidence JSON must be an object")
    return value


def _index(root, descriptor):
    index_path = _anchored_file(root, descriptor)
    index = _json(index_path)
    if index.get("training_approved") is not False or not isinstance(index.get("files"), list):
        raise ValueError("Evidence index must withhold approval and list files")
    files, names, receipts, unavailable = {}, set(), {}, []
    for entry in index["files"]:
        path = _anchored_file(index_path.parent, entry)
        relative = entry["path"]
        if relative.casefold() in names:
            raise ValueError("Duplicate evidence index path")
        names.add(relative.casefold())
        count = entry.get("size_bytes", entry.get("bytes"))
        if type(count) is not int or count < 0 or path.stat().st_size != count:
            raise ValueError("Evidence byte count differs from its index")
        files[relative] = path
    for relative, path in files.items():
        if not relative.endswith(".receipt.json"):
            continue
        raw_name = relative.removesuffix(".receipt.json")
        if raw_name not in files:
            unavailable.append(relative)
            continue
        receipt = _json(path)
        raw = files[raw_name]
        if _sha(raw) != receipt.get("sha256"):
            raise ValueError("Receipt SHA differs from its raw response")
        count = receipt.get("size_bytes", receipt.get("bytes"))
        if type(count) is not int or raw.stat().st_size != count:
            raise ValueError("Receipt byte count differs from its raw response")
        receipts[raw_name] = receipt
    return files, receipts, _sha(index_path), unavailable


class _Cells(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.row, self.cell = [], None, None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        elif tag in {"td", "th"} and self.row is not None:
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.row is not None and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def _smbs_table(path):
    text = Path(path).read_bytes().decode("cp949")

    def literal(match):
        # These fixed marker encodings occur in the archived table's d1..d4
        # literal renderers. Never evaluate scripts or follow embedded links.
        decoded = re.sub(r"%u?_[A-Z]([0-9a-fA-F]+)", lambda value: chr(int(value[1], 16)), match[1])
        return unquote(decoded)

    text = re.sub(
        r"<script[^>]*>\s*d\d*\(\s*'([^']*)'\s*\)\s*;?\s*</script>",
        literal,
        text,
        flags=re.I,
    )
    parser = _Cells()
    parser.feed(text)
    values = {}
    for row in parser.rows:
        if not row or not re.fullmatch(r"\d{4}\.\d{2}\.\d{2}", row[0]):
            continue
        if len(row) < 3 or "USD" not in row[1]:
            raise ValueError("Unsupported SMBS period-table currency/schema")
        date = pd.Timestamp(row[0])
        if date in values:
            raise ValueError("Duplicate SMBS observation in raw table")
        values[date] = float(row[2].replace(",", ""))
    if not values:
        raise ValueError("No recognized SMBS MAR observations in archived HTML")
    return pd.Series(values, dtype=float)


def audit_fx_evidence(root, *, source, csv_descriptor, index_descriptor):
    """Verify indexed files/receipts and each archived quote; preserve unknown releases.

    The CSV descriptor and evidence index require independent SHA anchors. ALFRED
    per-row source_file/source_sha256 must identify the requested date-only
    vintage. SMBS evidence_sha must identify a collected USD MAR table. This
    validates the frozen response chain, not HTTPS authenticity or historical
    availability. Unknown/missing proof is never synthesized.
    """
    issues = []
    try:
        root = Path(root)
        if source not in {"smbs", "alfred"}:
            raise ValueError("Unsupported FX evidence scenario")
        csv_path = _anchored_file(root, csv_descriptor)
        files, receipts, index_digest, unavailable = _index(root, index_descriptor)
        raw = pd.read_csv(csv_path)
        if raw.empty:
            raise ValueError("Observed FX CSV is empty")
        dates_name, rate_name = (
            ("date", "mar_krw_per_usd") if source == "smbs" else ("observation_date", "DEXKOUS")
        )
        dates = pd.DatetimeIndex(pd.to_datetime(raw[dates_name]))
        if (
            dates.hasnans
            or dates.tz is not None
            or not dates.is_unique
            or not dates.is_monotonic_increasing
            or not dates.equals(dates.normalize())
        ):
            raise ValueError("FX observations require ordered unique naive daily dates")
        rates = raw[rate_name].to_numpy(dtype=float)
        if np.isinf(rates).any() or (rates[np.isfinite(rates)] <= 0).any():
            raise ValueError("FX observations must be finite positive values or explicit nulls")
        used_files, matched, http_sources = set(), 0, 0
        groups = raw.groupby(
            "evidence_sha" if source == "smbs" else "source_file", dropna=False, sort=False
        )
        for identity, group in groups:
            if not isinstance(identity, str):
                raise ValueError("Every FX observation needs an original-response pointer")
            if source == "smbs":
                matches = [
                    name
                    for name, path in files.items()
                    if path.suffix == ".html" and _sha(path) == identity
                ]
                if len(matches) != 1:
                    raise ValueError(
                        "SMBS evidence_sha does not uniquely identify an indexed HTML response"
                    )
                name = matches[0]
                expected = _smbs_table(files[name])
                url = urlsplit(receipts.get(name, {}).get("url", ""))
                if (
                    url.hostname not in {"www.smbs.biz", "smbs.biz"}
                    or url.path != "/ExRate/StdExRate.jsp"
                    or parse_qs(url.query).get("tongwha_code") != ["USD"]
                ):
                    raise ValueError("SMBS receipt does not identify the USD period-table request")
            else:
                name = identity
                if name not in files:
                    raise ValueError(
                        "ALFRED source_file is absent from the anchored evidence index"
                    )
                digest = _sha(files[name])
                if not group["source_sha256"].eq(digest).all():
                    raise ValueError("ALFRED per-row source SHA mismatch")
                vintage = pd.DatetimeIndex(pd.to_datetime(group["vintage_date"]))
                if (
                    vintage.hasnans
                    or vintage.tz is not None
                    or not vintage.equals(vintage.normalize())
                    or vintage.nunique() != 1
                ):
                    raise ValueError("ALFRED group requires one explicit date-only vintage")
                day = vintage[0]
                group_dates = pd.DatetimeIndex(pd.to_datetime(group[dates_name]))
                if (group_dates > day).any():
                    raise ValueError("ALFRED observation occurs after its requested vintage")
                original = pd.read_csv(files[name], na_values=["."])
                column = "DEXKOUS_" + day.strftime("%Y%m%d")
                if set(original) != {"observation_date", column}:
                    raise ValueError("ALFRED original column does not match its requested vintage")
                original_dates = pd.DatetimeIndex(pd.to_datetime(original["observation_date"]))
                if original_dates.hasnans or not original_dates.is_unique:
                    raise ValueError("ALFRED raw observations require unique valid dates")
                expected = pd.Series(original[column].to_numpy(dtype=float), index=original_dates)
                url = urlsplit(receipts.get(name, {}).get("url", ""))
                query = parse_qs(url.query)
                if (
                    url.hostname != "alfred.stlouisfed.org"
                    or url.path != "/graph/alfredgraph.csv"
                    or query.get("id") != ["DEXKOUS"]
                    or query.get("vintage_date") != [day.date().isoformat()]
                ):
                    raise ValueError("ALFRED receipt request disagrees with its series/vintage")
            receipt = receipts.get(name)
            if receipt is None or receipt.get("http_status") != 200:
                raise ValueError("Selected FX response requires a successful collection receipt")
            http_sources += int(url.scheme == "http")
            wanted = pd.DatetimeIndex(pd.to_datetime(group[dates_name]))
            if not wanted.isin(expected.index).all():
                raise ValueError("FX CSV contains an observation absent from its original response")
            if not np.allclose(
                group[rate_name].to_numpy(dtype=float),
                expected.reindex(wanted).to_numpy(dtype=float),
                rtol=0,
                atol=0,
                equal_nan=True,
            ):
                raise ValueError("Observed FX value differs from its original response")
            matched += len(group)
            used_files.add(name)
        missing_actual = int(raw["actual_available_at"].isna().sum())
        if unavailable:
            issues.append(
                _issue(
                    "OPTIONAL_RESPONSE_UNAVAILABLE",
                    "Some indexed receipts have no preserved raw response",
                    severity="unverified",
                    receipts=unavailable,
                )
            )
        failed_requests = sum(r.get("http_status") != 200 for r in receipts.values())
        if failed_requests:
            issues.append(
                _issue(
                    "ARCHIVED_REQUEST_FAILED",
                    (
                        "Some archived document requests were unsuccessful; "
                        "no source claim is inferred"
                    ),
                    severity="unverified",
                    count=failed_requests,
                )
            )
        issues.append(
            _issue(
                "HISTORICAL_RELEASE_UNVERIFIED",
                "Receipts/vintage dates do not prove individual historical release timestamps",
                severity="unverified",
            )
        )
        if http_sources:
            issues.append(
                _issue(
                    "HTTP_TRANSPORT_UNAUTHENTICATED",
                    "Archived HTTP response hashes do not authenticate transport",
                    severity="unverified",
                )
            )
        return _report(
            issues,
            source=source,
            csv_sha256=csv_descriptor["sha256"],
            index_integrity_status="passed",
            preservation_status="partial" if unavailable else "passed",
            unavailable_responses=unavailable,
            original_values_status="passed",
            index_sha256=index_digest,
            indexed_files=len(files),
            receipt_count=len(receipts),
            response_count=len(used_files),
            matched_observations=matched,
            null_observations=int(np.isnan(rates).sum()),
            actual_release_missing=missing_actual,
            http_response_count=http_sources,
            receipt_http_status_counts=dict(
                Counter(str(r.get("http_status")) for r in receipts.values())
            ),
            release_interpretation=(
                "research assumptions remain separate; "
                "date-only vintage is not a publication timestamp"
            ),
        )
    except (ValueError, TypeError, KeyError, OSError, OverflowError) as exc:
        return _report([_issue("FX_EVIDENCE_INVALID", str(exc))])
