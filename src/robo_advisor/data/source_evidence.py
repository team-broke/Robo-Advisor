"""Offline follow-up evidence for #9/#11, without repairing or approving sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .audit import _anchored_file, _issue, _path, _report, audit_research_package
from .collectors import CollectionResult, OHLC, _numeric_report, _ohlc_report
from .fx_evidence import _json, audit_fx_evidence
from .snapshot import read_snapshot
from .validation import validate_price_data


def audit_yahoo_fx_origin(result):
    """Compare preserved Yahoo JSON with raw rows and requested-scope OHLCV.

    Collection metadata timezone is not necessarily the provider bar timezone.
    Counts distinguish unique missing dates from individual missing cells. A
    matching chain does not explain the provider's upstream error or releases.
    """
    try:
        if (
            not isinstance(result, CollectionResult)
            or result.metadata.get("symbol") != "KRW=X"
            or result.metadata.get("source") != "yahoo"
        ):
            raise ValueError("Expected the preserved Yahoo KRW=X CollectionResult")
        if not isinstance(result.raw_text, str) or not isinstance(result.raw_data, pd.DataFrame):
            unavailable = [
                _issue(
                    "YAHOO_ORIGINAL_UNAVAILABLE",
                    "Original JSON/raw frame was not preserved",
                    severity="unverified",
                )
            ]
            if result.status == "failed":
                unavailable.append(
                    _issue("PARENT_SOURCE_FAILED", "Preserve the declared failed parent source")
                )
            return _report(
                unavailable, source_status=result.status, original_chain_status="unverified"
            )
        payload = json.loads(result.raw_text)
        if not isinstance(payload, dict) or not isinstance(payload.get("chart"), dict):
            raise ValueError("Original Yahoo chart must be a JSON object")
        chart = payload["chart"]
        if (
            chart.get("error") is not None
            or not isinstance(chart["result"], list)
            or len(chart["result"]) != 1
        ):
            raise ValueError("Yahoo chart requires one successful original result")
        response = chart["result"][0]
        if not isinstance(response, dict) or not isinstance(response.get("meta"), dict):
            raise ValueError("Original Yahoo result/meta must be objects")
        meta, ticks = response["meta"], response["timestamp"]
        if (
            meta.get("symbol") != "KRW=X"
            or meta.get("currency") != "KRW"
            or meta.get("longName") != "USD/KRW"
        ):
            raise ValueError("Original quote identity/currency differs from USD/KRW")
        if not isinstance(ticks, list) or not ticks or any(type(v) is not int for v in ticks):
            raise ValueError("Original timestamps must be integer Unix seconds")
        quotes = response["indicators"]["quote"]
        if not isinstance(quotes, list) or len(quotes) != 1 or not isinstance(quotes[0], dict):
            raise ValueError("Original Yahoo quote arrays are ambiguous")
        fields = (*OHLC, "Volume")
        values = {}
        for name in fields:
            vector = quotes[0][name.lower()]
            if not isinstance(vector, list) or len(vector) != len(ticks):
                raise ValueError("Original Yahoo timestamp/quote lengths disagree")
            if any(
                v is not None
                and (isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v))
                for v in vector
            ):
                raise ValueError("Original quotes require numeric values or nulls")
            values[name] = vector
        dates = (
            pd.to_datetime(ticks, unit="s", utc=True)
            .tz_convert(meta["exchangeTimezoneName"])
            .tz_localize(None)
            .normalize()
        )
        if not dates.is_unique or not dates.is_monotonic_increasing:
            raise ValueError("Original Yahoo daily observations are unordered or duplicated")
        original = pd.DataFrame(values, index=dates, dtype=float)
        raw = result.raw_data
        if (
            len(raw) != len(original)
            or not np.array_equal(raw["ProviderTimestampUnix"], ticks)
            or not pd.DatetimeIndex(raw["ProviderBarDate"]).equals(dates)
        ):
            raise ValueError("Raw frame does not preserve original provider timestamps/dates")
        if not np.allclose(
            raw.loc[:, list(fields)].to_numpy(dtype=float),
            original.to_numpy(),
            rtol=0,
            atol=0,
            equal_nan=True,
        ):
            raise ValueError("Raw OHLCV differs from the original Yahoo response")
        start, end = (
            pd.Timestamp(result.metadata[k]) for k in ("requested_start", "requested_end_exclusive")
        )
        expected = original.loc[(dates >= start) & (dates < end)]
        data = result.data
        if not data.index.equals(expected.index) or not np.allclose(
            data.loc[:, list(fields)].to_numpy(dtype=float),
            expected.to_numpy(),
            rtol=0,
            atol=0,
            equal_nan=True,
        ):
            raise ValueError("Requested-scope data differs from preserved original quotes")
        price = validate_price_data(data, price_columns=OHLC)
        defects = []
        for issue in _ohlc_report(data)["issues"]:
            position = issue["position"]
            defects.append(
                {
                    **issue,
                    "date": data.index[position].date().isoformat(),
                    "values": {k: float(data.iloc[position][k]) for k in fields},
                }
            )
        missing = data.loc[:, list(OHLC)].isna().any(axis=1)
        issues = [
            _issue(
                "UPSTREAM_CAUSE_UNVERIFIED",
                (
                    "Matching preserved quotes locate the defect before normalization, "
                    "not its upstream cause"
                ),
                severity="unverified",
            ),
            _issue(
                "ACTUAL_RELEASE_UNVERIFIED",
                "Collection time and provider bar labels do not prove historical release",
                severity="unverified",
            ),
        ]
        issues.extend(price["issues"])
        issues.extend(_numeric_report(data, "Volume", allow_negative=False)["issues"])
        if defects or missing.any() or result.status == "failed":
            issues.append(_issue("PARENT_SOURCE_FAILED", "Preserved parent defects remain failed"))
        return _report(
            issues,
            original_chain_status="passed",
            source_status=result.status,
            provider_timezone=meta["exchangeTimezoneName"],
            label_timezone=result.metadata.get("timezone"),
            raw_rows=len(raw),
            data_rows=len(data),
            ohlc_defect_rows=len(defects),
            ohlc_defects=defects,
            missing_dates=[d.date().isoformat() for d in data.index[missing]],
            missing_price_cells=sum(i["code"] == "MISSING_PRICE" for i in price["issues"]),
            missing_volume_cells=int(data["Volume"].isna().sum()),
            raw_missing_rows=int(original.loc[:, list(OHLC)].isna().any(axis=1).sum()),
            original_currency=meta["currency"],
            original_quote_name=meta["longName"],
        )
    except (ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
        return _report([_issue("YAHOO_EVIDENCE_INVALID", str(exc))])


def audit_ktb_original(root, descriptor, candidate):
    """Compare the frozen CP949 KRX candidate CSV; never manufacture a receipt."""
    try:
        path = _anchored_file(Path(root), descriptor)
        raw = pd.read_csv(path, encoding="cp949", dtype=str)
        mapping = {
            "Open": "시가",
            "High": "고가",
            "Low": "저가",
            "Close": "종가",
            "Volume": "거래량",
        }
        dates = pd.DatetimeIndex(pd.to_datetime(raw["일자"], format="%Y/%m/%d"))
        if len(dates) != 1 or dates[0] != pd.Timestamp("2020-03-10"):
            raise ValueError("Expected the preserved single-date KTB correction candidate")
        original = pd.DataFrame(
            {
                name: raw[column].str.replace(",", "", regex=False).astype(float).to_numpy()
                for name, column in mapping.items()
            },
            index=dates,
        )
        if not isinstance(getattr(candidate, "datasets", None), dict):
            raise ValueError("Expected a verified candidate snapshot")
        result = candidate.datasets["KTB10Y_KRX_CORRECTION"]
        if (
            result.metadata.get("symbol") != "148070"
            or result.metadata.get("currency") != "KRW"
            or not result.data.index.equals(dates)
        ):
            raise ValueError("KTB candidate identity/currency/dates disagree")
        if not np.allclose(
            original, result.data.loc[:, list(mapping)], rtol=0, atol=0, equal_nan=True
        ):
            raise ValueError("KRX candidate snapshot differs from the frozen original CSV")
        if (
            validate_price_data(original, price_columns=OHLC)["status"] != "passed"
            or _ohlc_report(original)["status"] != "passed"
            or _numeric_report(original, "Volume", allow_negative=False)["status"] != "passed"
        ):
            raise ValueError("KRX original candidate contains numeric/OHLC errors")
        return _report(
            [
                _issue(
                    "CORRECTION_RECEIPT_UNVERIFIED",
                    (
                        "No independently anchored receipt was supplied for this correction; "
                        "do not infer it from other dates"
                    ),
                    severity="unverified",
                ),
                _issue(
                    "SAME_BASIS_ADOPTION_UNVERIFIED",
                    (
                        "Matching quoted values do not establish corporate-action basis "
                        "or historical availability"
                    ),
                    severity="unverified",
                ),
            ],
            original_values_status="passed",
            csv_sha256=descriptor["sha256"],
            low=float(original.iloc[0]["Low"]),
            collection_receipt_status="not_supplied",
            training_adoption="blocked",
        )
    except (ValueError, TypeError, KeyError, OSError, OverflowError) as exc:
        return _report([_issue("KRX_EVIDENCE_INVALID", str(exc))])


def audit_source_evidence(spec, *, root, base_audit):
    """Reuse #106 audit, then add bounded original-response and source comparisons."""
    reports = {}
    try:
        if (
            not isinstance(spec, dict)
            or type(spec.get("schema_version")) is not int
            or spec["schema_version"] != 1
        ):
            raise ValueError("Unsupported source evidence specification")
        if not isinstance(base_audit, dict) or any(
            not isinstance(base_audit.get(key), dict)
            for key in ("snapshots", "calendars", "scenarios")
        ):
            raise ValueError("Expected the existing base audit configuration")
        root = Path(root)
        reports["pipeline_audit"] = audit_research_package(base_audit, root=root)
        snapshots = {}
        for name, descriptor in {
            **base_audit["snapshots"],
            "candidate": spec["candidate_snapshot"],
        }.items():
            manifest = _anchored_file(
                root, {**descriptor, "path": descriptor["path"] + "/manifest.json"}
            )
            snapshots[name] = read_snapshot(
                manifest.parent, expected_manifest_sha256=descriptor["sha256"]
            )
        calendars = {
            market: pd.DatetimeIndex(
                pd.to_datetime(pd.read_csv(_anchored_file(root, entry))["session_date"])
            )
            for market, entry in base_audit["calendars"].items()
        }
        from .price_evidence import audit_price_evidence

        reports["prices"] = audit_price_evidence(
            snapshots["parent"], snapshots["selected"], snapshots["candidate"], calendars=calendars
        )
        reports["yahoo_fx_original"] = audit_yahoo_fx_origin(
            snapshots["parent"].datasets["USD_KRW"]
        )
        reports["ktb_original"] = audit_ktb_original(
            root, spec["ktb_original"], snapshots["candidate"]
        )
        if not isinstance(spec.get("fx_indexes"), dict) or set(spec["fx_indexes"]) != {
            "smbs",
            "alfred",
        }:
            raise ValueError("Both independent FX evidence indexes are required")
        for source in ("smbs", "alfred"):
            reports[source] = audit_fx_evidence(
                root,
                source=source,
                csv_descriptor=base_audit["scenarios"][source]["fx"],
                index_descriptor=spec["fx_indexes"][source],
            )
        issues = [
            _issue(
                "SOURCE_COMPONENT",
                "Source evidence requires attention",
                severity="error" if value["status"] == "failed" else "unverified",
                component=name,
            )
            for name, value in reports.items()
            if value["status"] != "passed"
        ]
        return _report(
            issues,
            components=reports,
            evidence_version="team12-source-evidence-v1",
            source_manifests={
                name: snapshot.manifest_sha256 for name, snapshot in snapshots.items()
            },
            ppo_status="blocked",
            snapshot_mutation=False,
            remaining_gates=[
                "actual_historical_release",
                "corporate_actions_and_price_basis",
                "official_full_market_calendars",
                "issue94_adapter_and_execution_policy",
            ],
        )
    except (ValueError, TypeError, KeyError, OSError, ImportError) as exc:
        return _report([_issue("SOURCE_EVIDENCE_INPUT_INVALID", str(exc))], components=reports)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        spec = _json(args.config)
        base = _json(_path(args.config.resolve().parent, spec["base_audit_config"]))
        result = audit_source_evidence(spec, root=args.root, base_audit=base)
        rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output is not None:
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(rendered + "\n")
        print(rendered)
        return {"passed": 0, "failed": 1, "unverified": 2}[result["status"]]
    except (ValueError, TypeError, KeyError, OSError) as exc:
        print(json.dumps(_report([_issue("SOURCE_EVIDENCE_INPUT_INVALID", str(exc))])))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
