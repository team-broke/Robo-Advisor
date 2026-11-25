"""Replay a pinned OpenDART corpus into a research-only event-study bundle."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

from robo_advisor.data.benchmark_audit import _csv, _indexed, _timestamps
from robo_advisor.research.dart_body import build_filing, sha256
from robo_advisor.research.dart_corpus import read_pinned_manifest, verify_body_corpus
from robo_advisor.research.event_study import (
    HORIZONS,
    StudyMarket,
    StudyPolicy,
    collect_candidates,
    iter_event_excerpts,
    study_events,
    summarize_events,
)
from robo_advisor.research.ingestion import KST

_KRX_HOSTS = {"data.krx.co.kr", "openapi.krx.co.kr", "global.krx.co.kr", "www.krx.co.kr"}
_REQUIRED_POLICY = "dart-event-study-v1"
_RESULT_FIELDS = (
    "event_id",
    "receipt",
    "event_type",
    "risk_tag_type",
    "asset",
    "split",
    "horizon_sessions",
    "status",
    "market_data_kind",
    "duplicate_of",
    "baseline_date",
    "first_session",
    "last_session",
    "asset_compound_return",
    "market_compound_return",
    "market_adjusted_horizon_return",
    "market_adjusted_last_day_ar",
    "market_adjusted_car",
    "market_model_status",
    "market_model_last_day_ar",
    "market_model_car",
    "alpha_daily",
    "beta",
    "estimation_start",
    "estimation_end",
    "estimation_n",
    "daily_realized_volatility",
    "max_drawdown",
    "overlap_cluster",
    "overlap_cluster_size",
)
_SUMMARY_FIELDS = (
    "event_type",
    "split",
    "horizon_sessions",
    "method",
    "candidate_count",
    "completed_count",
    "isolated_count",
    "status",
    "mean_car",
    "q25",
    "median",
    "q75",
    "isolated_mean_car",
    "ci95_lower",
    "ci95_upper",
)


def _is_official_url(url):
    parts = urlsplit(url)
    return (
        parts.scheme == "https"
        and parts.hostname in _KRX_HOSTS
        and parts.username is None
        and parts.password is None
        and parts.port in (None, 443)
    )


def _official_source(descriptor, *, identity, basis):
    if (
        not _is_official_url(descriptor["source_url"])
        or descriptor.get("identity") != identity
        or descriptor.get("basis") != basis
        or descriptor.get("verification") != "verified"
    ):
        raise ValueError(f"market input requires independently verified KRX source for {identity}")


def _prices(root, descriptor, sessions, identity, basis):
    _official_source(descriptor, identity=identity, basis=basis)
    if descriptor.get("availability_basis") not in {
        "verified_release_timestamp",
        "conservative_verified_bound",
    } or not _is_official_url(descriptor.get("availability_source_url", "")):
        raise ValueError(f"market input requires documented close availability for {identity}")
    frame = _indexed(_csv(root, descriptor), "session_date")
    if set(frame) != {"close", "available_at"} or not frame.index.isin(sessions.index).all():
        raise ValueError("prices require close and available_at for official sessions only")
    frame["available_at"] = _timestamps(frame["available_at"], required=True)
    close = pd.to_numeric(frame["close"], errors="raise")
    if (
        pd.api.types.is_bool_dtype(close)
        or not np.isfinite(close).all()
        or (close <= 0).any()
        or not frame["available_at"].ge(sessions.loc[frame.index, "close_at"]).all()
    ):
        raise ValueError("prices must be positive, finite, and available no earlier than close")
    frame["close"] = close.astype(float)
    return frame


def load_market(path, expected):
    manifest = read_pinned_manifest(path, expected)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "real"
        or manifest.get("training_approved") is not False
        or manifest.get("currency") != "KRW"
        or manifest.get("venue") != "KRX"
    ):
        raise ValueError("expected independently verified KRX research input")
    root = path.expanduser().parent
    calendar = manifest["calendar"]
    _official_source(calendar, identity="KRX", basis="official_session_open_close")
    sessions = _indexed(_csv(root, calendar), "session_date")
    if set(sessions) != {"open_at", "close_at"}:
        raise ValueError("calendar requires official open and close instants")
    for field in ("open_at", "close_at"):
        instants = _timestamps(sessions[field], required=True)
        local = pd.DatetimeIndex(instants).tz_convert(KST).tz_localize(None).normalize()
        if not local.equals(sessions.index):
            raise ValueError("session timestamps must match their Korean session date")
        sessions[field] = instants
    if not sessions["open_at"].lt(sessions["close_at"]).all():
        raise ValueError("every trading session must open before its close")
    benchmark = _prices(root, manifest["benchmark"], sessions, "KOSPI", "price_index")
    if not manifest["issuers"]:
        raise ValueError("price manifest requires at least one issuer price source")
    issuers = {}
    for code, descriptor in manifest["issuers"].items():
        if not isinstance(code, str) or not (len(code) == 6 and code.isascii() and code.isdigit()):
            raise ValueError("issuer series must use a six-digit DART stock code")
        issuers[code] = _prices(root, descriptor, sessions, code, "split_adjusted_price_return")
    return StudyMarket(sessions=sessions, prices=issuers, benchmark=benchmark, synthetic=False)


def scan_corpus(path, expected_hash, as_of):
    root = path.expanduser().parent
    manifest = read_pinned_manifest(path, expected_hash)
    if manifest.get("evidence_scope") != "dart_body":
        raise ValueError("a pinned OpenDART original-body corpus is required")
    filings, _ = verify_body_corpus(root, manifest, require_chunks=False)
    candidates, coverage = [], []
    for source in filings:
        document, filing = source.document, source.provenance
        coverage_row = {
            "receipt": filing["receipt"],
            "document_sha256": filing.get("document_sha256"),
            "source_url": filing["url"],
            "candidates": 0,
        }
        coverage.append(coverage_row)
        available = datetime.fromisoformat(filing["available_at"])
        if available > as_of:
            coverage_row.update(status="future_disclosure")
            continue
        if filing["status"] == "excluded" and filing["reason"] != "no_supported_body_evidence":
            coverage_row.update(status=filing["reason"])
            continue
        _, chunks = build_filing(
            document,
            source.original,
            source.acquisition,
            extract=iter_event_excerpts,
        )
        found, excluded = collect_candidates(
            chunks,
            stock_code=document.metadata.get("stock_code"),
            issuer_name=document.metadata.get("corp_name", ""),
            corpus_sha256=expected_hash,
        )
        coverage_row.update(status="scanned", candidates=len(found), excluded_mentions=excluded)
        candidates.extend(found)
    return tuple(candidates), coverage


def scan_corpora(inputs, as_of):
    """Combine bounded independently pinned archives without double-counting a receipt."""
    if len({anchor for _, anchor in inputs}) != len(inputs):
        raise ValueError("duplicate pinned corpus")
    if len(inputs) == 1:
        return scan_corpus(*inputs[0], as_of)
    candidates, coverage, versions = [], [], defaultdict(set)
    withdrawn_or_corrected = set()
    for path, anchor in inputs:
        found, filings = scan_corpus(path, anchor, as_of)
        candidates.extend(found)
        for filing in filings:
            coverage.append({**filing, "corpus_sha256": anchor})
            if filing["status"] == "future_disclosure":
                continue
            if filing["status"] in {
                "unlinked_correction",
                "later_correction_reported",
                "withdrawal_reported",
            }:
                withdrawn_or_corrected.add(filing["receipt"])
            if filing["document_sha256"] is not None:
                versions[filing["receipt"]].add(filing["document_sha256"])
    conflicts = {receipt for receipt, digests in versions.items() if len(digests) > 1}
    for filing in coverage:
        if filing["status"] == "scanned":
            receipt = filing["receipt"]
            if receipt in withdrawn_or_corrected:
                filing.update(status="version_flagged_in_another_corpus", candidates=0)
            elif receipt in conflicts:
                filing.update(status="conflicting_document_versions", candidates=0)
    unique = {}
    for event in sorted(candidates, key=lambda item: (item.available_at, item.trace_id)):
        if event.receipt not in withdrawn_or_corrected | conflicts:
            unique.setdefault(event.event_id, event)
    return tuple(unique.values()), coverage


def _json(path, data):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def _csv_output(path, fields, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _report(manifest):
    lines = [
        "# 공시 이벤트 연구 및 RiskTag 보정 검토",
        "",
        f"- 평가 마감 시각 `{manifest['as_of']}`",
        f"- 보정 상태 `{manifest['calibration_status']}`",
        f"- 원문 검토 {manifest['scanned_filings']}건, 후보 {manifest['candidate_count']}건, "
        f"수익률 산출 {manifest['completed_horizon_count']}개 구간",
        "- 승인 상태 `training_approved=false`",
        "",
        "## 원문 검토 기록",
        "",
    ]
    for item in manifest["filings"]:
        lines.append(f"- {item['receipt']} · {item['status']} · 후보 {item['candidates']}건")
        for reason, count in item.get("excluded_mentions", {}).items():
            lines.append(f"  - `{reason}` {count}건")
    lines.extend(("", "## 현재 차단 사유", ""))
    lines.extend(f"- `{reason}`" for reason in manifest["limitations"])
    lines.extend(
        (
            "",
            "## 점수 후보와 후속 보정",
            "",
            "Severity는 사전 확인된 이벤트의 경제적 영향 크기를 뜻한다. Train에서만 "
            "원문 변수와 변동성·시장효과를 통제한 사후 손실·CAR의 관계를 탐색한다.",
            "",
            "Confidence는 이벤트 분류·직접 자산 연결의 증거 강도를 뜻한다. 독립 검토자의 "
            "일치도, 원문과 정정 이력, 시각과 자산 매핑을 확인하고 보정 오차를 평가한다.",
            "",
            "사전 확정한 정의를 Validation에서 검토하고 Test는 최종 평가에만 사용한다. "
            "일별 시장 수익률과 연구 결과는 사후 검증용이며 PPO 관측으로 내보내지 않는다.",
            "",
            "현재 점수를 발행하지 않는다. 충분한 독립 표본과 사건 라벨이 필요하다.",
            "",
        )
    )
    return "\n".join(lines)


def run(
    *,
    corpus,
    corpus_sha256,
    output,
    as_of,
    source_kind="unverified",
    market_input=None,
    policy_input=None,
    additional_corpora=(),
):
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("evaluation cutoff requires a timezone")
    if output.exists():
        raise ValueError("output directory already exists")
    if source_kind not in {"synthetic", "unverified", "real_opendart"}:
        raise ValueError("unknown research source kind")
    if market_input is not None and source_kind != "real_opendart":
        raise ValueError("real market outcomes require operator-attested OpenDART originals")
    policy = StudyPolicy()
    if policy_input is not None:
        policy = StudyPolicy.model_validate(read_pinned_manifest(*policy_input))
    if market_input is not None and not policy.splits:
        raise ValueError("price analysis requires a pinned train/validation/test split policy")
    market = load_market(*market_input) if market_input else None
    inputs = ((corpus, corpus_sha256), *additional_corpora)
    candidates, coverage = scan_corpora(inputs, as_of)
    rows = study_events(candidates, market, policy, as_of=as_of)
    summaries = summarize_events(rows, policy.min_ci_events)
    finished = sum(item["status"] == "completed" for item in rows)
    reasons = ["independent_event_labels_and_calibration_unavailable"]
    if not candidates:
        reasons.append("no_candidate_in_supported_body_passages")
    if market is None:
        reasons.append("verified_issuer_and_kospi_prices_and_calendar_unavailable")
    elif not finished:
        reasons.append("no_completed_event_window")
    if not policy.splits:
        reasons.append("project_study_splits_unavailable")
    output.mkdir(parents=True, exist_ok=False)
    candidate_path = output / "event_candidates.jsonl"
    with candidate_path.open("x", encoding="utf-8", newline="\n") as stream:
        for event in candidates:
            stream.write(event.model_dump_json() + "\n")
    _csv_output(output / "event_study_results.csv", _RESULT_FIELDS, rows)
    _csv_output(output / "event_type_summary.csv", _SUMMARY_FIELDS, summaries)
    manifest = {
        "schema_version": 1,
        "method_version": _REQUIRED_POLICY,
        "as_of": as_of.isoformat(),
        "source_kind": source_kind,
        "training_approved": False,
        "calibration_status": "insufficient_evidence",
        "market_data_status": "verified_input" if market else "unavailable",
        "score_status": "not_produced",
        "inputs": {
            "corpus_sha256": corpus_sha256,
            **(
                {"additional_corpus_sha256": [anchor for _, anchor in additional_corpora]}
                if additional_corpora
                else {}
            ),
            "market_manifest_sha256": market_input[1] if market_input else None,
            "policy_sha256": policy_input[1] if policy_input else None,
        },
        "method_policy": policy.model_dump(mode="json"),
        "horizons": HORIZONS,
        "candidate_count": len(candidates),
        "completed_horizon_count": finished,
        "scanned_filings": len(
            {item["receipt"] for item in coverage if item["status"] == "scanned"}
        ),
        "outcome_status_counts": dict(Counter(item["status"] for item in rows)),
        "filings": coverage,
        "limitations": reasons,
    }
    with (output / "calibration_report.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(_report(manifest))
    manifest["outputs_sha256"] = {
        item.name: sha256(item.read_bytes()) for item in sorted(output.iterdir())
    }
    _json(output / "manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, action="append", required=True)
    parser.add_argument("--corpus-sha256", action="append", required=True)
    parser.add_argument("--as-of", type=datetime.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-kind", choices=("unverified", "synthetic", "real_opendart"), default="unverified"
    )
    parser.add_argument("--market", type=Path)
    parser.add_argument("--market-sha256")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--policy-sha256")
    args = parser.parse_args(argv)
    if len(args.corpus) != len(args.corpus_sha256):
        parser.error("supply one --corpus-sha256 per --corpus, in the same order")
    for label in ("market", "policy"):
        if bool(getattr(args, label)) != bool(getattr(args, f"{label}_sha256")):
            parser.error(f"--{label} and --{label}-sha256 must be supplied together")
    try:
        result = run(
            corpus=args.corpus[0],
            corpus_sha256=args.corpus_sha256[0],
            additional_corpora=tuple(zip(args.corpus[1:], args.corpus_sha256[1:])),
            as_of=args.as_of,
            source_kind=args.source_kind,
            output=args.output,
            market_input=(args.market, args.market_sha256) if args.market else None,
            policy_input=(args.policy, args.policy_sha256) if args.policy else None,
        )
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"event study blocked ({type(error).__name__}): {error}\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "candidate_count",
                    "scanned_filings",
                    "completed_horizon_count",
                    "calibration_status",
                )
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
