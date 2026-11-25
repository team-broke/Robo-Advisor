import io
import json
from datetime import timedelta
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from robo_advisor.research.dart_body import Acquisition, build_filing, filing_url, sha256
from robo_advisor.research.documents import RawDocument, SourceKind
from robo_advisor.research.event_study import (
    StudyMarket,
    StudyPolicy,
    StudySplit,
    collect_candidates,
    iter_event_excerpts,
    study_events,
    summarize_events,
)
from robo_advisor.research.ingestion import KST, normalize
from robo_advisor.schemas.risk import RiskEventType
from scripts.prepare_dart_body import build_manifest, write_json
from scripts.run_dart_event_study import load_market, run, scan_corpora

_DAYS = pd.bdate_range("2024-03-04", periods=106)
_ALPHA, _BETA = 0.001, 1.5


def _splits(boundary=80):
    return StudyPolicy(
        splits=(
            StudySplit(name="train", start=_DAYS[0].date(), end_exclusive=_DAYS[boundary].date()),
            StudySplit(
                name="validation", start=_DAYS[boundary].date(), end_exclusive=_DAYS[95].date()
            ),
            StudySplit(
                name="test",
                start=_DAYS[95].date(),
                end_exclusive=_DAYS[-1].date() + timedelta(days=1),
            ),
        ),
        estimation_sessions=25,
        pre_event_gap_sessions=4,
    )


def _market(shocks=None):
    opened = _DAYS.tz_localize(KST) + pd.Timedelta(hours=9)
    sessions = pd.DataFrame(
        {"open_at": opened, "close_at": opened + pd.Timedelta(hours=6, minutes=30)},
        index=_DAYS,
    )
    reference = (np.arange(len(_DAYS)) % 5 - 2) * 0.003
    firm = _ALPHA + _BETA * reference
    for index, shock in (shocks or {}).items():
        firm[index] += shock
    prices = [
        pd.DataFrame(
            {"close": 100 * np.cumprod(1 + series), "available_at": sessions["close_at"]},
            index=_DAYS,
        )
        for series in (firm, reference)
    ]
    return StudyMarket(sessions, {"005930": prices[0]}, prices[1], synthetic=True)


def _original(when, statements, *, n=1, stock="005930", section="주요사항", remarks=""):
    issued = pd.Timestamp(when).tz_convert(KST).normalize() - pd.Timedelta(days=1)
    receipt = issued.strftime("%Y%m%d") + f"{n:06d}"
    document = normalize(
        RawDocument(
            source=SourceKind.DART,
            source_name="dart",
            source_id=receipt,
            title="보고서",
            summary="목록",
            url=filing_url(receipt),
            published_precision="date",
            published_at=issued.to_pydatetime(),
            fetched_at=when,
            metadata={"rm": remarks, "corp_name": "예시회사", "stock_code": stock},
        )
    )
    rows = "<P>(단위 : 백만원)</P><TABLE><THEAD><TR><TH>항목</TH><TH>당기</TH></TR></THEAD>"
    rows += "<TBODY><TR><TD>매출액</TD><TD>100</TD></TR></TBODY></TABLE>"
    paragraphs = "".join(f"<P>{line}</P>" for line in statements)
    xml = (
        f"<DOCUMENT><BODY><SECTION-1><TITLE>{section}</TITLE>{paragraphs}</SECTION-1>"
        f"<SECTION-1><TITLE>연결 손익계산서</TITLE>{rows}</SECTION-1></BODY></DOCUMENT>"
    ).encode()
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr(f"{receipt}.xml", xml)
    payload = buffer.getvalue()
    acquired = Acquisition(receipt=receipt, received_at=when, sha256=sha256(payload))
    return document, payload, acquired


def _collect(document, original, acquisition):
    _, excerpts = build_filing(document, original, acquisition, extract=iter_event_excerpts)
    return collect_candidates(
        excerpts,
        stock_code=document.metadata["stock_code"],
        issuer_name="예시회사",
        corpus_sha256="a" * 64,
    )


def _time(position, *, hour=8):
    return (_DAYS[position].tz_localize(KST) + pd.Timedelta(hours=hour)).to_pydatetime()


def _event(position=57, text="당사는 주요 공급 계약을 해지하였습니다.", **kwargs):
    candidates, excluded = _collect(*_original(_time(position), [text], **kwargs))
    assert not excluded
    return candidates[0]


def test_extracts_supported_and_unmapped_types_without_scores():
    source = _original(
        _time(57),
        [
            "당사는 신규 법령 개정이 확정되었음을 공시합니다.",
            "당사의 신규 공급 계약을 해지하였습니다.",
            "당사의 신규 공급 계약을 해지하였습니다.",
            "예시회사는 영업이익이 감소했습니다.",
            "향후 규제 강화 가능성이 있습니다.",
        ],
    )
    candidates, excluded = _collect(*source)
    kinds = {item.event_type: item for item in candidates}
    assert not excluded
    assert set(kinds) == {"regulation_change", "contract_termination", "earnings_deterioration"}
    assert kinds["regulation_change"].risk_tag_type == RiskEventType.REGULATION_CHANGE
    assert kinds["contract_termination"].risk_tag_type is None
    assert kinds["earnings_deterioration"].risk_tag_type is None
    assert len(kinds["contract_termination"].evidence) == 1
    candidate = kinds["contract_termination"]
    assert candidate.asset == "005930" and candidate.available_at == _time(57)
    assert candidate.evidence[0].body.quote == "당사의 신규 공급 계약을 해지하였습니다."
    assert candidate.trace_id == _collect(*source)[0][0].trace_id
    assert {"severity", "confidence", "risk_score"}.isdisjoint(candidate.model_dump())


@pytest.mark.parametrize(
    ("text", "section", "reason"),
    [
        (
            "2023년 3월 21일 계열사 대표이사가 사임했습니다.",
            "VII. 주주에 관한 사항",
            "related_entity_not_linked_to_issuer",
        ),
        (
            "2023년 3월 21일 예시회사의 대표이사가 사임했습니다.",
            "경영진",
            "historical_reference_needs_first_disclosure",
        ),
    ],
)
def test_context_does_not_attribute_old_or_related_entities_to_issuer(text, section, reason):
    events, excluded = _collect(*_original(_time(57), [text], section=section))
    assert not events
    assert excluded == {reason: 1}


def test_both_adjustments_compounding_drawdown_and_volatility():
    first = 57
    market = _market({first: -0.10, first + 1: 0.04, first + 2: 0.02})
    rows = study_events((_event(first),), market, _splits(), as_of=_time(100))
    assert [row["horizon_sessions"] for row in rows] == [1, 3, 5, 10]
    one, three, _, _ = rows
    assert all(
        row["status"] == "completed" and row["market_data_kind"] == "synthetic" for row in rows
    )
    assert one["baseline_date"] == _DAYS[first - 1].date()
    assert one["first_session"] == _DAYS[first].date()
    assert one["daily_realized_volatility"] is None
    assert one["max_drawdown"] == pytest.approx(0.099)
    assert three["alpha_daily"] == pytest.approx(_ALPHA)
    assert three["beta"] == pytest.approx(_BETA)
    assert three["market_model_car"] == pytest.approx(-0.04)
    assert three["market_model_last_day_ar"] == pytest.approx(0.02)
    assert three["estimation_end"] == _DAYS[first - 5].date()
    base = slice(first - 1, first + 3)
    x = market.prices["005930"]["close"].iloc[base].to_numpy()
    y = market.benchmark["close"].iloc[base].to_numpy()
    daily, reference = x[1:] / x[:-1] - 1, y[1:] / y[:-1] - 1
    assert three["market_adjusted_car"] == pytest.approx((daily - reference).sum())
    assert three["asset_compound_return"] == pytest.approx(x[-1] / x[0] - 1)
    assert three["market_adjusted_horizon_return"] == pytest.approx(x[-1] / x[0] - y[-1] / y[0])
    assert three["daily_realized_volatility"] == pytest.approx(daily.std(ddof=1))


@pytest.mark.leakage
def test_event_horizon_cannot_cross_validation_boundary():
    rows = study_events((_event(57),), _market(), _splits(60), as_of=_time(100))
    assert [row["status"] for row in rows] == [
        "completed",
        "completed",
        "cross_split_horizon",
        "cross_split_horizon",
    ]
    assert all("asset_compound_return" not in item for item in rows[2:])


@pytest.mark.leakage
def test_future_disclosure_future_prices_and_late_estimation_are_excluded():
    first = 57
    candidate = _event(first)
    market = _market()
    policy = _splits()
    prior = study_events((candidate,), market, policy, as_of=_time(first, hour=7))
    assert {row["status"] for row in prior} == {"future_disclosure"}
    ongoing = study_events((candidate,), market, policy, as_of=_time(first, hour=14))
    assert {row["status"] for row in ongoing} == {"followup_not_observed"}

    firm = market.prices["005930"]
    firm.loc[_DAYS[first - 5], "available_at"] = candidate.available_at
    rows = study_events((candidate,), market, policy, as_of=_time(100))
    assert rows[0]["status"] == "completed"
    assert rows[0]["market_model_status"] == "estimation_price_not_available_at_cutoff"
    assert "market_model_car" not in rows[0]

    firm.loc[_DAYS[first - 1], "available_at"] = _time(first, hour=10)
    late_base = study_events((candidate,), market, policy, as_of=_time(100))
    assert {row["status"] for row in late_base} == {"baseline_price_not_available_at_cutoff"}


@pytest.mark.leakage
def test_after_open_uses_next_session_and_missing_session_is_not_bridged():
    candidate = _collect(*_original(_time(57, hour=11), ["이 공급 계약을 해지하였습니다."]))[0][0]
    market = _market()
    rows = study_events((candidate,), market, _splits(), as_of=_time(100))
    assert rows[0]["baseline_date"] == _DAYS[57].date()
    assert rows[0]["first_session"] == _DAYS[58].date()
    market.prices["005930"].drop(_DAYS[59], inplace=True)
    missing = study_events((candidate,), market, _splits(), as_of=_time(100))
    assert [row["status"] for row in missing[:2]] == ["completed", "missing_price_observation"]


@pytest.mark.leakage
def test_correction_with_unlinked_original_cannot_become_candidate():
    dates = _time(56)
    original = _original(dates, ["이 공급 계약을 해지하였습니다."], remarks="유정")
    assert _collect(*original) == ((), {})


def test_exact_duplicate_excluded_and_overlapping_windows_not_used_for_ci():
    items = (
        _event(57, n=3),
        _event(58, n=4),
        _event(58, text="당사는 유통 계약을 해지했습니다.", n=5),
    )
    rows = study_events(items, _market(), _splits(), as_of=_time(100))
    duplicate = [item for item in rows if item["status"] == "duplicate_disclosure"]
    assert len(duplicate) == 4
    assert duplicate[0]["duplicate_of"] == items[0].event_id
    complete = [item for item in rows if item["status"] == "completed"]
    assert {item["overlap_cluster_size"] for item in complete} == {2}
    summaries = summarize_events(rows, 3)
    assert all(item["isolated_count"] == 0 for item in summaries)
    assert all(item["status"] == "insufficient_evidence" for item in summaries)


def test_ci_centers_on_independent_sample_mean_only():
    rows = [
        dict(
            event_type="regulation_change",
            split="train",
            horizon_sessions=1,
            overlap_cluster_size=1,
            market_adjusted_car=value,
        )
        for value in (0.01, 0.03, 0.05)
    ]
    rows.append({**rows[0], "market_adjusted_car": 2.0, "overlap_cluster_size": 2})
    adjusted, model = summarize_events(rows, 3)
    assert adjusted["completed_count"] == 4
    assert adjusted["isolated_mean_car"] == pytest.approx(0.03)
    assert adjusted["ci95_lower"] < 0.03 < adjusted["ci95_upper"]
    assert model["status"] == "insufficient_evidence"


def _bundle(tmp_path, *, source=None, listing_hash="b" * 64):
    source = source or _original(_time(57), ["당사는 납품 계약을 해지하였습니다."])
    document, archive, acquired = source
    root = tmp_path / "corpus"
    slot = root / "raw" / acquired.receipt
    slot.mkdir(parents=True)
    (slot / "original.zip").write_bytes(archive)
    write_json(slot / "acquisition.json", acquired.model_dump(mode="json"))
    info = build_manifest(root, (document,), listing_hash)
    return root / "research_manifest.json", info["manifest_sha256"]


def test_offline_bundle_hashes_reproducible_without_market_data(tmp_path):
    corpus, anchor = _bundle(tmp_path)
    first = tmp_path / "first"
    args = dict(corpus=corpus, corpus_sha256=anchor, as_of=_time(90), source_kind="synthetic")
    result = run(output=first, **args)
    second = run(output=tmp_path / "second", **args)
    assert result == second
    assert result["candidate_count"] == 1 and result["completed_horizon_count"] == 0
    assert result["source_kind"] == "synthetic"
    assert result["training_approved"] is False
    assert result["calibration_status"] == "insufficient_evidence"
    assert result["outcome_status_counts"] == {"market_data_unavailable": 4}
    for filename, digest in result["outputs_sha256"].items():
        assert sha256((first / filename).read_bytes()) == digest
    assert len(list(first.iterdir())) == 5
    with pytest.raises(ValueError, match="already exists"):
        run(output=first, **args)
    with pytest.raises(ValueError, match="operator-attested"):
        run(output=tmp_path / "mixed", market_input=(first / "manifest.json", anchor), **args)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        run(output=tmp_path / "bad", **{**args, "corpus_sha256": "3" * 64})


@pytest.mark.leakage
def test_multi_corpus_deduplicates_and_does_not_apply_future_correction(tmp_path):
    text = "당사는 납품 계약을 해지하였습니다."
    original = _original(_time(57), [text])
    first = _bundle(tmp_path / "original", source=original)
    duplicate = _bundle(tmp_path / "duplicate", source=original, listing_hash="d" * 64)
    corrected = _original(_time(57, hour=18), [text], remarks="정")
    second = _bundle(tmp_path / "correction", source=corrected)
    assert len(scan_corpora((first, duplicate, second), _time(57, hour=16))[0]) == 1
    events, filings = scan_corpora((first, duplicate, second), _time(57, hour=19))
    assert not events
    assert {item["status"] for item in filings} == {
        "later_correction_reported",
        "version_flagged_in_another_corpus",
    }


def test_conflicting_original_versions_cannot_produce_an_outcome(tmp_path):
    first = _bundle(tmp_path / "first")
    other = _original(_time(57, hour=10), ["당사는 신규 계약을 해제하였습니다."])
    second = _bundle(tmp_path / "other", source=other)
    events, filings = scan_corpora((first, second), _time(57, hour=19))
    assert not events
    assert {item["status"] for item in filings} == {"conflicting_document_versions"}


def _market_file(tmp_path):
    market = _market()
    base = {"verification": "verified", "source_url": "https://data.krx.co.kr/"}

    def descriptor(frame, name, identity, basis):
        path = tmp_path / f"{name}.csv"
        # Use exact ISO timestamp strings in value columns.
        text = frame.astype(str).copy()
        text.index = frame.index.strftime("%Y-%m-%d")
        text.to_csv(path, index_label="session_date")
        return {
            **base,
            "path": path.name,
            "sha256": sha256(path.read_bytes()),
            "identity": identity,
            "basis": basis,
            "availability_basis": "verified_release_timestamp",
            "availability_source_url": "https://openapi.krx.co.kr/",
        }

    return {
        "schema_version": 1,
        "kind": "real",
        "currency": "KRW",
        "venue": "KRX",
        "training_approved": False,
        "calendar": descriptor(market.sessions, "calendar", "KRX", "official_session_open_close"),
        "benchmark": descriptor(market.benchmark, "market", "KOSPI", "price_index"),
        "issuers": {
            "005930": descriptor(
                market.prices["005930"], "stock", "005930", "split_adjusted_price_return"
            )
        },
    }


def test_price_inputs_validate_hashes_basis_and_timestamps(tmp_path):
    spec = _market_file(tmp_path)
    path = tmp_path / "market.json"

    def read():
        path.write_text(json.dumps(spec), encoding="utf-8")
        return load_market(path, sha256(path.read_bytes()))

    assert "005930" in read().prices
    spec["benchmark"]["basis"] = "total_return_index"
    with pytest.raises(ValueError, match="KOSPI"):
        read()
    spec["benchmark"]["basis"] = "price_index"
    spec["benchmark"]["availability_basis"] = "undocumented"
    with pytest.raises(ValueError, match="documented close availability for KOSPI"):
        read()
    spec["benchmark"]["availability_basis"] = "verified_release_timestamp"
    prices = tmp_path / "stock.csv"
    prices.write_bytes(prices.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        read()
    spec["kind"] = "synthetic"
    with pytest.raises(ValueError, match="verified KRX"):
        read()


def test_rejects_price_available_before_krx_close(tmp_path):
    spec = _market_file(tmp_path)
    path = tmp_path / "prices.json"
    asset = spec["issuers"]["005930"]
    target = tmp_path / asset["path"]
    record = pd.read_csv(target)
    record.loc[0, "available_at"] = f"{_DAYS[0].date()}T10:00:00+09:00"
    record.to_csv(target, index=False)
    asset["sha256"] = sha256(target.read_bytes())
    path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="no earlier than close"):
        load_market(path, sha256(path.read_bytes()))
