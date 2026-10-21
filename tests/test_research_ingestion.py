"""문서 정제·중복 제거와 시점 조회 차단 검증."""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from robo_advisor.research.documents import RawDocument
from robo_advisor.research.ingestion import (
    ResearchDocument,
    canonical_url,
    ingest,
    load_documents,
    normalize,
    save_documents,
    visible_documents,
)

KST = timezone(timedelta(hours=9))


def dart(rcept_no="20260105000123", summary="삼성전자 주요사항보고서", day=5, **changes):
    values = dict(
        source="dart",
        source_name="dart",
        source_id=rcept_no,
        title="주요사항보고서",
        summary=summary,
        url=f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}",
        published_at=datetime(2026, 1, day, tzinfo=KST),
        published_precision="date",
        fetched_at=datetime(2026, 10, 18, 9, tzinfo=KST),
    )
    values.update(changes)
    return RawDocument(**values)


def rss(url="https://news.example.com/a?id=1", summary="환율 급등", published=None, fetched=None):
    published = published or datetime(2026, 1, 5, 10, tzinfo=KST)
    return RawDocument(
        source="rss",
        source_name="naver-finance",
        title="환율 급등에 증시 하락",
        summary=summary,
        url=url,
        published_at=published,
        fetched_at=fetched or published + timedelta(minutes=5),
    )


def test_canonical_url_removes_presentation_differences():
    left = canonical_url("HTTPS://News.Example.com/a?b=2&utm_source=x&a=1#top")
    right = canonical_url("https://news.example.com/a?a=1&b=2")
    assert left == right == "https://news.example.com/a?a=1&b=2"


@pytest.mark.leakage
def test_date_only_official_publication_is_available_next_kst_day():
    document = normalize(dart())
    assert document.availability_basis == "official_publication"
    assert document.available_at == datetime(2026, 1, 6, tzinfo=KST)
    assert document.available_at > document.published_at


@pytest.mark.leakage
def test_backfilled_feed_items_are_not_available_before_first_fetch():
    late = datetime(2026, 3, 1, 9, tzinfo=KST)
    document = normalize(rss(fetched=late))
    assert document.availability_basis == "first_fetch"
    assert document.available_at == late
    future_dated = normalize(rss(published=late, fetched=late - timedelta(minutes=1)))
    assert future_dated.available_at == late


def test_duplicate_urls_keep_the_earliest_available_copy():
    early = rss(url="https://news.example.com/a?utm_medium=rss&id=1")
    late = rss(fetched=datetime(2026, 1, 7, tzinfo=KST))
    result = ingest([late, early])
    assert len(result.documents) == 1
    assert result.documents[0].available_at == early.fetched_at
    assert [item.reason for item in result.dropped] == ["duplicate_url"]


def test_duplicate_content_across_urls_is_removed():
    first = rss(url="https://news.example.com/a")
    copy = rss(
        url="https://mirror.example.com/b",
        fetched=first.fetched_at + timedelta(hours=1),
    )
    result = ingest([copy, first])
    assert [str(item.url) for item in result.documents] == ["https://news.example.com/a"]
    assert result.dropped[0].reason == "duplicate_content"
    assert result.dropped[0].kept_doc_id == result.documents[0].doc_id


def test_empty_summaries_are_deduplicated_by_url_only():
    result = ingest([dart("1", summary=""), dart("2", summary="")])
    assert len(result.documents) == 2
    assert result.dropped == ()


@pytest.mark.leakage
def test_reingestion_never_moves_availability_earlier():
    late = rss(fetched=datetime(2026, 2, 1, tzinfo=KST))
    stored = ingest([late]).documents
    again = ingest([rss()], existing=stored)
    assert again.documents == stored
    assert again.added == ()
    assert again.dropped[0].reason == "duplicate_url"


@pytest.mark.leakage
def test_new_content_duplicate_cannot_displace_stored_document():
    stored = ingest([rss(fetched=datetime(2026, 2, 1, tzinfo=KST))]).documents
    earlier_copy = rss(url="https://mirror.example.com/b")
    result = ingest([earlier_copy], existing=stored)
    assert result.documents == stored
    assert result.dropped[0].reason == "duplicate_content"


def test_ingestion_is_idempotent_and_ordered():
    raws = [rss(url="https://news.example.com/z", summary="다른 기사"), dart(), rss()]
    first = ingest(raws)
    second = ingest(raws, existing=first.documents)
    assert second.documents == first.documents
    assert second.added == ()
    times = [item.available_at for item in first.documents]
    assert times == sorted(times)
    assert set(first.added) == {item.doc_id for item in first.documents}


def test_existing_store_must_not_contain_duplicate_ids():
    document = normalize(rss())
    with pytest.raises(ValueError, match="duplicate"):
        ingest([], existing=[document, document])


@pytest.mark.leakage
def test_point_in_time_query_blocks_documents_not_yet_available():
    documents = ingest([dart(), rss()]).documents
    before_close = datetime(2026, 1, 5, 15, 30, tzinfo=KST)
    assert [item.source.value for item in visible_documents(documents, before_close)] == ["rss"]
    next_day = datetime(2026, 1, 6, tzinfo=KST)
    assert len(visible_documents(documents, next_day)) == 2
    with pytest.raises(ValueError, match="timezone"):
        visible_documents(documents, datetime(2026, 1, 6))


def test_available_at_cannot_precede_publication():
    values = normalize(rss()).model_dump()
    values["available_at"] = values["published_at"] - timedelta(seconds=1)
    with pytest.raises(ValidationError, match="available_at"):
        ResearchDocument(**values)


def test_jsonl_roundtrip_and_missing_store(tmp_path):
    documents = ingest([dart(), rss()]).documents
    path = save_documents(documents, tmp_path / "research" / "documents.jsonl")
    assert load_documents(path) == documents
    assert load_documents(tmp_path / "missing.jsonl") == ()
    assert not list(path.parent.glob("*.tmp"))
