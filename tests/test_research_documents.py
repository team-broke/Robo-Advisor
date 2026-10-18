"""원천 문서 계약 검증."""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from robo_advisor.research.documents import SUMMARY_MAX_LENGTH, RawDocument, SourceKind

KST = timezone(timedelta(hours=9))


def document(**changes):
    values = dict(
        source="dart",
        source_name="dart",
        source_id="20260105000123",
        title="주요사항보고서",
        summary="요약",
        url="https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20260105000123",
        published_at=datetime(2026, 1, 5, tzinfo=KST),
        published_precision="date",
        fetched_at=datetime(2026, 1, 6, 9, tzinfo=KST),
    )
    values.update(changes)
    return RawDocument(**values)


def test_times_are_normalized_to_utc_and_values_frozen():
    item = document()
    assert item.source is SourceKind.DART
    assert item.published_at == datetime(2026, 1, 4, 15, tzinfo=timezone.utc)
    assert item.fetched_at.tzinfo == timezone.utc
    with pytest.raises(ValidationError):
        item.title = "changed"


@pytest.mark.parametrize(
    "changes",
    [
        {"summary": "가" * (SUMMARY_MAX_LENGTH + 1)},
        {"title": " "},
        {"url": "not-a-url"},
        {"published_at": datetime(2026, 1, 5)},
        {"fetched_at": datetime(2026, 1, 6)},
        {"source": "blog"},
        {"published_precision": "month"},
        {"body": "원문 전체"},
    ],
)
def test_invalid_documents_are_rejected(changes):
    with pytest.raises(ValidationError):
        document(**changes)


def test_summary_at_limit_is_allowed():
    assert len(document(summary="가" * SUMMARY_MAX_LENGTH).summary) == SUMMARY_MAX_LENGTH
