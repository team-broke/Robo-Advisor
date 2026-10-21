"""원천 문서 정제·중복 제거와 시점 검증.

available_at은 백테스트 관측 시점에서 문서를 볼 수 있게 되는 가장 이른 시각이다.
- DART처럼 공식 게시 기록이 남는 출처(official_publication): 게시 시각을 믿는다.
  날짜만 있는 경우 장 마감 이후 공시일 수 있으므로 다음 날 0시(KST)부터 쓴다.
- RSS처럼 최신 항목만 제공하는 출처(first_fetch): 게시 시각과 처음 받은 시각 중
  늦은 쪽부터 쓴다. 과거 기사를 나중에 받아도 과거 관측에 소급 삽입되지 않는다.
이미 정제된 문서를 다시 받으면 처음 정한 available_at을 유지한다.

중복 URL은 정규화한 URL(스킴·호스트 소문자, fragment와 utm_ 파라미터 제거, 쿼리
정렬)이 같은 문서다. 중복 콘텐츠는 정규화한 제목과 요약이 모두 같은 문서이며, 요약이
비어 있으면 서로 다른 공시가 같은 서식 제목을 쓸 수 있어 URL로만 판단한다. 중복이면
available_at이 가장 이른 문서를 남긴다.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from zoneinfo import ZoneInfo

from pydantic import model_validator

from robo_advisor.research.documents import RawDocument, SourceKind
from robo_advisor.schemas.common import UTCDateTime

KST = ZoneInfo("Asia/Seoul")
AvailabilityBasis = Literal["official_publication", "first_fetch"]
AVAILABILITY_BASIS: dict[SourceKind, AvailabilityBasis] = {
    SourceKind.DART: "official_publication",
    SourceKind.RSS: "first_fetch",
}


class ResearchDocument(RawDocument):
    """정제를 마쳐 시점 조회와 색인에 쓰는 문서."""

    doc_id: str
    canonical_url: str
    content_hash: str
    available_at: UTCDateTime
    availability_basis: AvailabilityBasis

    @model_validator(mode="after")
    def availability_after_publication(self) -> "ResearchDocument":
        if self.available_at < self.published_at:
            raise ValueError("available_at must be greater than or equal to published_at")
        return self


def canonical_url(url: str) -> str:
    """같은 문서를 가리키는 URL 표기 차이를 없앤다."""
    parts = urlsplit(str(url).strip())
    query = sorted(
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
    )
    path = parts.path or "/"
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), path, urlencode(query, doseq=True), "")
    )


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip().casefold()


def content_hash(title: str, summary: str) -> str:
    """정규화한 제목과 요약의 해시."""
    text = f"{_normalize_text(title)}\n{_normalize_text(summary)}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def available_at(raw: RawDocument) -> datetime:
    """출처 유형별 규칙으로 가장 이른 사용 가능 시각을 정한다."""
    if AVAILABILITY_BASIS[raw.source] == "official_publication":
        if raw.published_precision == "date":
            day = raw.published_at.astimezone(KST).date() + timedelta(days=1)
            return datetime.combine(day, time(), KST).astimezone(timezone.utc)
        return raw.published_at
    return max(raw.published_at, raw.fetched_at)


def normalize(raw: RawDocument) -> ResearchDocument:
    """원천 문서에 식별자·해시·available_at을 붙인다."""
    url = canonical_url(str(raw.url))
    return ResearchDocument(
        **raw.model_dump(),
        doc_id=hashlib.sha256(url.encode("utf-8")).hexdigest(),
        canonical_url=url,
        content_hash=content_hash(raw.title, raw.summary),
        available_at=available_at(raw),
        availability_basis=AVAILABILITY_BASIS[raw.source],
    )


@dataclass(frozen=True)
class DroppedDocument:
    url: str
    reason: Literal["duplicate_url", "duplicate_content"]
    kept_doc_id: str


@dataclass(frozen=True)
class IngestionResult:
    documents: tuple[ResearchDocument, ...]
    added: tuple[str, ...]
    dropped: tuple[DroppedDocument, ...] = field(default_factory=tuple)


def _priority(document: ResearchDocument):
    return (document.available_at, document.fetched_at, document.doc_id)


def ingest(
    raw_documents: Iterable[RawDocument], existing: Sequence[ResearchDocument] = ()
) -> IngestionResult:
    """기존 문서를 보존하면서 새 문서를 정제하고 중복을 제거한다.

    결과는 available_at, doc_id 순으로 정렬한다. 같은 입력을 다시 넣어도 결과가 같다.
    """
    by_url: dict[str, ResearchDocument] = {document.doc_id: document for document in existing}
    if len(by_url) != len(existing):
        raise ValueError("existing documents contain duplicate doc_id values")
    existing_ids = set(by_url)
    dropped: list[DroppedDocument] = []
    for raw in raw_documents:
        candidate = normalize(raw)
        current = by_url.get(candidate.doc_id)
        if current is None:
            by_url[candidate.doc_id] = candidate
            continue
        # 기존 available_at을 앞당기는 재수집은 소급 삽입이므로 받아들이지 않는다.
        if current.doc_id in existing_ids or _priority(current) <= _priority(candidate):
            dropped.append(DroppedDocument(str(raw.url), "duplicate_url", current.doc_id))
        else:
            dropped.append(DroppedDocument(str(current.url), "duplicate_url", candidate.doc_id))
            by_url[candidate.doc_id] = candidate
    kept: dict[str, ResearchDocument] = {}
    # 기존 문서를 먼저 두어 새 문서가 이미 쓰인 문서를 밀어내지 못하게 한다.
    ordered = sorted(
        by_url.values(), key=lambda item: (item.doc_id not in existing_ids, *_priority(item))
    )
    for document in ordered:
        if not document.summary:
            kept[f"url:{document.doc_id}"] = document
            continue
        key = f"content:{document.content_hash}"
        if key in kept:
            dropped.append(
                DroppedDocument(str(document.url), "duplicate_content", kept[key].doc_id)
            )
        else:
            kept[key] = document
    documents = tuple(sorted(kept.values(), key=lambda item: (item.available_at, item.doc_id)))
    added = tuple(item.doc_id for item in documents if item.doc_id not in existing_ids)
    return IngestionResult(documents, added, tuple(dropped))


def visible_documents(
    documents: Iterable[ResearchDocument], as_of: datetime
) -> tuple[ResearchDocument, ...]:
    """as_of 시점에 실제로 볼 수 있었던 문서만 돌려준다."""
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must include timezone information")
    return tuple(document for document in documents if document.available_at <= as_of)


def save_documents(documents: Iterable[ResearchDocument], path: str | Path) -> Path:
    """JSON Lines로 원자적으로 저장한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for document in documents:
            stream.write(json.dumps(document.model_dump(mode="json"), ensure_ascii=False))
            stream.write("\n")
    temporary.replace(path)
    return path


def load_documents(path: str | Path) -> tuple[ResearchDocument, ...]:
    """저장된 문서를 검증하며 읽는다. 파일이 없으면 빈 결과다."""
    path = Path(path)
    if not path.exists():
        return ()
    with path.open(encoding="utf-8") as stream:
        return tuple(ResearchDocument.model_validate_json(line) for line in stream if line.strip())
