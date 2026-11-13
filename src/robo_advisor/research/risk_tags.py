"""검증된 공개 주장에 연결된 위험 이벤트를 시점 리스크 태그로 변환한다."""

from __future__ import annotations

from collections.abc import Iterable
import re
from typing import Literal

from pydantic import Field, HttpUrl, ValidationError

from robo_advisor.research.contracts import ResearchModel, ResearchResult
from robo_advisor.research.ingestion import canonical_url
from robo_advisor.research.trace import public_url, redact
from robo_advisor.schemas.common import AssetSymbol, NonEmptyString, TraceId, UTCDateTime
from robo_advisor.schemas.risk import RiskEventType, RiskTag

_PATTERNS = {
    RiskEventType.EARNINGS_SHOCK: (
        r"실적\s*쇼크|어닝\s*쇼크|매출\s*감소|적자\s*전환|"
        r"earnings\s+(?:shock|miss)|profit\s+warning"
    ),
    RiskEventType.REGULATION_CHANGE: r"규제|제재|법안|regulat\w*|sanctions?",
    RiskEventType.SHARP_MOVE: r"급등|급락|폭락|급변|surge|plunge|sell.off|sharp\s+move",
    RiskEventType.GEOPOLITICAL: r"지정학|전쟁|군사|geopolit\w*|\bwar\b|military",
}


class RiskEvent(ResearchModel):
    """공개 주장 문구와 출처 trace를 가진 이벤트 후보. 시각은 원 출처 기준이다."""

    description: NonEmptyString
    asset: AssetSymbol
    severity: float = Field(ge=0, le=1, strict=True)
    confidence: float = Field(ge=0, le=1, strict=True)
    published_at: UTCDateTime
    available_at: UTCDateTime
    source_doc_id: NonEmptyString
    source_url: HttpUrl
    source_trace_id: TraceId | None = None


class EventExclusion(ResearchModel):
    """민감한 원문 없이 입력 위치와 제외 사유만 기록한다."""

    index: int = Field(ge=0)
    reason: Literal[
        "invalid_event",
        "failed_research",
        "missing_source_trace",
        "source_mismatch",
        "source_time_mismatch",
        "unavailable_at_cutoff",
        "invalid_availability",
        "below_confidence_threshold",
        "unsupported_event",
        "unclassified_event",
        "sensitive_asset",
        "duplicate_tag",
    ]


class RiskTagConversion(ResearchModel):
    trace_id: TraceId
    as_of: UTCDateTime
    tags: tuple[RiskTag, ...]
    excluded: tuple[EventExclusion, ...]


class _ConversionOptions(ResearchModel):
    as_of: UTCDateTime
    confidence_threshold: float = Field(default=0.7, ge=0, le=1, strict=True)


def classify_event(description: str) -> RiskEventType | None:
    """명시적인 사건 표현을 분류한다. 미분류·복수 유형은 추정하지 않는다."""
    matched = [kind for kind, pattern in _PATTERNS.items() if re.search(pattern, description, re.I)]
    return matched[0] if len(matched) == 1 else None


def _normalized(text):
    return " ".join(text.split()).casefold()


def convert_risk_events(
    events: Iterable[RiskEvent | dict],
    research: ResearchResult,
    *,
    as_of=None,
    confidence_threshold: float = 0.7,
) -> RiskTagConversion:
    """인용·검증·합성 trace가 있는 공개 주장만 RiskTag로 연결한다.

    이벤트 문구는 같은 문서가 인용된 공개 주장과 일치해야 한다. 게시 시각은
    citation과 같고 가용 시각은 원 문서보다 빠를 수 없다. as_of는 리서치가
    확인한 시점보다 넓히지 않는다. 누락·미분류·낮은 신뢰도 후보는 사유와 함께
    제외하며, 실적 쇼크·규제 변경·급등락·지정학 표현이 겹치면 자동 선택하지 않는다.
    """
    options = _ConversionOptions(
        as_of=research.as_of if as_of is None else as_of,
        confidence_threshold=confidence_threshold,
    )
    cutoff = min(options.as_of, research.as_of)
    verified = set()
    synthesized = set()
    for step in research.trace:
        if step.retries != research.retries or step.validation != ("passed",):
            continue
        if step.step == "verify":
            verified.update(step.source_ids)
        elif step.step == "synthesize":
            synthesized.update(step.source_ids)
    sources = {}
    for claim in research.claims:
        for citation in claim.citations:
            sources.setdefault(citation.doc_id, []).append((citation, _normalized(claim.text)))
    tags, excluded, seen = [], [], set()

    def reject(index, reason):
        excluded.append(EventExclusion(index=index, reason=reason))

    for index, candidate in enumerate(events):
        try:
            event = RiskEvent.model_validate(candidate)
        except (ValidationError, TypeError):
            reject(index, "invalid_event")
            continue
        if research.status != "completed":
            reject(index, "failed_research")
            continue
        if (
            event.source_trace_id != research.trace_id
            or event.source_doc_id not in verified & synthesized
        ):
            reject(index, "missing_source_trace")
            continue
        linked = sources.get(event.source_doc_id, ())
        if not linked:
            reject(index, "source_mismatch")
            continue
        citation = linked[0][0]
        try:
            source_url = public_url(str(citation.url))
            if any(item[0] != citation for item in linked) or canonical_url(
                public_url(str(event.source_url))
            ) != canonical_url(source_url):
                raise ValueError("source mismatch")
        except ValueError:
            reject(index, "source_mismatch")
            continue
        if event.published_at != citation.published_at:
            reject(index, "source_time_mismatch")
            continue
        if event.available_at < event.published_at or event.available_at < citation.available_at:
            reject(index, "invalid_availability")
            continue
        if event.available_at > cutoff or citation.available_at > cutoff:
            reject(index, "unavailable_at_cutoff")
            continue
        if event.confidence < options.confidence_threshold:
            reject(index, "below_confidence_threshold")
            continue
        if _normalized(event.description) not in {item[1] for item in linked}:
            reject(index, "unsupported_event")
            continue
        event_type = classify_event(event.description)
        if event_type is None:
            reject(index, "unclassified_event")
            continue
        if redact(event.asset) != event.asset:
            reject(index, "sensitive_asset")
            continue
        tag = RiskTag(
            event_type=event_type,
            asset=event.asset,
            severity=event.severity,
            confidence=event.confidence,
            published_at=event.published_at,
            available_at=event.available_at,
            source_url=source_url,
            trace_id=research.trace_id,
        )
        identity = tag.model_dump_json()
        if identity in seen:
            reject(index, "duplicate_tag")
            continue
        seen.add(identity)
        tags.append(tag)
    return RiskTagConversion(
        trace_id=research.trace_id, as_of=cutoff, tags=tuple(tags), excluded=tuple(excluded)
    )
