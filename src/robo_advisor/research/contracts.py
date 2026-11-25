"""리서치 그래프의 구조화 응답과 공개 결과 계약."""

from __future__ import annotations

import hashlib
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from robo_advisor.research.trace import TraceStep
from robo_advisor.schemas.common import NonEmptyString, TraceId, UTCDateTime

EvidenceScope = Literal["dart_listing_metadata", "dart_body"]


class ResearchModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, allow_inf_nan=False, hide_input_in_errors=True
    )


class ResearchOptions(ResearchModel):
    """재검색 상한과 서로 다른 근거의 수·관련도 기준."""

    max_retries: int = Field(default=2, ge=0, le=10, strict=True)
    n_results: int = Field(default=5, ge=1, le=50, strict=True)
    min_evidence: int = Field(default=2, ge=1, strict=True)
    min_relevance: float = Field(default=0.5, ge=0, le=1)
    max_distance: float = Field(default=1.0, ge=0, le=2)

    @model_validator(mode="after")
    def evidence_within_search_size(self):
        if self.min_evidence > self.n_results:
            raise ValueError("min_evidence must not exceed n_results")
        return self


class ResearchRequest(ResearchModel):
    question: NonEmptyString
    as_of: UTCDateTime


class QueryPlan(ResearchModel):
    query: str = Field(min_length=1, max_length=1000, pattern=r"\S")


class DartBodyEvidence(ResearchModel):
    receipt: str = Field(pattern=r"^[0-9]{14}$")
    receipt_date: date
    first_fetched_at: UTCDateTime
    member: NonEmptyString
    section: NonEmptyString
    locator: NonEmptyString
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    quote_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    quote: str = Field(min_length=1, max_length=1800, pattern=r"\S")

    @model_validator(mode="after")
    def verify_quote_hash(self):
        if hashlib.sha256(self.quote.encode("utf-8")).hexdigest() != self.quote_sha256:
            raise ValueError("body excerpt SHA-256 mismatch")
        return self


class Citation(ResearchModel):
    """저장소에서 검증한 식별자·원문 URL·게시 및 가용 시각."""

    doc_id: NonEmptyString
    url: HttpUrl
    published_at: UTCDateTime
    available_at: UTCDateTime
    body: DartBodyEvidence | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def publication_before_availability(self):
        if self.available_at < self.published_at:
            raise ValueError("available_at must not precede published_at")
        if self.body is not None:
            from robo_advisor.research.ingestion import KST

            expected_url = f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={self.body.receipt}"
            local_published = self.published_at.astimezone(KST)
            if (
                str(self.url) != expected_url
                or local_published.date() != self.body.receipt_date
                or any(
                    (
                        local_published.hour,
                        local_published.minute,
                        local_published.second,
                        local_published.microsecond,
                    )
                )
                or self.available_at < self.body.first_fetched_at
                or self.available_at.astimezone(KST).date() <= self.body.receipt_date
            ):
                raise ValueError("body citation does not match its receipt or availability")
        return self


class EvidenceAssessment(ResearchModel):
    doc_id: NonEmptyString
    relevance: float = Field(ge=0, le=1, strict=True)


class Conflict(ResearchModel):
    left: NonEmptyString
    right: NonEmptyString


class EvidenceReview(ResearchModel):
    evidence: tuple[EvidenceAssessment, ...]
    conflicts: tuple[Conflict, ...]


class DraftClaim(ResearchModel):
    text: NonEmptyString
    source_ids: tuple[NonEmptyString, ...] = Field(min_length=1)


class Synthesis(ResearchModel):
    claims: tuple[DraftClaim, ...] = Field(min_length=1, max_length=20)


class ResearchClaim(ResearchModel):
    text: NonEmptyString
    citations: tuple[Citation, ...] = Field(min_length=1)


class BodyRiskCandidate(ResearchModel):
    event_type: NonEmptyString
    source_id: NonEmptyString


class ResearchResult(ResearchModel):
    """성공한 인용 주장 또는 명시적 실패와 공개 실행 기록."""

    trace_id: TraceId
    question: NonEmptyString
    as_of: UTCDateTime
    status: Literal["completed", "failed"]
    claims: tuple[ResearchClaim, ...] = ()
    reasons: tuple[NonEmptyString, ...] = ()
    retries: int = Field(ge=0)
    trace: tuple[TraceStep, ...]
    evidence_scope: EvidenceScope | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    risk_candidates: tuple[BodyRiskCandidate, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )

    @model_validator(mode="after")
    def completed_claims_only(self):
        if self.status == "completed" and (not self.claims or self.reasons):
            raise ValueError("completed research requires cited claims without failures")
        if self.status == "failed" and (self.claims or not self.reasons):
            raise ValueError("failed research must withhold claims and explain failure")
        for claim in self.claims:
            if any(citation.available_at > self.as_of for citation in claim.citations):
                raise ValueError("future citations are forbidden")
            if self.evidence_scope == "dart_body" and (
                len(claim.citations) != 1
                or claim.citations[0].body is None
                or claim.text != claim.citations[0].body.quote
            ):
                raise ValueError("body claims must reproduce exactly one verified excerpt")
            if self.evidence_scope != "dart_body" and any(item.body for item in claim.citations):
                raise ValueError("body citations require the body evidence scope")
        if self.risk_candidates:
            cited = {item.doc_id for claim in self.claims for item in claim.citations}
            if self.evidence_scope != "dart_body" or any(
                item.source_id not in cited for item in self.risk_candidates
            ):
                raise ValueError("risk candidates require cited body evidence")
        return self
