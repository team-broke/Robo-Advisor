"""리서치 그래프의 구조화 응답과 공개 결과 계약."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from robo_advisor.research.trace import TraceStep
from robo_advisor.schemas.common import NonEmptyString, TraceId, UTCDateTime


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


class Citation(ResearchModel):
    """저장소에서 검증한 식별자·원문 URL·게시 및 가용 시각."""

    doc_id: NonEmptyString
    url: HttpUrl
    published_at: UTCDateTime
    available_at: UTCDateTime

    @model_validator(mode="after")
    def publication_before_availability(self):
        if self.available_at < self.published_at:
            raise ValueError("available_at must not precede published_at")
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

    @model_validator(mode="after")
    def completed_claims_only(self):
        if self.status == "completed" and (not self.claims or self.reasons):
            raise ValueError("completed research requires cited claims without failures")
        if self.status == "failed" and (self.claims or not self.reasons):
            raise ValueError("failed research must withhold claims and explain failure")
        for claim in self.claims:
            if any(citation.available_at > self.as_of for citation in claim.citations):
                raise ValueError("future citations are forbidden")
        return self
