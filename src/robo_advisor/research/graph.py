"""계획·검색·근거 검증·합성과 상한이 있는 재검색 그래프."""

from __future__ import annotations

import argparse
import json
import time
from typing import TypedDict
from uuid import uuid4

from langgraph.graph import END, StateGraph

from robo_advisor.config import get_settings
from robo_advisor.research.contracts import (
    BodyRiskCandidate,
    Citation,
    EvidenceReview,
    EvidenceScope,
    QueryPlan,
    ResearchClaim,
    ResearchOptions,
    ResearchRequest,
    ResearchResult,
    Synthesis,
)
from robo_advisor.research.dart_body import BodyChunk, is_explicit_risk_excerpt
from robo_advisor.research.documents import SUMMARY_MAX_LENGTH
from robo_advisor.research.ingestion import KST, canonical_url
from robo_advisor.research.llm import HttpResearchLLM, LLMResponseError, ResearchLLM
from robo_advisor.research.risk_tags import classify_event
from robo_advisor.research.trace import TraceStep, public_url, redact
from robo_advisor.research.vector_store import ResearchVectorStore, SearchHit


class ResearchState(TypedDict):
    """실행 중 상태. hits와 모델 원문은 공개 결과에 포함하지 않는다."""

    trace_id: str
    query: str
    hits: tuple[SearchHit, ...]
    verified: tuple[SearchHit, ...]
    claims: tuple[ResearchClaim, ...]
    problems: tuple[str, ...]
    retries: int
    trace: tuple[TraceStep, ...]


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _json_response(text, schema):
    def nonfinite(_):
        raise ValueError("nonfinite JSON")

    data = json.loads(text, object_pairs_hook=_unique_object, parse_constant=nonfinite)
    return schema.model_validate(data)


def _llm_problem(error, fallback):
    if isinstance(error, LLMResponseError) and error.code in {
        "authentication",
        "permission",
        "model_unavailable",
        "rate_or_quota_limit",
        "timeout",
    }:
        return (f"llm_{error.code}",)
    return (fallback,)


def run_research(
    question_or_ticker: str,
    as_of,
    *,
    store=None,
    llm: ResearchLLM | None = None,
    options: ResearchOptions | None = None,
    evidence_scope: EvidenceScope | None = None,
) -> ResearchResult:
    """관측 시점까지의 근거만 사용하고 실패를 명시한 인용 결과를 반환한다.

    max_retries는 최초 검색 이후 허용하는 쿼리 재작성 횟수다. 구조화 응답 오류와
    검색·모델 오류도 같은 예산을 쓰며, 원문 오류 메시지는 기록하지 않는다.
    """
    request = ResearchRequest(question=question_or_ticker, as_of=as_of)
    options = options or ResearchOptions()
    settings = get_settings()
    secrets = tuple(
        item.get_secret_value()
        for item in (settings.llm_api_key, settings.dart_api_key, settings.naver_client_secret)
        if item
    )
    llm = llm if llm is not None else HttpResearchLLM(settings)
    store = store if store is not None else ResearchVectorStore()
    question = redact(request.question, secrets=secrets)

    def record(state, step, started, problems, *, query=None, hits=()):
        tool = "llm"
        if step == "retrieve":
            tool = "vector_store"
        elif step == "synthesize" and evidence_scope == "dart_body":
            tool = "verified_excerpts"
        return (
            *state["trace"],
            TraceStep.model_validate(
                redact(
                    {
                        "step": step,
                        "tool": tool,
                        "query": state["query"] if query is None else query,
                        "source_ids": [hit.doc_id for hit in hits],
                        "validation": list(problems) or ["passed"],
                        "retries": state["retries"],
                        "elapsed_ms": max(0.0, (time.monotonic() - started) * 1000),
                    },
                    secrets=secrets,
                )
            ),
        )

    def generate(stage, state, schema, hits=()):
        payload = {
            "trace_id": state["trace_id"],
            "question": question,
            "query": state["query"],
            "as_of": request.as_of.isoformat(),
            "problems": list(state["problems"]),
            "schema": schema.model_json_schema(),
            "sources": [
                {
                    "doc_id": hit.doc_id,
                    "title": hit.title,
                    "summary": hit.summary,
                    "url": hit.url,
                    "published_at": hit.published_at,
                    "available_at": hit.available_at,
                    **(
                        {"body": {"section": hit.body.section, "quote": hit.body.quote}}
                        if hit.body
                        else {}
                    ),
                }
                for hit in hits
            ],
        }
        return _json_response(llm.generate(stage, payload), schema)

    def plan(state):
        started = time.monotonic()
        try:
            query = redact(generate("plan", state, QueryPlan).query.strip(), secrets=secrets)
            problems = ()
        except Exception as error:
            query, problems = state["query"], _llm_problem(error, "invalid_plan_response")
        return {
            "query": query,
            "problems": problems,
            "trace": record(state, "plan", started, problems, query=query),
        }

    def retrieve(state):
        started, hits, rejected = time.monotonic(), {}, []
        urls = set()
        try:
            found = store.search(state["query"], as_of=request.as_of, n_results=options.n_results)
            for hit in found:
                try:
                    if redact(hit.doc_id, secrets=secrets) != hit.doc_id:
                        raise ValueError("source ID must not contain sensitive data")
                    body = getattr(hit, "body", None)
                    if (evidence_scope == "dart_body") != (body is not None):
                        raise ValueError("source evidence scope mismatch")
                    citation = Citation(
                        doc_id=redact(hit.doc_id, secrets=secrets),
                        url=public_url(hit.url, secrets=secrets),
                        published_at=hit.published_at,
                        available_at=hit.available_at,
                        body=body,
                    )
                    if body is not None:
                        BodyChunk(citation=citation, title=hit.title, summary=hit.summary)
                        provenance = body.model_dump(mode="json")
                        if redact(provenance, secrets=secrets) != provenance:
                            raise ValueError("body evidence must not contain sensitive data")
                    if (
                        citation.published_at > request.as_of
                        or citation.available_at > request.as_of
                    ):
                        rejected.append("future_source_excluded")
                        continue
                    if not 0 <= hit.distance <= options.max_distance:
                        rejected.append("low_similarity_excluded")
                        continue
                    url_identity = canonical_url(str(citation.url))
                    if citation.doc_id in hits or (body is None and url_identity in urls):
                        rejected.append("duplicate_source_excluded")
                        continue
                    urls.add(url_identity)
                    hits[citation.doc_id] = SearchHit(
                        citation.doc_id,
                        hit.distance,
                        str(citation.url),
                        redact(hit.title, secrets=secrets),
                        redact(hit.summary[:SUMMARY_MAX_LENGTH], secrets=secrets),
                        hit.source,
                        citation.published_at.isoformat(),
                        citation.available_at.isoformat(),
                        body,
                    )
                except (ValueError, TypeError, AttributeError):
                    rejected.append("invalid_source_excluded")
            problems = ()
        except Exception:
            hits, problems = {}, ("retrieval_error",)
        hits = tuple(hits.values())
        return {
            "hits": hits,
            "verified": (),
            "claims": (),
            "problems": problems,
            "trace": record(
                state, "retrieve", started, (*problems, *dict.fromkeys(rejected)), hits=hits
            ),
        }

    def verify(state):
        started, hits, verified = time.monotonic(), state["hits"], ()
        problems = state["problems"]
        if not hits:
            problems = (*problems, "missing_citations")
        elif len(hits) < options.min_evidence:
            problems = (*problems, "insufficient_evidence")
        else:
            try:
                review = generate("verify", state, EvidenceReview, hits)
                ids = [item.doc_id for item in review.evidence]
                expected = {hit.doc_id for hit in hits}
                if len(ids) != len(set(ids)) or set(ids) != expected:
                    raise ValueError("review must assess every retrieved source exactly once")
                if any(
                    pair.left not in expected
                    or pair.right not in expected
                    or pair.left == pair.right
                    for pair in review.conflicts
                ):
                    raise ValueError("conflicts must reference distinct retrieved sources")
                selected = {
                    item.doc_id
                    for item in review.evidence
                    if item.relevance >= options.min_relevance
                }
                verified = tuple(hit for hit in hits if hit.doc_id in selected)
                if review.conflicts:
                    problems = ("conflicting_evidence",)
                elif len(verified) < options.min_evidence:
                    problems = ("insufficient_relevance",)
            except Exception as error:
                problems = _llm_problem(error, "invalid_verification_response")
        return {
            "problems": problems,
            "verified": verified,
            "trace": record(state, "verify", started, problems, hits=verified),
        }

    def rewrite(state):
        started = time.monotonic()
        next_state = {**state, "retries": state["retries"] + 1}
        try:
            query = redact(generate("rewrite", state, QueryPlan).query.strip(), secrets=secrets)
            problems = ()
        except Exception as error:
            query, problems = state["query"], _llm_problem(error, "invalid_rewrite_response")
        return {
            "retries": next_state["retries"],
            "query": query,
            "problems": problems,
            "trace": record(next_state, "rewrite", started, problems, query=query),
        }

    def synthesize(state):
        started, claims = time.monotonic(), ()
        try:
            sources = {
                hit.doc_id: Citation(
                    doc_id=hit.doc_id,
                    url=hit.url,
                    published_at=hit.published_at,
                    available_at=hit.available_at,
                    body=hit.body,
                )
                for hit in state["verified"]
            }
            if evidence_scope == "dart_body":
                claims = tuple(
                    ResearchClaim(text=item.body.quote, citations=(item,))
                    for item in sources.values()
                )
            else:
                synthesis = generate("synthesize", state, Synthesis, state["verified"])
                selected = dict.fromkeys(
                    source_id for claim in synthesis.claims for source_id in claim.source_ids
                )
                if not selected.keys() <= sources.keys():
                    raise ValueError("unknown citation")
                if evidence_scope == "dart_listing_metadata":
                    hits = {hit.doc_id: hit for hit in state["verified"]}
                    claims = tuple(
                        ResearchClaim(
                            text=(
                                f"접수일 {sources[source_id].published_at.astimezone(KST).date()}"
                                f" · {hits[source_id].title}"
                            ),
                            citations=(sources[source_id],),
                        )
                        for source_id in selected
                    )
                else:
                    claims = tuple(
                        ResearchClaim(
                            text=redact(claim.text, secrets=secrets),
                            citations=tuple(
                                sources[source_id] for source_id in dict.fromkeys(claim.source_ids)
                            ),
                        )
                        for claim in synthesis.claims
                    )
            problems = ()
        except Exception as error:
            problems = _llm_problem(error, "invalid_synthesis_or_citations")
        return {
            "claims": claims,
            "problems": problems,
            "trace": record(state, "synthesize", started, problems, hits=state["verified"]),
        }

    def retry_or_end(state):
        return "rewrite" if state["retries"] < options.max_retries else END

    graph = StateGraph(ResearchState)
    for name, node in (
        ("plan", plan),
        ("retrieve", retrieve),
        ("verify", verify),
        ("rewrite", rewrite),
        ("synthesize", synthesize),
    ):
        graph.add_node(name, node)
    graph.set_entry_point("plan")
    graph.add_conditional_edges(
        "plan", lambda state: retry_or_end(state) if state["problems"] else "retrieve"
    )
    graph.add_edge("retrieve", "verify")
    graph.add_conditional_edges(
        "verify", lambda state: retry_or_end(state) if state["problems"] else "synthesize"
    )
    graph.add_conditional_edges(
        "rewrite", lambda state: retry_or_end(state) if state["problems"] else "retrieve"
    )
    graph.add_conditional_edges(
        "synthesize", lambda state: retry_or_end(state) if state["problems"] else END
    )
    initial = ResearchState(
        trace_id=str(uuid4()),
        query=question,
        hits=(),
        verified=(),
        claims=(),
        problems=(),
        retries=0,
        trace=(),
    )
    final = graph.compile().invoke(initial, {"recursion_limit": 8 + options.max_retries * 5})
    failed = bool(final["problems"])
    claims = () if failed else final["claims"]
    candidates = []
    if evidence_scope == "dart_body":
        for claim in claims:
            if is_explicit_risk_excerpt(claim.text) and (kind := classify_event(claim.text)):
                candidates.append(
                    BodyRiskCandidate(event_type=kind.value, source_id=claim.citations[0].doc_id)
                )
    return ResearchResult(
        evidence_scope=evidence_scope,
        trace_id=initial["trace_id"],
        question=question,
        as_of=request.as_of,
        status="failed" if failed else "completed",
        claims=claims,
        risk_candidates=tuple(candidates),
        reasons=(*final["problems"], "retry_limit_exceeded") if failed else (),
        retries=final["retries"],
        trace=final["trace"],
    )


def main(argv=None):
    """시점과 재검색 상한을 명시한 리서치 결과 JSON을 출력한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question_or_ticker")
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--path")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--min-evidence", type=int, default=2)
    parser.add_argument("--base-url")
    args = parser.parse_args(argv)
    try:
        result = run_research(
            args.question_or_ticker,
            args.as_of,
            store=ResearchVectorStore(args.path),
            llm=HttpResearchLLM(base_url=args.base_url),
            options=ResearchOptions(max_retries=args.max_retries, min_evidence=args.min_evidence),
        )
    except ValueError:
        parser.exit(2, "리서치 입력 또는 LLM_PROVIDER/LLM_MODEL/LLM_API_KEY 설정을 확인하세요.\n")
    print(result.model_dump_json(indent=2))
    return 1 if result.status == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
