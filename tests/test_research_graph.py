"""외부 호출 없이 계획·재검색·시점 인용과 공개 trace를 검증한다."""

from dataclasses import replace
from datetime import datetime, timezone
import json
import socket
from uuid import UUID

import pytest

from robo_advisor.config import Settings
from robo_advisor.research import graph
from robo_advisor.research.contracts import ResearchOptions
from robo_advisor.research.graph import run_research
from robo_advisor.research.vector_store import ResearchVectorStore, SearchHit
from tests.test_research_vector_store import documents

AS_OF = datetime(2026, 1, 8, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(graph, "get_settings", lambda: Settings(_env_file=None))


def hit(doc_id="one", **changes):
    return replace(
        SearchHit(
            doc_id,
            0.1,
            f"https://news.example.com/{doc_id}",
            "실적 발표",
            "매출 감소 공시",
            "dart",
            "2026-01-05T00:00:00Z",
            "2026-01-05T01:00:00Z",
        ),
        **changes,
    )


class FakeStore:
    def __init__(self, batches=None):
        self.batches = batches if batches is not None else [(hit(), hit("two"))]
        self.calls = []

    def search(self, query, as_of, n_results):
        self.calls.append((query, as_of, n_results))
        return self.batches[min(len(self.calls) - 1, len(self.batches) - 1)]


class FakeLLM:
    def __init__(self, overrides=None):
        self.overrides = overrides or {}
        self.calls = []

    def generate(self, stage, payload):
        self.calls.append((stage, payload))
        if stage in self.overrides:
            answer = self.overrides[stage]
            if callable(answer):
                answer = answer(payload)
            if isinstance(answer, Exception):
                raise answer
            return answer if isinstance(answer, str) else json.dumps(answer)
        if stage in {"plan", "rewrite"}:
            return json.dumps(
                {"query": "실적 공시" if stage == "plan" else "실적 매출 감소 재확인"}
            )
        if stage == "verify":
            return json.dumps(
                {
                    "evidence": [
                        {"doc_id": source["doc_id"], "relevance": 0.9}
                        for source in payload["sources"]
                    ],
                    "conflicts": [],
                }
            )
        return json.dumps(
            {
                "claims": [
                    {
                        "text": "공시에 매출 감소가 보고됐다.",
                        "source_ids": [item["doc_id"] for item in payload["sources"]],
                    }
                ]
            }
        )


def test_all_nodes_return_citations_trace_and_serializable_result():
    llm, store = FakeLLM(), FakeStore()
    result = run_research("SPY", AS_OF, llm=llm, store=store)
    assert result.status == "completed" and result.retries == 0
    UUID(result.trace_id)
    assert [step.step for step in result.trace] == ["plan", "retrieve", "verify", "synthesize"]
    assert result.claims[0].citations[0].doc_id == "one"
    assert str(result.claims[0].citations[0].url) == hit().url
    assert result.claims[0].citations[0].published_at < AS_OF
    assert store.calls[0] == ("실적 공시", AS_OF, 5)
    assert all(call[1]["trace_id"] == result.trace_id for call in llm.calls)
    assert all(step.elapsed_ms >= 0 and step.retries == 0 for step in result.trace)
    assert json.loads(result.model_dump_json())["trace"][1]["tool"] == "vector_store"
    assert run_research("SPY", AS_OF, llm=FakeLLM(), store=FakeStore()).trace_id != result.trace_id


@pytest.mark.parametrize(
    "first,reason", [([], "missing_citations"), ([hit()], "insufficient_evidence")]
)
def test_missing_evidence_rewrites_and_searches_again(first, reason):
    store, llm = FakeStore([first, (hit(), hit("two"))]), FakeLLM()
    result = run_research("SPY", AS_OF, store=store, llm=llm)
    assert result.status == "completed" and result.retries == 1
    assert store.calls[0][0] != store.calls[1][0]
    assert reason in result.trace[2].validation
    assert next(payload for stage, payload in llm.calls if stage == "rewrite")["problems"] == [
        reason
    ]


@pytest.mark.parametrize("problem", ["insufficient_relevance", "conflicting_evidence"])
def test_relevance_and_conflict_trigger_bounded_correction(problem):
    def review(payload):
        return {
            "evidence": [
                {
                    "doc_id": source["doc_id"],
                    "relevance": 0.1 if problem == "insufficient_relevance" else 1.0,
                }
                for source in payload["sources"]
            ],
            "conflicts": (
                [{"left": "one", "right": "two"}] if problem == "conflicting_evidence" else []
            ),
        }

    store = FakeStore()
    result = run_research(
        "SPY",
        AS_OF,
        store=store,
        llm=FakeLLM({"verify": review}),
        options=ResearchOptions(max_retries=2),
    )
    assert result.status == "failed" and not result.claims
    assert result.retries == 2 and len(store.calls) == 3
    assert result.reasons == (problem, "retry_limit_exceeded")


@pytest.mark.parametrize("stage", ["plan", "rewrite", "verify", "synthesize"])
@pytest.mark.parametrize(
    "invalid",
    ["not JSON", '{"query":"a","query":"b"}', '{"query":NaN}', {"reasoning": "private thoughts"}],
)
def test_invalid_structured_response_never_silently_succeeds(stage, invalid):
    store = FakeStore([[]]) if stage == "rewrite" else FakeStore()
    result = run_research(
        "SPY",
        AS_OF,
        store=store,
        llm=FakeLLM({stage: invalid}),
        options=ResearchOptions(max_retries=1),
    )
    # 계획 한 번의 오류는 정상 재작성으로 회복할 수 있다.
    if stage == "plan":
        assert result.status == "completed" and result.retries == 1
    else:
        assert result.status == "failed" and result.retries == 1 and not result.claims
    assert "private thoughts" not in result.model_dump_json()


@pytest.mark.parametrize(
    "draft",
    [
        {"claims": [{"text": "없는 근거", "source_ids": []}]},
        {"claims": [{"text": "없는 근거", "source_ids": ["invented"]}]},
        {"claims": []},
    ],
)
def test_uncited_or_fabricated_claims_are_withheld(draft):
    result = run_research(
        "SPY",
        AS_OF,
        store=FakeStore(),
        llm=FakeLLM({"synthesize": draft}),
        options=ResearchOptions(max_retries=0),
    )
    assert result.status == "failed" and not result.claims
    assert "invalid_synthesis_or_citations" in result.reasons


@pytest.mark.leakage
@pytest.mark.parametrize(
    "changes",
    [
        {"published_at": "2026-01-09T00:00:00Z", "available_at": "2026-01-09T01:00:00Z"},
        {"available_at": "2026-01-09T00:00:00Z"},
        {"published_at": "2026-01-05"},
        {"available_at": "2026-01-04T00:00:00Z"},
        {"url": "javascript:alert(1)"},
        {"distance": float("nan")},
    ],
)
def test_bad_or_future_sources_do_not_reach_llm_or_citations(changes):
    llm = FakeLLM()
    result = run_research(
        "SPY", AS_OF, store=FakeStore([(hit("bad", **changes), hit(), hit("two"))]), llm=llm
    )
    assert result.status == "completed"
    assert all(
        source["doc_id"] != "bad" for _, payload in llm.calls for source in payload["sources"]
    )
    assert all(citation.doc_id != "bad" for claim in result.claims for citation in claim.citations)


def test_duplicate_documents_do_not_satisfy_minimum_evidence():
    for repeated in (hit(), hit("other", url=hit().url)):
        result = run_research(
            "SPY",
            AS_OF,
            store=FakeStore([(hit(), repeated)]),
            llm=FakeLLM(),
            options=ResearchOptions(max_retries=0),
        )
        assert result.status == "failed" and "insufficient_evidence" in result.reasons


def test_model_and_store_errors_are_explicit_without_secret_exception_text():
    class BrokenStore:
        def search(self, *args, **kwargs):
            raise RuntimeError("private@example.com sk-super-secret")

    for store, llm in (
        (BrokenStore(), FakeLLM()),
        (FakeStore(), FakeLLM({"verify": RuntimeError("private@example.com")})),
    ):
        result = run_research(
            "SPY", AS_OF, store=store, llm=llm, options=ResearchOptions(max_retries=1)
        )
        assert result.status == "failed"
        assert "private@example.com" not in result.model_dump_json()


def test_real_vector_store_integration_excludes_future_documents(tmp_path):
    store = ResearchVectorStore(tmp_path / "vectors")
    store.index(documents())
    result = run_research(
        "환율",
        AS_OF,
        store=store,
        llm=FakeLLM(),
        options=ResearchOptions(min_evidence=1, max_distance=2),
    )
    assert result.status == "completed"
    assert all(
        citation.available_at <= AS_OF for claim in result.claims for citation in claim.citations
    )


def test_trace_and_result_redact_known_keys_and_contact_data(monkeypatch):
    secret = "unstructured-configuration-secret"
    monkeypatch.setattr(graph, "get_settings", lambda: Settings(_env_file=None, llm_api_key=secret))
    llm = FakeLLM(
        {
            "plan": {"query": f"SPY {secret} person@example.com 010-1234-5678"},
            "synthesize": {"claims": [{"text": f"공시 {secret}", "source_ids": ["one"]}]},
        }
    )
    result = run_research(f"SPY {secret} person@example.com", AS_OF, store=FakeStore(), llm=llm)
    serialized = result.model_dump_json()
    assert (
        secret not in serialized
        and "person@example.com" not in serialized
        and "010-1234-5678" not in serialized
    )
    assert result.status == "completed"


def test_cli_prints_json_and_failed_exit_status(monkeypatch, capsys):
    result = run_research(
        "SPY", AS_OF, store=FakeStore([[]]), llm=FakeLLM(), options=ResearchOptions(max_retries=0)
    )
    monkeypatch.setattr(graph, "ResearchVectorStore", lambda _: FakeStore())
    monkeypatch.setattr(graph, "HttpResearchLLM", lambda **kwargs: FakeLLM())
    monkeypatch.setattr(graph, "run_research", lambda *args, **kwargs: result)
    assert graph.main(["SPY", "--as-of", AS_OF.isoformat()]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "failed"


def test_naive_cutoff_and_bad_options_fail_before_tools():
    with pytest.raises(ValueError):
        run_research("SPY", datetime(2026, 1, 8), store=FakeStore(), llm=FakeLLM())
    with pytest.raises(ValueError):
        ResearchOptions(min_evidence=6)


@pytest.mark.parametrize(
    "review",
    [
        {"evidence": [{"doc_id": "invented", "relevance": 1.0}], "conflicts": []},
        {"evidence": [{"doc_id": "one", "relevance": 1.0}] * 2, "conflicts": []},
        {
            "evidence": [{"doc_id": "one", "relevance": True}, {"doc_id": "two", "relevance": 0.9}],
            "conflicts": [],
        },
        {
            "evidence": [{"doc_id": "one", "relevance": 1.0}, {"doc_id": "two", "relevance": 0.9}],
            "conflicts": [{"left": "one", "right": "invented"}],
        },
    ],
)
def test_verification_cannot_invent_omit_or_repeat_sources(review):
    result = run_research(
        "SPY",
        AS_OF,
        store=FakeStore(),
        llm=FakeLLM({"verify": review}),
        options=ResearchOptions(max_retries=0),
    )
    assert result.status == "failed" and result.reasons[0] == "invalid_verification_response"


def test_invalid_synthesis_can_recover_after_retrieval():
    calls = []

    def synthesize(payload):
        calls.append(payload)
        return {"claims": [{"text": "공시", "source_ids": [] if len(calls) == 1 else ["one"]}]}

    store = FakeStore()
    result = run_research("SPY", AS_OF, store=store, llm=FakeLLM({"synthesize": synthesize}))
    assert result.status == "completed" and result.retries == 1 and len(store.calls) == 2
