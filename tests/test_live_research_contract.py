"""Network-free checks for the live research service boundaries."""

from datetime import datetime, timezone
import json

from fastapi.testclient import TestClient
import httpx
import pytest

from robo_advisor.api.app import create_app
from robo_advisor.api.services import Services
from robo_advisor.config import Settings
from robo_advisor.research.graph import run_research
from robo_advisor.research.llm import HttpResearchLLM, LLMResponseError
from robo_advisor.research.risk_tags import convert_risk_events
from robo_advisor.research.sources.dart import DartAdapter
from robo_advisor.research.sources.http import PoliteClient
from robo_advisor.research.sources.naver_news import NaverNewsSearch
from tests.test_research_graph import FakeLLM, FakeStore, hit


def _gemini():
    return Settings(
        _env_file=None,
        llm_provider="gemini",
        llm_model="gemini-3.5-flash-lite",
        llm_api_key="fixture-google-key",
    )


def test_gemini_uses_official_endpoint_header_and_json_schema():
    schema = {"type": "object"}

    def send(request):
        assert request.url.host == "generativelanguage.googleapis.com"
        assert request.url.path.endswith("/gemini-3.5-flash-lite:generateContent")
        assert not request.url.query and request.headers["x-goog-api-key"] == "fixture-google-key"
        body = json.loads(request.content)
        assert body["generationConfig"]["responseJsonSchema"] == schema
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {
                            "parts": [
                                {"text": "hidden", "thought": True},
                                {"text": '{"query":"삼성전자"}'},
                            ]
                        },
                    }
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(send)) as client:
        result = HttpResearchLLM(_gemini(), client=client).generate("plan", {"schema": schema})
    assert json.loads(result) == {"query": "삼성전자"}


@pytest.mark.parametrize(
    "status,reason,code",
    [
        (400, "API_KEY_INVALID", "authentication"),
        (404, None, "model_unavailable"),
        (429, None, "rate_or_quota_limit"),
        (None, None, "timeout"),
    ],
)
def test_gemini_errors_are_classified_without_secret_or_retry(status, reason, code):
    calls = []

    def send(request):
        calls.append(request)
        if status is None:
            raise httpx.ReadTimeout("fixture-google-key")
        return httpx.Response(
            status,
            json={
                "error": {
                    "details": [{"reason": reason}],
                    "message": "fixture-google-key",
                }
            },
        )

    with httpx.Client(transport=httpx.MockTransport(send)) as client:
        with pytest.raises(LLMResponseError) as caught:
            HttpResearchLLM(_gemini(), client=client).generate("plan", {})
    assert caught.value.code == code
    assert "fixture-google-key" not in str(caught.value)
    assert len(calls) == 1


def test_news_http_reads_once_and_is_marked_no_store():
    calls = []

    def send(request):
        calls.append(request)
        assert request.url.host == "openapi.naver.com"
        assert request.headers["x-naver-client-secret"] == "fixture-naver-secret"
        assert request.url.params["display"] == "5"
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "title": "<b>ETF</b> &amp; 채권",
                        "description": "<script>bad()</script>기사 요약",
                        "pubDate": "Tue, 24 Nov 2026 17:00:00 +0900",
                        "originallink": "https://example.com/article",
                        "link": "https://news.naver.com/article",
                    }
                ]
            },
        )

    settings = Settings(
        _env_file=None, naver_client_id="fixture", naver_client_secret="fixture-naver-secret"
    )
    with httpx.Client(transport=httpx.MockTransport(send)) as upstream:
        service = NaverNewsSearch(
            settings.naver_client_id, settings.naver_client_secret, client=upstream
        )
        app = TestClient(create_app(Services(news=service)))
        assert app.get("/news", params={"query": "  "}).status_code == 422
        response = app.get("/news", params={"query": "ETF"})
    assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
    assert response.json()["items"] == [
        {
            "title": "ETF & 채권",
            "summary": "기사 요약",
            "published_at": "2026-11-24T08:00:00Z",
            "url": "https://example.com/article",
            "original_url": "https://example.com/article",
        }
    ]
    assert len(calls) == 1
    assert "fixture-naver-secret" not in response.text


def test_registered_dart_list_is_bounded_and_preserves_provider_status():
    def send(request):
        assert request.url.path == "/api/list.json"
        assert request.url.params["crtfc_key"] == "fixture-dart-key"
        assert request.url.params["page_no"] == "1"
        return httpx.Response(
            200,
            json={
                "status": "000",
                "total_page": 15,
                "list": [
                    {
                        "rcept_no": "20261124000001",
                        "rcept_dt": "20261124",
                        "corp_code": "00126380",
                        "corp_name": "삼성전자",
                        "report_nm": "분기보고서",
                    }
                ],
            },
        )

    settings = Settings(_env_file=None, dart_api_key="fixture-dart-key")
    with (
        httpx.Client(transport=httpx.MockTransport(send)) as upstream,
        PoliteClient(upstream) as client,
    ):
        response = DartAdapter(client, max_pages=1, settings=settings, registered_api=True).fetch(
            "20261101", "20261124", corp_code="00126380"
        )
    assert response.provider_status == "000" and len(response.request_logs) == 1
    assert "fixture-dart-key" not in response.request_logs[0].path
    assert response.documents[0].published_precision == "date"
    assert not response.documents[0].related_assets


@pytest.mark.leakage
def test_listing_scope_constrains_synthesis_and_blocks_risk_tags(monkeypatch):
    monkeypatch.setattr(
        "robo_advisor.research.graph.get_settings", lambda: Settings(_env_file=None)
    )
    result = run_research(
        "공시 제목",
        datetime(2026, 1, 8, tzinfo=timezone.utc),
        store=FakeStore([(hit(), hit("two"), hit("future", available_at="2026-01-09T00:00:00Z"))]),
        llm=FakeLLM(),
        evidence_scope="dart_listing_metadata",
    )
    assert result.status == "completed"
    assert all(claim.text == "접수일 2026-01-05 · 실적 발표" for claim in result.claims)
    assert {citation.doc_id for claim in result.claims for citation in claim.citations} == {
        "one",
        "two",
    }
    converted = convert_risk_events([{}], result)
    assert not converted.tags and converted.excluded[0].reason == "listing_metadata_only"
