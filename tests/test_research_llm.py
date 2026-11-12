"""두 제공자의 HTTP 형식·오류·비밀정보를 실제 네트워크 없이 검증한다."""

import json
import socket

import httpx
import pytest

from robo_advisor.config import Settings
from robo_advisor.research.llm import HttpResearchLLM, LLMConfigurationError, LLMResponseError


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)


def settings(provider="openai", **kwargs):
    return Settings(
        _env_file=None,
        llm_provider=provider,
        llm_model="configured-model",
        llm_api_key="private-key-value",
        **kwargs,
    )


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_provider_request_and_json_response(provider):
    def handle(request):
        body = json.loads(request.content)
        assert body["model"] == "configured-model"
        assert "JSON" in str(body)
        assert json.loads(body["messages"][-1]["content"])["stage"] == "plan"
        if provider == "openai":
            assert request.url == "https://compatible.example/v1/chat/completions"
            assert request.headers["Authorization"] == "Bearer private-key-value"
            assert body["response_format"] == {"type": "json_object"}
            data = {
                "choices": [{"finish_reason": "stop", "message": {"content": '{"query":"SPY"}'}}]
            }
        else:
            assert request.url == "https://compatible.example/v1/messages"
            assert request.headers["x-api-key"] == "private-key-value"
            assert request.headers["anthropic-version"] == "2023-06-01"
            assert body["max_tokens"] > 0 and "system" in body
            data = {
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": '{"query":"SPY"}'}],
            }
        return httpx.Response(200, json=data)

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        adapter = HttpResearchLLM(
            settings(provider), client=client, base_url="https://compatible.example/v1"
        )
        assert json.loads(adapter.generate("plan", {"schema": {}})) == {"query": "SPY"}
        assert "private-key-value" not in repr(adapter)
        assert not client.is_closed


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize(
    "data", [{}, {"choices": []}, {"content": []}, {"choices": [{"finish_reason": "length"}]}]
)
def test_bad_envelope_or_truncated_response_is_rejected(provider, data):
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=data))
    ) as client:
        with pytest.raises(LLMResponseError, match="invalid"):
            HttpResearchLLM(settings(provider), client=client).generate("plan", {})


@pytest.mark.parametrize("status", [302, 401, 429, 500])
def test_http_errors_do_not_include_raw_response_or_key(status):
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(status, text="private-key-value person@example.com")
        )
    ) as client:
        with pytest.raises(LLMResponseError) as error:
            HttpResearchLLM(settings(), client=client).generate("plan", {})
        assert "private-key-value" not in str(error.value)
        assert "person@example.com" not in str(error.value)
        assert str(status) in str(error.value)


def test_network_failure_does_not_expose_exception():
    def fail(request):
        raise httpx.ConnectError("private-key-value", request=request)

    with httpx.Client(transport=httpx.MockTransport(fail)) as client:
        with pytest.raises(LLMResponseError) as error:
            HttpResearchLLM(settings(), client=client).generate("plan", {})
        assert "private-key-value" not in str(error.value)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("llm_api_key", None, "LLM_API_KEY"),
        ("llm_api_key", " ", "LLM_API_KEY"),
        ("llm_model", None, "LLM_MODEL"),
        ("llm_provider", "unsupported", "LLM_PROVIDER"),
    ],
)
def test_missing_configuration_is_explicit(field, value, message):
    values = settings().model_dump()
    values[field] = value
    with pytest.raises(LLMConfigurationError, match=message):
        HttpResearchLLM(Settings(_env_file=None, **values))


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/v1",
        "https://key:secret@example.com/v1",
        "https://example.com/v1?api_key=secret",
    ],
)
def test_private_or_insecure_endpoints_are_rejected(url):
    with pytest.raises(LLMConfigurationError):
        HttpResearchLLM(settings(), base_url=url)
