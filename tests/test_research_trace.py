"""공개 실행 기록에서 비밀·연락처·토큰을 제거한다."""

import json
from urllib.parse import quote

import pytest

from robo_advisor.research.trace import TraceStep, public_url, redact


@pytest.mark.parametrize(
    "secret",
    [
        "person@example.com",
        "010-1234-5678",
        "01012345678",
        "+1 415 555 2671",
        "sk-ant-api03-abcdef-secret",
        "ghp_abcdef123456",
        "Bearer opaque-secret",
        "eyJhbGciOiJIUzI1NiJ9.payload.signature",
        "api_key=my-private-key",
        "abcdefghijklmnopqrstuvwxyz0123456789ABC",
    ],
)
def test_text_redacts_sensitive_patterns(secret):
    assert secret not in redact(f"질의 {secret} SPY")


def test_nested_json_and_registered_keys_are_redacted():
    secret = "a secret/with+punctuation"
    value = {
        "api_key": "arbitrary",
        "headers": {"Authorization": "opaque"},
        "query": [secret, quote(secret, safe=""), "person@example.com"],
        "retries": 2,
    }
    result = redact(value, secrets=(secret,))
    assert result["api_key"] == "[redacted]"
    assert result["headers"]["Authorization"] == "[redacted]"
    assert result["retries"] == 2
    assert secret not in json.dumps(result)
    assert all(item == "[redacted]" for item in result["query"])


def test_public_urls_drop_sensitive_query_and_fragment():
    assert (
        public_url("https://example.com/doc?id=3&token=secret&email=person%40example.com#private")
        == "https://example.com/doc?id=3"
    )
    with pytest.raises(ValueError):
        public_url("https://person:password@example.com/doc")


def test_document_hashes_dates_and_trace_metadata_survive():
    doc_id = "abc12345" * 8
    assert redact(doc_id) == doc_id
    assert redact("2026-01-05T01:00:00+00:00") == "2026-01-05T01:00:00+00:00"
    step = TraceStep(
        step="verify",
        tool="llm",
        query="SPY",
        source_ids=(doc_id,),
        validation=("passed",),
        retries=1,
        elapsed_ms=0.5,
    )
    assert json.loads(step.model_dump_json())["source_ids"] == [doc_id]
    with pytest.raises(ValueError):
        TraceStep(**{**step.model_dump(), "reasoning": "private"})


def test_trace_id_is_preserved_and_sensitive_url_paths_are_rejected():
    trace_id = "c8fc4067-e3b1-4d71-9c84-682fd346af90"
    assert redact(trace_id) == trace_id
    with pytest.raises(ValueError):
        public_url("https://example.com/person%40example.com")


@pytest.mark.parametrize("value", ["token=short-private", '"api_key": "short-private"'])
def test_inline_labeled_secrets_are_redacted(value):
    assert "short-private" not in redact(value)
