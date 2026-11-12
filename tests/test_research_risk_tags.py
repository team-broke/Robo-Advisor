"""위험 이벤트의 출처 trace·분류·신뢰도와 시점 차단을 검증한다."""

from datetime import timedelta
import json
import socket

import pytest

from robo_advisor.research.contracts import ResearchOptions
from robo_advisor.research.graph import run_research
from robo_advisor.research.risk_tags import classify_event, convert_risk_events
from robo_advisor.schemas.risk import RiskEventType
from tests.test_research_graph import AS_OF, FakeLLM, FakeStore


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)


def research(text="실적 쇼크로 매출 감소를 발표했다."):
    return run_research(
        "SPY",
        AS_OF,
        store=FakeStore(),
        llm=FakeLLM(
            {
                "synthesize": {"claims": [{"text": text, "source_ids": ["one", "two"]}]},
            }
        ),
    )


def event(result, **changes):
    citation = result.claims[0].citations[0]
    return {
        "description": result.claims[0].text,
        "asset": "SPY",
        "severity": 0.8,
        "confidence": 0.9,
        "published_at": citation.published_at,
        "available_at": citation.available_at,
        "source_doc_id": citation.doc_id,
        "source_url": str(citation.url),
        "source_trace_id": result.trace_id,
        **changes,
    }


@pytest.mark.parametrize(
    "text,kind",
    [
        ("실적 쇼크로 매출 감소를 발표했다.", RiskEventType.EARNINGS_SHOCK),
        ("정부가 규제 변경을 발표했다.", RiskEventType.REGULATION_CHANGE),
        ("주가가 급락했다.", RiskEventType.SHARP_MOVE),
        ("지정학 전쟁 위험이 커졌다.", RiskEventType.GEOPOLITICAL),
        ("Earnings miss was reported.", RiskEventType.EARNINGS_SHOCK),
    ],
)
def test_classification_and_cited_risk_tag_fields(text, kind):
    result = research(text)
    converted = convert_risk_events([event(result)], result)
    assert not converted.excluded and len(converted.tags) == 1
    tag = converted.tags[0]
    assert tag.event_type == kind and tag.asset == "SPY"
    assert tag.severity == 0.8 and tag.confidence == 0.9
    assert tag.risk_score == pytest.approx(0.72)
    assert tag.trace_id == result.trace_id
    assert str(tag.source_url) == str(result.claims[0].citations[0].url)
    assert tag.available_at >= tag.published_at
    assert json.loads(converted.model_dump_json())["tags"][0]["trace_id"] == result.trace_id


@pytest.mark.parametrize("confidence,count", [(0.6999, 0), (0.7, 1), (1.0, 1)])
def test_confidence_threshold_boundary(confidence, count):
    result = research()
    converted = convert_risk_events(
        [event(result, confidence=confidence)], result, confidence_threshold=0.7
    )
    assert len(converted.tags) == count
    if not count:
        assert converted.excluded[0].reason == "below_confidence_threshold"


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"source_trace_id": None}, "missing_source_trace"),
        ({"source_trace_id": "different-trace"}, "missing_source_trace"),
        ({"source_doc_id": "invented"}, "missing_source_trace"),
        ({"source_url": "https://example.com/invented"}, "source_mismatch"),
        ({"description": "공시에 없는 규제 변경"}, "unsupported_event"),
        ({"asset": "person@example.com"}, "sensitive_asset"),
    ],
)
def test_untraceable_or_unsupported_candidates_are_excluded(changes, reason):
    result = research()
    converted = convert_risk_events([event(result, **changes)], result)
    assert not converted.tags and converted.excluded[0].reason == reason
    assert "person@example.com" not in converted.model_dump_json()


@pytest.mark.parametrize("missing", ["verify", "synthesize"])
def test_trace_id_alone_without_validated_source_steps_is_insufficient(missing):
    original = research()
    result = original.model_copy(
        update={"trace": tuple(step for step in original.trace if step.step != missing)}
    )
    converted = convert_risk_events([event(result)], result)
    assert not converted.tags and converted.excluded[0].reason == "missing_source_trace"


@pytest.mark.parametrize("text", ["일반 공시가 발표됐다.", "규제 변경 후 주가 급락을 발표했다."])
def test_unclassified_or_ambiguous_events_are_not_guessed(text):
    result = research(text)
    assert classify_event(text) is None
    converted = convert_risk_events([event(result)], result)
    assert not converted.tags and converted.excluded[0].reason == "unclassified_event"


@pytest.mark.leakage
def test_backdated_publication_and_availability_are_excluded():
    result = research()
    citation = result.claims[0].citations[0]
    for changes, reason in (
        ({"published_at": citation.published_at - timedelta(seconds=1)}, "source_time_mismatch"),
        ({"available_at": citation.published_at - timedelta(seconds=1)}, "invalid_availability"),
        ({"available_at": citation.available_at - timedelta(seconds=1)}, "invalid_availability"),
        ({"available_at": AS_OF + timedelta(seconds=1)}, "unavailable_at_cutoff"),
    ):
        converted = convert_risk_events([event(result, **changes)], result)
        assert not converted.tags and converted.excluded[0].reason == reason


@pytest.mark.leakage
def test_cutoff_cannot_extend_original_research_horizon():
    result = research()
    converted = convert_risk_events(
        [event(result, available_at=AS_OF + timedelta(seconds=1))],
        result,
        as_of=AS_OF + timedelta(days=1),
    )
    assert converted.as_of == AS_OF and not converted.tags
    early = convert_risk_events(
        [event(result)], result, as_of=result.claims[0].citations[0].published_at
    )
    assert not early.tags and early.excluded[0].reason == "unavailable_at_cutoff"
    boundary = convert_risk_events(
        [event(result)], result, as_of=result.claims[0].citations[0].available_at
    )
    assert len(boundary.tags) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"severity": -0.1},
        {"confidence": 1.1},
        {"severity": float("nan")},
        {"confidence": True},
        {"published_at": "2026-01-05"},
        {"source_url": "invalid"},
    ],
)
def test_invalid_events_are_reported_without_aborting_other_events(changes):
    result = research()
    converted = convert_risk_events([event(result, **changes), event(result)], result)
    assert len(converted.tags) == 1
    assert converted.excluded[0].index == 0 and converted.excluded[0].reason == "invalid_event"


def test_failed_research_does_not_emit_tags_and_duplicates_are_removed():
    result = research()
    candidate = event(result)
    failed = run_research(
        "SPY", AS_OF, store=FakeStore([[]]), llm=FakeLLM(), options=ResearchOptions(max_retries=0)
    )
    converted = convert_risk_events([candidate], failed)
    assert not converted.tags and converted.excluded[0].reason == "failed_research"
    duplicate = convert_risk_events([candidate, candidate], result)
    assert len(duplicate.tags) == 1 and duplicate.excluded[0].reason == "duplicate_tag"


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), True])
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(ValueError):
        convert_risk_events([], research(), confidence_threshold=threshold)
