"""Tests for risk-event schema contracts."""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from robo_advisor.schemas.risk import RiskEventType, RiskTag

PUBLISHED_AT = datetime(2026, 10, 5, 1, tzinfo=timezone.utc)


def make_risk_tag(**overrides: object) -> RiskTag:
    values: dict[str, object] = {
        "event_type": RiskEventType.EARNINGS_SHOCK,
        "severity": 0.8,
        "confidence": 0.5,
        "asset": "AAA",
        "published_at": PUBLISHED_AT,
        "available_at": PUBLISHED_AT,
        "source_url": "https://example.com/risk/42",
        "trace_id": "request-42",
    }
    values.update(overrides)
    return RiskTag(**values)


@pytest.mark.parametrize("event_type", list(RiskEventType))
def test_all_event_types_are_accepted(event_type: RiskEventType) -> None:
    assert make_risk_tag(event_type=event_type).event_type is event_type


def test_unknown_event_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_risk_tag(event_type="unknown")


@pytest.mark.parametrize("field", ["severity", "confidence"])
@pytest.mark.parametrize("value", [0.0, 1.0])
def test_score_components_include_boundaries(field: str, value: float) -> None:
    assert getattr(make_risk_tag(**{field: value}), field) == value


@pytest.mark.parametrize("field", ["severity", "confidence"])
@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_score_components_reject_values_outside_unit_interval(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        make_risk_tag(**{field: value})


def test_risk_score_is_severity_times_confidence() -> None:
    assert make_risk_tag(severity=0.8, confidence=0.25).risk_score == pytest.approx(0.2)


def test_available_at_may_equal_or_follow_published_at() -> None:
    assert make_risk_tag(available_at=PUBLISHED_AT).available_at == PUBLISHED_AT
    later = PUBLISHED_AT + timedelta(minutes=3)
    assert make_risk_tag(available_at=later).available_at == later


def test_available_at_before_published_at_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_risk_tag(available_at=PUBLISHED_AT - timedelta(microseconds=1))


def test_datetimes_are_normalized_to_utc() -> None:
    korea = timezone(timedelta(hours=9))
    tag = make_risk_tag(
        published_at=datetime(2026, 10, 5, 10, tzinfo=korea),
        available_at=datetime(2026, 10, 5, 10, 3, tzinfo=korea),
    )

    assert tag.published_at == datetime(2026, 10, 5, 1, tzinfo=timezone.utc)
    assert tag.available_at == datetime(2026, 10, 5, 1, 3, tzinfo=timezone.utc)


@pytest.mark.parametrize("field", ["published_at", "available_at"])
def test_naive_datetimes_are_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        make_risk_tag(**{field: datetime(2026, 10, 5, 1)})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_url", "not-a-url"),
        ("asset", "   "),
        ("trace_id", ""),
    ],
)
def test_invalid_required_identity_fields_are_rejected(field: str, value: str) -> None:
    with pytest.raises(ValidationError):
        make_risk_tag(**{field: value})


def test_source_url_is_required() -> None:
    values = make_risk_tag().model_dump()
    del values["source_url"]

    with pytest.raises(ValidationError):
        RiskTag(**values)
