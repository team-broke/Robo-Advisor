"""Tests for shared domain schema contracts."""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import TypeAdapter, ValidationError

from robo_advisor.schemas.common import (
    WEIGHT_SUM_TOLERANCE,
    AssetWeight,
    PortfolioWeights,
    Source,
    TraceId,
    UTCDateTime,
)


def test_portfolio_accepts_assets_cash_and_floating_point_total() -> None:
    portfolio = PortfolioWeights(
        asset_weights=(
            AssetWeight(asset="AAA", weight=0.1),
            AssetWeight(asset="BBB", weight=0.2),
        ),
        cash_weight=0.7 + WEIGHT_SUM_TOLERANCE / 2,
    )

    assert portfolio.cash_weight > 0
    assert all(item.weight >= 0 for item in portfolio.asset_weights)


@pytest.mark.parametrize(
    ("asset_weights", "cash_weight"),
    [
        ((AssetWeight(asset="AAA", weight=0.5),), -0.1),
        (({"asset": "AAA", "weight": -0.1},), 1.1),
        ((AssetWeight(asset="AAA", weight=0.5),), 0.4),
    ],
)
def test_portfolio_rejects_negative_or_invalid_total_weights(
    asset_weights: tuple[object, ...], cash_weight: float
) -> None:
    with pytest.raises(ValidationError):
        PortfolioWeights(asset_weights=asset_weights, cash_weight=cash_weight)


def test_non_empty_asset_and_trace_id_are_normalized() -> None:
    weight = AssetWeight(asset="  AAA  ", weight=1)
    trace_id = TypeAdapter(TraceId).validate_python("  request-42  ")

    assert weight.asset == "AAA"
    assert trace_id == "request-42"


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_asset_and_trace_id_are_rejected(value: str) -> None:
    with pytest.raises(ValidationError):
        AssetWeight(asset=value, weight=1)
    with pytest.raises(ValidationError):
        TypeAdapter(TraceId).validate_python(value)


def test_aware_datetime_is_normalized_to_utc() -> None:
    korea_time = datetime(2026, 10, 5, 12, tzinfo=timezone(timedelta(hours=9)))

    result = TypeAdapter(UTCDateTime).validate_python(korea_time)

    assert result == datetime(2026, 10, 5, 3, tzinfo=timezone.utc)
    assert result.tzinfo is timezone.utc


def test_naive_datetime_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(UTCDateTime).validate_python(datetime(2026, 10, 5, 12))


def test_source_requires_valid_url_and_keeps_identifiers() -> None:
    source = Source(
        source_url="https://example.com/articles/42",
        source_name="Example News",
        source_id="article-42",
    )

    assert str(source.source_url) == "https://example.com/articles/42"
    assert source.source_name == "Example News"


def test_source_rejects_invalid_url() -> None:
    with pytest.raises(ValidationError):
        Source(source_url="not-a-url")
