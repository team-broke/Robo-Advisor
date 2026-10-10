"""RSI/MACD 기술지표 테스트."""

import numpy as np
import pandas as pd
import pytest


from robo_advisor.features.indicators import calculate_macd, calculate_rsi


def test_rsi_manual_reference_values():
    """작은 예제로 RSI 기준값을 직접 검증한다."""
    close = pd.Series([10.0, 11.0, 10.0, 12.0, 11.0, 13.0])

    result = calculate_rsi(close, period=3)

    expected = pd.Series(
        [
            np.nan,
            np.nan,
            np.nan,
            83.33333333333333,
            60.60606060606061,
            78.33333333333333,
        ],
        name="rsi",
    )

    np.testing.assert_allclose(
        result.to_numpy(),
        expected.to_numpy(),
        rtol=1e-10,
        atol=1e-10,
        equal_nan=True,
    )


def test_macd_manual_reference_values():
    """작은 예제로 일반 MACD 기준값을 직접 검증한다."""
    close = pd.Series([1.0, 2.0, 3.0, 4.0])

    result = calculate_macd(
        close,
        fast=2,
        slow=3,
        signal=2,
        normalize=False,
    )

    expected_macd = np.array(
        [
            0.0,
            0.16666666666666674,
            0.30555555555555536,
            0.39351851851851816,
        ]
    )
    expected_signal = np.array(
        [
            0.0,
            0.11111111111111116,
            0.24074074074074063,
            0.3425925925925923,
        ]
    )

    np.testing.assert_allclose(result["macd"], expected_macd, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        result["macd_signal"],
        expected_signal,
        rtol=1e-10,
        atol=1e-10,
    )


def test_macd_is_normalized_by_price_by_default():
    """프로젝트 v5 규격에서는 MACD를 가격으로 정규화한다."""
    close = pd.Series([1.0, 2.0, 3.0, 4.0])

    raw = calculate_macd(close, fast=2, slow=3, signal=2, normalize=False)
    normalized = calculate_macd(close, fast=2, slow=3, signal=2)

    np.testing.assert_allclose(normalized["macd"], raw["macd"] / close)
    np.testing.assert_allclose(
        normalized["macd_signal"],
        raw["macd_signal"] / close,
    )


@pytest.mark.leakage
def test_rsi_does_not_reference_future_values():
    """미래 가격을 바꿔도 과거 RSI가 변하지 않아야 한다."""
    close = pd.Series(np.linspace(100.0, 130.0, 80))
    changed = close.copy()

    original = calculate_rsi(close, period=14)

    # 50번째 이후의 미래 값만 크게 변경한다.
    changed.iloc[50:] = changed.iloc[50:] * 10.0 + 500.0
    modified = calculate_rsi(changed, period=14)

    pd.testing.assert_series_equal(
        original.iloc[:50],
        modified.iloc[:50],
        check_exact=True,
    )


@pytest.mark.leakage
def test_macd_does_not_reference_future_values():
    """미래 가격을 바꿔도 과거 MACD와 Signal은 변하지 않아야 한다."""
    close = pd.Series(np.linspace(100.0, 150.0, 80))
    changed = close.copy()

    original = calculate_macd(close)

    # 50번째 이후 미래 값만 변경
    changed.iloc[50:] = changed.iloc[50:] * 10.0 + 500.0
    modified = calculate_macd(changed)

    pd.testing.assert_frame_equal(
        original.iloc[:50],
        modified.iloc[:50],
        check_exact=True,
    )
