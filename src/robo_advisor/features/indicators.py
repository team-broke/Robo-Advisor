"""RSI와 MACD 기술지표 계산 모듈.

모든 계산은 현재 시점과 과거 값만 사용한다.
프로젝트 v5 Feature 규격에 맞춰 MACD는 기본적으로 가격으로 정규화한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_close(close: pd.Series) -> pd.Series:
    """종가 입력을 float Series로 검증해 반환한다."""
    if not isinstance(close, pd.Series):
        raise TypeError("close must be a pandas Series")
    if close.empty:
        raise ValueError("close must not be empty")

    values = close.astype(float)

    if not np.isfinite(values.dropna().to_numpy()).all():
        raise ValueError("close contains inf or -inf")

    return values


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder 방식에 가까운 EWM 기반 RSI를 계산한다.

    Parameters
    ----------
    close:
        시간순으로 정렬된 종가 Series.
    period:
        RSI 기간. 기본값은 14.

    Returns
    -------
    pandas.Series
        0~100 범위의 RSI. 충분한 과거 데이터가 없는 초반 구간은 NaN이다.

    Notes
    -----
    - ``ewm(alpha=1/period, adjust=False)``를 사용한다.
    - 미래 값을 shift(-1), bfill 등으로 참조하지 않는다.
    """
    close = _validate_close(close)

    if period <= 0:
        raise ValueError("period must be greater than 0")

    delta = close.diff()

    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()
    avg_loss = loss.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    relative_strength = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + relative_strength))

    # 분모가 0인 명확한 경계 사례를 안정적으로 처리한다.
    only_gain = (avg_gain > 0.0) & (avg_loss == 0.0)
    only_loss = (avg_gain == 0.0) & (avg_loss > 0.0)
    no_move = (avg_gain == 0.0) & (avg_loss == 0.0)

    rsi = rsi.mask(only_gain, 100.0)
    rsi = rsi.mask(only_loss, 0.0)
    rsi = rsi.mask(no_move, 50.0)

    rsi.name = "rsi"
    return rsi


def calculate_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    normalize: bool = True,
) -> pd.DataFrame:
    """MACD와 Signal line을 계산한다.

    Parameters
    ----------
    close:
        시간순으로 정렬된 종가 Series.
    fast:
        단기 EMA 기간. 기본값 12.
    slow:
        장기 EMA 기간. 기본값 26.
    signal:
        MACD Signal EMA 기간. 기본값 9.
    normalize:
        True이면 프로젝트 v5 규격처럼 MACD와 Signal을 현재 가격으로 나눈다.
        False이면 일반적인 원 단위 MACD를 반환한다.

    Returns
    -------
    pandas.DataFrame
        ``macd``와 ``macd_signal`` 두 컬럼.

    Notes
    -----
    EMA는 ``adjust=False``를 사용하며 현재 시점까지의 데이터만 참조한다.
    """
    close = _validate_close(close)

    if fast <= 0 or slow <= 0 or signal <= 0:
        raise ValueError("fast, slow and signal must be greater than 0")
    if fast >= slow:
        raise ValueError("fast must be smaller than slow")

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    macd_raw = ema_fast - ema_slow
    signal_raw = macd_raw.ewm(span=signal, adjust=False).mean()

    if normalize:
        denominator = close.replace(0.0, np.nan)
        macd = macd_raw / denominator
        macd_signal = signal_raw / denominator
    else:
        macd = macd_raw
        macd_signal = signal_raw

    return pd.DataFrame(
        {
            "macd": macd,
            "macd_signal": macd_signal,
        },
        index=close.index,
    )
