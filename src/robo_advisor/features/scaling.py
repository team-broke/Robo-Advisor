"""학습 구간 전용 Z-score 정규화.

기술지표 계산 시 과거 관측치를 참조하는 것은 허용된다. 하지만 scaler의
mean/std를 학습하는 fit 범위는 반드시 해당 Walk-Forward Window의 Train
기간으로 제한한다. Validation/Test는 미래 평가 구간이므로 transform만 한다.
각 Window에서 scaler를 새로 만들고 해당 Train으로 fit해야 하며, 이전 Window의
학습 통계를 다음 Window에 그대로 재사용하면 안 된다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


def _validate_features(features: pd.DataFrame) -> pd.DataFrame:
    """NaN은 허용하고 무한대는 거부하는 float DataFrame을 반환한다."""
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    if features.empty:
        raise ValueError("features must not be empty")
    if not features.columns.is_unique:
        raise ValueError("feature columns must be unique")
    if np.iscomplexobj(features.to_numpy()):
        raise TypeError("features must contain real numeric values")

    values = features.astype(float)
    if np.isinf(values.to_numpy()).any():
        raise ValueError("features contain inf or -inf")
    return values


class TrainOnlyStandardScaler:
    """DataFrame의 index/columns를 보존하는 StandardScaler wrapper.

    호출자가 Train 구간을 선택해 ``fit`` 또는 ``fit_transform``에 전달한다.
    이 클래스는 입력만으로 Train/Validation/Test 구분을 추론하지 않는다.
    ``transform``은 학습 통계를 변경하지 않고 동일한 컬럼 순서를 요구한다.

    StandardScaler와 동일하게 population variance(ddof=0)를 사용한다.
    지표 워밍업 NaN은 fit 통계에서 제외하고 transform 결과에 보존하며,
    별도 보정은 하지 않는다. Train 전체가 NaN인 컬럼은 통계를 정의할 수
    없어 거부한다. 상수 컬럼의 scale은 1이다.

    Examples
    --------
    각 Walk-Forward Window에서 다음 순서로 사용한다::

        scaler = TrainOnlyStandardScaler()
        scaler.fit(features.iloc[window.train])
        train_scaled = scaler.transform(features.iloc[window.train])
        validation_scaled = scaler.transform(validation_features)
        test_scaled = scaler.transform(features.iloc[window.test])
    """

    def __init__(self):
        self._scaler = StandardScaler()
        self._columns: pd.Index | None = None

    def fit(self, train_features: pd.DataFrame) -> TrainOnlyStandardScaler:
        """해당 Window의 Train만으로 통계를 학습하고 self를 반환한다.

        다시 호출하면 이전 통계를 대체한다. Validation/Test나 전체 기간을
        합친 데이터를 전달해서는 안 된다.
        """
        values = _validate_features(train_features)
        if values.isna().all().any():
            raise ValueError("each training feature must contain a non-NaN value")

        scaler = StandardScaler().fit(values.to_numpy())
        self._scaler = scaler
        self._columns = values.columns.copy()
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Train에서 학습한 통계만 적용하며 fit 이전에는 NotFittedError를 낸다."""
        check_is_fitted(self._scaler)
        values = _validate_features(features)
        if not values.columns.equals(self._columns):
            raise ValueError("feature columns must match fitted columns in the same order")

        scaled = self._scaler.transform(values.to_numpy())
        return pd.DataFrame(scaled, index=features.index, columns=features.columns)

    def fit_transform(self, train_features: pd.DataFrame) -> pd.DataFrame:
        """Train에만 fit한 뒤 같은 Train을 변환한다. 평가 구간에는 사용하지 않는다."""
        return self.fit(train_features).transform(train_features)

    @property
    def mean_(self) -> np.ndarray:
        """Train에서 학습한 feature별 평균의 복사본."""
        check_is_fitted(self._scaler)
        return self._scaler.mean_.copy()

    @property
    def scale_(self) -> np.ndarray:
        """Train에서 학습한 feature별 scale의 복사본."""
        check_is_fitted(self._scaler)
        return self._scaler.scale_.copy()
