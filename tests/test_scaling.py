"""Train 전용 정규화의 계산, Walk-Forward 사용 및 미래 정보 누수 검증."""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

from robo_advisor.backtest.walk_forward import walk_forward_split
from robo_advisor.features.scaling import TrainOnlyStandardScaler


@pytest.fixture
def train_features():
    return pd.DataFrame(
        {"A": [1.0, 2.0, 3.0], "B": [10.0, 20.0, 30.0]},
        index=pd.date_range("2020-01-01", periods=3, name="date"),
    )


def test_z_score_manual_reference_values(train_features):
    """평균 [2, 20], 분산 [2/3, 200/3] 기준으로 직접 검증한다."""
    scaler = TrainOnlyStandardScaler()
    assert scaler.fit(train_features) is scaler

    result = scaler.transform(train_features)
    expected = pd.DataFrame(
        {"A": [-np.sqrt(1.5), 0.0, np.sqrt(1.5)], "B": [-np.sqrt(1.5), 0.0, np.sqrt(1.5)]},
        index=train_features.index,
    )

    np.testing.assert_allclose(scaler.mean_, [2.0, 20.0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        scaler.scale_, np.sqrt([2.0 / 3.0, 200.0 / 3.0]), rtol=1e-12, atol=1e-12
    )
    pd.testing.assert_frame_equal(result, expected, rtol=1e-12, atol=1e-12)


def test_transformed_train_mean_is_zero(train_features):
    scaler = TrainOnlyStandardScaler().fit(train_features)

    np.testing.assert_allclose(scaler.transform(train_features).mean(), 0.0, atol=1e-12)


def test_transformed_train_population_std_is_one(train_features):
    scaler = TrainOnlyStandardScaler().fit(train_features)

    np.testing.assert_allclose(
        scaler.transform(train_features).std(ddof=0), 1.0, rtol=1e-12, atol=1e-12
    )


def test_fit_transform_matches_standard_scaler(train_features):
    original = train_features.copy(deep=True)
    scaler = TrainOnlyStandardScaler()
    reference = StandardScaler().fit(train_features)

    result = scaler.fit_transform(train_features)

    np.testing.assert_allclose(result, reference.transform(train_features))
    np.testing.assert_allclose(scaler.mean_, reference.mean_)
    np.testing.assert_allclose(scaler.scale_, reference.scale_)
    pd.testing.assert_index_equal(result.index, train_features.index)
    pd.testing.assert_index_equal(result.columns, train_features.columns)
    pd.testing.assert_frame_equal(train_features, original, check_exact=True)


def test_future_test_values_do_not_change_training_statistics(train_features):
    """Test만 극단적으로 바꿔도 두 scaler의 통계와 Train 결과는 같아야 한다."""
    test_a = pd.DataFrame({"A": [4.0, 5.0], "B": [40.0, 50.0]})
    test_b = test_a * 1000.0 + 1_000_000.0
    scalers = []
    train_results = []
    test_results = []

    for test in (test_a, test_b):
        scaler = TrainOnlyStandardScaler().fit(train_features)
        mean_before, scale_before = scaler.mean_, scaler.scale_
        train_before = scaler.transform(train_features)
        test_results.append(scaler.transform(test))

        np.testing.assert_array_equal(scaler.mean_, mean_before)
        np.testing.assert_array_equal(scaler.scale_, scale_before)
        pd.testing.assert_frame_equal(
            scaler.transform(train_features), train_before, check_exact=True
        )
        scalers.append(scaler)
        train_results.append(train_before)

    np.testing.assert_array_equal(scalers[0].mean_, scalers[1].mean_)
    np.testing.assert_array_equal(scalers[0].scale_, scalers[1].scale_)
    np.testing.assert_allclose(scalers[0].mean_, [2.0, 20.0])
    np.testing.assert_allclose(scalers[0].scale_, np.sqrt([2.0 / 3.0, 200.0 / 3.0]))
    pd.testing.assert_frame_equal(train_results[0], train_results[1], check_exact=True)
    assert not test_results[0].equals(test_results[1])


def test_validation_and_test_only_transform(train_features, monkeypatch):
    """평가 구간 변환 중 fit/fit_transform이 호출되면 즉시 실패한다."""
    scaler = TrainOnlyStandardScaler().fit(train_features)
    mean_before, scale_before = scaler.mean_, scaler.scale_
    validation = pd.DataFrame({"A": [4.0, 5.0], "B": [40.0, 50.0]})
    test = pd.DataFrame({"A": [6.0, 7.0], "B": [60.0, 70.0]})

    def fail_on_fit(*args, **kwargs):
        pytest.fail("Validation/Test must only use transform")

    for cls in (TrainOnlyStandardScaler, StandardScaler):
        monkeypatch.setattr(cls, "fit", fail_on_fit)
        monkeypatch.setattr(cls, "fit_transform", fail_on_fit)

    for evaluation in (validation, test):
        original = evaluation.copy(deep=True)
        expected = (evaluation - [2.0, 20.0]) / np.sqrt([2.0 / 3.0, 200.0 / 3.0])

        result = scaler.transform(evaluation)

        pd.testing.assert_frame_equal(result, expected, rtol=1e-12, atol=1e-12)
        assert (result.mean() > 0.0).all()
        np.testing.assert_array_equal(scaler.mean_, mean_before)
        np.testing.assert_array_equal(scaler.scale_, scale_before)
        pd.testing.assert_frame_equal(evaluation, original, check_exact=True)


def test_each_walk_forward_window_fits_its_own_train_only():
    """분할기의 Train 위치만 fit하며 이전 Window 통계를 재사용하지 않는다."""
    dates = pd.date_range("2020-01-01", "2025-12-31")
    values = np.arange(len(dates), dtype=float)
    features = pd.DataFrame({"A": values, "B": values**2}, index=dates)
    windows = walk_forward_split(dates, start_date="2020-01-01", warmup_periods=20)
    means = []

    for window in windows:
        train = features.iloc[window.train]
        scaler = TrainOnlyStandardScaler().fit(train)
        mean_before, scale_before = scaler.mean_, scaler.scale_
        train_before = scaler.transform(train)
        changed = features.copy()
        changed.iloc[window.test] = changed.iloc[window.test] * 1000.0 + 1_000_000.0
        changed_scaler = TrainOnlyStandardScaler().fit(changed.iloc[window.train])

        np.testing.assert_allclose(scaler.mean_, train.mean())
        np.testing.assert_allclose(scaler.scale_, train.std(ddof=0))
        for evaluation in (features.iloc[window.test], features.iloc[window.warmup]):
            expected = (evaluation - train.mean()) / train.std(ddof=0)
            pd.testing.assert_frame_equal(scaler.transform(evaluation), expected)
        scaler.transform(changed.iloc[window.test])

        np.testing.assert_array_equal(scaler.mean_, mean_before)
        np.testing.assert_array_equal(scaler.scale_, scale_before)
        np.testing.assert_array_equal(scaler.mean_, changed_scaler.mean_)
        np.testing.assert_array_equal(scaler.scale_, changed_scaler.scale_)
        pd.testing.assert_frame_equal(scaler.transform(train), train_before, check_exact=True)
        pd.testing.assert_frame_equal(
            changed_scaler.transform(train), train_before, check_exact=True
        )
        means.append(scaler.mean_)

    assert not np.array_equal(means[0], means[1])


def test_refit_replaces_previous_training_statistics(train_features):
    scaler = TrainOnlyStandardScaler().fit(train_features)
    next_train = train_features * 10.0 + 100.0

    result = scaler.fit_transform(next_train)

    np.testing.assert_allclose(scaler.mean_, [120.0, 300.0])
    np.testing.assert_allclose(scaler.scale_, 10.0 * np.sqrt([2.0 / 3.0, 200.0 / 3.0]))
    np.testing.assert_allclose(result.mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(result.std(ddof=0), 1.0, atol=1e-12)


def test_transform_before_fit_raises(train_features):
    with pytest.raises(NotFittedError):
        TrainOnlyStandardScaler().transform(train_features)


@pytest.mark.parametrize("attribute", ["mean_", "scale_"])
def test_statistics_before_fit_raise(attribute):
    with pytest.raises(NotFittedError):
        getattr(TrainOnlyStandardScaler(), attribute)


@pytest.mark.parametrize("columns", [["B", "A"], ["A"], ["A", "C"], ["A", "B", "C"]])
def test_transform_rejects_changed_columns(train_features, columns):
    scaler = TrainOnlyStandardScaler().fit(train_features)
    evaluation = pd.DataFrame([[1.0] * len(columns)], columns=columns)

    with pytest.raises(ValueError, match="columns must match"):
        scaler.transform(evaluation)


@pytest.mark.parametrize("method", ["fit", "transform"])
@pytest.mark.parametrize(
    "features,error,match",
    [
        (np.array([[1.0, 2.0]]), TypeError, "pandas DataFrame"),
        (pd.Series([1.0, 2.0]), TypeError, "pandas DataFrame"),
        (pd.DataFrame(columns=["A", "B"]), ValueError, "must not be empty"),
        (pd.DataFrame(index=[0]), ValueError, "must not be empty"),
        (pd.DataFrame([[1.0, 2.0]], columns=["A", "A"]), ValueError, "must be unique"),
        (pd.DataFrame({"A": [np.inf], "B": [1.0]}), ValueError, "inf or -inf"),
        (pd.DataFrame({"A": [-np.inf], "B": [1.0]}), ValueError, "inf or -inf"),
        (pd.DataFrame({"A": ["invalid"], "B": [1.0]}), ValueError, None),
        (pd.DataFrame({"A": [1.0 + 2.0j], "B": [1.0]}), TypeError, "real numeric"),
    ],
)
def test_invalid_features_raise(train_features, method, features, error, match):
    scaler = TrainOnlyStandardScaler().fit(train_features)

    with pytest.raises(error, match=match):
        getattr(scaler, method)(features)


def test_nan_is_ignored_during_fit_and_preserved_during_transform():
    train = pd.DataFrame({"A": [np.nan, 1.0, 3.0], "B": [10.0, np.nan, 30.0]})
    scaler = TrainOnlyStandardScaler().fit(train)
    reference = StandardScaler().fit(train)

    result = scaler.transform(train)
    evaluation = pd.DataFrame({"A": [np.nan, 5.0], "B": [np.nan, 50.0]})
    evaluation_result = scaler.transform(evaluation)

    np.testing.assert_allclose(scaler.mean_, [2.0, 20.0])
    np.testing.assert_allclose(scaler.scale_, [1.0, 10.0])
    np.testing.assert_allclose(result, reference.transform(train), equal_nan=True)
    np.testing.assert_allclose(evaluation_result, reference.transform(evaluation), equal_nan=True)
    pd.testing.assert_frame_equal(result.isna(), train.isna())
    pd.testing.assert_frame_equal(evaluation_result.isna(), evaluation.isna())
    np.testing.assert_allclose(scaler.mean_, [2.0, 20.0])
    np.testing.assert_allclose(scaler.scale_, [1.0, 10.0])


def test_fit_rejects_training_column_without_observations():
    train = pd.DataFrame({"A": [np.nan, np.nan], "B": [1.0, 2.0]})

    with pytest.raises(ValueError, match="non-NaN value"):
        TrainOnlyStandardScaler().fit(train)


def test_constant_feature_uses_unit_scale(train_features):
    train = train_features.assign(B=10.0)
    scaler = TrainOnlyStandardScaler().fit(train)
    evaluation = pd.DataFrame({"A": [4.0, 5.0], "B": [10.0, 15.0]})
    reference = StandardScaler().fit(train)

    assert scaler.scale_[1] == 1.0
    np.testing.assert_array_equal(scaler.transform(train)["B"], 0.0)
    np.testing.assert_allclose(scaler.transform(evaluation), reference.transform(evaluation))


def test_exposed_statistics_cannot_mutate_fitted_scaler(train_features):
    scaler = TrainOnlyStandardScaler().fit(train_features)
    expected = scaler.transform(train_features)

    scaler.mean_[:] = 1_000_000.0
    scaler.scale_[:] = 1.0

    np.testing.assert_allclose(scaler.mean_, [2.0, 20.0])
    np.testing.assert_allclose(scaler.scale_, np.sqrt([2.0 / 3.0, 200.0 / 3.0]))
    pd.testing.assert_frame_equal(scaler.transform(train_features), expected, check_exact=True)
