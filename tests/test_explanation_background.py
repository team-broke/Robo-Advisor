"""설명 배경집합의 학습 구간 제한과 표집 재현성을 검증한다."""

import numpy as np
import pandas as pd
import pytest


def inputs():
    from robo_advisor.explain.policy import ObservationLayout

    layout = ObservationLayout(("A",), lookback=2)
    index = pd.date_range("2026-01-01", periods=14, tz="UTC")
    # Rows 0..1 and 12..13 are not part of model fit.
    values = np.arange(14 * 7, dtype=float).reshape(14, 7)
    observations = layout.to_frame(values, index=index)
    strata = pd.Series(
        ["outside"] * 2 + ["up"] * 6 + ["down"] * 3 + ["sideways"] + ["future"] * 2, index=index
    )
    return layout, observations, strata


def sample(observations, strata, **overrides):
    from robo_advisor.explain.background import sample_training_background

    schema, _, _ = inputs()
    kwargs = dict(
        layout=schema,
        train_start="2026-01-03T00:00:00Z",
        train_end="2026-01-12T00:00:00Z",
        n_samples=6,
        seed=42,
        expected_strata=("up", "down", "sideways"),
    )
    kwargs.update(overrides)
    return sample_training_background(observations, strata, **kwargs)


def test_stratification_covers_small_strata_without_replacement_and_is_reproducible():
    _, obs, labels = inputs()
    result = sample(obs, labels)
    assert result.strata.value_counts().to_dict() == {"up": 3, "down": 2, "sideways": 1}
    assert len(result.observations) == 6
    assert result.observations.index.is_unique
    assert result.observations.index.is_monotonic_increasing
    assert result.observations.index.min() >= obs.index[2]
    assert result.observations.index.max() <= obs.index[11]
    pd.testing.assert_frame_equal(result.observations, obs.loc[result.observations.index])
    again = sample(obs, labels)
    pd.testing.assert_frame_equal(again.observations, result.observations)
    assert again.sample_id == result.sample_id
    assert len(result.sample_id) == 64
    assert result.train_start == obs.index[2]
    assert result.train_end == obs.index[11]
    before = obs.copy()
    result.observations.iloc[0, 0] = -99
    pd.testing.assert_frame_equal(obs, before)


@pytest.mark.leakage
def test_values_labels_and_rows_outside_fit_do_not_change_background():
    _, obs, labels = inputs()
    expected = sample(obs, labels)
    polluted = obs.copy()
    polluted.iloc[:2] = np.nan
    polluted.iloc[12:] = np.inf
    changed_labels = labels.copy()
    changed_labels.iloc[:2] = None
    changed_labels.iloc[12:] = "NEW-FUTURE-REGIME"
    extra = obs.iloc[-1:].copy()
    extra.index = extra.index + pd.Timedelta(days=1)
    polluted = pd.concat([polluted, extra])
    changed_labels = pd.concat([changed_labels, pd.Series("later", index=extra.index)])
    actual = sample(polluted, changed_labels)
    pd.testing.assert_frame_equal(actual.observations, expected.observations)
    pd.testing.assert_series_equal(actual.strata, expected.strata)
    assert actual.sample_id == expected.sample_id


def test_boundaries_are_inclusive_and_timezone_equivalent():
    _, obs, labels = inputs()
    result = sample(
        obs,
        labels,
        n_samples=10,
        train_start="2026-01-03T09:00:00+09:00",
        train_end="2026-01-12T09:00:00+09:00",
    )
    pd.testing.assert_frame_equal(result.observations, obs.iloc[2:12], check_freq=False)


@pytest.mark.parametrize(
    "change",
    [
        {"n_samples": 2},
        {"n_samples": 11},
        {"n_samples": True},
        {"n_samples": 0},
        {"seed": -1},
        {"seed": True},
        {"train_end": "2026-01-12"},
        {"train_start": "2026-01-13T00:00:00Z"},
    ],
)
def test_invalid_sampling_contract_is_rejected(change):
    _, obs, labels = inputs()
    with pytest.raises((TypeError, ValueError)):
        sample(obs, labels, **change)


def test_only_fit_rows_require_finite_inputs_and_nonempty_strata():
    _, obs, labels = inputs()
    bad = obs.copy()
    bad.iloc[3, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        sample(bad, labels)
    bad_labels = labels.copy()
    bad_labels.iloc[3] = ""
    with pytest.raises(ValueError, match="strat"):
        sample(obs, bad_labels)
    with pytest.raises(ValueError, match="index"):
        sample(obs, labels.iloc[::-1])
    with pytest.raises(ValueError, match="columns"):
        sample(obs.iloc[:, ::-1], labels)
    with pytest.raises(ValueError, match="index"):
        sample(obs.iloc[::-1], labels.iloc[::-1])


def test_sample_identity_changes_with_fit_data_or_seed():
    _, obs, labels = inputs()
    first = sample(obs, labels)
    changed = obs.copy()
    changed.loc[first.observations.index[0], changed.columns[0]] += 1
    assert sample(changed, labels).sample_id != first.sample_id
    assert sample(obs, labels, seed=43).sample_id != first.sample_id


@pytest.mark.leakage
def test_out_of_fit_object_cells_do_not_reject_numeric_fit_rows():
    _, obs, labels = inputs()
    expected = sample(obs, labels)
    changed = obs.astype(object)
    changed.iloc[0, 0] = None
    changed.iloc[-1, 0] = "bad outside fit"
    actual = sample(changed, labels)
    pd.testing.assert_frame_equal(actual.observations, expected.observations)
    assert actual.sample_id == expected.sample_id


def test_expected_strata_cannot_silently_disappear_after_truncated_collection():
    _, obs, labels = inputs()
    # A collector that stops before the final sideways day must fail loudly.
    with pytest.raises(ValueError, match="missing.*strata"):
        sample(obs.iloc[:11], labels.iloc[:11], expected_strata=("up", "down", "sideways"))


def test_selected_labels_and_identity_track_the_corresponding_rows():
    _, obs, labels = inputs()
    result = sample(obs, labels)
    assert result.strata.tolist() == [
        "up" if stamp.day < 9 else "down" if stamp.day < 12 else "sideways"
        for stamp in result.observations.index
    ]
    other_labels = labels.replace({"up": "down", "down": "up"})
    assert sample(obs, other_labels).sample_id != result.sample_id
    # Retain all fit rows: changing an otherwise identical bound must change the ID.
    assert sample(obs, labels, train_start="2026-01-02T12:00:00Z").sample_id != result.sample_id


def test_expected_strata_accepts_a_numpy_catalog():
    _, obs, labels = inputs()
    expected = sample(obs, labels)
    actual = sample(obs, labels, expected_strata=np.array(["up", "down", "sideways"]))
    assert actual.sample_id == expected.sample_id
