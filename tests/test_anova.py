"""고정 합성 독립 표본의 ANOVA·진단·조건부 사후 검정과 PNG 저장 검증."""

from hashlib import sha256
import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from robo_advisor.experiments import anova


@pytest.fixture(autouse=True)
def headless(monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")


@pytest.fixture
def different():
    rng = np.random.default_rng(44)
    return pd.DataFrame(
        {
            "strategy": np.repeat(["DRL", "MVO", "equal"], 30),
            "reward": np.repeat(["return", "sharpe", "penalty"], 30),
            "score": np.concatenate([rng.normal(mean, 0.4, 30) for mean in (0.0, 2.0, 4.0)]),
            "seed": np.tile(np.arange(30), 3),
        }
    )


@pytest.fixture
def same():
    values = np.random.default_rng(45).normal(size=30)
    return pd.DataFrame(
        {"strategy": np.repeat(["DRL", "MVO", "equal"], 30), "score": np.tile(values, 3)}
    )


@pytest.fixture
def interaction():
    rng = np.random.default_rng(46)
    rows = []
    for strategy in ("DRL", "MVO"):
        for regime in ("bull", "bear"):
            mean = 0.0 if (strategy == "DRL") == (regime == "bull") else 5.0
            for value in rng.normal(mean, 0.2, 25):
                rows.append({"strategy": strategy, "regime": regime, "score": value})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("fixture", ["different", "same"])
def test_one_way_matches_scipy_and_manual_effect_size(request, fixture):
    samples = request.getfixturevalue(fixture)
    result = anova.strategy_anova(samples, "score")
    groups = [group.score.to_numpy() for _, group in samples.groupby("strategy")]
    expected = stats.f_oneway(*groups)
    effect = result.effects.set_index("effect_id").loc["factor_0"]
    total = ((samples.score - samples.score.mean()) ** 2).sum()
    between = sum(len(values) * (values.mean() - samples.score.mean()) ** 2 for values in groups)
    assert effect.F == pytest.approx(expected.statistic, rel=1e-10, abs=1e-20)
    assert effect.p == pytest.approx(expected.pvalue)
    assert effect.eta_squared == pytest.approx(between / total, rel=1e-10, abs=1e-20)
    assert effect.df == 2
    assert result.effects.set_index("effect_id").loc["residual", "df"] == len(samples) - 3
    assert result.group_counts.n.tolist() == [30, 30, 30]
    assert result.alpha == 0.05 and result.metric == "score"


def test_significant_effect_runs_tukey_with_exact_values_and_effect_size(different):
    result = anova.reward_anova(different, "score", alpha=0.01)
    reference = pairwise_tukeyhsd(different.score, different.reward, alpha=0.01)
    effect = result.effects.iloc[0]
    assert effect.tukey_performed and effect.tukey_reason == "performed"
    assert len(result.tukey) == 3
    np.testing.assert_allclose(result.tukey.meandiff, reference.meandiffs)
    np.testing.assert_allclose(result.tukey.p_adj, reference.pvalues)
    np.testing.assert_allclose(result.tukey[["ci_lower", "ci_upper"]], reference.confint)
    np.testing.assert_array_equal(result.tukey.reject, reference.reject)
    assert (result.tukey.eta_squared == effect.eta_squared).all()
    assert set(result.tukey.eta_squared_label) == {"large"}


def test_non_significant_effect_never_calls_tukey(same, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Tukey must be conditional")

    monkeypatch.setattr(anova, "pairwise_tukeyhsd", forbidden)
    result = anova.strategy_anova(same, "score")
    assert result.tukey.empty
    assert not result.effects.iloc[0].tukey_performed
    assert result.effects.iloc[0].tukey_reason == "effect_not_significant"


def test_two_way_detects_interaction_and_compares_cells(interaction):
    result = anova.strategy_regime_anova(interaction, "score")
    effects = result.effects.set_index("effect_id")
    assert effects.loc["interaction", "p"] < 0.001
    assert effects.loc["interaction", "partial_eta_squared"] > 0.9
    pairs = result.tukey.loc[result.tukey.effect_id.eq("interaction")]
    assert len(pairs) == 6
    assert all(len(json.loads(label)) == 2 for label in [*pairs.group1, *pairs.group2])
    assert result.group_counts.n.tolist() == [25] * 4
    for effect in result.effects.itertuples():
        if effect.effect_id == "residual":
            continue
        assert effect.tukey_performed == (effect.p < result.alpha)
        assert len(result.tukey.loc[result.tukey.effect_id.eq(effect.effect_id)]) == (
            (6 if effect.effect_id == "interaction" else 1) if effect.p < result.alpha else 0
        )


def test_unbalanced_two_way_matches_type_two_and_correct_total_ss():
    rng = np.random.default_rng(47)
    rows = []
    for i, size in enumerate((7, 12, 15, 24, 9, 17)):
        left, right = divmod(i, 2)
        for value in rng.normal(2 * left + right + left * right, 0.8, size):
            rows.append({"A": str(left), "B": str(right), "score": value})
    samples = pd.DataFrame(rows)
    result = anova.two_way_anova(samples, "score", "A", "B")
    reference = anova_lm(ols("score ~ C(A) * C(B)", data=samples).fit(), typ=2)
    total = np.square(samples.score - samples.score.mean()).sum()
    assert not np.isclose(reference.sum_sq.sum(), total)
    residual = reference.loc["Residual", "sum_sq"]
    for position, term in enumerate(("C(A)", "C(B)", "C(A):C(B)")):
        actual = result.effects.iloc[position]
        expected = reference.loc[term]
        np.testing.assert_allclose(
            actual[["df", "sum_sq", "F", "p"]].to_numpy(dtype=float),
            expected[["df", "sum_sq", "F", "PR(>F)"]],
        )
        assert actual.eta_squared == pytest.approx(expected.sum_sq / total)
        assert actual.partial_eta_squared == pytest.approx(
            expected.sum_sq / (expected.sum_sq + residual)
        )
    assert result.group_counts.n.tolist() == [7, 12, 15, 24, 9, 17]


def test_diagnostics_match_residual_shapiro_and_median_levene(different):
    result = anova.strategy_anova(different, "score", alpha=0.1)
    residuals = different.score - different.groupby("strategy").score.transform("mean")
    shapiro = stats.shapiro(residuals)
    levene = stats.levene(
        *(group.score for _, group in different.groupby("strategy")), center="median"
    )
    diagnostics = result.diagnostics.set_index("test")
    for name, expected in (("shapiro", shapiro), ("levene", levene)):
        assert diagnostics.loc[name, "statistic"] == pytest.approx(expected.statistic)
        assert diagnostics.loc[name, "p"] == pytest.approx(expected.pvalue)
        assert diagnostics.loc[name, "violated"] == (expected.pvalue < 0.1)
        assert diagnostics.loc[name, "alpha"] == 0.1


def test_assumption_violations_are_reported_and_results_retained():
    rng = np.random.default_rng(48)
    samples = pd.DataFrame(
        {
            "group": np.repeat(["A", "B", "C"], 100),
            "score": np.concatenate(
                [rng.exponential(scale, 100) + i * 20 for i, scale in enumerate((1, 4, 12))]
            ),
        }
    )
    result = anova.one_way_anova(samples, "score", "group")
    assert result.diagnostics.violated.all()
    assert result.effects.iloc[0].p < 0.05 and not result.tukey.empty


def test_two_way_levene_uses_cells(interaction):
    result = anova.strategy_regime_anova(interaction, "score")
    expected = stats.levene(
        *(group.score for _, group in interaction.groupby(["strategy", "regime"])), center="median"
    )
    assert result.diagnostics.set_index("test").loc["levene", "p"] == pytest.approx(expected.pvalue)


def test_undefined_levene_is_explicit_and_two_samples_are_allowed():
    samples = pd.DataFrame({"group": ["A", "A", "B", "B"], "score": [-1.0, 1.0, -10.0, 10.0]})
    result = anova.one_way_anova(samples, "score", "group")
    diagnostic = result.diagnostics.set_index("test").loc["levene"]
    assert not diagnostic.valid and diagnostic.violated is None
    assert "undefined_statistic" in diagnostic.note


def test_shapiro_large_sample_reliability_note():
    samples = pd.DataFrame(
        {
            "group": np.repeat(["A", "B"], 2600),
            "score": np.tile(stats.norm.ppf((np.arange(2600) + 0.5) / 2600), 2),
        }
    )
    result = anova.one_way_anova(samples, "score", "group")
    assert "n_over_5000" in result.diagnostics.set_index("test").loc["shapiro", "note"]


@pytest.mark.parametrize(
    "eta,label",
    [
        (0, "negligible"),
        (0.0099, "negligible"),
        (0.01, "small"),
        (0.06, "medium"),
        (0.14, "large"),
        (1.0, "large"),
    ],
)
def test_effect_size_interpretation_boundaries(eta, label):
    assert anova.effect_size_label(eta) == label


@pytest.mark.parametrize("alpha", [0, 1, -0.1, float("nan"), float("inf"), True, "0.05"])
def test_invalid_alpha(different, alpha):
    with pytest.raises(ValueError, match="alpha"):
        anova.strategy_anova(different, "score", alpha=alpha)


@pytest.mark.parametrize(
    "kind",
    [
        "missing_metric",
        "nan",
        "infinite",
        "string_metric",
        "boolean_metric",
        "complex_metric",
        "missing_factor",
        "empty_factor",
        "infinite_factor",
        "single_group",
        "singleton",
        "empty",
        "duplicate_columns",
        "same_columns",
        "multiple_metrics",
        "constant",
        "perfect_fit",
    ],
)
def test_input_errors_are_explicit(different, kind):
    samples, metric, factor = different.copy(), "score", "strategy"
    if kind == "missing_metric":
        metric = "missing"
    elif kind in {"nan", "infinite"}:
        samples.loc[0, "score"] = np.nan if kind == "nan" else np.inf
    elif kind == "string_metric":
        samples["score"] = samples.score.astype(str)
    elif kind == "boolean_metric":
        samples["score"] = True
    elif kind == "complex_metric":
        samples["score"] = samples.score.astype(complex)
    elif kind == "missing_factor":
        samples.loc[0, factor] = None
    elif kind == "empty_factor":
        samples.loc[0, factor] = " "
    elif kind == "infinite_factor":
        samples[factor] = np.inf
    elif kind == "single_group":
        samples[factor] = "one"
    elif kind == "singleton":
        samples.loc[0, factor] = "singleton"
    elif kind == "empty":
        samples = samples.iloc[:0]
    elif kind == "duplicate_columns":
        samples.columns = ["score"] * len(samples.columns)
    elif kind == "same_columns":
        factor = metric
    elif kind == "multiple_metrics":
        metric = ["score", "seed"]
    elif kind == "constant":
        samples["score"] = 1.0
    else:
        samples["score"] = np.repeat([1.0, 2.0, 3.0], 30)
    with pytest.raises(ValueError):
        anova.one_way_anova(samples, metric, factor)


@pytest.mark.parametrize("kind", ["empty_cell", "singleton_cell", "same_factor", "unused_category"])
def test_two_way_empty_or_small_cells_are_rejected(interaction, kind):
    samples = interaction.copy()
    mask = samples.strategy.eq("DRL") & samples.regime.eq("bull")
    if kind == "empty_cell":
        samples = samples.loc[~mask]
    elif kind == "singleton_cell":
        samples = samples.loc[~mask | (samples.index == 0)]
    elif kind == "unused_category":
        samples.regime = pd.Categorical(samples.regime, categories=["bull", "bear", "unknown"])
    with pytest.raises(ValueError):
        anova.two_way_anova(
            samples, "score", "strategy", "strategy" if kind == "same_factor" else "regime"
        )


def test_unknown_labels_are_kept_and_column_names_are_not_formula_code(different):
    samples = different.rename(
        columns={"score": "value with spaces", "strategy": "factor: with spaces"}
    )
    samples.loc[samples["factor: with spaces"].eq("equal"), "factor: with spaces"] = "unknown"
    original = samples.copy(deep=True)
    result = anova.strategy_anova(samples, "value with spaces", strategy_col="factor: with spaces")
    assert "unknown" in result.group_counts.group.tolist()
    pd.testing.assert_frame_equal(samples, original)


def test_one_metric_only_and_full_input_hash_include_metadata(different):
    first = anova.strategy_anova(different, "score")
    samples = different.assign(other_metric=np.nan)
    second = anova.strategy_anova(samples, "score")
    pd.testing.assert_frame_equal(first.effects, second.effects)
    assert first.input_sha256 != second.input_sha256
    samples = different.copy()
    samples["seed"] += 1
    assert anova.strategy_anova(samples, "score").input_sha256 != first.input_sha256
    assert anova.strategy_anova(different, "score").input_sha256 == first.input_sha256


def test_categorical_factors_and_collision_free_cell_labels(interaction):
    samples = interaction.copy()
    samples.strategy = pd.Categorical(samples.strategy)
    result = anova.strategy_regime_anova(samples, "score")
    assert len(result.input_sha256) == 64
    samples.strategy = samples.strategy.astype(str).replace({"DRL": "a×b", "MVO": "a"})
    samples.regime = samples.regime.replace({"bull": "c", "bear": "b×c"})
    result = anova.strategy_regime_anova(samples, "score")
    assert result.group_counts.group.nunique() == 4


def test_saved_artifacts_hashes_pngs_and_overwrite_refusal(different, tmp_path):
    from matplotlib.image import imread

    result = anova.strategy_anova(different, "score")
    output = anova.save_anova(result, tmp_path / "report")
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["input_sha256"] == result.input_sha256
    assert manifest["metric"] == "score" and manifest["alpha"] == 0.05
    assert manifest["statsmodels_version"] == anova.statsmodels.__version__
    assert manifest["group_counts"] == result.group_counts.to_dict("records")
    assert set(manifest["files"]) == {
        "results.csv",
        "diagnostics.csv",
        "group_counts.csv",
        "tukey.csv",
        "plots/box_factor_0.png",
        "plots/tukey_factor_0.png",
    }
    for filename, digest in manifest["files"].items():
        assert sha256((output / filename).read_bytes()).hexdigest() == digest
        if filename.endswith(".png"):
            image = imread(output / filename)
            assert image.shape[0] > 100 and image.shape[1] > 100 and image.std() > 0.05
    before = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        anova.save_anova(result, output)
    assert (output / "manifest.json").read_bytes() == before
    pd.testing.assert_frame_equal(
        pd.read_csv(output / "tukey.csv").fillna(""), result.tukey.fillna(""), check_dtype=False
    )


def test_non_significant_save_has_header_only_tukey_and_no_ci_plot(same, tmp_path):
    output = anova.save_anova(anova.strategy_anova(same, "score"), tmp_path / "report")
    assert pd.read_csv(output / "tukey.csv").empty
    assert not list((output / "plots").glob("tukey_*.png"))
    assert (output / "plots" / "box_factor_0.png").exists()


def test_two_way_save_has_cell_and_factor_boxplots(interaction, tmp_path):
    output = anova.save_anova(
        anova.strategy_regime_anova(interaction, "score"), tmp_path / "report"
    )
    assert all(
        (output / "plots" / f"box_{name}.png").exists()
        for name in ("factor_0", "factor_1", "interaction")
    )
    assert (output / "plots" / "tukey_interaction.png").exists()


def test_failed_save_cleans_only_new_directory(different, tmp_path, monkeypatch):
    result = anova.strategy_anova(different, "score")
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("keep")

    def fail(*args, **kwargs):
        raise OSError("disk failure")

    monkeypatch.setattr(anova, "_plots", fail)
    with pytest.raises(OSError, match="disk failure"):
        anova.save_anova(result, tmp_path / "report")
    assert not (tmp_path / "report").exists() and sentinel.read_text() == "keep"
