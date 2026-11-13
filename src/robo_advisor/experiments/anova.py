"""독립 표본 long-format 입력의 ANOVA·진단·조건부 Tukey와 결과 저장.

한 번에 주 종속변수 한 열만 분석한다. 한 행의 독립성과 제외 국면 필터는
호출자가 #42 표본 생성 단계에서 확정해야 하며 여기서 표본을 만들거나 버리지 않는다.
결측·비유한 검증은 지정한 지표·요인에 적용하고 나머지 열은 입력 해시에만 포함한다.
Type II SS는 불균형 설계에서 가산적이지 않으므로 eta²의 분모는 관측값의
수정 총제곱합이다. partial eta²는 효과 SS / (효과 SS + 잔차 SS)로 계산한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from itertools import combinations, product
import json
from numbers import Real
from pathlib import Path
import shutil
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype
from scipy import stats
import statsmodels
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from robo_advisor.data.snapshot import FINGERPRINT_FORMAT, dataframe_fingerprint

_TUKEY_COLUMNS = [
    "effect_id",
    "effect",
    "group1",
    "group2",
    "meandiff",
    "ci_lower",
    "ci_upper",
    "p_adj",
    "reject",
    "eta_squared",
    "eta_squared_label",
    "partial_eta_squared",
    "partial_eta_squared_label",
]


@dataclass(frozen=True)
class AnovaResult:
    """효과·가정 진단·표본 수·사후 검정과 재현 가능한 입력 계보."""

    metric: str
    factors: tuple[str, ...]
    alpha: float
    effects: pd.DataFrame
    diagnostics: pd.DataFrame
    group_counts: pd.DataFrame
    tukey: pd.DataFrame
    input_sha256: str
    total_sum_sq: float
    _data: pd.DataFrame = field(repr=False)
    _levels: tuple[tuple[str, ...], ...] = field(repr=False)


def effect_size_label(eta_squared: float) -> str:
    """0.01·0.06·0.14 경계에서 negligible·small·medium·large를 구분한다."""
    if (
        not isinstance(eta_squared, Real)
        or not np.isfinite(eta_squared)
        or not 0 <= eta_squared <= 1
    ):
        raise ValueError("eta_squared must be finite and in [0, 1]")
    if eta_squared < 0.01:
        return "negligible"
    if eta_squared < 0.06:
        return "small"
    return "medium" if eta_squared < 0.14 else "large"


def _sample_hash(samples):
    normalized = samples.copy(deep=True)
    categories = {}
    for name in normalized:
        if isinstance(normalized[name].dtype, pd.CategoricalDtype):
            categories[name] = {
                "ordered": normalized[name].cat.ordered,
                "levels": [
                    (type(value).__name__, str(value)) for value in normalized[name].cat.categories
                ],
            }
            normalized[name] = normalized[name].astype("string")
    payload = {
        "format": "anova-input-v1",
        "fingerprint_format": FINGERPRINT_FORMAT,
        "frame": dataframe_fingerprint(normalized),
        "categories": categories,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode()
    ).hexdigest()


def _validated_input(samples, metric, factors, alpha):
    if not isinstance(samples, pd.DataFrame) or samples.empty or not samples.columns.is_unique:
        raise ValueError("samples must be a nonempty DataFrame with unique columns")
    if (
        not isinstance(metric, str)
        or not metric
        or any(not isinstance(name, str) or not name for name in factors)
    ):
        raise ValueError("metric and factor names must be nonempty strings")
    if len(set((metric, *factors))) != len(factors) + 1:
        raise ValueError("metric and factors must be distinct columns")
    if any(name not in samples for name in (metric, *factors)):
        raise ValueError("metric or factor column is missing")
    if (
        not isinstance(alpha, Real)
        or isinstance(alpha, bool)
        or not np.isfinite(alpha)
        or not 0 < alpha < 1
    ):
        raise ValueError("alpha must be finite and in (0, 1)")
    selected = samples.loc[:, [metric, *factors]]
    if selected.isna().any().any():
        raise ValueError("metric and factor columns cannot contain missing values")
    values = selected[metric]
    if (
        not is_numeric_dtype(values.dtype)
        or is_bool_dtype(values.dtype)
        or is_complex_dtype(values.dtype)
    ):
        raise ValueError("metric must contain real numeric values")
    values = values.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("metric cannot contain nonfinite values")
    data = pd.DataFrame({"_response": values})
    levels = []
    for position, name in enumerate(factors):
        series = selected[name]
        labels = series.tolist()
        textual = all(isinstance(value, str) and value.strip() for value in labels)
        numeric = all(isinstance(value, Real) and np.isfinite(value) for value in labels)
        if not textual and not numeric:
            raise ValueError("factor labels must be nonempty text or finite real numbers")
        if (
            isinstance(series.dtype, pd.CategoricalDtype)
            and not (series.value_counts(sort=False) > 0).all()
        ):
            raise ValueError("factor contains an empty declared category")
        codes, unique = pd.factorize(series, sort=False)
        if len(unique) < 2:
            raise ValueError("each factor requires at least two groups")
        data[f"_factor_{position}"] = codes
        levels.append(tuple(str(value) for value in unique))
    groups = _groups(data, tuple(levels), "interaction" if len(factors) == 2 else "factor_0")
    if any(len(values) < 2 for _, values in groups):
        raise ValueError(
            "every group or two-way cell requires at least two samples; empty cells are forbidden"
        )
    return data, tuple(levels), float(alpha)


def _groups(data, levels, effect_id):
    if effect_id == "interaction":
        result = []
        for left, right in product(range(len(levels[0])), range(len(levels[1]))):
            mask = data._factor_0.eq(left) & data._factor_1.eq(right)
            label = json.dumps([levels[0][left], levels[1][right]], ensure_ascii=False)
            result.append((label, data.loc[mask, "_response"].to_numpy()))
        return result
    position = int(effect_id.rsplit("_", 1)[1])
    return [
        (label, data.loc[data[f"_factor_{position}"].eq(code), "_response"].to_numpy())
        for code, label in enumerate(levels[position])
    ]


def _diagnostics(residuals, groups, alpha):
    records = []
    for name, function, args in (
        ("shapiro", stats.shapiro, [residuals]),
        ("levene", stats.levene, [values for _, values in groups]),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = function(*args, **({"center": "median"} if name == "levene" else {}))
        valid = bool(np.isfinite(result.statistic) and np.isfinite(result.pvalue))
        notes = []
        if not valid:
            notes.append("undefined_statistic")
        if name == "shapiro" and len(residuals) > 5000:
            notes.append("p_value_unreliable_n_over_5000")
        if caught:
            notes.append("scipy_warning")
        records.append(
            {
                "test": name,
                "statistic": float(result.statistic) if valid else np.nan,
                "p": float(result.pvalue) if valid else np.nan,
                "alpha": alpha,
                "violated": bool(result.pvalue < alpha) if valid else None,
                "valid": valid,
                "note": ";".join(notes),
                "n": len(residuals),
                "grouping": "model_residuals" if name == "shapiro" else "groups_or_cells",
            }
        )
    return pd.DataFrame(records)


def _tukey(effect, groups, alpha):
    values = np.concatenate([values for _, values in groups])
    labels = np.concatenate([np.repeat(label, len(values)) for label, values in groups])
    result = pairwise_tukeyhsd(values, labels, alpha=alpha)
    rows = []
    for index, (left, right) in enumerate(combinations(result.groupsunique, 2)):
        rows.append(
            {
                "effect_id": effect["effect_id"],
                "effect": effect["effect"],
                "group1": left,
                "group2": right,
                "meandiff": float(result.meandiffs[index]),
                "ci_lower": float(result.confint[index, 0]),
                "ci_upper": float(result.confint[index, 1]),
                "p_adj": float(result.pvalues[index]),
                "reject": bool(result.reject[index]),
                **{
                    name: effect[name]
                    for name in (
                        "eta_squared",
                        "eta_squared_label",
                        "partial_eta_squared",
                        "partial_eta_squared_label",
                    )
                },
            }
        )
    return rows


def _anova(samples, metric, factors, alpha):
    data, levels, alpha = _validated_input(samples, metric, factors, alpha)
    formula = "_response ~ " + " * ".join(f"C(_factor_{index})" for index in range(len(factors)))
    with np.errstate(over="ignore", invalid="ignore"):
        total = float(np.square(data._response - data._response.mean()).sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("metric must have finite positive total variation")
    model = ols(formula, data=data).fit()
    if model.ssr <= np.finfo(float).eps * total or not np.isfinite(model.ssr):
        raise ValueError("ANOVA requires finite positive residual variance")
    table = anova_lm(model, typ=2)
    terms = [(f"C(_factor_{index})", f"factor_{index}", name) for index, name in enumerate(factors)]
    if len(factors) == 2:
        terms.append(("C(_factor_0):C(_factor_1)", "interaction", f"{factors[0]}:{factors[1]}"))
    effects, comparisons = [], []
    for term, effect_id, name in terms:
        row = table.loc[term]
        if not np.isfinite(row[["sum_sq", "df", "F", "PR(>F)"]]).all() or row.sum_sq < 0:
            raise ValueError("ANOVA produced an undefined effect statistic")
        eta = float(row.sum_sq / total)
        if eta > 1 + 1e-10:
            raise ValueError("ANOVA effect SS exceeds total variation")
        eta = min(eta, 1.0)
        partial = float(row.sum_sq / (row.sum_sq + model.ssr)) if len(factors) == 2 else np.nan
        significant = bool(row["PR(>F)"] < alpha)
        effect = {
            "effect_id": effect_id,
            "effect": name,
            "df": float(row.df),
            "sum_sq": float(row.sum_sq),
            "F": float(row.F),
            "p": float(row["PR(>F)"]),
            "eta_squared": eta,
            "eta_squared_label": effect_size_label(eta),
            "partial_eta_squared": partial,
            "partial_eta_squared_label": effect_size_label(partial) if len(factors) == 2 else None,
            "tukey_performed": significant,
            "tukey_reason": "performed" if significant else "effect_not_significant",
        }
        effects.append(effect)
        if significant:
            comparisons.extend(_tukey(effect, _groups(data, levels, effect_id), alpha))
    effects.append(
        {
            "effect_id": "residual",
            "effect": "Residual",
            "df": float(model.df_resid),
            "sum_sq": float(model.ssr),
            "F": np.nan,
            "p": np.nan,
            "eta_squared": np.nan,
            "eta_squared_label": None,
            "partial_eta_squared": np.nan,
            "partial_eta_squared_label": None,
            "tukey_performed": False,
            "tukey_reason": "residual_not_tested",
        }
    )
    groups = _groups(data, levels, "interaction" if len(factors) == 2 else "factor_0")
    return AnovaResult(
        metric,
        factors,
        alpha,
        pd.DataFrame(effects),
        _diagnostics(model.resid.to_numpy(), groups, alpha),
        pd.DataFrame([{"group": label, "n": len(values)} for label, values in groups]),
        pd.DataFrame(comparisons, columns=_TUKEY_COLUMNS),
        _sample_hash(samples),
        total,
        data,
        levels,
    )


def one_way_anova(samples: pd.DataFrame, metric: str, factor: str, *, alpha=0.05) -> AnovaResult:
    """한 요인의 Type II OLS ANOVA와 잔차 진단·조건부 Tukey를 계산한다."""
    return _anova(samples, metric, (factor,), alpha)


def two_way_anova(
    samples: pd.DataFrame, metric: str, factor_a: str, factor_b: str, *, alpha=0.05
) -> AnovaResult:
    """불균형 표본의 두 주효과와 상호작용을 Type II로 검정한다."""
    return _anova(samples, metric, (factor_a, factor_b), alpha)


def reward_anova(samples, metric, *, reward_col="reward", alpha=0.05) -> AnovaResult:
    """검정 1: 호출자가 고른 보상함수 열의 One-way 검정."""
    return one_way_anova(samples, metric, reward_col, alpha=alpha)


def strategy_anova(samples, metric, *, strategy_col="strategy", alpha=0.05) -> AnovaResult:
    """검정 2: DRL·MVO·동일가중 등 전략 열의 One-way 검정."""
    return one_way_anova(samples, metric, strategy_col, alpha=alpha)


def strategy_regime_anova(
    samples, metric, *, strategy_col="strategy", regime_col="regime", alpha=0.05
) -> AnovaResult:
    """검정 3: 전략×시장국면 상호작용을 포함한 Two-way 검정."""
    return two_way_anova(samples, metric, strategy_col, regime_col, alpha=alpha)


def _plots(result, directory):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    for effect in result.effects.to_dict("records"):
        if effect["effect_id"] == "residual":
            continue
        groups = _groups(result._data, result._levels, effect["effect_id"])
        figure = Figure(figsize=(max(6, min(len(groups), 20) * 0.7), 4.5))
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        axis.boxplot([values for _, values in groups], tick_labels=[label for label, _ in groups])
        random = np.random.default_rng(0)
        for position, (_, values) in enumerate(groups, 1):
            axis.scatter(
                position + random.uniform(-0.12, 0.12, len(values)), values, alpha=0.6, s=12
            )
        axis.set(xlabel=effect["effect"], ylabel=result.metric)
        axis.tick_params(axis="x", labelrotation=25)
        figure.tight_layout()
        figure.savefig(directory / f"box_{effect['effect_id']}.png", dpi=140)
        figure.clear()
        pairs = result.tukey.loc[result.tukey.effect_id.eq(effect["effect_id"])]
        if pairs.empty:
            continue
        figure = Figure(figsize=(8, max(3, len(pairs) * 0.35)))
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        for position, pair in enumerate(pairs.itertuples()):
            axis.plot([pair.ci_lower, pair.ci_upper], [position, position], color="C0")
            axis.scatter(pair.meandiff, position, color="C0")
        axis.axvline(0, color="gray", linestyle="--")
        axis.set_yticks(
            range(len(pairs)), [f"{row.group2} - {row.group1}" for row in pairs.itertuples()]
        )
        axis.set(
            xlabel=f"Mean difference ({1 - result.alpha:.1%} simultaneous CI)",
            title=effect["effect"],
        )
        figure.tight_layout()
        figure.savefig(directory / f"tukey_{effect['effect_id']}.png", dpi=140)
        figure.clear()


def save_anova(result: AnovaResult, output_dir) -> Path:
    """새 디렉터리에 표·PNG·계보 manifest를 저장하고 기존 출력은 거부한다."""
    output = Path(output_dir).expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    try:
        result.effects.to_csv(output / "results.csv", index=False)
        result.diagnostics.to_csv(output / "diagnostics.csv", index=False)
        result.group_counts.to_csv(output / "group_counts.csv", index=False)
        result.tukey.to_csv(output / "tukey.csv", index=False)
        (output / "plots").mkdir()
        _plots(result, output / "plots")
        manifest = {
            "schema_version": 1,
            "input_sha256": result.input_sha256,
            "input_hash_format": "anova-input-v1; full ordered DataFrame via snapshot fingerprint",
            "metric": result.metric,
            "factors": list(result.factors),
            "alpha": result.alpha,
            "statsmodels_version": statsmodels.__version__,
            "ss_type": 2,
            "interaction_included": len(result.factors) == 2,
            "diagnostic_specification": {
                "shapiro": "OLS residuals",
                "levene_center": "median",
                "levene_grouping": "cells" if len(result.factors) == 2 else "groups",
            },
            "total_sum_sq": result.total_sum_sq,
            "eta_squared_formula": "effect SS / corrected total SS",
            "partial_eta_squared_formula": "effect SS / (effect SS + residual SS)",
            "effect_size_thresholds": {"small": 0.01, "medium": 0.06, "large": 0.14},
            "cell_label_encoding": "JSON array of factor labels in factor order",
            "sample_unit": (
                "one independent sample per row; independence and exclusions supplied by caller"
            ),
            "group_counts": result.group_counts.to_dict("records"),
            "files": {
                path.relative_to(output).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(output.rglob("*"))
                if path.is_file()
            },
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
    except Exception:
        shutil.rmtree(output)
        raise
    return output
