"""Summary and force plots drawn only from stored SHAP results, with sidecar provenance."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from robo_advisor.explain.shap_values import ShapResult

_PROVENANCE_KEYS = (
    "model_id",
    "background_id",
    "background_train_start",
    "background_train_end",
    "seed",
    "nsamples",
    "shap_version",
    "manifest_sha256",
)


def _new_file(path) -> Path:
    path = Path(path).expanduser().absolute()
    if path.suffix.lower() != ".png":
        raise ValueError("plot path must end with .png")
    sidecar = path.with_suffix(".json")
    for target in (path, sidecar):
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Plot output already exists: {target}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save(figure, path: Path, provenance: dict) -> Path:
    try:
        figure.savefig(path, dpi=150, bbox_inches="tight")
        path.with_suffix(".json").write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
    except Exception:
        path.unlink(missing_ok=True)
        path.with_suffix(".json").unlink(missing_ok=True)
        raise
    finally:
        plt.close(figure)
    return path


def _provenance(result: ShapResult, kind: str, output: str, timestamps) -> dict:
    if output not in result.values:
        raise ValueError(f"unknown explained output: {output!r}")
    return {
        "kind": kind,
        "output": output,
        "timestamps": [stamp.isoformat() for stamp in timestamps],
        **{key: result.metadata.get(key) for key in _PROVENANCE_KEYS},
    }


def save_summary_plot(result: ShapResult, output: str, path, *, max_display: int = 20) -> Path:
    """Beeswarm summary of one output over every stored timestamp."""
    path = _new_file(path)
    provenance = _provenance(result, "summary", output, result.observations.index)
    figure = plt.figure()
    shap.summary_plot(
        result.values[output].to_numpy(),
        features=result.observations.to_numpy(),
        feature_names=list(result.observations.columns),
        max_display=max_display,
        show=False,
    )
    plt.title(f"SHAP summary: {output}")
    return _save(figure, path, provenance)


def save_force_plot(
    result: ShapResult, output: str, timestamp, path, *, max_labels: int = 8
) -> Path:
    """Force plot of one stored explanation at a single decision time.

    Drawn directly with matplotlib in shap's force layout: contributions push
    from the base value to the prediction, positive in red and negative in
    blue. shap 0.45's matplotlib force plot stalls when every contribution
    has the same sign, so it is not used here.
    """
    path = _new_file(path)
    explanation = result.explanation(output, timestamp)
    provenance = _provenance(result, "force", output, [pd.Timestamp(explanation["timestamp"])])
    base, prediction = explanation["base_value"], explanation["prediction"]
    provenance["base_value"] = base
    provenance["prediction"] = prediction
    contributions = explanation["contributions"]
    observation = explanation["observation"]
    ordered = contributions.loc[contributions.abs().sort_values(ascending=False).index]
    # Label only segments wide enough to hold text; the sidecar keeps the full picture.
    wide = ordered.abs() >= 0.08 * max(ordered.abs().sum(), 1e-12)
    labelled = set(ordered.index[wide][:max_labels])
    figure, axis = plt.subplots(figsize=(12, 3.2))
    # Positive pushes stack leftwards from the prediction, negative rightwards.
    for sign, color in ((1, "#ff0051"), (-1, "#008bfb")):
        edge, row = prediction, 0
        for name, value in ordered[np.sign(ordered) == sign].items():
            axis.barh(0, -value, left=edge, height=0.3, color=color, edgecolor="white")
            if name in labelled:
                # Alternate two label rows so neighbouring segments do not collide.
                axis.text(
                    edge - value / 2,
                    -sign * (0.3 + 0.18 * (row % 2)),
                    f"{name} = {observation[name]:.3g}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=color,
                )
                row += 1
            edge -= value
    for value, label, height in (
        (base, f"base value = {base:.4g}", 0.62),
        (prediction, f"f(x) = {prediction:.4g}", 0.8),
    ):
        axis.axvline(value, color="#555555", linewidth=0.8, linestyle="--")
        axis.text(value, height, label, ha="center", va="center", fontsize=8)
    positive, negative = (
        contributions[contributions > 0].sum(),
        contributions[contributions < 0].sum(),
    )
    low, high = min(base, prediction - positive), max(base, prediction - negative)
    margin = 0.08 * max(high - low, 1e-12)
    axis.set_xlim(low - margin, high + margin)
    axis.set_ylim(-0.75, 0.95)
    axis.set_yticks([])
    axis.set_title(f"SHAP force: {output} at {explanation['timestamp']}", fontsize=9)
    return _save(figure, path, provenance)
