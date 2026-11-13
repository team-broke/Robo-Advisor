"""Precomputed KernelSHAP for PPO mean actions, stored with verifiable provenance.

KernelSHAP is far too slow for request-time serving, so explanations are
computed offline, checked for additivity, saved once and only read afterwards.
"""

from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import time

import numpy as np
import pandas as pd
import shap

from robo_advisor.data.snapshot import read_frame, write_frame
from robo_advisor.explain.background import BackgroundSample
from robo_advisor.explain.policy import ObservationLayout, PPOPolicyWrapper

SCHEMA_VERSION = 1
_OUTPUT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


@dataclass(frozen=True)
class ShapResult:
    """SHAP values per explained output, in raw policy-mean units.

    ``values`` maps an output name to a (timestamps, features) frame whose rows
    sum, together with ``base_values[output]``, to ``predictions[output]``.
    """

    values: dict[str, pd.DataFrame]
    base_values: dict[str, float]
    predictions: pd.DataFrame
    observations: pd.DataFrame
    feature_groups: tuple[str, ...]
    metadata: dict

    @property
    def outputs(self) -> tuple[str, ...]:
        return tuple(self.values)

    def explanation(self, output: str, timestamp) -> dict:
        """Return one stored explanation; never recomputes SHAP."""
        if output not in self.values:
            raise ValueError(f"unknown explained output: {output!r}")
        stamp = pd.Timestamp(timestamp)
        if pd.isna(stamp) or stamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        stamp = stamp.tz_convert("UTC")
        if stamp not in self.observations.index:
            raise ValueError(f"no precomputed explanation at {stamp.isoformat()}")
        contributions = self.values[output].loc[stamp]
        return {
            "output": output,
            "timestamp": stamp.isoformat(),
            "base_value": self.base_values[output],
            "prediction": float(self.predictions.loc[stamp, output]),
            "contributions": contributions.copy(),
            "group_contributions": group_contributions(contributions, self.feature_groups),
            "observation": self.observations.loc[stamp].copy(),
        }


def group_contributions(contributions: pd.Series, feature_groups: Sequence[str]) -> pd.Series:
    """Sum feature SHAP values by layout group, keeping the layout's group order."""
    if len(contributions) != len(feature_groups):
        raise ValueError("feature groups must align with contributions")
    groups = pd.Series(contributions.to_numpy(dtype=float), index=list(feature_groups))
    order = list(dict.fromkeys(feature_groups))
    return groups.groupby(level=0, sort=False).sum().reindex(order)


def model_fingerprint(path) -> str:
    """SHA-256 of a saved checkpoint file, usable as ``model_id``."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _seeded_global_rng(seed: int):
    # shap 0.45 KernelExplainer samples coalitions from the global NumPy RNG.
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def _validated_index(frame: pd.DataFrame, name: str) -> pd.DatetimeIndex:
    index = frame.index
    if (
        not isinstance(index, pd.DatetimeIndex)
        or index.tz is None
        or index.hasnans
        or not index.is_unique
        or not index.is_monotonic_increasing
        or not len(index)
    ):
        raise ValueError(f"{name} index must be nonempty, ordered, unique and timezone-aware")
    return index.tz_convert("UTC")


def compute_kernel_shap(
    wrapper: PPOPolicyWrapper,
    background: BackgroundSample,
    observations: pd.DataFrame,
    *,
    model_id: str,
    outputs: Sequence[str] | None = None,
    nsamples: int | None = None,
    seed: int = 0,
    tolerance: float = 1e-4,
) -> ShapResult:
    """Explain each output's raw Gaussian mean with KernelSHAP and check additivity.

    The background must come from ``sample_training_background`` so that its
    rows lie inside the model's fit interval. ``nsamples`` is the coalition
    budget per row (shap's ``auto`` when None). Results whose
    ``base + sum(values)`` differs from the policy mean by more than
    ``tolerance`` are rejected rather than stored.
    """
    layout = wrapper.layout
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must identify the explained checkpoint")
    if not isinstance(background, BackgroundSample):
        raise TypeError("background must be a BackgroundSample")
    if tuple(background.observations.columns) != layout.feature_names:
        raise ValueError("background columns must match the wrapper layout")
    if not isinstance(observations, pd.DataFrame):
        raise TypeError("observations must be a DataFrame")
    index = _validated_index(observations, "observations")
    batch = layout.as_batch(observations)
    reference = layout.as_batch(background.observations)
    if outputs is None:
        outputs = wrapper.output_names
    if isinstance(outputs, str):
        raise ValueError("outputs must be a sequence of output names")
    outputs = tuple(outputs)
    if (
        not outputs
        or len(set(outputs)) != len(outputs)
        or any(name not in wrapper.output_names for name in outputs)
    ):
        raise ValueError("outputs must be unique names from the wrapper outputs")
    for name, value, minimum in (("seed", seed, 0), ("nsamples", nsamples, 1)):
        if value is None and name == "nsamples":
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if seed > 2**32 - 1:
        raise ValueError("seed must be <= 2**32 - 1")
    if isinstance(tolerance, bool) or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be a positive finite number")

    predictions = pd.DataFrame(
        wrapper.predict_mean_actions(batch), index=index, columns=wrapper.output_names
    ).loc[:, list(outputs)]
    values, base_values, errors, seconds = {}, {}, {}, {}
    options = {"silent": True} if nsamples is None else {"silent": True, "nsamples": nsamples}
    for output in outputs:
        started = time.perf_counter()
        with _seeded_global_rng(seed):
            explainer = shap.KernelExplainer(wrapper.for_asset(output), reference)
            matrix = np.asarray(explainer.shap_values(batch, **options), dtype=np.float64)
        seconds[output] = time.perf_counter() - started
        base = float(np.asarray(explainer.expected_value, dtype=np.float64).reshape(-1)[0])
        if matrix.shape != batch.shape or not np.isfinite(matrix).all() or not np.isfinite(base):
            raise ValueError(f"KernelSHAP returned invalid values for {output}")
        error = np.abs(base + matrix.sum(axis=1) - predictions[output].to_numpy())
        if error.max() > tolerance:
            raise ValueError(
                f"SHAP additivity error {error.max():.3g} exceeds tolerance for {output}"
            )
        values[output] = pd.DataFrame(matrix, index=index, columns=layout.feature_names)
        base_values[output] = base
        errors[output] = float(error.max())
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "method": "KernelSHAP",
        "target": "raw_gaussian_policy_mean",
        "model_id": model_id,
        "assets": list(layout.assets),
        "lookback": layout.lookback,
        "outputs": list(outputs),
        "background_id": background.sample_id,
        "background_rows": len(background.observations),
        "background_seed": background.seed,
        "background_train_start": background.train_start.isoformat(),
        "background_train_end": background.train_end.isoformat(),
        "seed": seed,
        "nsamples": nsamples if nsamples is not None else "auto",
        "tolerance": tolerance,
        "max_additivity_error": errors,
        "compute_seconds": seconds,
        "first_timestamp": index[0].isoformat(),
        "last_timestamp": index[-1].isoformat(),
        "rows": len(index),
        "shap_version": shap.__version__,
    }
    return ShapResult(
        values,
        base_values,
        predictions,
        layout.to_frame(batch, index=index),
        layout.feature_groups,
        metadata,
    )


def save_shap_result(result: ShapResult, directory) -> Path:
    """Write a new result directory; an existing path is never overwritten."""
    output = Path(directory).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"SHAP output already exists: {output}")
    if any(not _OUTPUT_NAME.fullmatch(name) for name in result.outputs):
        raise ValueError("output names must be safe file name components")
    output.mkdir(parents=True)
    try:
        files = {
            "observations.parquet": write_frame(
                result.observations, output / "observations.parquet"
            ),
            "predictions.parquet": write_frame(result.predictions, output / "predictions.parquet"),
        }
        for name, frame in result.values.items():
            files[f"shap__{name}.parquet"] = write_frame(frame, output / f"shap__{name}.parquet")
        manifest = {
            **result.metadata,
            "base_values": result.base_values,
            "feature_groups": list(result.feature_groups),
            "files": files,
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
    except Exception:
        shutil.rmtree(output)
        raise
    return output


def load_shap_result(directory, *, expected_manifest_sha256: str | None = None) -> ShapResult:
    """Read a saved result after verifying the manifest and every file hash."""
    path = Path(directory).expanduser()
    payload = (path / "manifest.json").read_bytes()
    if (
        expected_manifest_sha256 is not None
        and hashlib.sha256(payload).hexdigest() != expected_manifest_sha256
    ):
        raise ValueError("SHAP manifest SHA256 mismatch")
    manifest = json.loads(payload.decode("utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported SHAP result schema version")
    files = manifest["files"]
    outputs = manifest["outputs"]
    if set(files) != {
        "observations.parquet",
        "predictions.parquet",
        *(f"shap__{name}.parquet" for name in outputs),
    }:
        raise ValueError("SHAP manifest files do not match its outputs")
    frames = {name: read_frame(path / name, descriptor) for name, descriptor in files.items()}
    layout = ObservationLayout(manifest["assets"], manifest["lookback"])
    observations = frames["observations.parquet"]
    if tuple(observations.columns) != layout.feature_names:
        raise ValueError("Stored observations do not match the recorded layout")
    if tuple(manifest["feature_groups"]) != layout.feature_groups:
        raise ValueError("Stored feature groups do not match the recorded layout")
    metadata = {
        key: value
        for key, value in manifest.items()
        if key not in {"base_values", "feature_groups", "files"}
    }
    metadata["manifest_sha256"] = hashlib.sha256(payload).hexdigest()
    return ShapResult(
        {name: frames[f"shap__{name}.parquet"] for name in outputs},
        {name: float(manifest["base_values"][name]) for name in outputs},
        frames["predictions.parquet"],
        observations,
        layout.feature_groups,
        metadata,
    )
