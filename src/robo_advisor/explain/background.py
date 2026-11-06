"""Reproducible, balanced background sampling within an explicit model-fit interval."""

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd

from robo_advisor.explain.policy import ObservationLayout, PPOPolicyWrapper


def _timestamp(value, name: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError(f"{name} must be a timezone-aware timestamp")
    return stamp.tz_convert("UTC")


@dataclass(frozen=True)
class BackgroundSample:
    """Selected rows and their reference distribution; no model approval is implied.

    Frames are independent copies of the caller's data. The ID is computed from
    current contents, so editing a returned frame also changes its identity.
    """

    observations: pd.DataFrame
    strata: pd.Series
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    seed: int

    @property
    def sample_id(self) -> str:
        payload = {
            "schema_version": 1,
            "columns": list(self.observations.columns),
            "timestamps": [stamp.isoformat() for stamp in self.observations.index],
            "values": self.observations.to_numpy(dtype=np.float64).tolist(),
            "strata": self.strata.tolist(),
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "seed": self.seed,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sample_training_background(
    observations: pd.DataFrame,
    strata: pd.Series,
    *,
    layout: ObservationLayout,
    train_start,
    train_end,
    n_samples: int,
    seed: int,
    expected_strata: Sequence[str],
) -> BackgroundSample:
    """Sample causal strata without replacement, exclusively from model-fit rows.

    Both bounds are inclusive and must describe actual fit, not a parent range
    that also contains validation. The caller supplies genuine environment
    observations and labels computed without future data. This function cannot
    infer or certify those upstream facts. Equal-stratum allocation changes the
    background distribution compared with empirical regime frequencies.
    """
    if not isinstance(observations, pd.DataFrame):
        raise TypeError("observations must be a DataFrame")
    index = observations.index
    if (
        not isinstance(index, pd.DatetimeIndex)
        or index.tz is None
        or index.hasnans
        or not index.is_unique
        or not index.is_monotonic_increasing
    ):
        raise ValueError("index must be ordered, unique and timezone-aware")
    if not isinstance(strata, pd.Series) or not strata.index.equals(index):
        raise ValueError("strata index must match observations exactly")
    for name, value, minimum in (("n_samples", n_samples, 1), ("seed", seed, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if seed > 2**32 - 1:
        raise ValueError("seed must be <= 2**32 - 1")
    start, end = _timestamp(train_start, "train_start"), _timestamp(train_end, "train_end")
    if start > end:
        raise ValueError("train_start must be <= train_end")

    # Slice first: excluded rows cannot contribute values, labels, or strata.
    mask = (index >= start) & (index <= end)
    candidates = observations.loc[mask]
    values = layout.as_batch(candidates.infer_objects())
    labels = strata.loc[mask]
    if any(not isinstance(value, str) or not value.strip() for value in labels):
        raise ValueError("training strata must be nonempty strings")
    if isinstance(expected_strata, str):
        raise ValueError("expected_strata must be a sequence of stratum names")
    expected_strata = tuple(expected_strata)
    if (
        not expected_strata
        or any(not isinstance(name, str) or not name.strip() for name in expected_strata)
        or len(set(expected_strata)) != len(expected_strata)
    ):
        raise ValueError("expected_strata must contain nonempty unique stratum names")
    names = sorted(labels.unique())
    missing = set(expected_strata) - set(names)
    if missing:
        raise ValueError(f"missing expected training strata: {sorted(missing)}")
    if set(names) - set(expected_strata):
        raise ValueError("training strata contain unexpected labels")
    if not len(names) <= n_samples <= len(candidates):
        raise ValueError("n_samples must cover all strata and not exceed training rows")
    positions = {name: np.flatnonzero(labels.to_numpy() == name) for name in names}
    counts = dict.fromkeys(names, 0)
    for _ in range(n_samples):
        available = [name for name in names if counts[name] < len(positions[name])]
        # Equalize counts, then prefer larger remaining capacity, then label.
        name = min(available, key=lambda name: (counts[name], -len(positions[name]), name))
        counts[name] += 1
    rng = np.random.default_rng(seed)
    selected = sorted(
        int(position)
        for name in names
        for position in rng.choice(positions[name], size=counts[name], replace=False)
    )
    selected_index = candidates.index[selected].tz_convert("UTC")
    frame = pd.DataFrame(
        values[selected].copy(), index=selected_index, columns=layout.feature_names
    )
    selected_labels = pd.Series(labels.iloc[selected].to_numpy(copy=True), index=selected_index)
    return BackgroundSample(frame, selected_labels, start, end, seed)


def collect_training_observations(
    model,
    dataset,
    config,
    *,
    layout: ObservationLayout,
    train_start,
    train_end,
) -> pd.DataFrame:
    """Replay a fixed checkpoint chronologically inside its actual fit interval.

    This is an explicit reference trajectory, not the randomized training-episode
    distribution. Disable random starts/length limits, keep the MDD safeguard,
    and fail rather than silently sampling a truncated trajectory. The last
    dataset row is a terminal state, not a decision with an available next step.
    """
    from robo_advisor.rl.evaluation import evaluation_env_config
    from robo_advisor.rl.training import TrainingDataset, _make_env

    start, end = _timestamp(train_start, "train_start"), _timestamp(train_end, "train_end")
    if start > end:
        raise ValueError("train_start must be <= train_end")
    if tuple(dataset.returns.columns) != tuple(layout.assets):
        raise ValueError("dataset asset order must match layout")
    if config.lookback != layout.lookback:
        raise ValueError("environment lookback must match layout")
    mask = (dataset.returns.index >= start) & (dataset.returns.index <= end)
    selected = TrainingDataset(
        *(frame.loc[mask].copy() for frame in (dataset.returns, dataset.rsi, dataset.macd)),
        tuple(tag for tag in dataset.risk_tags if tag.available_at <= end),
        dataset.file_sha256,
    )
    wrapper = PPOPolicyWrapper(model, layout)
    effective = evaluation_env_config(config)
    env = _make_env(selected, effective, seed=0)
    observations, dates = [], []
    try:
        observation, info = env.reset(seed=0)
        while True:
            observations.append(observation.copy())
            dates.append(info["timestamp"])
            action = np.clip(
                wrapper.predict_mean_actions(observation)[0],
                model.action_space.low,
                model.action_space.high,
            )
            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                if info["timestamp"] < selected.returns.index[-1]:
                    raise ValueError(
                        f"training replay ended early: {info['reason']}; "
                        "a complete reference background is unavailable"
                    )
                break
    finally:
        env.close()
    result = layout.to_frame(np.asarray(observations), index=pd.DatetimeIndex(dates))
    result.attrs["collection"] = {
        "policy": "clipped_deterministic_mean",
        "env": effective.model_dump(mode="json"),
        "train_start": start.isoformat(),
        "train_end": end.isoformat(),
    }
    return result
