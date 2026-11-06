"""Named PortfolioEnv inputs and a fixed, scalar-selectable Gaussian PPO mean."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from gymnasium import spaces
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import FlattenExtractor


@dataclass(frozen=True)
class ObservationLayout:
    """Name existing environment observations without fitting or normalizing.

    Asset order must come from the checkpoint's training dataset/manifest.
    The model's numeric spaces alone cannot recover asset identities.
    """

    assets: Sequence[str]
    lookback: int

    def __post_init__(self):
        if isinstance(self.assets, str):
            raise ValueError("assets must be a sequence of unique names")
        assets = tuple(self.assets)
        if (
            not assets
            or any(not isinstance(asset, str) or not asset.strip() for asset in assets)
            or len(set(assets)) != len(assets)
            or "CASH" in assets
        ):
            raise ValueError("assets must be nonempty unique names excluding reserved CASH")
        if (
            isinstance(self.lookback, bool)
            or not isinstance(self.lookback, int)
            or self.lookback < 1
        ):
            raise ValueError("lookback must be a positive integer")
        object.__setattr__(self, "assets", assets)

    @property
    def output_names(self) -> tuple[str, ...]:
        return (*self.assets, "CASH")

    @property
    def feature_names(self) -> tuple[str, ...]:
        return (
            *(
                f"simple_return_lag_{lag}__{asset}"
                for lag in range(self.lookback - 1, -1, -1)
                for asset in self.assets
            ),
            *(f"weight__{asset}" for asset in self.output_names),
            *(f"rsi_fraction__{asset}" for asset in self.assets),
            *(f"macd_price_normalized__{asset}" for asset in self.assets),
            *(f"risk_score__{asset}" for asset in self.assets),
        )

    @property
    def feature_groups(self) -> tuple[str, ...]:
        count = len(self.assets)
        return (
            ("returns",) * (self.lookback * count)
            + ("weights",) * (count + 1)
            + ("rsi",) * count
            + ("macd",) * count
            + ("risk",) * count
        )

    def as_batch(self, observations) -> np.ndarray:
        """Return a copied (rows, features) float64 matrix, preserving model units."""
        if isinstance(observations, pd.DataFrame) and tuple(observations.columns) != (
            self.feature_names
        ):
            raise ValueError("observation columns must match layout in the same order")
        if isinstance(observations, pd.Series) and tuple(observations.index) != self.feature_names:
            raise ValueError("observation Series index must match layout in the same order")
        values = np.asarray(observations)
        if values.dtype.kind not in "iuf":
            raise TypeError("observations must contain real numeric values")
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != len(self.feature_names) or not len(values):
            raise ValueError("observations must be a nonempty batch matching layout dimensions")
        values = np.array(values, dtype=np.float64, copy=True)
        if not np.isfinite(values).all():
            raise ValueError("observations must be finite")
        return values

    def to_frame(self, observations, *, index=None) -> pd.DataFrame:
        """Attach names to actual environment vectors; do not rebuild their state."""
        return pd.DataFrame(self.as_batch(observations), index=index, columns=self.feature_names)


class PPOPolicyWrapper:
    """Explain raw Gaussian means, separately from executed portfolio weights.

    This supports the project's plain, flat MlpPolicy PPO without VecNormalize,
    SDE, or action squashing. Inputs are already-preprocessed environment
    observations. Callers must not mutate/train the model while explaining it.
    """

    def __init__(self, model: PPO, layout: ObservationLayout):
        if not isinstance(model, PPO):
            raise TypeError("model must be an SB3 PPO")
        policy = model.policy
        if (
            type(policy) is not ActorCriticPolicy
            or type(policy.action_dist) is not DiagGaussianDistribution
            or policy.squash_output
            or not isinstance(policy.features_extractor, FlattenExtractor)
        ):
            raise ValueError("only plain Gaussian PPO MlpPolicy without squashing/SDE is supported")
        if model.get_vec_normalize_env() is not None:
            raise ValueError("VecNormalize preprocessing is not supported by this wrapper")
        obs, action = model.observation_space, model.action_space
        if (
            not isinstance(obs, spaces.Box)
            or obs.shape != (len(layout.feature_names),)
            or not isinstance(action, spaces.Box)
            or action.shape != (len(layout.output_names),)
            or not np.all(action.low == -20)
            or not np.all(action.high == 20)
        ):
            raise ValueError("model spaces must match the PortfolioEnv layout and action bounds")
        self.model = model
        self.layout = layout
        self.output_names = layout.output_names

    def predict_mean_actions(self, observations) -> np.ndarray:
        """Return unclipped distribution means, not samples or percentages."""
        batch = self.layout.as_batch(observations)
        policy = self.model.policy
        was_training = policy.training
        try:
            policy.set_training_mode(False)
            with torch.no_grad():
                tensor, _ = policy.obs_to_tensor(batch)
                if not torch.isfinite(tensor.float()).all():
                    raise ValueError("observations must remain finite in policy precision")
                distribution = policy.get_distribution(tensor)
                means = distribution.distribution.mean.cpu().numpy().astype(np.float64, copy=True)
            if means.shape != (len(batch), len(self.output_names)) or not np.isfinite(means).all():
                raise ValueError("policy mean must be a finite row per observation")
            return means
        finally:
            policy.set_training_mode(was_training)

    def for_asset(self, asset: str) -> Callable:
        """Return a (rows,) mean-action function for one asset, including CASH."""
        if asset not in self.output_names:
            raise ValueError(f"unknown asset: {asset!r}")
        position = self.output_names.index(asset)

        def predict(observations):
            return self.predict_mean_actions(observations)[:, position]

        return predict

    def predict_target_weights(self, observations) -> np.ndarray:
        """Reference only: softmax of clipped means, as in deterministic execution."""
        means = self.predict_mean_actions(observations)
        logits = np.clip(means, self.model.action_space.low, self.model.action_space.high)
        weights = np.exp(logits - logits.max(axis=1, keepdims=True))
        return weights / weights.sum(axis=1, keepdims=True)
