"""Point-in-time portfolio environment using the shared NAV accounting contract."""

from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from robo_advisor.rl.accounting import rebalance_step
from robo_advisor.rl.rewards import RewardVariant, calculate_reward
from robo_advisor.rl.safeguard import MDDSafeguard
from robo_advisor.schemas.risk import RiskTag


class PortfolioEnv(gym.Env):
    """Allocate long-only weights to assets and zero-return cash.

    Inputs are aligned DataFrames with identical, unique asset columns and a
    strictly increasing timezone-aware DatetimeIndex (converted to UTC).
    returns[t] is the simple return ending at t; rsi[t] and macd[t] must be
    computed using information available by t. MACD is the normalized macd
    line from features.indicators, not its signal line. Market-close alignment
    and any overseas lag must be applied upstream before supplying the frames.

    At reset, t = lookback - 1. Observation is flattened oldest-first returns
    [t-N+1:t+1], current drift weights (assets then cash), RSI/100, MACD, and
    one risk_score per asset: A*(N+4)+1 float64 values. Missing tags are zero;
    the latest available event persists, with maximum score for timestamp ties.
    Only tags with available_at <= t are visible. Leading indicator warmup NaNs
    before the initial observation are allowed; no filling from the future occurs.

    Actions are A+1 logits in [-20, 20]. Stable softmax produces nonnegative
    weights summing to one, without an individual weight cap. step rebalances
    at t, deducts costs, applies returns[t+1], and observes at t+1. Reward and
    safeguard use that same net NAV. MDD termination takes precedence over
    exhaustion truncation; either requires reset before another step.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        returns: pd.DataFrame,
        rsi: pd.DataFrame,
        macd: pd.DataFrame,
        *,
        lookback: int,
        n_assets: int,
        risk_tags: Sequence[RiskTag] = (),
        initial_nav: float = 1.0,
        initial_weights=None,
        commission_rate: float = 0.00015,
        slippage_rate: float = 0.0005,
        reward_variant: RewardVariant | str = RewardVariant.RETURN,
        penalty_lambda: float = 1.0,
        mdd_limit: float = 0.15,
        trace_id: str = "portfolio-environment",
    ):
        super().__init__()
        for name, value in (("lookback", lookback), ("n_assets", n_assets)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(returns, pd.DataFrame):
            raise ValueError("returns must be a DataFrame")
        index = returns.index
        if (
            not isinstance(index, pd.DatetimeIndex)
            or index.tz is None
            or index.hasnans
            or not index.is_unique
            or not index.is_monotonic_increasing
            or len(index) <= lookback
        ):
            raise ValueError("index must be ordered, unique, timezone-aware and include a step")
        if (
            len(returns.columns) != n_assets
            or not returns.columns.is_unique
            or any(not isinstance(asset, str) or not asset.strip() for asset in returns.columns)
        ):
            raise ValueError("columns must contain n_assets unique asset names")
        matrices = []
        for name, frame in (("returns", returns), ("rsi", rsi), ("macd", macd)):
            if (
                not isinstance(frame, pd.DataFrame)
                or not frame.index.equals(index)
                or not frame.columns.equals(returns.columns)
            ):
                raise ValueError(f"{name} must match returns index and columns exactly")
            values = frame.to_numpy(dtype=np.float64, copy=True)
            active = values if name == "returns" else values[lookback - 1 :]
            if not np.isfinite(active).all():
                raise ValueError(f"{name} observations must be finite")
            if name == "returns" and np.any(values < -1):
                raise ValueError("simple returns must be at least -1")
            if name == "rsi" and np.any((active < 0) | (active > 100)):
                raise ValueError("RSI must be in [0, 100]")
            matrices.append(values)
        self._returns, self._rsi, self._macd = matrices
        self._index = index.tz_convert("UTC").copy()
        self._assets = tuple(returns.columns)
        self._tags = tuple(risk_tags)
        if any(not isinstance(tag, RiskTag) or tag.asset not in self._assets for tag in self._tags):
            raise ValueError("risk_tags must contain RiskTag objects for known assets")
        self.lookback = lookback
        self.n_assets = n_assets
        self._initial_nav = float(initial_nav)
        weights = np.zeros(n_assets + 1) if initial_weights is None else initial_weights
        if initial_weights is None:
            weights[-1] = 1
        # Reuse accounting validation without performing an initial trade.
        validated = rebalance_step(
            self._initial_nav, weights, weights, np.zeros(n_assets), commission_rate, slippage_rate
        )
        self._initial_weights = validated.weights.copy()
        self._commission = commission_rate
        self._slippage = slippage_rate
        self._variant = RewardVariant(reward_variant)
        calculate_reward(0, self._variant, penalty_lambda=penalty_lambda)
        self._penalty = penalty_lambda
        self._limit = mdd_limit
        self._trace_id = trace_id
        MDDSafeguard(self._initial_nav, trace_id, mdd_limit)
        self.action_space = spaces.Box(-20.0, 20.0, (n_assets + 1,), dtype=np.float64)
        size = n_assets * (lookback + 4) + 1
        low = np.full(size, -np.inf)
        high = np.full(size, np.inf)
        low[: lookback * n_assets] = -1
        offset = lookback * n_assets
        low[offset : offset + 2 * n_assets + 1] = 0
        high[offset : offset + 2 * n_assets + 1] = 1
        low[-n_assets:] = 0
        high[-n_assets:] = 1
        self.observation_space = spaces.Box(low, high, dtype=np.float64)
        self._done = True

    def _observation(self):
        scores = np.zeros(self.n_assets)
        for i, asset in enumerate(self._assets):
            available = [
                tag
                for tag in self._tags
                if tag.asset == asset and tag.available_at <= self._index[self._cursor]
            ]
            if available:
                latest = max(tag.available_at for tag in available)
                scores[i] = max(tag.risk_score for tag in available if tag.available_at == latest)
        return np.concatenate(
            (
                self._returns[self._cursor - self.lookback + 1 : self._cursor + 1].ravel(),
                self._weights,
                self._rsi[self._cursor] / 100,
                self._macd[self._cursor],
                scores,
            )
        )

    def reset(self, *, seed=None, options=None):
        """Reset to the first complete window; seed the Gymnasium RNG."""
        super().reset(seed=seed)
        if options:
            raise ValueError("reset options are not supported")
        self._cursor = self.lookback - 1
        self._nav = self._initial_nav
        self._weights = self._initial_weights.copy()
        self._history = []
        self._guard = MDDSafeguard(self._nav, self._trace_id, self._limit)
        self._done = False
        return self._observation(), {
            "nav": self._nav,
            "weights": self._weights.copy(),
            "timestamp": self._index[self._cursor],
        }

    def step(self, action):
        """Return observation, reward, terminated, truncated, info."""
        if self._done:
            raise gym.error.ResetNeeded("reset is required before stepping")
        logits = np.asarray(action, dtype=np.float64)
        if not self.action_space.contains(logits):
            raise ValueError("action must be a finite vector of A+1 logits in [-20, 20]")
        target = np.exp(logits - logits.max())
        target /= target.sum()
        result = rebalance_step(
            self._nav,
            self._weights,
            target,
            self._returns[self._cursor + 1],
            self._commission,
            self._slippage,
        )
        log_return = float(np.log(result.nav) - np.log(self._nav))
        safeguard = self._guard.update(result.nav)
        reward = calculate_reward(
            log_return,
            self._variant,
            return_history=self._history,
            drawdown=safeguard.drawdown,
            penalty_lambda=self._penalty,
        )
        self._history.append(log_return)
        self._nav, self._weights = result.nav, result.weights
        self._cursor += 1
        terminated = self._guard.termination_result is not None
        truncated = not terminated and self._cursor == len(self._index) - 1
        self._done = terminated or truncated
        return (
            self._observation(),
            reward,
            terminated,
            truncated,
            {
                "nav": self._nav,
                "weights": self._weights.copy(),
                "target_weights": target.copy(),
                "turnover": result.turnover,
                "cost": result.cost,
                "portfolio_log_return": log_return,
                "timestamp": self._index[self._cursor],
                "safeguard": safeguard.model_dump(),
                "termination_result": (
                    self._guard.termination_result.model_dump() if terminated else None
                ),
                "reason": (
                    "mdd_limit_exceeded" if terminated else "data_exhausted" if truncated else None
                ),
            },
        )
