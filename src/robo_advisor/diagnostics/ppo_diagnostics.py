"""기존 PPO 환경을 관찰하는 추론 전용 진단. 학습·원천 승인·모델 선택은 하지 않는다."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from robo_advisor.backtest.metrics import calculate_metrics
from robo_advisor.data.snapshot import dataframe_fingerprint
from robo_advisor.experiments.regimes import label_market_regimes
from robo_advisor.rl.curves import judge_convergence, load_episodes
from robo_advisor.rl.evaluation import evaluation_env_config
from robo_advisor.rl.training import EnvironmentConfig, TrainingDataset, _make_env


@dataclass(frozen=True)
class DiagnosticPeriod:
    """수익률 전이 종료일 기준 [start, end). 직전 관측은 문맥으로만 사용한다."""

    name: str
    start: str | pd.Timestamp
    end: str | pd.Timestamp


@dataclass(frozen=True)
class ReferenceReturns:
    """동일 통화·비용 회계로 이미 계산된 순 단순수익률. 날짜를 보간하지 않는다."""

    returns: pd.Series
    commission_rate: float
    slippage_rate: float
    currency: str
    cost_basis: str = "net_same_accounting"
    return_starts: pd.Series | None = None


@dataclass(frozen=True)
class DiagnosticResult:
    summary: dict
    period_metrics: pd.DataFrame
    allocation_history: pd.DataFrame
    allocation_summary: pd.DataFrame
    safeguard_events: pd.DataFrame


def _timestamp(value):
    value = pd.Timestamp(value)
    if pd.isna(value) or value.tzinfo is None:
        raise ValueError("dates must be finite timezone-aware timestamps")
    return value.tz_convert("UTC")


def validate_weights(weights, n_assets: int) -> np.ndarray:
    """위험자산 다음 마지막 열은 현금이다. 잘못된 값을 정규화해 숨기지 않는다."""
    values = np.asarray(weights)
    if values.shape != (n_assets + 1,) or values.dtype.kind not in "fiu":
        raise ValueError("weights must be a numeric asset+cash vector")
    values = values.astype(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("weights must be finite and nonnegative")
    if not np.isclose(values.sum(), 1, rtol=0, atol=1e-10):
        raise ValueError("weights must sum to one")
    return values


def allocation_statistics(target, actual, previous_target=None) -> dict:
    """HHI는 현금을 포함한다. 단일 위험자산 최대치와 현금 포함 최대치를 분리한다."""
    n = len(target) - 1
    if n < 1:
        raise ValueError("at least one risky asset is required")
    target = validate_weights(target, n)
    actual = validate_weights(actual, n)
    risky_equal = np.r_[np.full(n, 1 / n), 0.0]
    return {
        "target_cash": target[-1],
        "actual_cash": actual[-1],
        "max_risky_weight": actual[:-1].max(),
        "max_weight_including_cash": actual.max(),
        "hhi": np.dot(actual, actual),
        "target_hhi": np.dot(target, target),
        "risky_equal_l1": np.abs(target - risky_equal).sum(),
        "all_equal_l1": np.abs(target - 1 / (n + 1)).sum(),
        "target_change_l1": (
            np.nan
            if previous_target is None
            else np.abs(target - validate_weights(previous_target, n)).sum()
        ),
    }


def _validate_contract(dataset, manifest, config, periods):
    frames = [dataset.returns, dataset.rsi, dataset.macd]
    dates, assets = frames[0].index, frames[0].columns
    if (
        not isinstance(dates, pd.DatetimeIndex)
        or dates.tz is None
        or dates.empty
        or dates.hasnans
        or not dates.is_unique
        or not dates.is_monotonic_increasing
        or not assets.is_unique
        or "CASH" in assets
        or len(assets) == 0
        or any(not isinstance(asset, str) or not asset for asset in assets)
    ):
        raise ValueError("dataset requires ordered aware dates and unique non-CASH asset names")
    for frame in frames:
        if not frame.index.equals(dates) or not frame.columns.equals(assets):
            raise ValueError("dataset dates and asset order must match")
        if any(dtype.kind not in "fiu" for dtype in frame.dtypes):
            raise ValueError("dataset must be numeric")
    if not np.isfinite(dataset.returns.to_numpy()).all():
        raise ValueError("returns must be finite; missing returns are not zero-filled")
    # PortfolioEnv validates RSI/MACD from the first observable row of each rollout.
    if manifest.get("status") != "completed":
        raise ValueError("training manifest must describe a completed run")
    seed = manifest.get("seed")
    if type(seed) is not int or not 0 <= seed < 2**32 or seed != manifest["config"].get("seed"):
        raise ValueError("training seed must match the recorded configuration")
    if not dataset.file_sha256 or dataset.file_sha256 != manifest.get("data_file_sha256"):
        raise ValueError("dataset version/hash mismatch")
    if list(assets) != manifest.get("data_cutoff", {}).get("assets"):
        raise ValueError("manifest asset order mismatch")
    recorded = EnvironmentConfig.model_validate(manifest["config"]["env"])
    if recorded != config:
        raise ValueError("environment/reward/cost configuration mismatch")
    if [p.name for p in periods] != ["train", "validation", "test"]:
        raise ValueError("provide train, validation, test in that order")
    bounds = [(_timestamp(p.start), _timestamp(p.end)) for p in periods]
    if any(a >= b for a, b in bounds) or any(bounds[i][1] > bounds[i + 1][0] for i in range(2)):
        raise ValueError("periods must be ordered and nonoverlapping")
    validation = manifest["config"].get("validation")
    if validation is None:
        raise ValueError("recorded validation boundary is required; do not invent a split")
    data = manifest["config"]["data"]
    if data.get("train_start") and _timestamp(data["train_start"]) != bounds[0][0]:
        raise ValueError("train start differs from manifest")
    if bounds[0][1] != bounds[1][0] or bounds[1][0] != _timestamp(validation["start"]):
        raise ValueError("validation boundary differs from manifest")
    cutoff = _timestamp(data["train_end"])
    if cutoff < bounds[1][0] or cutoff >= bounds[2][0]:
        raise ValueError("test overlaps training/model-selection cutoff")
    train_last = _timestamp(manifest["data_cutoff"]["actual_last_index"])
    if train_last >= bounds[1][0] or train_last < bounds[0][0]:
        raise ValueError("recorded training data crosses validation boundary")
    observed_train = dates[(dates >= bounds[0][0]) & (dates < bounds[0][1])]
    if observed_train.empty or observed_train[-1] != train_last:
        raise ValueError("recorded training coverage differs from input")
    if observed_train[0] != _timestamp(manifest["data_cutoff"]["actual_first_index"]):
        raise ValueError("recorded training start differs from input")
    val_dates = dates[(dates >= bounds[1][0]) & (dates < bounds[1][1])]
    if len(val_dates) and val_dates[-1] > cutoff:
        raise ValueError("validation extends beyond recorded selection period")
    recorded_val_dates = dates[(dates >= bounds[1][0]) & (dates <= cutoff)]
    if not val_dates.equals(recorded_val_dates):
        raise ValueError("validation period omits recorded selection rows")
    return bounds


def _period_dataset(dataset, start, end, lookback):
    dates = dataset.returns.index
    requested = dates[(dates >= start) & (dates < end)]
    begin, stop = dates.searchsorted(start), dates.searchsorted(end)
    first = max(begin, lookback)
    if first >= stop:
        return None, requested, requested
    # observation at first-1 only sees the preceding lookback rows.
    frames = [
        frame.iloc[first - lookback : stop].copy()
        for frame in (dataset.returns, dataset.rsi, dataset.macd)
    ]
    tags = tuple(tag for tag in dataset.risk_tags if _timestamp(tag.available_at) < end)
    subset = TrainingDataset(*frames, tags, dataset.file_sha256.copy())
    return subset, requested, requested[requested < dates[first]]


def trace_policy(model, dataset: TrainingDataset, config: EnvironmentConfig) -> pd.DataFrame:
    """evaluate_policy와 같은 환경/결정론적 추론으로 기존 info만 추가 기록한다."""
    env = _make_env(dataset, evaluation_env_config(config), seed=0)
    rows, previous_target = [], None
    try:
        observation, previous = env.reset(seed=0)
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(action)
            row = {
                "date": info["timestamp"],
                "decision_at": previous["timestamp"],
                "return_start": previous["timestamp"],
                "nav": info["nav"],
                "return": info["nav"] / previous["nav"] - 1,
                "turnover": info["turnover"],
                "cost": info["cost"],
                "drawdown": info["safeguard"]["drawdown"],
                "peak_nav": info["safeguard"]["peak_nav"],
                "episode_step": info["episode_step"],
                "terminated": terminated,
                "truncated": truncated,
                "reason": info["reason"],
                **allocation_statistics(info["target_weights"], info["weights"], previous_target),
            }
            for name, target, actual in zip(
                [*dataset.returns.columns, "CASH"], info["target_weights"], info["weights"]
            ):
                row[f"target__{name}"] = target
                row[f"actual__{name}"] = actual
            rows.append(row)
            previous_target, previous = info["target_weights"].copy(), info
            if terminated or truncated:
                break
    finally:
        env.close()
    trace = pd.DataFrame(rows).set_index("date")
    if not np.isfinite(trace[["return", "nav", "cost", "turnover"]].to_numpy()).all():
        raise ValueError("nonfinite evaluation results")
    return trace


def _drawdown_crossings(trace):
    # Use the native safeguard's NAV comparison even when the active limit is 1.0.
    # Computing drawdown > 0.15 can misclassify the exact boundary through rounding.
    threshold = trace.peak_nav * (1 - 0.15)
    result = {}
    for label, mask in (
        ("first_reached_15pct", trace.nav <= threshold),
        ("first_exceeded_15pct", trace.nav < threshold),
    ):
        hits = trace.index[mask]
        result[label] = hits[0].isoformat() if len(hits) else None
    return result


def _metrics(returns):
    if len(returns) < 2:
        return {
            "cumulative_return": None if len(returns) == 0 else float((1 + returns).prod() - 1),
            "sharpe": None,
            "mdd": None,
        }
    metrics = calculate_metrics(returns, pd.Series(0.0, index=returns.index))
    return {key: metrics[key] for key in ("cumulative_return", "sharpe", "mdd")}


def _reference_comparison(trace, requested, reference, config, currency, complete):
    values = reference.returns.copy()
    if not isinstance(values.index, pd.DatetimeIndex) or values.index.tz is None:
        raise ValueError("reference dates must be timezone-aware")
    values.index = values.index.tz_convert("UTC")
    if (
        not values.index.equals(requested)
        or values.dtype.kind not in "fiu"
        or not np.isfinite(values.to_numpy()).all()
        or (values < -1).any()
    ):
        raise ValueError("reference must have exactly the full evaluation dates and finite returns")
    if (
        reference.commission_rate != config.commission_rate
        or reference.slippage_rate != config.slippage_rate
        or reference.currency != currency
        or reference.cost_basis != "net_same_accounting"
    ):
        raise ValueError("reference currency or cost conditions differ")
    starts = reference.return_starts
    if starts is None or not isinstance(starts.index, pd.DatetimeIndex) or starts.index.tz is None:
        raise ValueError("reference return-start dates are required")
    starts = starts.copy()
    starts.index = starts.index.tz_convert("UTC")
    if not starts.index.equals(requested):
        raise ValueError("reference return-start dates are required")
    starts = pd.DatetimeIndex([_timestamp(value) for value in starts])
    if (starts >= requested).any() or not starts.is_unique or not starts.is_monotonic_increasing:
        raise ValueError("invalid reference return intervals")
    if trace is not None and not starts[requested.get_indexer(trace.index)].equals(
        pd.DatetimeIndex(trace.return_start)
    ):
        raise ValueError("reference return intervals differ")
    return {
        "status": "matched" if complete else "not_comparable_incomplete",
        "reference_metrics": _metrics(values),
        "excess_cumulative_return": (
            _metrics(trace["return"])["cumulative_return"] - _metrics(values)["cumulative_return"]
            if complete
            else None
        ),
    }


def summarize_training_episodes(run_dir):
    """Monitor의 종료 사유를 집계한다. 기록되지 않은 시장 날짜/NAV는 추정하지 않는다."""
    if run_dir is None:
        return {"status": "unavailable", "episodes": None, "early_end_rate": None}, []
    try:
        episodes = load_episodes(run_dir)
    except FileNotFoundError:
        return {"status": "unavailable", "episodes": None, "early_end_rate": None}, []
    if episodes.empty:
        return {"status": "empty", "episodes": 0, "early_end_rate": None}, []
    for column in ("reward", "length", "elapsed_seconds"):
        values = pd.to_numeric(episodes[column], errors="raise")
        if not np.isfinite(values).all() or (column != "reward" and (values < 0).any()):
            raise ValueError("invalid Monitor episode values")
    known = episodes["reason"].isin(["mdd_limit_exceeded", "data_exhausted", "episode_limit"])
    early = episodes["reason"].eq("mdd_limit_exceeded")
    events = [
        {
            "scope": "training",
            "episode": int(row.episode),
            "environment": row.env,
            "date": None,
            "nav": None,
            "drawdown": None,
            "regime": "unknown",
            "reason": row.reason,
        }
        for row in episodes.loc[early].itertuples()
    ]
    return {
        "status": "recorded" if known.all() else "incomplete_reasons",
        "scope": "provided_completed_monitor_episodes",
        "episodes": len(episodes),
        "early_ends": int(early.sum()),
        "unknown_reasons": int((~known).sum()),
        "early_end_rate": float(early.mean()) if known.all() else None,
        "convergence": judge_convergence(episodes["reward"]),
        "market_dates_nav_mdd": "unavailable_in_native_monitor",
    }, events


def _allocation_summary(history, assets):
    rows = []
    if history.empty:
        return pd.DataFrame(
            columns=["period", "mode", "regime", "asset", "mean_target", "mean_actual"]
        )
    for (period, mode), group in history.groupby(["period", "mode"], sort=False):
        for label, part in [("all", group), *list(group.groupby("regime", sort=False))]:
            for asset in [*assets, "CASH"]:
                rows.append(
                    {
                        "period": period,
                        "mode": mode,
                        "regime": label,
                        "asset": asset,
                        "observations": len(part),
                        "mean_target": part[f"target__{asset}"].mean(),
                        "mean_actual": part[f"actual__{asset}"].mean(),
                        "mean_hhi": part.hhi.mean(),
                        "max_hhi": part.hhi.max(),
                        "max_risky_weight": part.max_risky_weight.max(),
                        "mean_turnover": part.turnover.mean(),
                        "total_cost": part.cost.sum(),
                        "mean_target_change_l1": part.target_change_l1.mean(),
                        "mean_risky_equal_l1": part.risky_equal_l1.mean(),
                        "mean_all_equal_l1": part.all_equal_l1.mean(),
                    }
                )
    return pd.DataFrame(rows)


def diagnose_ppo(
    model,
    dataset: TrainingDataset,
    training_manifest: dict,
    config: EnvironmentConfig,
    periods: list[DiagnosticPeriod],
    *,
    provenance: dict,
    policy_version: str,
    currency: str,
    sensitivity: bool = True,
    references: dict | None = None,
    regime_benchmark: pd.Series | None = None,
    regime_rules: dict | None = None,
    training_run_dir: str | Path | None = None,
) -> DiagnosticResult:
    """동일 체크포인트를 매 구간 NAV=1/현금=1로 평가한다. Test로 설정을 선택하지 않는다.

    provenance의 해시 검증은 파일 진입점(run_from_spec)이 담당한다. 직접 전달한 객체는
    파일 검증의 대체물이 아니며 결과는 항상 unverified/training_approved=false다.
    """
    if (
        not isinstance(policy_version, str)
        or not policy_version.strip()
        or not isinstance(currency, str)
        or not currency.strip()
        or not isinstance(training_manifest.get("run_id"), str)
        or not training_manifest["run_id"]
        or provenance.get("run_id") != training_manifest.get("run_id")
    ):
        raise ValueError("policy version, currency and matching model run_id are required")
    if type(sensitivity) is not bool:
        raise ValueError("sensitivity must be a boolean")
    if provenance.get("kind") not in ("synthetic", "checkpoint_inference"):
        raise ValueError("explicit synthetic/checkpoint_inference provenance is required")
    raw_source = provenance.get("source_validation", {})
    if isinstance(raw_source, str):
        raw_source = {"status": raw_source}
    source_validation = {
        "reported_status": raw_source.get("reported_status", raw_source.get("status")),
        "reported_training_approved": raw_source.get(
            "reported_training_approved", raw_source.get("training_approved")
        ),
        "status": "unverified",
        "training_approved": False,
    }
    provenance = {**provenance, "source_validation": source_validation}
    bounds = _validate_contract(dataset, training_manifest, config, periods)
    # Match the native training loader's inclusive start. Earlier rows are not trained context.
    start_mask = dataset.returns.index >= bounds[0][0]
    evaluation_data = TrainingDataset(
        *(frame.loc[start_mask].copy() for frame in (dataset.returns, dataset.rsi, dataset.macd)),
        dataset.risk_tags,
        dataset.file_sha256.copy(),
    )
    for frame in (evaluation_data.returns, evaluation_data.rsi, evaluation_data.macd):
        frame.index = frame.index.tz_convert("UTC")
    labels = None
    if regime_benchmark is not None:
        if regime_rules is None:
            raise ValueError("explicit regime rules are required")
        if (
            not isinstance(regime_benchmark.index, pd.DatetimeIndex)
            or regime_benchmark.index.tz is None
        ):
            raise ValueError("regime benchmark dates must be timezone-aware")
        regime_benchmark = regime_benchmark.copy()
        regime_benchmark.index = regime_benchmark.index.tz_convert("UTC")
        labels = label_market_regimes(
            regime_benchmark,
            trading_calendar=dataset.returns.index.tz_convert("UTC"),
            **regime_rules,
        ).labels
    elif regime_rules is not None:
        raise ValueError("regime benchmark is required")
    modes = [("configured", config)]
    if sensitivity and config.mdd_limit < 1:
        modes.append(("no_early_stop_inference", config.model_copy(update={"mdd_limit": 1.0})))
    metrics, histories, details = [], [], []
    if training_run_dir is not None:
        count = training_manifest.get("n_envs")
        if (
            type(count) is not int
            or count < 1
            or count != training_manifest["config"].get("n_envs")
        ):
            raise ValueError("native Monitor coverage requires matching n_envs metadata")
        expected = {f"env-{rank:03d}.monitor.csv" for rank in range(count)}
        found = {path.name for path in (Path(training_run_dir) / "monitor").glob("*.monitor.csv")}
        if found != expected:
            raise ValueError(
                "all native environment Monitor files are required; "
                "partial episode rates are unsafe"
            )
    training, events = summarize_training_episodes(training_run_dir)
    for period, (start, end) in zip(periods, bounds):
        subset, requested, missing = _period_dataset(evaluation_data, start, end, config.lookback)
        for mode, mode_config in modes:
            trace = trace_policy(model, subset, mode_config) if subset is not None else None
            complete = trace is not None and trace.index.equals(requested) and missing.empty
            values = trace["return"] if trace is not None else pd.Series(dtype=float)
            role = {
                "train": "in_sample_reference",
                "validation": "model_selection",
                "test": "held_out_test",
            }
            metrics.append(
                {
                    "period": period.name,
                    "mode": mode,
                    "role": role[period.name],
                    "requested_start": start.isoformat(),
                    "requested_end_exclusive": end.isoformat(),
                    "first_return": None if trace is None else trace.index[0].isoformat(),
                    "last_return": None if trace is None else trace.index[-1].isoformat(),
                    "observations": len(values),
                    "complete": complete,
                    "coverage_basis": "input_index",
                    **_metrics(values),
                }
            )
            detail = {
                "period": period.name,
                "mode": mode,
                "complete": complete,
                "coverage_basis": "input_index",
                "requested_observations": len(requested),
                "evaluated_observations": len(values),
                "missing_context_dates": [d.isoformat() for d in missing],
                "reason": (
                    "insufficient_context_or_empty" if trace is None else trace.iloc[-1]["reason"]
                ),
                "early_terminated": (
                    trace is not None
                    and bool(trace.iloc[-1].terminated)
                    and len(values) < len(requested) - len(missing)
                ),
                "evaluation_environment": evaluation_env_config(mode_config).model_dump(
                    mode="json"
                ),
                "comparisons": {
                    name: {"status": "unavailable"} for name in ("equal_weight", "benchmark")
                },
            }
            for name, ref in (references or {}).get(period.name, {}).items():
                detail["comparisons"][name] = _reference_comparison(
                    trace, requested, ref, config, currency, complete
                )
                metrics[-1][name + "_excess_cumulative_return"] = detail["comparisons"][name][
                    "excess_cumulative_return"
                ]
            if trace is not None:
                trace["period"], trace["mode"], trace["regime"] = period.name, mode, "unknown"
                if labels is not None:
                    # Decision-time labels use only pre-block history, never the ensuing return.
                    trace["regime"] = (
                        labels.reindex(pd.DatetimeIndex(trace.decision_at))
                        .fillna("unknown")
                        .to_numpy()
                    )
                detail.update(_drawdown_crossings(trace))
                for date, row in trace.loc[trace.terminated].iterrows():
                    events.append(
                        {
                            "scope": "evaluation",
                            "period": period.name,
                            "mode": mode,
                            "date": date.isoformat(),
                            "nav": row.nav,
                            "drawdown": row.drawdown,
                            "reason": row.reason,
                            "regime": row.regime,
                        }
                    )
                histories.append(trace.reset_index())
            details.append(detail)
    history = (
        pd.concat(histories, ignore_index=True)
        if histories
        else pd.DataFrame(columns=["date", "period", "mode"])
    )
    summary = {
        "schema_version": 1,
        "status": "unverified",
        "training_approved": False,
        "kind": provenance["kind"],
        "provenance": provenance,
        "policy_version": policy_version,
        "currency": currency,
        "seed": training_manifest.get("seed"),
        "environment": config.model_dump(mode="json"),
        "training_config_sha256": training_manifest.get("config_sha256"),
        "training_hyperparameters": training_manifest["config"].get("ppo"),
        "training_package_versions": training_manifest.get("package_versions"),
        "evaluation_package_versions": {
            "python": sys.version.split()[0],
            **{
                name: version(name)
                for name in ("stable-baselines3", "torch", "gymnasium", "numpy", "pandas")
            },
        },
        "dataset_file_sha256": dataset.file_sha256,
        "assets": list(dataset.returns.columns),
        "dataset_fingerprints": {
            name: dataframe_fingerprint(frame)
            for name, frame in (
                ("returns", dataset.returns),
                ("rsi", dataset.rsi),
                ("macd", dataset.macd),
            )
        },
        "regime_rules": regime_rules,
        "periods": details,
        "training_episodes": training,
        "source_validation": source_validation,
        "causal_attribution": "unverified",
        "limitations": [
            "Train은 in-sample 참고, Validation은 모델 선택, Test는 고정 설정의 평가다.",
            "Validation은 재평가다. 첫날 수익률을 포함하므로 native validation.best/history 점수를 직접 재현한 값이 아니다.",
            "미완주 구간은 완주한 전략과 직접 비교하지 않으며 수익률을 채우지 않는다.",
            "에피소드 종료는 실제 보유자산의 자동 매도가 아니다.",
            "mdd_limit=1 비교는 같은 모델의 추론 민감도이며 Safe-Guard 없는 재학습 실험이 아니다.",
            "원천 공개시각·지표 인과성·Train 전용 정규화는 파일 해시만으로 증명되지 않는다.",
            "과적합·학습 부족·보상·데이터 오류의 인과적 원인은 이 보고서만으로 확정하지 않는다.",
            "기존 Monitor에는 시장 날짜/NAV/MDD가 없어 학습 종료의 국면별 빈도는 미확인이다.",
            "균등 비중 유사성은 L1 거리 원시 수치이며 정체/집중 판단 임계값은 팀 합의가 필요하다.",
            "평가 대상 날짜 수는 입력 데이터 인덱스 기준이며 거래일 누락 여부는 별도 원천 감사를 요구한다.",
        ],
    }
    indicator_columns = [
        "hhi",
        "max_risky_weight",
        "target_cash",
        "actual_cash",
        "target_change_l1",
        "risky_equal_l1",
        "all_equal_l1",
    ]
    summary["allocation_indicators"] = (
        history.groupby(["period", "mode"], sort=False)[indicator_columns]
        .mean()
        .reset_index()
        .to_dict(orient="records")
        if not history.empty
        else []
    )
    event_frame = (
        pd.DataFrame(events)
        if events
        else pd.DataFrame(
            columns=["scope", "period", "mode", "date", "nav", "drawdown", "reason", "regime"]
        )
    )
    return DiagnosticResult(
        summary,
        pd.DataFrame(metrics),
        history,
        _allocation_summary(history, dataset.returns.columns),
        event_frame,
    )


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def compare_experiment_settings(left: DiagnosticResult, right: DiagnosticResult) -> dict:
    """다른 실행의 조건 차이만 나열한다. 다른 조건의 성과를 자동으로 비교하지 않는다."""
    fields = (
        "policy_version",
        "currency",
        "seed",
        "environment",
        "training_hyperparameters",
        "dataset_file_sha256",
        "dataset_fingerprints",
        "assets",
        "regime_rules",
    )
    mismatches = [
        name
        for name in fields
        if _json_safe(left.summary.get(name)) != _json_safe(right.summary.get(name))
    ]
    columns = [
        "period",
        "mode",
        "requested_start",
        "requested_end_exclusive",
        "first_return",
        "last_return",
        "complete",
    ]
    if not left.period_metrics[columns].equals(right.period_metrics[columns]):
        mismatches.append("evaluation_coverage")
    return {
        "same_conditions": not mismatches,
        "mismatches": mismatches,
        "performance_comparison": "not_performed",
    }


def save_diagnostics(result: DiagnosticResult, output_dir: str | Path) -> Path:
    """새 디렉터리에만 저장한다. 기존 보고서·모델·데이터는 덮어쓰지 않는다."""
    if (
        result.summary.get("status") != "unverified"
        or result.summary.get("training_approved") is not False
    ):
        raise ValueError("diagnostics cannot certify performance or approve training data")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    created = []

    def write(name, content):
        path = output / name
        with path.open("x", encoding="utf-8", newline="") as stream:
            created.append(path)
            stream.write(content)

    def dump(value):
        return (
            json.dumps(
                _json_safe(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
            )
            + "\n"
        )

    try:
        write("summary.json", dump(result.summary))
        for name, frame in (
            ("period_metrics", result.period_metrics),
            ("allocation_history", result.allocation_history),
            ("allocation_summary", result.allocation_summary),
            ("safeguard_events", result.safeguard_events),
        ):
            labeled = frame.assign(result_kind=result.summary["kind"])
            write(name + ".csv", labeled.to_csv(index=False, lineterminator="\n"))
        report = "# PPO 모델 신뢰성 진단\n\n"
        report += (
            f"자료 구분: {result.summary['kind']} / 상태: unverified / training_approved=false\n\n"
        )
        report += (
            "## 모델·원천 검증 상태\n\n```json\n" + dump(result.summary["provenance"]) + "```\n\n"
        )
        report += (
            "## 구간 성과\n\n```csv\n"
            + result.period_metrics.to_csv(index=False, lineterminator="\n")
            + "```\n\n"
        )
        report += (
            "## 학습 종료 기록\n\n```json\n" + dump(result.summary["training_episodes"]) + "```\n\n"
        )
        report += (
            "## 비중 관찰 지표(구간 평균)\n\n```json\n"
            + dump(result.summary["allocation_indicators"])
            + "```\n\n"
        )
        report += (
            "## 비교 가능 여부와 종료 시점\n\n```json\n"
            + dump(result.summary["periods"])
            + "```\n\n"
        )
        report += (
            "## 해석 한계\n\n"
            + "\n".join("- " + item for item in result.summary["limitations"])
            + "\n"
        )
        write("diagnostic_report.md", report)
        hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in created}
        write(
            "manifest.json",
            dump(
                {
                    "schema_version": 1,
                    "kind": result.summary["kind"],
                    "status": "unverified",
                    "training_approved": False,
                    "policy_version": result.summary["policy_version"],
                    "inputs": result.summary["provenance"],
                    "environment": result.summary["environment"],
                    "dataset_file_sha256": result.summary["dataset_file_sha256"],
                    "output_sha256": hashes,
                    "limitations": result.summary["limitations"],
                }
            ),
        )
    except Exception:
        for path in reversed(created):
            path.unlink()
        output.rmdir()
        raise
    return output
