"""전략 거래 경로를 보존하는 단일 윈도우 비교 백테스트."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from robo_advisor.backtest.equal_weight_backtest import run_equal_weight_backtest
from robo_advisor.backtest.metrics import calculate_metrics
from robo_advisor.backtest.mvo_backtest import run_mvo_backtest
from robo_advisor.data.benchmark_audit import audit_benchmark_data, validate_comparison_periods
from robo_advisor.diagnostics.inputs import prepare_diagnostic_inputs, read_pinned
from robo_advisor.diagnostics.ppo_diagnostics import (
    DiagnosticPeriod,
    _drawdown_crossings,
    _json_safe,
    _validate_contract,
    diagnose_ppo,
)
from robo_advisor.experiments.anova_samples import (
    average_seed_returns,
    block_samples,
    reward_samples,
)


@dataclass(frozen=True)
class ComparisonPolicy:
    version: str
    first_return: str  # include_first_session / after_first_observation
    safeguard: str  # configured / observe_only
    return_type: str  # PR / TR
    fx_source: str  # alfred / smbs
    mvo_lookback: int = 252
    mvo_max_weight: float = 0.4

    def __post_init__(self):
        if (
            not self.version.strip()
            or self.first_return not in {"include_first_session", "after_first_observation"}
            or self.safeguard not in {"configured", "observe_only"}
            or self.return_type not in {"PR", "TR"}
            or self.fx_source not in {"alfred", "smbs"}
            or type(self.mvo_lookback) is not int
            or self.mvo_lookback < 2
            or not 0 < self.mvo_max_weight <= 1
        ):
            raise ValueError("invalid candidate comparison policy")


@dataclass(frozen=True)
class ComparisonResult:
    manifest: dict
    tables: dict[str, pd.DataFrame]


def _aware(values):
    dates = [pd.Timestamp(value) for value in values]
    if any(pd.isna(date) or date.tzinfo is None for date in dates):
        raise ValueError("evaluation timestamps must be timezone aware")
    return pd.to_datetime(dates, utc=True)


def _schedule_intervals(schedule, start, end, first_return):
    index = _aware(schedule)
    if not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError("evaluation schedule must be ordered and unique")
    local = index.tz_convert("Asia/Seoul").normalize()
    if not local.is_unique:
        raise ValueError("expected one evaluation per local session")
    start, end = _aware([start, end])
    positions = np.flatnonzero((index >= start) & (index < end))
    if first_return == "after_first_observation":
        positions = positions[1:]
    if len(positions) < 2 or positions[0] == 0:
        raise ValueError("missing predecessor or fewer than two evaluation transitions")
    return pd.Series(index[positions - 1], index=index[positions], name="return_start")


def _validate_transitions(left, right):
    if left.empty and right.empty:
        return

    def sessions(series):
        def local_dates(index):
            return index.tz_convert("Asia/Seoul").tz_localize(None).normalize()

        return pd.Series(local_dates(_aware(series)), index=local_dates(series.index))

    validate_comparison_periods(sessions(left), sessions(right))
    if not left.index.equals(right.index) or not pd.DatetimeIndex(left).equals(
        pd.DatetimeIndex(right)
    ):
        raise ValueError("return transition timestamps must match exactly")


def composite_benchmark(report, expected, policy, *, test_end=None):
    """감사에서 재현한 두 단순수익률을 결측 보간 없이 일별 50:50으로 합성한다."""
    if not report or report["status"] == "failed":
        return pd.Series(np.nan, index=expected.index, name="benchmark")
    if report["status"] not in {"passed", "unverified"}:
        raise ValueError("unknown benchmark audit status")
    evidence = report["metrics"]
    if evidence["fx"]["source"] != policy.fx_source:
        raise ValueError("benchmark FX source differs from candidate policy")
    if any(
        evidence["sources"][leg]["return_type"] != policy.return_type for leg in ("KOSPI", "SP500")
    ):
        raise ValueError("benchmark and strategy return types differ")
    rows = pd.DataFrame(evidence["rows"])
    dates = _aware(rows["evaluation_at"])
    sessions = pd.DatetimeIndex(pd.to_datetime(rows["session_date"]))
    if (
        not dates.is_unique
        or not dates.is_monotonic_increasing
        or not sessions.equals(dates.tz_convert("Asia/Seoul").tz_localize(None).normalize())
    ):
        raise ValueError("benchmark session and evaluation timestamps differ")
    planned = expected.index.insert(0, expected.iloc[0])
    audit_scope = dates >= planned[0]
    if test_end is None:
        audit_scope &= dates <= planned[-1]
    else:
        audit_scope &= dates < _aware([test_end])[0]
    local_plan = planned.tz_convert("Asia/Seoul").tz_localize(None).normalize()
    if not sessions[audit_scope].difference(local_plan).empty:
        raise ValueError("audit calendar contains missing comparison sessions")
    by_session = dict(zip(sessions, dates))
    starts = pd.Series(
        [by_session.get(pd.Timestamp(value), pd.NaT) for value in rows["return_start_date"]],
        index=dates,
        dtype="datetime64[ns, UTC]",
    )
    legs = rows[["kospi_simple_return", "sp500_krw_simple_return"]].astype(float)
    legs.index = dates
    if np.isinf(legs.to_numpy()).any() or (legs < -1).any().any():
        raise ValueError("invalid benchmark returns")
    usable = legs.notna().all(axis=1)
    matched_ends = expected.index.intersection(dates[usable])
    _validate_transitions(expected.loc[matched_ends], starts.loc[matched_ends])
    # Reject mismatched times on the same session, rather than treating them as missing.
    actual_by_day = dict(zip(sessions, dates))
    for instant in expected.index.difference(dates):
        day = instant.tz_convert("Asia/Seoul").tz_localize(None).normalize()
        if day in actual_by_day:
            raise ValueError("benchmark evaluation time differs on the same session")
    return legs.mean(axis=1, skipna=False).reindex(expected.index).rename("benchmark")


def _strategy_path(result, starts):
    path = pd.DataFrame({"return": result.returns, "nav": result.nav})
    path.index = path.index.tz_convert("UTC")
    if not path.index.equals(starts.index):
        raise ValueError("baseline evaluation dates differ")
    path["return_start"] = starts
    rebalances = result.rebalances.join(result.weights)
    rebalances.index = rebalances.index.tz_convert("UTC")
    rebalances = rebalances.rename(columns={"cost": "cost_fraction"})
    prior_nav = path.nav.shift(fill_value=1.0)
    rebalances["cost"] = rebalances.cost_fraction * prior_nav.loc[rebalances.index]
    return path, rebalances


def compare_window(*, schedule, benchmark_audit, policy: ComparisonPolicy, **diagnostic_inputs):
    """후보 정책으로 한 윈도우를 실행하고 실제 거래 경로의 공통 구간만 비교한다."""
    args = diagnostic_inputs
    if args["currency"] != "KRW":
        raise ValueError("comparison requires KRW inputs")
    periods = args["periods"]
    dataset, config = args["dataset"], args["config"]
    _validate_contract(dataset, args["training_manifest"], config, periods)
    test = periods[-1]
    starts = _schedule_intervals(schedule, test.start, test.end, policy.first_return)
    calendar = _aware(schedule)
    lower, upper = _aware([periods[0].start, test.end])
    expected_dates = calendar[(calendar >= lower) & (calendar < upper)]
    asset_returns = dataset.returns.copy()
    asset_returns.index = _aware(asset_returns.index)
    asset_returns = asset_returns.loc[
        (asset_returns.index >= lower) & (asset_returns.index < upper)
    ]
    if not asset_returns.index.equals(expected_dates):
        raise ValueError("asset input has missing or off-calendar sessions; gaps cannot be rolled")
    if starts.iloc[0] not in asset_returns.index:
        raise ValueError("first transition predecessor missing from asset inputs")
    benchmark = composite_benchmark(benchmark_audit, starts, policy, test_end=test.end)
    effective_test = DiagnosticPeriod("test", starts.index[0], test.end)
    diagnostic = diagnose_ppo(
        **{
            **args,
            "periods": [*periods[:-1], effective_test],
            "sensitivity": policy.safeguard == "observe_only",
            "references": None,
        }
    )
    mode = (
        "no_early_stop_inference"
        if policy.safeguard == "observe_only" and config.mdd_limit < 1
        else "configured"
    )
    history = diagnostic.allocation_history
    trace = history.loc[history.period.eq("test") & history["mode"].eq(mode)].set_index("date")
    if trace.empty:
        trace.index = pd.DatetimeIndex([], tz="UTC", name="date")
        trace = trace.reindex(columns=["return", "nav", "return_start", "cost", "turnover"])
    elif not trace["return_start"].equals(starts.reindex(trace.index)):
        raise ValueError("PPO transitions differ from comparison calendar")
    costs = {"commission_rate": config.commission_rate, "slippage_rate": config.slippage_rate}
    local = asset_returns.tz_convert("Asia/Seoul")
    mvo = run_mvo_backtest(
        local,
        start=starts.index[0],
        end=test.end,
        lookback=policy.mvo_lookback,
        max_weight=policy.mvo_max_weight,
        **costs,
    )
    equal = run_equal_weight_backtest(local.loc[starts.index.tz_convert("Asia/Seoul")], **costs)
    mvo_path, mvo_trades = _strategy_path(mvo, starts)
    equal_path, equal_trades = _strategy_path(equal, starts)
    paths = {"ppo": trace, "mvo": mvo_path, "equal_weight": equal_path}
    all_strategies_complete = trace.index.equals(starts.index)
    benchmark_complete = bool(benchmark.notna().all())
    comparable = all_strategies_complete and benchmark_complete
    paths["benchmark"] = pd.DataFrame(
        {
            "return": benchmark,
            "nav": (1 + benchmark).cumprod() if benchmark_complete else np.nan,
            "return_start": starts,
        }
    )
    metrics, events, nav_rows = [], [], []
    coverage = pd.DataFrame({"return_start": starts})
    for name, path in paths.items():
        observed = path["return"].dropna()
        full = observed.index.equals(starts.index)
        coverage[name + "_observed"] = starts.index.isin(observed.index)
        benchmark_ref = benchmark if comparable and name != "benchmark" else None
        statistics = calculate_metrics(observed, benchmark_ref) if full else {}
        metrics.append(
            {
                "strategy": name,
                "complete": full,
                "benchmark_comparable": benchmark_ref is not None,
                **statistics,
            }
        )
        nav = path.nav.dropna()
        if not nav.empty:
            initial = starts.loc[nav.index[0]]
            nav_rows.append(
                pd.concat([pd.Series([1.0], index=[initial]), nav])
                .rename("nav")
                .to_frame()
                .assign(strategy=name)
            )
            peaks = nav.cummax().clip(lower=1.0)
            events.append(
                {
                    "strategy": name,
                    **_drawdown_crossings(pd.DataFrame({"nav": nav, "peak_nav": peaks})),
                }
            )
    coverage["all_available"] = coverage.filter(like="_observed").all(axis=1)
    statistics = pd.DataFrame(metrics)
    statistics["excess_vs_equal_weight"] = np.nan
    if all_strategies_complete:
        perfect = statistics.set_index("strategy")["cumulative_return"]
        statistics.loc[statistics.strategy.eq("ppo"), "excess_vs_equal_weight"] = (
            perfect["ppo"] - perfect["equal_weight"]
        )
    ppo_trades = trace.loc[trace.turnover.gt(0)].copy()
    if not ppo_trades.empty:
        ppo_trades["cost_fraction"] = ppo_trades.cost / trace.nav.shift(fill_value=1.0)
    trade_tables = {"ppo": ppo_trades, "mvo": mvo_trades, "equal_weight": equal_trades}
    tables = {
        "daily_returns": pd.concat(
            [path[["return_start", "return"]].assign(strategy=name) for name, path in paths.items()]
        )
        .rename_axis("return_end")
        .reset_index(),
        "nav_history": pd.concat(nav_rows).rename_axis("evaluation_at").reset_index(),
        "strategy_metrics": statistics,
        "rebalance_history": pd.concat(
            [path.assign(strategy=name) for name, path in trade_tables.items()]
        )
        .rename_axis("return_end")
        .reset_index(),
        "safeguard_events": pd.DataFrame(events),
        "evaluation_coverage": coverage.rename_axis("return_end").reset_index(),
        "ppo_period_metrics": diagnostic.period_metrics,
        "ppo_allocation_summary": diagnostic.allocation_summary,
    }
    audit_metadata = dict(benchmark_audit or {})
    if benchmark_audit:
        audit_metadata["metrics"] = {
            key: value for key, value in benchmark_audit["metrics"].items() if key != "rows"
        }
        audit_metadata["training_approved"] = False
    manifest = {
        "schema_version": 1,
        "kind": diagnostic.summary["kind"],
        "status": "unverified",
        "training_approved": False,
        "policy_status": "candidate",
        "performance_decision": "pending_protocol_and_evidence",
        "comparison_policy": asdict(policy),
        "rebalance_policy": {
            "ppo": "deterministic_each_transition",
            "mvo": "first_transition_and_first_session_of_seoul_month",
            "equal_weight": "first_transition_and_first_session_of_seoul_month",
            "benchmark": "daily_50_50",
        },
        "ppo_evaluation_mode": mode,
        "comparable": comparable,
        "strategy_comparison_complete": all_strategies_complete,
        "benchmark_calculation": "complete" if benchmark_complete else "incomplete",
        "benchmark_source_status": (benchmark_audit or {}).get("status", "unavailable"),
        "benchmark": audit_metadata,
        "costs": {**costs, "benchmark": "cost_free_index"},
        "initial_nav": 1.0,
        "initial_cash_weight": 1.0,
        "currency": "KRW",
        "requested_test": {"start": _aware([test.start])[0], "end": _aware([test.end])[0]},
        "first_return_start": starts.iloc[0],
        "first_return_end": starts.index[0],
        "last_return_end": starts.index[-1],
        "evaluation_schedule_sha256": hashlib.sha256(calendar.asi8.tobytes()).hexdigest(),
        "ppo": diagnostic.summary,
        "anova": {"status": "not_run", "reason": "single_window_and_checkpoint"},
    }
    return ComparisonResult(manifest, tables)


def prepare_anova_reward_samples(
    runs, *, rewards, seeds, windows, selected_reward=None, block_size=None
):
    """전체 설계가 충족되면 #42의 표본을 반환한다. 검정은 호출하지 않는다."""
    expected = set(product(rewards, seeds, windows))
    if len(set(rewards)) < 2 or not seeds or not windows:
        raise ValueError("explicit rewards, seeds and windows are required")
    if set(runs) - expected:
        raise ValueError("unexpected reward/seed/window result")
    if (selected_reward is None) != (block_size is None) or (
        selected_reward is not None and selected_reward not in rewards
    ):
        raise ValueError("selected reward and block size must be specified together")
    missing, incomplete = sorted(expected - set(runs)), []
    returns, protocols, run_ids = {}, set(), set()
    window_paths, window_inputs = {}, {}
    for key, result in runs.items():
        metadata = result.manifest
        if not metadata["comparable"] or metadata["kind"] == "synthetic":
            incomplete.append(key)
            continue
        ppo = metadata["ppo"]
        run_id = ppo["provenance"]["run_id"]
        if (
            run_id in run_ids
            or ppo["seed"] != key[1]
            or (ppo["environment"]["reward_variant"] != key[0])
        ):
            raise ValueError("ANOVA must identify distinct runs with matching reward and seed")
        run_ids.add(run_id)
        protocols.add(
            json.dumps([metadata["comparison_policy"], metadata["costs"]], sort_keys=True)
        )
        path = result.tables["daily_returns"].query("strategy == 'ppo'").set_index("return_end")
        window = key[-1]
        hashes = (ppo["dataset_file_sha256"], metadata["evaluation_schedule_sha256"])
        if window in window_paths:
            _validate_transitions(window_paths[window], path.return_start)
            if hashes != window_inputs[window]:
                raise ValueError("ANOVA window inputs differ between runs")
        window_paths[window] = path.return_start
        window_inputs[window] = hashes
        returns[key] = path["return"]
    if len(protocols) > 1:
        raise ValueError("ANOVA inputs use different candidate policies")
    if missing or incomplete:
        return {
            "status": "insufficient",
            "missing": missing,
            "incomplete": sorted(incomplete),
            "samples": None,
        }
    result = {"status": "candidate_samples", "samples": reward_samples(returns), "blocks": None}
    if selected_reward is not None:
        frames = {}
        for window in windows:
            selected = [runs[selected_reward, seed, window] for seed in seeds]
            baseline = None
            for run in selected:
                daily = run.tables["daily_returns"]
                pair = daily.query("strategy in ['mvo', 'equal_weight']").pivot(
                    index="return_end", columns="strategy", values="return"
                )
                if baseline is not None and not pair.equals(baseline):
                    raise ValueError("ANOVA benchmark strategies differ across seeds")
                baseline = pair
            seed_returns = pd.DataFrame(
                {seed: returns[selected_reward, seed, window] for seed in seeds}
            )
            frames[window] = baseline.assign(ppo=average_seed_returns(seed_returns))
        result["blocks"] = block_samples(frames, block_size=block_size)
    return result


def save_comparison(result, output):
    """새 디렉터리에만 쓰고 각 파일의 SHA-256을 manifest에 남긴다."""
    metadata = result.manifest
    if metadata["status"] != "unverified" or metadata["training_approved"] is not False:
        raise ValueError("comparison output cannot certify training or performance")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, table in result.tables.items():
        path = output / f"{name}.csv"
        table.assign(
            result_kind=metadata["kind"],
            policy_version=metadata["comparison_policy"]["version"],
            status="unverified",
            training_approved=False,
        ).to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = output / "comparison_report.md"
    report.write_text(
        "# 통합 비교 백테스트\n\n"
        f"- 자료 유형: {metadata['kind']}\n"
        f"- 4개 경로 비교 가능: {metadata['comparable']}\n"
        "- 후보 정책. 성능 판정 보류. 원천 상태 unverified, training_approved=false.\n"
        "- 거래 경로를 유지하며 빠진 벤치마크 구간을 보간하지 않는다.\n",
        encoding="utf-8",
        newline="\n",
    )
    files[report.name] = hashlib.sha256(report.read_bytes()).hexdigest()
    (output / "manifest.json").write_text(
        json.dumps(
            _json_safe({**metadata, "output_sha256": files}),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args(argv)
    if cli.output.exists():
        raise FileExistsError(cli.output)
    spec_bytes = cli.spec.read_bytes()
    spec = json.loads(spec_bytes)
    schedule = pd.read_csv(io.BytesIO(read_pinned(cli.root, spec["schedule"])))
    if list(schedule.columns) != ["evaluation_at"]:
        raise ValueError("schedule CSV must have one evaluation_at column")
    benchmark_spec = spec.get("benchmark")
    audit = (
        audit_benchmark_data(json.loads(read_pinned(cli.root, benchmark_spec)), root=cli.root)
        if benchmark_spec
        else None
    )
    diagnostic_bytes = read_pinned(cli.root, spec["diagnostic"])
    with prepare_diagnostic_inputs(diagnostic_bytes, root=cli.root) as arguments:
        result = compare_window(
            schedule=schedule.evaluation_at,
            benchmark_audit=audit,
            policy=ComparisonPolicy(**spec["policy"]),
            **arguments,
        )
    result.manifest["file_inputs"] = {
        name: spec.get(name) for name in ("schedule", "benchmark", "diagnostic")
    }
    result.manifest["comparison_spec_sha256"] = hashlib.sha256(spec_bytes).hexdigest()
    print(save_comparison(result, cli.output))


if __name__ == "__main__":
    main()
