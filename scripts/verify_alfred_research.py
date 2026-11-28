"""Read-only PPO handoff checks for the recovered ALFRED study."""

import argparse
from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from recover_team12_research import DOMESTIC, END, START, US, WINDOWS, _prices

from robo_advisor.data.alignment import convert_prices_to_krw
from robo_advisor.data.audit import _anchored_file, _fx_input, _fx_replay, _sha
from robo_advisor.data.snapshot import read_snapshot
from robo_advisor.features.dataset import WARMUP_ROWS, assess_dataset_readiness
from robo_advisor.features.dataset_io import FeatureDatasetIOError, read_prepared_research_dataset
from robo_advisor.features.ppo_input import convert_ppo_inputs
from robo_advisor.features.scaling import TrainOnlyStandardScaler


def _check_values(actual, expected, label):
    if not actual.index.equals(expected.index) or not np.allclose(
        actual, expected, atol=1e-11, rtol=1e-10
    ):
        raise ValueError(f"Recalculation disagrees with saved {label}")
    return float((actual - expected).abs().to_numpy().max())


def _check_domestic(snapshot, proof, corporate_actions, prices, segment, features):
    output = {}
    for asset in DOMESTIC:
        item = snapshot.datasets[asset]
        old = item.raw_data
        added = item.data.index.difference(old.index)
        applied = proof.loc[(proof["asset"] == asset) & proof["state"].eq("recovered")]
        cols = ["Open", "High", "Low", "Close", "Volume"]
        pd.testing.assert_frame_equal(
            item.data.loc[added, cols], applied[cols], check_dtype=False, check_names=False
        )
        pd.testing.assert_series_equal(item.data.loc[old.index, "Close"], old["Close"])
        if not item.data.loc[added, "Adj Close"].isna().all():
            raise ValueError("Recovered market quotes must preserve the missing adjusted fields")

        close = prices.loc[segment, asset]
        close_delta = close.diff()
        gain = close_delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        loss = -close_delta.clip(upper=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        rsi = (100 * gain / (gain + loss)).mask(gain.eq(0) & loss.eq(0), 50)
        macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        trailing = np.log(close / prices[asset].shift().reindex(segment))
        offsets = features.index.get_indexer(added)
        if (offsets < 0).any():
            raise ValueError("A recovered quote is outside the observed evaluation grid")
        affected = features.index[
            np.unique(np.clip((offsets[:, None] + [-1, 0, 1]).ravel(), 0, len(features) - 1))
        ]
        expected = {
            "log_return": trailing,
            "rsi14": rsi,
            "macd": macd / close,
            "macd_signal": macd_signal / close,
            "macd_histogram": (macd - macd_signal) / close,
        }
        residuals = {
            kind: _check_values(
                features.loc[affected, f"{asset}__{kind}"], values.loc[affected], f"{asset} {kind}"
            )
            for kind, values in expected.items()
        }
        action_days = pd.DatetimeIndex(
            [
                entry["local_date"]
                for entry in corporate_actions[asset]["corporate_actions_reported_by_yahoo"]
            ]
        )
        if added.intersection(action_days).size:
            raise ValueError("Recovered price collides with an event requiring separate review")
        event_closes = item.data.loc[:, ["Close", "Adj Close"]]
        event_returns = event_closes.pct_change(fill_method=None).loc[
            event_closes.index.intersection(action_days)
        ]
        event_returns = event_returns.loc[START:END].dropna()
        simple = np.expm1(features[f"{asset}__log_return"])
        output[asset] = {
            "recovered_rows": len(added),
            "audited_recovery_neighborhood_rows": len(affected),
            "maximum_feature_recalculation_residual": residuals,
            "affected_max_absolute_simple_return": float(simple.loc[affected].abs().max()),
            "all_max_absolute_simple_return": float(simple.abs().max()),
            "all_dates_at_or_above_10pct_absolute_return": int(simple.abs().ge(0.10).sum()),
            "provider_action_dates_in_period": len(event_returns),
            "max_raw_vs_adjusted_return_difference_on_reported_events": (
                float(event_returns["Close"].sub(event_returns["Adj Close"]).abs().max())
                if not event_returns.empty
                else None
            ),
        }
    return output


def _check_windows(frames, lineage):
    results = {}
    root = frames["features"]
    for window in WINDOWS:
        prefix = f"windows/{window.window_id}"
        splits, rewards = frames[f"{prefix}/splits"], frames[f"{prefix}/rewards"]
        bounds = [
            pd.Timestamp(d)
            for d in (window.train_start, window.train_end, window.validation_end, window.test_end)
        ]
        plan = dict(zip(("train", "validation", "test"), zip(bounds[:-1], bounds[1:])))
        if not splits.index.equals(
            root.loc[(root.index >= bounds[0]) & (root.index < bounds[-1])].index
        ):
            raise ValueError("Window data does not exactly cover the declared range")
        if not rewards.loc[~splits["reward_eligible"]].isna().all().all():
            raise ValueError("Reward crosses a split or data boundary")
        results[window.window_id] = {}
        for name, (start, end) in plan.items():
            rows = splits.loc[splits["split"].eq(name)]
            expected = root.loc[(root.index >= start) & (root.index < end)]
            if not rows.index.equals(expected.index) or not rows["eligible_observation"].all():
                raise ValueError(f"Unexpected split date coverage for {window.window_id} {name}")
            if not rewards.loc[rows.index[-1]].isna().all():
                raise ValueError(f"Last reward must be unavailable for {window.window_id} {name}")
            results[window.window_id][name] = {
                "start": str(start.date()),
                "end_exclusive": str(end.date()),
                "rows": len(rows),
                "longest_contiguous_rows": len(rows),
            }
            if name == "train":
                record = lineage["windows"][window.window_id]["scaler"]
                scaler = TrainOnlyStandardScaler().fit(expected)
                if (
                    record["fit_rows"] != len(rows)
                    or not np.allclose(scaler.mean_, record["mean"])
                    or not np.allclose(scaler.scale_, record["scale"])
                ):
                    raise ValueError("Window scaler must be fit on Train only")
            if name == "test":
                monthly = rows.index[~rows.index.to_period("M").duplicated()]
                history = frames["rows"].loc[monthly, "warmup_position"]
                results[window.window_id][name].update(
                    {
                        "mvo_first_session_preceding_returns": int(history.iloc[0]),
                        "mvo_month_starts": len(monthly),
                        "mvo_month_starts_with_252": int(history.ge(252).sum()),
                    }
                )
    return results


def verify(args):
    cfg = json.loads(args.config.read_bytes())
    data = read_prepared_research_dataset(args.dataset, expected_lineage_sha256=args.lineage_sha256)
    frames, lineage = data["frames"], data["lineage"]
    if (
        lineage["alignment"]["fx_scenario"] != "alfred"
        or lineage["warmup"]["rows_per_segment"] != WARMUP_ROWS
    ):
        raise ValueError("Expected the recovered ALFRED 33-observation study")
    snapshot = read_snapshot(args.snapshot, expected_manifest_sha256=args.snapshot_sha256)
    info = json.loads(args.evidence_lineage.read_bytes())
    proof_hash = _sha(args.recovery_evidence)
    if (
        lineage["source_snapshot"]["manifest_sha256"] != args.snapshot_sha256
        or info["restored_snapshot_sha256"] != args.snapshot_sha256
        or info["recovery_evidence_sha256"] != proof_hash
        or lineage["alignment"]["recovery_evidence_sha256"] != proof_hash
    ):
        raise ValueError("Snapshot, derivative and recovery evidence pins disagree")

    calendars = [
        pd.read_csv(
            _anchored_file(args.archive, cfg["calendars"][market]), index_col=0, parse_dates=True
        )
        for market in ("KRX", "US")
    ]
    raw, evaluation, _ = _prices(snapshot, {}, calendars)
    rate_spec = cfg["scenarios"]["alfred"]["fx"]
    if lineage["alignment"]["fx_file_sha256"] != rate_spec["sha256"]:
        raise ValueError("Independent FX source pin differs from derivative")
    observations = _fx_input(pd.read_csv(_anchored_file(args.archive, rate_spec)), "alfred")
    fx = _fx_replay(observations, evaluation, "USD/KRW", lineage["alignment"]["fx_max_age_days"])
    prices = raw.copy()
    prices[US] = convert_prices_to_krw(raw[US], fx["rate"], fx_direction="USD/KRW")
    complete = prices.notna().all(axis=1)
    eligible = complete & complete.shift(1, fill_value=False)
    groups = (~eligible).cumsum()
    dates = frames["rows"].index
    segment = prices.index[eligible & groups.eq(groups.loc[dates[0]])]
    if not segment[segment >= START].equals(dates) or not frames["rows"]["feature_ready"].all():
        raise ValueError("Model dates must form a single feature-ready contiguous segment")
    prior = segment[segment < dates[0]]
    recorded = frames["rows"]["warmup_position"]
    if len(prior) < WARMUP_ROWS or not np.array_equal(
        recorded, np.arange(len(prior), len(segment))
    ):
        raise ValueError("Recorded warmup cannot be reconstructed from pre-model observations")
    stored = frames["rows"]
    for label in (
        "rate",
        "observation_age_hours",
        "observation_date",
        "actual_available_at",
        "assumed_usable_at",
        "reference_at",
    ):
        pd.testing.assert_series_equal(
            stored[f"FX__{label}"], fx.loc[dates, label], check_names=False
        )
    actual = pd.to_datetime(stored["FX__actual_available_at"], utc=True)
    effective = actual.fillna(stored["FX__assumed_usable_at"])
    if (
        effective.gt(stored["evaluation_at"]).any()
        or effective.lt(stored["FX__reference_at"]).any()
    ):
        raise ValueError("FX release chronology disagrees with the chosen evaluation time")
    expected = (dates + pd.Timedelta(hours=15, minutes=30)).tz_localize("Asia/Seoul")
    trailing = np.log(prices).diff().loc[dates]
    all_return_residual = _check_values(
        frames["features"][trailing.columns + "__log_return"].set_axis(trailing.columns, axis=1),
        trailing,
        "Team12 returns after FX conversion",
    )

    proof = pd.read_csv(args.recovery_evidence, index_col="date", parse_dates=True)
    read_gate = assess_dataset_readiness(
        args.snapshot, expected_manifest_sha256=args.snapshot_sha256
    )
    with TemporaryDirectory() as work:
        native = Path(work) / "native"
        try:
            convert_ppo_inputs(args.dataset, native, expected_lineage_sha256=args.lineage_sha256)
        except FeatureDatasetIOError as error:
            native_blocker = str(error)
        else:
            raise ValueError("The strict converter unexpectedly accepted research evidence")
        if native.exists():
            raise ValueError("Rejected conversion produced native PPO files")
    return {
        "training_approved": False,
        "native_ppo_inputs": None,
        "source_snapshot_sha256": args.snapshot_sha256,
        "derived_lineage_sha256": args.lineage_sha256,
        "source_eligibility": read_gate["status"],
        "source_blocker_counts": dict(Counter(reason["code"] for reason in read_gate["reasons"])),
        "strict_converter_blocker": native_blocker,
        "feature_rows": len(dates),
        "pre_evaluation_observations": len(prior),
        "pre_evaluation_first": str(prior[0].date()),
        "pre_evaluation_last": str(prior[-1].date()),
        "initial_warmup_end": str(prior[WARMUP_ROWS - 1].date()),
        "stored_seoul_clocks": stored["evaluation_at"]
        .dt.tz_convert("Asia/Seoul")
        .dt.strftime("%H:%M")
        .value_counts()
        .to_dict(),
        "domestic_assumed_release_later_than_1530": {
            asset: int(
                stored[f"{asset}__assumed_usable_at"].gt(pd.Series(expected, index=dates)).sum()
            )
            for asset in DOMESTIC
        },
        "fx_actual_available_at_verified_rows": int(
            stored["FX__actual_available_at"].notna().sum()
        ),
        "fx_reference_age_over_seven_days": int(
            stored["FX__observation_age_hours"].gt(7 * 24).sum()
        ),
        "fx_max_reference_age_hours": float(stored["FX__observation_age_hours"].max()),
        "max_team12_log_return_recalculation_residual": all_return_residual,
        "domestic": _check_domestic(
            snapshot, proof, info["vendor_evidence"], prices, segment, frames["features"]
        ),
        "windows": _check_windows(frames, lineage),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "snapshot", "archive", "recovery-evidence", "evidence-lineage"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--snapshot-sha256", required=True)
    parser.add_argument("--lineage-sha256", required=True)
    parser.add_argument("--config", type=Path, default=Path("config/team12_audit.json"))
    print(json.dumps(verify(parser.parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
