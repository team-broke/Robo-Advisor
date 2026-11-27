"""Recover research-only Team12 prices from preserved KRX exports and vendor evidence."""

import argparse
import ast
import json
from pathlib import Path

import pandas as pd

from robo_advisor.data.alignment import align_trading_calendars, convert_prices_to_krw
from robo_advisor.data.audit import _anchored_file, _fx_input, _fx_replay, _sha
from robo_advisor.data.collectors import CollectionResult
from robo_advisor.data.snapshot import read_snapshot, write_snapshot
from robo_advisor.features.audit import read_audited_research_archive
from robo_advisor.features.dataset import (
    TEAM12_ASSETS,
    DatasetWindow,
    FeatureDataset,
    _window_features,
)
from robo_advisor.features.dataset_io import _write_prepared_research
from robo_advisor.features.research_recovery import (
    OHLCV,
    crosscheck_missing_ohlcv,
    replay_observed_prices,
)

DOMESTIC = dict(zip(TEAM12_ASSETS[-2:], ("278530", "148070")))
US = list(TEAM12_ASSETS[:-2])
START, END = pd.Timestamp("2020-01-01"), pd.Timestamp("2026-01-01")
WINDOWS = (
    DatasetWindow("W1", "2020-01-01", "2023-01-01", "2024-01-01", "2025-01-01"),
    DatasetWindow("W2", "2021-01-01", "2024-01-01", "2025-01-01", "2026-01-01"),
)


def _json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_received(root, filename, receipt_name):
    path = root / filename
    receipt = _json(root / receipt_name)
    if _sha(path) != receipt["sha256"]:
        raise ValueError(f"Receipt hash mismatch: {path.name}")
    return path, receipt


def _vendors(source_dir, symbol):
    yahoo_path, yahoo_receipt = _read_received(
        source_dir, f"{symbol}.KS.yahoo-chart.json", f"{symbol}.KS.yahoo-receipt.json"
    )
    naver_path, naver_receipt = _read_received(
        source_dir, f"{symbol}.naver-day.txt", f"{symbol}.naver-day-receipt.json"
    )
    chart = _json(yahoo_path)["chart"]
    if chart["error"] is not None or len(chart["result"]) != 1:
        raise ValueError("Yahoo chart error or ambiguous result")
    chart = chart["result"][0]
    meta = chart["meta"]
    if (meta["symbol"], meta["currency"], meta["exchangeTimezoneName"]) != (
        symbol + ".KS",
        "KRW",
        "Asia/Seoul",
    ):
        raise ValueError("Unexpected Yahoo chart identity")

    def domestic_day(timestamps):
        return (
            pd.to_datetime(timestamps, unit="s", utc=True)
            .tz_convert("Asia/Seoul")
            .tz_localize(None)
            .normalize()
        )

    yahoo = pd.DataFrame(
        chart["indicators"]["quote"][0], index=domestic_day(chart["timestamp"])
    ).rename(columns=str.title)
    yahoo["Adj Close"] = chart["indicators"]["adjclose"][0]["adjclose"]
    events = [
        {**event, "kind": kind, "local_date": domestic_day([event["date"]])[0].date().isoformat()}
        for kind, group in chart.get("events", {}).items()
        for event in group.values()
    ]
    records = ast.literal_eval(naver_path.read_text(encoding="utf-8").strip())
    if records[0][:6] != ["날짜", "시가", "고가", "저가", "종가", "거래량"]:
        raise ValueError("Unexpected Naver price schema")
    naver = pd.DataFrame(
        [row[1:6] for row in records[1:]],
        index=pd.to_datetime([row[0] for row in records[1:]], format="%Y%m%d"),
        columns=OHLCV,
        dtype=float,
    )
    return yahoo, naver, events, {"yahoo": yahoo_receipt, "naver": naver_receipt}


def _official(root, symbol, expected_dates):
    quotes, receipts = [], {}
    korean_fields = ["시가", "고가", "저가", "종가", "거래량"]
    for day in expected_dates:
        stem = f"KRX_{symbol}_{day:%Y%m%d}"
        path = root / f"{stem}.csv"
        receipt = _json(root / f"{stem}.request.json")
        if (receipt["symbol"], receipt["date"], receipt["sha256"]) != (
            symbol,
            day.strftime("%Y%m%d"),
            _sha(path),
        ) or receipt.get("historical_available_at") is not None:
            raise ValueError(f"Unexpected KRX evidence or date: {stem}")
        raw = pd.read_csv(path, encoding="cp949")
        if len(raw) != 1 or pd.Timestamp(raw["일자"].iloc[0]) != day:
            raise ValueError(f"Unexpected KRX row count/date: {stem}")
        values = raw[korean_fields].iloc[0].to_numpy(dtype=float)
        quotes.append(values)
        receipts[day] = receipt
    return pd.DataFrame(quotes, index=expected_dates, columns=OHLCV), receipts


def _recover(snapshot, calendars, gap_report, official_root, source_dir):
    common = align_trading_calendars(*[table.index for table in calendars])
    losses = pd.read_csv(gap_report, parse_dates=["date"])
    missing = pd.DatetimeIndex(
        losses.loc[losses["row_category"].eq("direct_price_missing"), "date"].unique()
    ).sort_values()
    model_days = common[(common >= START) & (common < END)]
    series, evidence, vendors = {}, [], {}
    for asset, symbol in DOMESTIC.items():
        archive = snapshot.datasets[asset].data
        days = calendars[0].index.difference(archive.index)
        if not missing.equals(model_days.difference(archive.index)):
            raise ValueError(f"Gap audit disagrees with source snapshot: {asset}")
        krx, receipts = _official(official_root, symbol, days)
        yahoo, naver, events, provider_receipts = _vendors(source_dir, symbol)
        check = crosscheck_missing_ohlcv(
            archive,
            yahoo,
            naver,
            krx,
            [pd.Timestamp(event["local_date"]) for event in events],
            adjusted=asset == "KTB10Y",
        )
        for day, row in check.iterrows():
            usable = day in missing and row["basis_compatible"]
            evidence.append(
                {
                    "asset": asset,
                    "symbol": symbol,
                    "date": day,
                    **krx.loc[day].to_dict(),
                    **row.to_dict(),
                    "state": (
                        "recovered"
                        if usable
                        else ("blocked" if day in missing else "outside_common_evaluation")
                    ),
                    "krx_file": receipts[day]["file"],
                    "krx_sha256": receipts[day]["sha256"],
                    "krx_retrieved_at": receipts[day]["received_at_utc"],
                    "historical_available_at": None,
                    "assumed_usable_at": pd.Timestamp(calendars[0].loc[day, "close"])
                    + pd.Timedelta(minutes=30),
                    "yahoo_has_raw_close": bool(pd.notna(yahoo["Close"].get(day))),
                    "naver_close_for_comparison": naver["Close"].get(day),
                }
            )
        use = check.index[check["basis_compatible"] & check.index.isin(missing)]
        recovered = krx.loc[use].astype(archive[OHLCV].dtypes.to_dict())
        data = pd.concat([archive, recovered]).sort_index()
        pd.testing.assert_frame_equal(data.loc[archive.index], archive, check_names=False)
        series[asset] = data
        vendors[asset] = {
            "receipts": provider_receipts,
            "corporate_actions_reported_by_yahoo": events,
        }
    return series, pd.DataFrame(evidence), vendors


def _prices(snapshot, domestic, calendars):
    kr, us = calendars
    common = align_trading_calendars(kr.index, us.index)
    evaluation = pd.to_datetime(kr.loc[common, "close"], utc=True) + pd.Timedelta(minutes=30)
    lag = us.index.to_series().shift(1).reindex(common)
    usable_us = pd.to_datetime(us["scheduled_close_utc"], utc=True) + pd.Timedelta(hours=2)
    usable_us = usable_us.reindex(lag).set_axis(common)
    prices = pd.DataFrame(index=common)
    timing = pd.DataFrame(index=common)
    timing["next_evaluation_at"] = evaluation.shift(-1)
    for asset in TEAM12_ASSETS:
        item = snapshot.datasets[asset]
        if item.metadata["available_at"] is not None:
            raise ValueError("Archived assumed-clock replay cannot replace actual availability")
        if asset in DOMESTIC:
            prices[asset] = domestic.get(asset, item.data)["Close"].reindex(common)
        else:
            observations = item.data["Close"].reindex(lag).set_axis(common)
            prices[asset] = observations.where(usable_us.le(evaluation))
        source = pd.Series(common, index=common) if asset in DOMESTIC else lag
        usable = evaluation if asset in DOMESTIC else usable_us
        timing[f"{asset}__source_date"] = source.where(prices[asset].notna())
        timing[f"{asset}__assumed_usable_at"] = usable.where(prices[asset].notna())
    return prices, evaluation, timing


def _counts(replay, prices, fx, *, scenario, variant):
    rows = replay.rows.loc[START : END - pd.Timedelta(days=1)]
    prices, fx = prices.reindex(rows.index), fx.reindex(rows.index)
    missing, stale = prices.isna().any(axis=1), fx["rate"].isna()
    ready = rows["feature_ready"]
    lengths = rows.loc[ready].groupby("segment_id").size()
    aligned = rows.loc[rows["alignment_eligible"]].groupby("segment_id").size()
    return {
        "scenario": scenario,
        "variant": variant,
        "common_days": len(rows),
        "direct_price_missing_days": int(missing.sum()),
        "fx_unusable_days": int(stale.sum()),
        "prior_price_unusable_days": int((~(missing | stale) & ~rows["alignment_eligible"]).sum()),
        "alignment_segments": len(aligned),
        "longest_aligned": int(aligned.max()),
        "feature_segments": len(lengths),
        "longest_feature_segment": int(lengths.max()),
        "warmup_lost": int(rows["warmup"].sum()),
        "features": int(ready.sum()),
        "segments_with_20_plus_60": int(lengths.ge(80).sum()),
    }


def _write_derived(replay, output, windows, lineage, *, dataset_id, timing):
    left = (replay.rows.index >= START) & (replay.rows.index < END)
    rows, features = replay.rows.loc[left].join(timing), replay.features.loc[left]
    rewards = replay.rewards.loc[left]
    frames = {"rows": rows, "features": features, "rewards": rewards}
    window_frames, descriptions = _window_features(features, rows, windows)
    frames.update(window_frames)
    partitions = []
    for window_id in descriptions:
        prefix = f"windows/{window_id}"
        splits = frames[f"{prefix}/splits"]
        frames[f"{prefix}/rewards"] = rewards.loc[splits.index].where(
            splits["reward_eligible"], axis=0
        )
        for partition, data in splits.groupby("split"):
            accepted = rows.loc[data.index[data["eligible_observation"]]]
            lengths = accepted.groupby("segment_id").size()
            months = accepted.loc[~accepted.index.to_period("M").duplicated()]
            first, max_history = int(months["warmup_position"].iloc[0]), int(
                months["warmup_position"].max()
            )
            partitions.append(
                {
                    "dataset": dataset_id,
                    "window": window_id,
                    "split": partition,
                    "features": len(accepted),
                    "longest_segment": int(lengths.max()),
                    "segments_with_20_plus_60": int(lengths.ge(80).sum()),
                    "mvo_history_first_month_start": first,
                    "mvo_max_history_at_month_starts": max_history,
                    "mvo_month_starts_with_252": int(months["warmup_position"].ge(252).sum()),
                }
            )
    if output is not None:
        lineage.update(
            {
                "feature_columns": features.columns.tolist(),
                "reward_columns": rewards.columns.tolist(),
                "asset_order": list(TEAM12_ASSETS),
                "windows": descriptions,
                "data_period": {
                    "start": str(rows.index[0].date()),
                    "end_inclusive": str(rows.index[-1].date()),
                    "rows": len(rows),
                },
                "cutoff_at": rows["evaluation_at"].iloc[-1].isoformat(),
            }
        )
        _write_prepared_research(FeatureDataset(frames, lineage), output, dataset_id=dataset_id)
    return partitions


def _sensitivity(base, twenty, asset_history, scenario, variant):
    output = []
    cols = base.features.columns
    macd, rsi, vol = [
        cols[cols.str.endswith(kind)] for kind in ("__macd", "__rsi14", "__volatility20")
    ]
    for window in WINDOWS:
        for split, start, end in (
            ("train", window.train_start, window.train_end),
            ("validation", window.train_end, window.validation_end),
        ):
            model = (base.rows.index >= pd.Timestamp(start)) & (base.rows.index < pd.Timestamp(end))
            add = model & twenty.rows["feature_ready"] & ~base.rows["feature_ready"]
            common = model & base.rows["feature_ready"] & twenty.rows["feature_ready"]
            both = common & asset_history.rows["feature_ready"]
            newly_ready = twenty.features.loc[add]
            output.append(
                {
                    "scenario": scenario,
                    "scope": variant,
                    "window": window.window_id,
                    "split": split,
                    "added_rows_at_20": int(add.sum()),
                    "added_rows_rsi_valid": int(
                        ((newly_ready[rsi] >= 0) & (newly_ready[rsi] <= 100)).all(axis=1).sum()
                    ),
                    "added_rows_volatility_valid": int(newly_ready[vol].notna().all(axis=1).sum()),
                    "max_macd_difference_20_vs_33_on_shared_dates": float(
                        (base.features.loc[common, macd] - twenty.features.loc[common, macd])
                        .abs()
                        .max()
                        .max()
                    ),
                    "max_macd_difference_per_asset_vs_shared": float(
                        (base.features.loc[both, macd] - asset_history.features.loc[both, macd])
                        .abs()
                        .max()
                        .max()
                    ),
                }
            )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--vendor-evidence", required=True, type=Path)
    parser.add_argument("--gap-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", default=Path("config/team12_audit.json"), type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    if any(out.is_relative_to(root.resolve()) for root in (args.archive, args.vendor_evidence)):
        raise ValueError("Use an output directory separate from the inputs")
    cfg = _json(args.config)
    original = cfg["snapshots"]["selected"]
    snapshot = read_snapshot(
        args.archive / original["path"], expected_manifest_sha256=original["sha256"]
    )
    calendars = []
    for market in ("KRX", "US"):
        frame = pd.read_csv(
            _anchored_file(args.archive, cfg["calendars"][market]), index_col=0, parse_dates=True
        )
        calendars.append(frame)
    domestic, evidence, vendor_lineage = _recover(
        snapshot,
        calendars,
        args.gap_report,
        args.archive / "evidence/official-KRX-originals",
        args.vendor_evidence,
    )
    old_prices, evaluation, old_timing = _prices(snapshot, {}, calendars)
    new_prices, _, new_timing = _prices(snapshot, domestic, calendars)
    out.mkdir(parents=True, exist_ok=False)
    evidence_path = out / "recovery-evidence.csv"
    evidence.to_csv(evidence_path, index=False, encoding="utf-8-sig")
    evidence_hash = _sha(evidence_path)
    results = dict(snapshot.datasets)
    for asset, data in domestic.items():
        old = snapshot.datasets[asset]
        metadata = {
            **old.metadata,
            "source": "archived_yahoo_and_krx_market_quotes",
            "price_basis": (
                "Archived Yahoo Close and KRX market OHLCV on corroborated missing dates. "
                "Price returns exclude cash distributions. Historical availability is unverified."
            ),
            "row_count": len(data),
        }
        results[asset] = CollectionResult(
            data,
            old.raw_data,
            metadata,
            {
                "status": "unverified",
                "training_approved": False,
                "historical_release_verified": False,
                "research_recovery_evidence_sha256": evidence_hash,
                "archived_parent_manifest_sha256": snapshot.manifest_sha256,
                "original_validation": old.validation,
            },
        )
    restored = write_snapshot(results, out / "snapshots")
    restored_digest = _sha(restored / "manifest.json")
    read_snapshot(restored, expected_manifest_sha256=restored_digest)
    counts, partitions, sensitivity, segments = [], [], [], []
    for name, spec in cfg["scenarios"].items():
        archived, lineage = read_audited_research_archive(
            args.archive / spec["dataset"]["path"],
            expected_lineage_sha256=spec["dataset"]["sha256"],
        )
        observations = _fx_input(pd.read_csv(_anchored_file(args.archive, spec["fx"])), name)
        fx = _fx_replay(observations, evaluation, "USD/KRW", spec["max_age_days"])
        for recovered, prices, timing in (
            (False, old_prices, old_timing),
            (True, new_prices, new_timing),
        ):
            unit = prices.copy()
            unit[US] = convert_prices_to_krw(prices[US], fx["rate"], fx_direction="USD/KRW")
            base = replay_observed_prices(unit, evaluation)
            twenty = replay_observed_prices(unit, evaluation, warmup_rows=20)
            per_asset = replay_observed_prices(unit, evaluation, asset_history=True)
            if not recovered:
                retained = base.features.loc[archived["features"].index]
                retained.columns = retained.columns.str.replace("__macd_histogram", "__macd_hist")
                pd.testing.assert_frame_equal(retained, archived["features"], check_names=False)
            scope = "recovered" if recovered else "baseline"
            sensitivity.extend(_sensitivity(base, twenty, per_asset, name, scope))
            for label, replay in (
                ("33-shared", base),
                ("20-shared", twenty),
                ("33-per-asset", per_asset),
            ):
                dataset_id = f"{name}-{scope}-{label}"
                counts.append(
                    _counts(replay, prices, fx, scenario=name, variant=f"{scope}-{label}")
                )
                for selected_id, group in replay.rows.loc[
                    START : END - pd.Timedelta(days=1)
                ].groupby("segment_id"):
                    aligned = group.loc[group["alignment_eligible"]]
                    if aligned.empty:
                        continue
                    ready = aligned.index[aligned["feature_ready"]]
                    segments.append(
                        {
                            "dataset": dataset_id,
                            "segment": selected_id,
                            "start": aligned.index[0].date(),
                            "end": aligned.index[-1].date(),
                            "aligned_rows": len(aligned),
                            "features": len(ready),
                            "first_feature": str(ready[0].date()) if len(ready) else "",
                        }
                    )
                target = out / "datasets" if recovered and label == "33-shared" else None
                partitions.extend(
                    _write_derived(
                        replay,
                        target,
                        WINDOWS,
                        {
                            "schema_version": 1,
                            "feature_version": lineage["feature_version"],
                            "dataset_kind": "experimental_real",
                            "training_approved": False,
                            "validation": {
                                "status": "unverified",
                                "native_ppo_status": "not_exported",
                            },
                            "research_archive": {"lineage_sha256": spec["dataset"]["sha256"]},
                            "source_snapshot": {
                                "snapshot_id": restored.name,
                                "manifest_sha256": restored_digest,
                            },
                            "alignment": {
                                "evaluation": (
                                    "Archived KRX reference close + 30 minutes. "
                                    "Historical availability is unverified."
                                ),
                                "us_price_lag": "one US session",
                                "fx_scenario": name,
                                "fx_max_age_days": spec["max_age_days"],
                                "fx_file_sha256": spec["fx"]["sha256"],
                                "calendars": cfg["calendars"],
                                "recovery_evidence_sha256": evidence_hash,
                            },
                            "warmup": {"rows_per_segment": 33, "asset_history": "shared"},
                            "market_regimes": {"status": "not_generated"},
                            "reward_kind": "next_evaluation_log_price_return",
                        },
                        dataset_id=dataset_id,
                        timing=timing.join(fx.add_prefix("FX__")),
                    )
                )
    for filename, records in (
        ("comparison", counts),
        ("partitions", partitions),
        ("segments", segments),
        ("warmup-sensitivity", sensitivity),
    ):
        pd.DataFrame(records).to_csv(out / f"{filename}.csv", index=False)
    (out / "evidence-lineage.json").write_text(
        json.dumps(
            {
                "training_approved": False,
                "original_snapshot_sha256": snapshot.manifest_sha256,
                "restored_snapshot_sha256": restored_digest,
                "recovery_evidence_sha256": evidence_hash,
                "vendor_evidence": vendor_lineage,
                "gap_audit_sha256": _sha(args.gap_report),
                "scope": (
                    "Market price returns under archived release assumptions, "
                    "excluding distribution reinvestment."
                ),
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    print(out)


if __name__ == "__main__":
    main()
