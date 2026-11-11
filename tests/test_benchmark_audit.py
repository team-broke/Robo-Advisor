"""Offline tests for the two benchmark legs and their return intervals."""

import hashlib
import json

import pandas as pd
import pytest

from robo_advisor.data import benchmark_audit
from robo_advisor.data.benchmark_audit import (
    audit_benchmark_data,
    main,
    validate_comparison_periods,
)


def _write(root, filename, frame):
    path = root / filename
    frame.to_csv(path, index=False)
    return {"path": filename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _clock(dates, hours, timezone):
    return (pd.DatetimeIndex(dates) + pd.Timedelta(hours=hours)).tz_localize(timezone)


@pytest.fixture
def inputs(tmp_path):
    # Small, discontinuous reference schedule for exercising boundaries without a provider.
    kr = [
        "2022-01-03",
        "2022-01-04",
        "2022-01-05",
        "2022-05-09",
        "2022-05-10",
        "2022-05-11",
        "2022-06-30",
        "2022-07-04",
        "2022-07-05",
        "2022-07-06",
    ]
    us = sorted(
        set(kr) - {"2022-07-04"}
        | {
            "2021-12-31",
            "2022-05-06",
            "2022-06-29",
            "2022-07-01",
        }
    )
    config = {
        "schema_version": 1,
        "start": "2022-01-01",
        "end_exclusive": "2022-07-07",
        "benchmarks": {
            "KOSPI": {
                **_write(
                    tmp_path,
                    "kospi.csv",
                    pd.DataFrame(
                        {"Date": kr, "Close": 3000.0, "Ticker": "^KS11", "Source": "Yahoo Finance"}
                    ),
                ),
                "index_id": "KOSPI",
                "source": "yahoo",
                "symbol": "^KS11",
                "currency": "KRW",
                "return_type": "PR",
            },
            "SP500": {
                **_write(tmp_path, "sp500.csv", pd.DataFrame({"Date": us, "Close": 4000.0})),
                "index_id": "SP500",
                "source": "unknown",
                "currency": "USD",
                "return_type": "PR",
            },
        },
        "calendars": {
            "KRX": _write(
                tmp_path,
                "kr.csv",
                pd.DataFrame(
                    {
                        "session_date": kr,
                        "close": _clock(kr, 15.5, "Asia/Seoul"),
                    }
                ),
            ),
            "US": _write(
                tmp_path,
                "us.csv",
                pd.DataFrame(
                    {
                        "session_date": us,
                        "scheduled_close_utc": _clock(us, 16, "America/New_York"),
                    }
                ),
            ),
        },
        "evaluation": _write(
            tmp_path,
            "evaluation.csv",
            pd.DataFrame(
                {
                    "session_date": [day for day in kr if day in us],
                    "evaluation_at": _clock([day for day in kr if day in us], 16, "Asia/Seoul"),
                }
            ),
        ),
        "fx": {
            **_write(
                tmp_path,
                "fx.csv",
                pd.DataFrame(
                    {
                        "date": kr,
                        "mar_krw_per_usd": 1000.0,
                        "source_identity": "smbs:USD_KRW_MAR:public_period_table",
                        "actual_available_at": _clock(kr, 12, "Asia/Seoul"),
                    }
                ),
            ),
            "source": "smbs",
            "max_age_days": 7,
        },
    }
    return tmp_path, config


def _edit(root, descriptor, change):
    frame = pd.read_csv(root / descriptor["path"])
    descriptor.update(_write(root, descriptor["path"], change(frame)))


def _rows(report):
    assert report["status"] == "unverified", report["issues"]
    assert report["training_approved"] is False
    return {row["session_date"][:10]: row for row in report["metrics"]["rows"]}


def test_kospi_missing_sessions_invalidate_both_return_endpoints(inputs):
    root, config = inputs
    missing = ["2022-01-03", "2022-05-09"]
    _edit(root, config["benchmarks"]["KOSPI"], lambda f: f.loc[~f["Date"].isin(missing)])
    report = audit_benchmark_data(config, root=root)
    rows = _rows(report)
    assert report["metrics"]["sources"]["KOSPI"]["absent_dates"] == missing
    assert report["metrics"]["affected_return_dates"] == [
        "2022-01-03",
        "2022-01-04",
        "2022-05-09",
        "2022-05-10",
    ]
    for absent, following in zip(missing, ("2022-01-04", "2022-05-10")):
        assert "KOSPI_CLOSE_MISSING" in rows[absent]["exclusion_reasons"]
        assert "PREVIOUS_EVALUATION_INCOMPLETE" in rows[following]["exclusion_reasons"]
        assert rows[following]["kospi_simple_return"] is None
        assert rows[following]["sp500_krw_simple_return"] is None
    assert rows["2022-05-11"]["kospi_simple_return"] == 0.0


def test_initial_previous_us_close_is_required(inputs):
    root, config = inputs
    _edit(root, config["benchmarks"]["SP500"], lambda f: f.loc[f["Date"] != "2021-12-31"])
    report = audit_benchmark_data(config, root=root)
    rows = _rows(report)
    assert report["metrics"]["sources"]["SP500"]["absent_dates"] == ["2021-12-31"]
    assert rows["2022-01-03"]["sp500_source_date"].startswith("2021-12-31")
    assert rows["2022-01-03"]["sp500_krw"] is None
    assert rows["2022-01-04"]["sp500_krw_simple_return"] is None
    assert rows["2022-01-05"]["sp500_krw_simple_return"] == 0


def test_fx_only_move_changes_us_leg(inputs):
    root, config = inputs
    _edit(
        root,
        config["fx"],
        lambda f: f.assign(
            mar_krw_per_usd=f["mar_krw_per_usd"].where(f["date"] != "2022-01-04", 1100)
        ),
    )
    rows = _rows(audit_benchmark_data(config, root=root))
    row = rows["2022-01-04"]
    assert row["sp500_krw"] == 4_400_000
    assert row["sp500_krw_simple_return"] == pytest.approx(0.10)
    assert row["kospi_simple_return"] == 0


@pytest.mark.leakage
def test_smbs_observation_age_excludes_current_and_next_return(inputs):
    root, config = inputs
    fx = config["fx"]
    _edit(
        root,
        fx,
        lambda frame: frame.loc[frame["date"] != "2022-01-04"].replace(
            {
                "date": {"2022-01-03": "2021-12-28"},
                "actual_available_at": {"2022-01-03 12:00:00+09:00": "2021-12-28 12:00:00+09:00"},
            }
        ),
    )
    rows = _rows(audit_benchmark_data(config, root=root))
    assert rows["2022-01-03"]["fx_krw_per_usd"] == 1000
    assert rows["2022-01-04"]["fx_observation_date"] == "2021-12-28"
    assert rows["2022-01-04"]["fx_observation_age_hours"] == 184
    assert rows["2022-01-04"]["fx_krw_per_usd"] is None
    assert "FX_STALE" in rows["2022-01-04"]["exclusion_reasons"]
    assert rows["2022-01-05"]["fx_krw_per_usd"] == 1000
    assert rows["2022-01-05"]["sp500_krw_simple_return"] is None
    assert "PREVIOUS_EVALUATION_INCOMPLETE" in rows["2022-01-05"]["exclusion_reasons"]


@pytest.mark.leakage
def test_full_us_session_lag_respects_holiday_and_dst(inputs):
    root, config = inputs
    rows = _rows(audit_benchmark_data(config, root=root))
    assert "2022-07-04" not in rows
    # July 1 is on the US schedule but not on the fixture's Korean schedule.
    assert rows["2022-07-05"]["sp500_source_date"].startswith("2022-07-01")
    winter = pd.Timestamp(rows["2022-01-04"]["sp500_available_at"])
    summer = pd.Timestamp(rows["2022-07-05"]["sp500_available_at"])
    assert winter == pd.Timestamp("2022-01-03T21:00Z")
    assert summer == pd.Timestamp("2022-07-01T20:00Z")


@pytest.mark.leakage
def test_actual_late_fx_overrides_research_assumption_and_future_values(inputs):
    root, config = inputs
    # The Jan 3 quote actually arrives after the Jan 4 evaluation (16:00 KST).
    # Its next-day 10:00 research assumption would have incorrectly admitted it.
    _edit(
        root,
        config["fx"],
        lambda f: f.assign(
            actual_available_at=f["actual_available_at"].mask(
                f["date"] == "2022-01-03", "2022-01-04T17:00:00+09:00"
            )
        ),
    )
    _edit(
        root,
        config["fx"],
        lambda f: f.assign(
            actual_available_at=f["actual_available_at"].mask(
                f["date"] == "2022-01-04", "2022-01-04 18:00:00+09:00"
            )
        ),
    )
    before = _rows(audit_benchmark_data(config, root=root))
    assert before["2022-01-04"]["fx_krw_per_usd"] is None
    assert "FX_NOT_AVAILABLE" in before["2022-01-04"]["exclusion_reasons"]
    _edit(
        root,
        config["fx"],
        lambda f: f.assign(
            mar_krw_per_usd=f["mar_krw_per_usd"].where(f["date"] != "2022-01-03", 50_000)
        ),
    )
    after = _rows(audit_benchmark_data(config, root=root))
    assert after["2022-01-03"] == before["2022-01-03"]
    assert after["2022-01-04"] == before["2022-01-04"]


@pytest.mark.leakage
def test_late_us_quote_is_excluded_without_backfilling(inputs):
    root, config = inputs
    _edit(
        root,
        config["benchmarks"]["SP500"],
        lambda f: f.assign(
            available_at=pd.Series(index=f.index, dtype="object").mask(
                f["Date"] == "2022-01-03", "2022-01-04 17:00:00+09:00"
            )
        ),
    )
    row = _rows(audit_benchmark_data(config, root=root))["2022-01-04"]
    assert row["sp500_source_date"].startswith("2022-01-03")
    assert row["sp500_krw"] is None
    assert "SP500_AFTER_CUTOFF" in row["exclusion_reasons"]


def test_same_end_dates_cannot_hide_different_return_starts():
    dates = pd.to_datetime(["2022-01-04", "2022-01-05"])
    strategy = pd.Series(pd.to_datetime(["2022-01-03", "2022-01-04"]), index=dates)
    benchmark = strategy.copy()
    validate_comparison_periods(strategy, benchmark)
    benchmark.iloc[0] = pd.Timestamp("2021-12-31")
    with pytest.raises(ValueError, match="intervals must match"):
        validate_comparison_periods(strategy, benchmark)
    with pytest.raises(ValueError, match="intervals must match"):
        validate_comparison_periods(strategy.iloc[1:], strategy)


@pytest.mark.parametrize("target", ["KOSPI", "SP500", "fx", "KRX", "US", "evaluation"])
def test_hash_mismatch_blocks_replay(inputs, target):
    root, config = inputs
    descriptors = {**config["benchmarks"], **config["calendars"], **config}
    descriptor = descriptors[target]
    path = root / descriptor["path"]
    path.write_bytes(path.read_bytes() + b"\n")
    report = audit_benchmark_data(config, root=root)
    assert report["status"] == "failed"
    assert report["issues"][-1]["code"] == "EVIDENCE_HASH_MISMATCH"
    assert "rows" not in report["metrics"]


def test_bytes_read_after_anchor_check_must_also_match(inputs, monkeypatch):
    root, config = inputs
    check = benchmark_audit._anchored_file

    def replace_after_check(root, descriptor):
        path = check(root, descriptor)
        path.write_bytes(path.read_bytes() + b"\n")
        return path

    monkeypatch.setattr(benchmark_audit, "_anchored_file", replace_after_check)
    report = audit_benchmark_data(config, root=root)
    assert report["status"] == "failed"
    assert report["issues"][-1]["code"] == "EVIDENCE_HASH_MISMATCH"
    assert report["training_approved"] is False
    assert "rows" not in report["metrics"]


@pytest.mark.parametrize("problem", ["wrong_index", "mixed_return_types", "incorrect_ticker"])
def test_invalid_identity_or_return_type_fails(inputs, problem):
    root, config = inputs
    kospi = config["benchmarks"]["KOSPI"]
    if problem == "wrong_index":
        kospi["symbol"] = "KOSPI 200"
    elif problem == "mixed_return_types":
        config["benchmarks"]["SP500"].update(source="sp_dji", symbol="SPTR", return_type="TR")
    else:
        _edit(root, kospi, lambda f: f.assign(Ticker="^GSPC"))
    report = audit_benchmark_data(config, root=root)
    assert report["status"] == "failed"
    assert report["training_approved"] is False


@pytest.mark.leakage
def test_claims_are_checked_against_observations_and_krw_values(inputs):
    root, config = inputs
    rows = _rows(audit_benchmark_data(config, root=root))
    claims = pd.DataFrame(rows.values())
    config["claims"] = _write(root, "claims.csv", claims)
    report = audit_benchmark_data(config, root=root)
    assert report["status"] == "unverified"
    assert report["metrics"]["claims_check"] == "passed"
    claims.loc[0, "fx_observation_date"] = claims.loc[1, "fx_observation_date"]
    claims.loc[0, "sp500_krw"] = 0
    config["claims"].update(_write(root, "claims.csv", claims))
    report = audit_benchmark_data(config, root=root)
    assert report["status"] == "failed"
    assert report["metrics"]["claims_check"] == "failed"
    mismatches = [i for i in report["issues"] if i["code"] == "BENCHMARK_REPLAY_MISMATCH"]
    assert {i["column"] for i in mismatches} == {"fx_observation_date", "sp500_krw"}


def test_strategy_period_mismatch_is_reported_by_audit(inputs):
    root, config = inputs
    rows = _rows(audit_benchmark_data(config, root=root))
    strategy = pd.DataFrame(row for row in rows.values() if row["kospi_simple_return"] is not None)[
        ["session_date", "return_start_date"]
    ]
    config["comparison"] = _write(root, "strategy-periods.csv", strategy)
    report = audit_benchmark_data(config, root=root)
    assert report["metrics"]["periods_check"] == "passed"
    strategy.iloc[0, 1] = "2021-12-31"
    config["comparison"] = _write(root, "strategy-periods.csv", strategy)
    report = audit_benchmark_data(config, root=root)
    assert report["status"] == "failed"
    assert report["metrics"]["periods_check"] == "failed"
    assert report["issues"][-1]["code"] == "RETURN_INTERVAL_MISMATCH"


def test_cli_emits_json_with_unverified_exit_status(inputs, capsys):
    root, config = inputs
    path = root / "audit.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    assert main(["--root", str(root), "--config", str(path)]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "unverified"
    assert report["training_approved"] is False
