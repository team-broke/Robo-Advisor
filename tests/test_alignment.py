"""작은 DataFrame으로 거래 세션, 이용 가능 시각, 통화 및 결측 계약을 검증한다."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from robo_advisor.data.alignment import (
    AlignmentError,
    align_trading_calendars,
    calculate_log_returns,
    convert_prices_to_krw,
    lag_overseas_prices,
    preprocess_market_data,
)


@pytest.fixture
def inputs():
    dates = pd.to_datetime(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"]
    )
    return {
        "domestic_prices_krw": pd.DataFrame({"KR": 100 * 1.1 ** np.arange(6)}, index=dates),
        "overseas_prices_usd": pd.DataFrame({"US": 10 * 2.0 ** np.arange(6)}, index=dates),
        "fx_rates": pd.Series([1000, 1100, 1200, 1300, 1400, 1500], index=dates),
        "kr_calendar": dates.copy(),
        "us_calendar": dates.copy(),
        "fx_available_at": pd.Series(
            (dates + pd.Timedelta(hours=10)).tz_localize("Asia/Seoul"), index=dates
        ),
        "fx_direction": "USD/KRW",
    }


def reasons(result):
    return {row["date"]: row["reasons"] for row in result.diagnostics["excluded_dates"]}


def test_calendar_intersection():
    kr = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-06", "2024-01-07"])
    us = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-07", "2024-01-08"])
    assert align_trading_calendars(kr, us).equals(kr[[0, 1, 3]])


@pytest.mark.parametrize(
    "market, code", [("kr_calendar", "KRX_CLOSED"), ("us_calendar", "US_CLOSED")]
)
def test_market_holiday_excluded(inputs, market, code):
    # A supplied market schedule, rather than an observation gap, identifies a closure.
    day = inputs[market][2]
    inputs[market] = inputs[market].delete(2)
    key = "domestic_prices_krw" if market == "kr_calendar" else "overseas_prices_usd"
    inputs[key] = inputs[key].drop(day)
    result = preprocess_market_data(**inputs)
    assert day not in result.prices_krw.index
    assert reasons(result)[str(day.date())] == [code]


@pytest.mark.parametrize("market", ["kr_calendar", "us_calendar"])
@pytest.mark.parametrize("defect", ["duplicate", "reversed", "nat"])
def test_calendar_rejects_bad_order_and_dates(inputs, market, defect):
    dates = inputs[market]
    inputs[market] = {
        "duplicate": dates.insert(1, dates[0]),
        "reversed": dates[::-1],
        "nat": dates.insert(1, pd.NaT),
    }[defect]
    with pytest.raises(ValueError, match="unique, increasing"):
        preprocess_market_data(**inputs)


@pytest.mark.parametrize("key", ["domestic_prices_krw", "overseas_prices_usd", "fx_rates"])
@pytest.mark.parametrize("defect", ["duplicate", "reversed"])
def test_source_dates_are_not_silently_repaired(inputs, key, defect):
    data = inputs[key]
    inputs[key] = pd.concat([data.iloc[:1], data]) if defect == "duplicate" else data.iloc[::-1]
    with pytest.raises(AlignmentError) as error:
        preprocess_market_data(**inputs)
    report = next(iter(error.value.reports.values()))
    code = "DUPLICATE_DATE" if defect == "duplicate" else "OUT_OF_ORDER_DATE"
    assert code in {issue["code"] for issue in report["issues"]}


def test_timezone_normalization_preserves_market_session(inputs):
    local = pd.to_datetime(["2024-01-02", "2024-01-03"])
    kr = local.tz_localize("Asia/Seoul").tz_convert("UTC")
    us = local.tz_localize("America/New_York").tz_convert("UTC")
    assert align_trading_calendars(kr, us).equals(local)


def test_friday_to_monday_is_one_overseas_session(inputs):
    lag = lag_overseas_prices(inputs["overseas_prices_usd"], inputs["us_calendar"])
    assert lag.prices.loc["2024-01-08", "US"] == 80
    assert lag.source_dates.loc["2024-01-08", "US"] == pd.Timestamp("2024-01-05")
    assert lag.available_at.loc["2024-01-08", "US"] == pd.Timestamp("2024-01-05 21:00Z")
    assert pd.isna(lag.prices.iloc[0, 0])


def test_lag_precedes_intersection(inputs):
    # KR closes on Jan 3; its price still exists as US Jan 4's previous US session.
    inputs["kr_calendar"] = inputs["kr_calendar"].delete(1)
    inputs["domestic_prices_krw"] = inputs["domestic_prices_krw"].drop(pd.Timestamp("2024-01-03"))
    result = preprocess_market_data(**inputs)
    assert result.source_dates.loc["2024-01-05", "US"] == pd.Timestamp("2024-01-04")
    # Inspect the full lag grid as Jan 4 is excluded only for missing previous return price.
    lag = lag_overseas_prices(inputs["overseas_prices_usd"], inputs["us_calendar"])
    assert lag.prices.loc["2024-01-04", "US"] == 20
    assert lag.source_dates.loc["2024-01-04", "US"] == pd.Timestamp("2024-01-03")


def test_us_holiday_uses_previous_actual_session(inputs):
    inputs["us_calendar"] = inputs["us_calendar"].delete(2)
    inputs["overseas_prices_usd"] = inputs["overseas_prices_usd"].drop(pd.Timestamp("2024-01-04"))
    lag = lag_overseas_prices(inputs["overseas_prices_usd"], inputs["us_calendar"])
    assert lag.source_dates.loc["2024-01-05", "US"] == pd.Timestamp("2024-01-03")


def test_missing_row_is_a_session_not_an_extra_lag(inputs):
    inputs["overseas_prices_usd"] = inputs["overseas_prices_usd"].drop(pd.Timestamp("2024-01-03"))
    lag = lag_overseas_prices(inputs["overseas_prices_usd"], inputs["us_calendar"])
    assert lag.prices.loc["2024-01-04", "US"] == 10
    assert lag.source_dates.loc["2024-01-04", "US"] == pd.Timestamp("2024-01-02")
    assert lag.forward_filled.loc["2024-01-04", "US"]
    assert lag.diagnostics["missing_before_fill"]["metrics"]["columns"]["US"]["missing_count"] == 1


def test_default_us_close_respects_dst():
    dates = pd.to_datetime(["2024-07-05", "2024-07-08"])
    lag = lag_overseas_prices(pd.DataFrame({"US": [10, 20]}, index=dates), dates)
    assert lag.available_at.loc["2024-07-08", "US"] == pd.Timestamp("2024-07-05 20:00Z")


@pytest.mark.leakage
def test_same_day_us_close_is_never_used(inputs):
    result = preprocess_market_data(**inputs)
    assert (result.source_dates["US"] < result.prices_krw.index).all()
    assert result.available_at.le(result.evaluation_at, axis=0).all().all()
    assert result.prices_krw.loc["2024-01-04", "US"] == 20 * 1200


@pytest.mark.leakage
def test_future_overseas_prices_do_not_change_past_states(inputs):
    baseline = preprocess_market_data(**inputs)
    inputs["overseas_prices_usd"].loc["2024-01-05":] *= 999
    changed = preprocess_market_data(**inputs)
    # Jan 5's own US close is still in the future at Jan 5's Korean close.
    assert_frame_equal(
        baseline.prices_krw.loc[:"2024-01-05"], changed.prices_krw.loc[:"2024-01-05"]
    )
    assert_frame_equal(
        baseline.log_returns.loc[:"2024-01-05"], changed.log_returns.loc[:"2024-01-05"]
    )


@pytest.mark.leakage
def test_explicit_delayed_price_is_excluded_not_refilled(inputs):
    dates = inputs["us_calendar"]
    available = pd.Series(
        (dates + pd.Timedelta(hours=16)).tz_localize("America/New_York"), index=dates
    )
    available.loc["2024-01-03"] = pd.Timestamp("2024-01-04 16:00", tz="Asia/Seoul")
    inputs["overseas_available_at"] = available
    result = preprocess_market_data(**inputs)
    assert "2024-01-04" not in result.prices_krw.index
    assert "PRICE_NOT_AVAILABLE" in reasons(result)["2024-01-04"]
    assert "2024-01-05" not in result.prices_krw.index  # previous common price is unavailable
    assert "MISSING_PREVIOUS_PRICE" in reasons(result)["2024-01-05"]


@pytest.mark.parametrize(
    "direction, rates", [("USD/KRW", [1000, 1200]), ("KRW/USD", [1 / 1000, 1 / 1200])]
)
def test_currency_conversion_and_fx_only_return(direction, rates):
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    converted = convert_prices_to_krw(
        pd.DataFrame({"US": [10, 10]}, index=dates),
        pd.Series(rates, index=dates),
        fx_direction=direction,
    )
    np.testing.assert_allclose(converted["US"], [10000, 12000])
    assert calculate_log_returns(converted).iloc[1, 0] == pytest.approx(np.log(1.2))


@pytest.mark.parametrize("direction", ["USDKRW", "USD", "", None])
def test_unknown_fx_direction_is_rejected(inputs, direction):
    inputs["fx_direction"] = direction
    with pytest.raises(ValueError, match="fx_direction"):
        preprocess_market_data(**inputs)


def test_inverse_fx_pipeline_matches_direct_quote(inputs):
    direct = preprocess_market_data(**inputs)
    inputs["fx_rates"] = 1.0 / inputs["fx_rates"]
    inputs["fx_direction"] = "KRW/USD"
    inverse = preprocess_market_data(**inputs)
    np.testing.assert_allclose(direct.prices_krw, inverse.prices_krw)
    np.testing.assert_allclose(direct.log_returns, inverse.log_returns)


def test_domestic_prices_are_not_converted_and_benchmark_uses_same_policy(inputs):
    inputs["overseas_prices_usd"]["BENCHMARK_USD"] = inputs["overseas_prices_usd"]["US"] * 3
    result = preprocess_market_data(**inputs)
    assert_series_equal(
        result.prices_krw["KR"], inputs["domestic_prices_krw"]["KR"].loc[result.prices_krw.index]
    )
    np.testing.assert_allclose(result.prices_krw["BENCHMARK_USD"], result.prices_krw["US"] * 3)
    np.testing.assert_allclose(result.log_returns["BENCHMARK_USD"], result.log_returns["US"])


@pytest.mark.leakage
def test_future_fx_change_does_not_change_past(inputs):
    baseline = preprocess_market_data(**inputs)
    inputs["fx_rates"].loc["2024-01-08":] *= 10
    changed = preprocess_market_data(**inputs)
    assert_frame_equal(
        baseline.prices_krw.loc[:"2024-01-05"], changed.prices_krw.loc[:"2024-01-05"]
    )
    assert_frame_equal(
        baseline.log_returns.loc[:"2024-01-05"], changed.log_returns.loc[:"2024-01-05"]
    )


@pytest.mark.leakage
def test_late_fx_uses_last_released_observation_and_tracks_it(inputs):
    dates = inputs["kr_calendar"]
    inputs["fx_available_at"] = pd.Series(
        (dates + pd.Timedelta(hours=17)).tz_localize("Asia/Seoul"), index=dates
    )
    baseline = preprocess_market_data(**inputs)
    assert baseline.fx.loc["2024-01-05", "observation_date"] == pd.Timestamp("2024-01-04")
    assert baseline.fx.loc["2024-01-05", "rate"] == 1200
    assert baseline.fx.loc["2024-01-05", "available_at"] == pd.Timestamp("2024-01-04 08:00Z")
    assert baseline.fx.loc["2024-01-05", "forward_filled"]
    inputs["fx_rates"].loc["2024-01-05":] *= 10
    changed = preprocess_market_data(**inputs)
    assert_frame_equal(
        baseline.prices_krw.loc[:"2024-01-05"], changed.prices_krw.loc[:"2024-01-05"]
    )
    assert_frame_equal(
        baseline.log_returns.loc[:"2024-01-05"], changed.log_returns.loc[:"2024-01-05"]
    )


def test_fx_available_exactly_at_cutoff_can_be_used(inputs):
    inputs["fx_available_at"] = pd.Series(
        (inputs["kr_calendar"] + pd.Timedelta(hours=15, minutes=30)).tz_localize("Asia/Seoul"),
        index=inputs["kr_calendar"],
    )
    result = preprocess_market_data(**inputs)
    assert (result.fx["observation_date"] == result.prices_krw.index).all()


@pytest.mark.parametrize("defect", ["absent", "naive", "nat", "early", "reverse_release"])
def test_fx_requires_verified_availability(inputs, defect):
    dates = inputs["kr_calendar"]
    if defect == "absent":
        inputs["fx_available_at"] = None
    elif defect == "naive":
        inputs["fx_available_at"] = inputs["fx_available_at"].dt.tz_localize(None)
    elif defect == "nat":
        inputs["fx_available_at"].iloc[0] = pd.NaT
    elif defect == "early":
        inputs["fx_available_at"].iloc[1] = pd.Timestamp("2024-01-02 10:00", tz="Asia/Seoul")
    else:
        inputs["fx_available_at"].iloc[0] = pd.Timestamp("2024-01-03 11:00", tz="Asia/Seoul")
    with pytest.raises(ValueError):
        preprocess_market_data(**inputs)
    assert len(dates) == 6


def test_stale_fx_is_excluded(inputs):
    inputs["fx_rates"] = inputs["fx_rates"].iloc[:1]
    inputs["fx_available_at"] = inputs["fx_available_at"].iloc[:1]
    result = preprocess_market_data(**inputs, max_fx_age=pd.Timedelta(days=2))
    assert result.prices_krw.empty
    assert "FX_STALE" in reasons(result)["2024-01-05"]


def test_delayed_old_observation_cannot_reset_fx_age(inputs):
    inputs["fx_rates"] = pd.Series([1000.0], index=pd.to_datetime(["2023-12-01"]))
    inputs["fx_available_at"] = pd.Series(
        [pd.Timestamp("2024-01-02 10:00", tz="Asia/Seoul")], index=inputs["fx_rates"].index
    )
    result = preprocess_market_data(**inputs)
    assert result.prices_krw.empty
    assert "FX_STALE" in reasons(result)["2024-01-04"]


def test_missing_fx_uses_past_released_rate(inputs):
    inputs["fx_rates"] = inputs["fx_rates"].astype(float)
    inputs["fx_rates"].loc["2024-01-04"] = np.nan
    result = preprocess_market_data(**inputs)
    assert result.fx.loc["2024-01-04", "rate"] == 1100
    assert result.fx.loc["2024-01-04", "observation_date"] == pd.Timestamp("2024-01-03")
    assert result.diagnostics["fx_forward_fill_count"] == 1


def test_manual_prices_returns_and_metadata(inputs):
    result = preprocess_market_data(**inputs)
    assert result.prices_krw.index.equals(inputs["kr_calendar"][2:])
    np.testing.assert_allclose(result.prices_krw["US"], [24000, 52000, 112000, 240000])
    np.testing.assert_allclose(result.log_returns["KR"], np.log(1.1))
    np.testing.assert_allclose(
        result.log_returns["US"],
        np.log(np.array([24000 / 11000, 52000 / 24000, 112000 / 52000, 240000 / 112000])),
    )
    assert result.return_start_dates.iloc[0] == pd.Timestamp("2024-01-03")
    assert result.diagnostics["common_row_count"] == 6
    assert result.diagnostics["final_row_count"] == 4
    assert result.diagnostics["dropped_common_row_count"] == 2
    assert reasons(result)["2024-01-02"] == ["MISSING_ALIGNED_PRICE", "RETURN_WARMUP"]
    assert reasons(result)["2024-01-03"] == ["MISSING_PREVIOUS_PRICE"]
    assert result.diagnostics["finite_check"] == {"passed": True, "nan_count": 0, "inf_count": 0}


def test_log_returns_preserve_initial_nan_and_do_not_bridge_gap():
    prices = pd.DataFrame(
        {"A": [10.0, 20.0, np.nan, 80.0, 160.0]}, index=pd.date_range("2024-01-01", periods=5)
    )
    returns = calculate_log_returns(prices)
    np.testing.assert_allclose(
        returns["A"], [np.nan, np.log(2), np.nan, np.nan, np.log(2)], equal_nan=True
    )


@pytest.mark.leakage
def test_future_price_does_not_change_past_log_returns(inputs):
    prices = inputs["domestic_prices_krw"]
    baseline = calculate_log_returns(prices)
    changed = prices.copy()
    changed.iloc[-1] *= 1000
    assert_frame_equal(baseline.iloc[:-1], calculate_log_returns(changed).iloc[:-1])


def test_extreme_positive_prices_have_finite_log_returns():
    prices = pd.DataFrame({"A": [1e-300, 1e300]}, index=pd.date_range("2024-01-01", periods=2))
    assert calculate_log_returns(prices).iloc[1, 0] == pytest.approx(600 * np.log(10))


@pytest.mark.parametrize(
    "value, code",
    [
        (0, "NON_POSITIVE_PRICE"),
        (-1, "NON_POSITIVE_PRICE"),
        (np.inf, "NON_FINITE_PRICE"),
        (-np.inf, "NON_FINITE_PRICE"),
        (True, "INVALID_PRICE_TYPE"),
        ("10", "INVALID_PRICE_TYPE"),
    ],
)
def test_bad_price_is_rejected_with_existing_validator(value, code):
    prices = pd.DataFrame({"WTI": [value]}, index=pd.to_datetime(["2024-01-02"]))
    with pytest.raises(AlignmentError) as error:
        calculate_log_returns(prices)
    assert code in {issue["code"] for issue in error.value.reports["log_return_prices"]["issues"]}


@pytest.mark.parametrize("value", [0.0, -1.0, np.inf])
def test_invalid_fx_rejected(inputs, value):
    inputs["fx_rates"] = inputs["fx_rates"].astype(float)
    inputs["fx_rates"].iloc[1] = value
    with pytest.raises(AlignmentError):
        preprocess_market_data(**inputs)


def test_conversion_overflow_is_rejected():
    dates = pd.to_datetime(["2024-01-02"])
    with pytest.raises(AlignmentError, match="converted_prices"):
        convert_prices_to_krw(
            pd.DataFrame({"US": [1e300]}, index=dates),
            pd.Series([1e300], index=dates),
            fx_direction="USD/KRW",
        )


def test_forward_fill_tracks_actual_price_source_and_time(inputs):
    inputs["domestic_prices_krw"].loc["2024-01-04", "KR"] = np.nan
    inputs["overseas_prices_usd"].loc["2024-01-03", "US"] = np.nan
    result = preprocess_market_data(**inputs)
    assert result.prices_krw.loc["2024-01-04", "KR"] == pytest.approx(110)
    assert result.prices_krw.loc["2024-01-04", "US"] == 10 * 1200
    assert result.source_dates.loc["2024-01-04", "KR"] == pd.Timestamp("2024-01-03")
    assert result.source_dates.loc["2024-01-04", "US"] == pd.Timestamp("2024-01-02")
    assert result.available_at.loc["2024-01-04", "US"] == pd.Timestamp("2024-01-02 21:00Z")
    assert result.forward_filled.loc["2024-01-04"].all()
    assert result.log_returns.loc["2024-01-04", "KR"] == 0
    assert result.log_returns.loc["2024-01-04", "US"] == pytest.approx(np.log(1200 / 1100))
    assert result.diagnostics["forward_fill_counts"] == {"KR": 1, "US": 1}
    before = result.diagnostics["domestic"]["missing_before_fill"]["metrics"]["columns"]["KR"]
    after = result.diagnostics["domestic"]["missing_after_fill"]["metrics"]["columns"]["KR"]
    assert before["missing_count"] == 1 and after["missing_count"] == 0


def test_leading_nan_is_never_backward_filled(inputs):
    inputs["overseas_prices_usd"].iloc[:2] = np.nan
    lag = lag_overseas_prices(inputs["overseas_prices_usd"], inputs["us_calendar"])
    assert lag.prices.iloc[:3].isna().all().all()
    assert lag.source_dates.iloc[:3].isna().all().all()
    result = preprocess_market_data(**inputs)
    assert result.prices_krw.index.equals(inputs["kr_calendar"][4:])
    assert "MISSING_ALIGNED_PRICE" in reasons(result)["2024-01-04"]


@pytest.mark.parametrize(
    "key, report_key",
    [("domestic_prices_krw", "domestic"), ("overseas_prices_usd", "overseas"), ("fx_rates", "fx")],
)
def test_excessive_missing_intervals_fail_before_fill(inputs, key, report_key):
    inputs[key] = inputs[key].astype(float)
    inputs[key].iloc[1:4] = np.nan
    with pytest.raises(AlignmentError) as error:
        preprocess_market_data(**inputs, max_forward_fill=2)
    assert "EXCESSIVE_MISSING_INTERVAL" in {
        issue["code"] for issue in error.value.reports[report_key]["issues"]
    }


@pytest.mark.parametrize("limit", [-1, True, 1.5])
def test_invalid_forward_fill_limit(inputs, limit):
    with pytest.raises(ValueError, match="max_consecutive_missing"):
        preprocess_market_data(**inputs, max_forward_fill=limit)


def test_zero_fill_limit_supports_complete_inputs(inputs):
    result = preprocess_market_data(**inputs, max_forward_fill=0)
    assert len(result.prices_krw) == 4
    assert not result.forward_filled.any().any()


def test_non_session_price_is_a_data_error_not_a_holiday(inputs):
    extra = pd.DataFrame({"US": [7]}, index=pd.to_datetime(["2024-01-01"]))
    inputs["overseas_prices_usd"] = pd.concat([extra, inputs["overseas_prices_usd"]])
    with pytest.raises(AlignmentError) as error:
        preprocess_market_data(**inputs)
    assert "OUTSIDE_CALENDAR" in {
        issue["code"] for issue in error.value.reports["overseas"]["issues"]
    }


def test_calendar_coverage_exclusion_is_not_mislabeled_as_holiday(inputs):
    inputs["kr_calendar"] = inputs["kr_calendar"][1:]
    inputs["domestic_prices_krw"] = inputs["domestic_prices_krw"].iloc[1:]
    result = preprocess_market_data(**inputs)
    assert reasons(result)["2024-01-02"] == ["OUTSIDE_KRX_CALENDAR_RANGE"]


def test_disjoint_calendars_rejected(inputs):
    inputs["us_calendar"] = inputs["us_calendar"] + pd.Timedelta(days=100)
    with pytest.raises(AlignmentError, match="no common"):
        preprocess_market_data(**inputs)


def test_asset_name_collision_rejected(inputs):
    inputs["overseas_prices_usd"] = inputs["overseas_prices_usd"].rename(columns={"US": "KR"})
    with pytest.raises(ValueError, match="disjoint"):
        preprocess_market_data(**inputs)


def test_conversion_requires_same_dates(inputs):
    with pytest.raises(ValueError, match="match price dates"):
        convert_prices_to_krw(
            inputs["overseas_prices_usd"], inputs["fx_rates"].iloc[:-1], fx_direction="USD/KRW"
        )


@pytest.mark.parametrize("age", [pd.Timedelta(0), pd.Timedelta(days=-1), pd.NaT, "7D"])
def test_invalid_fx_age(inputs, age):
    with pytest.raises(ValueError, match="max_fx_age"):
        preprocess_market_data(**inputs, max_fx_age=age)


def test_inputs_are_unchanged_and_final_tables_are_aligned(inputs):
    originals = {
        key: value.copy(deep=True)
        for key, value in inputs.items()
        if isinstance(value, (pd.DataFrame, pd.Series, pd.DatetimeIndex))
    }
    result = preprocess_market_data(**inputs)
    for key, original in originals.items():
        if isinstance(original, pd.DataFrame):
            assert_frame_equal(inputs[key], original)
        elif isinstance(original, pd.Series):
            assert_series_equal(inputs[key], original)
        else:
            assert inputs[key].equals(original)
    for table in [
        result.log_returns,
        result.source_dates,
        result.available_at,
        result.forward_filled,
        result.fx,
    ]:
        assert table.index.equals(result.prices_krw.index)
    for table in [result.prices_krw, result.log_returns]:
        assert np.isfinite(table.to_numpy()).all()
    assert result.prices_krw.index.is_monotonic_increasing
    assert list(result.prices_krw.columns) == ["KR", "US"]


def test_single_session_has_no_invented_price_or_return(inputs):
    for key, value in list(inputs.items()):
        if isinstance(value, (pd.DataFrame, pd.Series)):
            inputs[key] = value.iloc[:1]
        elif isinstance(value, pd.DatetimeIndex):
            inputs[key] = value[:1]
    result = preprocess_market_data(**inputs)
    assert result.prices_krw.empty and result.log_returns.empty
    assert result.diagnostics["final_row_count"] == 0
    assert result.diagnostics["dropped_common_row_count"] == 1


def test_overseas_assets_keep_independent_missing_price_provenance(inputs):
    inputs["overseas_prices_usd"]["US2"] = inputs["overseas_prices_usd"]["US"] * 3
    inputs["overseas_prices_usd"].loc["2024-01-03", "US2"] = np.nan
    result = preprocess_market_data(**inputs)
    assert result.source_dates.loc["2024-01-04", "US"] == pd.Timestamp("2024-01-03")
    assert result.source_dates.loc["2024-01-04", "US2"] == pd.Timestamp("2024-01-02")
    assert not result.forward_filled.loc["2024-01-04", "US"]
    assert result.forward_filled.loc["2024-01-04", "US2"]


def test_domestic_fill_can_use_past_session_outside_common_grid(inputs):
    day = pd.Timestamp("2024-01-03")
    inputs["us_calendar"] = inputs["us_calendar"].delete(1)
    inputs["overseas_prices_usd"] = inputs["overseas_prices_usd"].drop(day)
    inputs["domestic_prices_krw"].loc["2024-01-04":"2024-01-05", "KR"] = np.nan
    result = preprocess_market_data(**inputs)
    assert result.source_dates.loc["2024-01-05", "KR"] == day
    assert result.prices_krw.loc["2024-01-05", "KR"] == pytest.approx(110)


@pytest.mark.leakage
def test_delayed_domestic_close_is_excluded(inputs):
    dates = inputs["kr_calendar"]
    available = pd.Series(
        (dates + pd.Timedelta(hours=15, minutes=30)).tz_localize("Asia/Seoul"), index=dates
    )
    available.loc["2024-01-04"] += pd.Timedelta(minutes=1)
    result = preprocess_market_data(**inputs, domestic_available_at=available)
    assert "PRICE_NOT_AVAILABLE" in reasons(result)["2024-01-04"]
    assert "MISSING_PREVIOUS_PRICE" in reasons(result)["2024-01-05"]
