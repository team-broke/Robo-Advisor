"""원천 시계열의 결함과 검증 근거를 네트워크 없이 확인한다."""

import numpy as np
import pandas as pd
import pytest

from robo_advisor.data.validation import (
    check_dividend_adjustment,
    validate_listing_date,
    validate_missing_intervals,
    validate_price_data,
)


def _codes(report):
    return {issue["code"] for issue in report["issues"]}


def _issues(report, code):
    return [issue for issue in report["issues"] if issue["code"] == code]


@pytest.fixture
def prices():
    return pd.DataFrame(
        {"Close": [100.0, 101.0, 102.0, 103.0], "Volume": [0, 10, 20, 0]},
        index=pd.date_range("2025-01-02", periods=4, name="Date"),
    )


@pytest.fixture
def dividend_inputs():
    dates = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"])
    raw = pd.Series([100.0, 101.0, 98.0, 99.0], index=dates, name="raw_close")
    reference = pd.Series([100.0, 101.0, 101.0, 101.0 * 99.0 / 98.0], index=dates)
    return {
        "prices": reference * 2.0,
        "raw_close": raw,
        "reinvested_close": reference,
        "dividends": pd.Series([3.0], index=pd.to_datetime(["2025-01-06"])),
        "trading_calendar": dates,
        "comparison_basis": "gross_total_return",
        "split_free": True,
        "sources": {
            "prices": "fixture:tested-series",
            "raw_close": "fixture:raw-close",
            "reinvested_close": "fixture:gross-total-return",
            "dividends": "fixture:cash-dividends",
        },
    }


@pytest.mark.parametrize("date_column", [None, "Date"])
def test_valid_prices_and_zero_volume_pass(prices, date_column):
    if date_column:
        prices = prices.reset_index()

    result = validate_price_data(prices, date_column=date_column)

    assert result["valid"] is True
    assert result["status"] == "passed"
    assert result["issues"] == []


def test_duplicate_trading_day_records_every_position(prices):
    prices.index = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-03", "2025-01-05"])

    result = validate_price_data(prices)

    assert result["valid"] is False
    assert result["status"] == "failed"
    duplicate = _issues(result, "DUPLICATE_DATE")
    assert {issue["position"] for issue in duplicate} == {1, 2}
    assert {issue["date"] for issue in duplicate} == {"2025-01-03"}
    assert "OUT_OF_ORDER_DATE" not in _codes(result)
    assert all(issue["message"] and issue["severity"] == "error" for issue in duplicate)


def test_intraday_observations_on_same_day_are_duplicates():
    data = pd.DataFrame(
        {"Close": [100.0, 101.0]},
        index=pd.to_datetime(["2025-01-02 09:00", "2025-01-02 15:00"]),
    )

    result = validate_price_data(data)

    assert "DUPLICATE_DATE" in _codes(result)
    assert "OUT_OF_ORDER_DATE" not in _codes(result)


def test_reversed_dates_are_detected_without_sorting(prices):
    prices.index = pd.to_datetime(["2025-01-02", "2025-01-04", "2025-01-03", "2025-01-05"])
    original = prices.copy(deep=True)

    result = validate_price_data(prices)

    assert result["status"] == "failed"
    reversed_dates = _issues(result, "OUT_OF_ORDER_DATE")
    assert any(issue["date"] == "2025-01-03" and issue["position"] == 2 for issue in reversed_dates)
    assert "DUPLICATE_DATE" not in _codes(result)
    pd.testing.assert_frame_equal(prices, original)


@pytest.mark.parametrize(
    ("value", "code"),
    [
        (0.0, "NON_POSITIVE_PRICE"),
        (-20.0, "NON_POSITIVE_PRICE"),
        (np.nan, "MISSING_PRICE"),
        (np.inf, "NON_FINITE_PRICE"),
        (-np.inf, "NON_FINITE_PRICE"),
    ],
)
def test_invalid_price_records_date_column_and_position(prices, value, code):
    prices.loc["2025-01-04", "Close"] = value

    result = validate_price_data(prices)

    assert result["valid"] is False
    assert result["status"] == "failed"
    issue = _issues(result, code)[0]
    assert issue["date"] == "2025-01-04"
    assert issue["position"] == 2
    assert issue["column"] == "Close"
    assert issue["message"]


@pytest.mark.parametrize(
    "values",
    [
        ["100", "101", "102", "103"],
        [100.0, "bad", 102.0, 103.0],
        [True, False, True, True],
        [100 + 0j, 101 + 0j, 102 + 0j, 103 + 0j],
    ],
)
def test_price_type_is_not_silently_coerced(prices, values):
    prices["Close"] = values

    result = validate_price_data(prices)

    assert result["status"] == "failed"
    assert "INVALID_PRICE_TYPE" in _codes(result)
    assert all(issue["column"] == "Close" for issue in _issues(result, "INVALID_PRICE_TYPE"))


def test_selected_ohlc_columns_use_same_price_rules(prices):
    for column in ("Open", "High", "Low"):
        prices[column] = prices["Close"]
    prices.loc["2025-01-03", ["Open", "High", "Low", "Close"]] = 0.0

    result = validate_price_data(prices, price_columns=("Open", "High", "Low", "Close"))

    invalid_columns = {issue["column"] for issue in _issues(result, "NON_POSITIVE_PRICE")}
    assert invalid_columns == {"Open", "High", "Low", "Close"}
    assert "Volume" not in invalid_columns


def test_nat_date_is_a_recorded_error(prices):
    prices.index = pd.DatetimeIndex(["2025-01-02", pd.NaT, "2025-01-04", "2025-01-05"])

    result = validate_price_data(prices)

    assert result["status"] == "failed"
    assert any(issue["position"] == 1 for issue in _issues(result, "INVALID_DATE"))


def test_numeric_date_column_is_not_interpreted_as_nanoseconds(prices):
    prices = prices.reset_index(drop=True)
    prices["Date"] = [20250102, 20250103, 20250104, 20250105]

    result = validate_price_data(prices, date_column="Date")

    assert result["status"] == "failed"
    assert "INVALID_DATE" in _codes(result)


def test_invalid_date_string_is_not_dropped(prices):
    prices = prices.reset_index()
    prices["Date"] = prices["Date"].astype(object)
    prices.loc[1, "Date"] = "not-a-date"

    result = validate_price_data(prices, date_column="Date")

    assert result["status"] == "failed"
    assert any(issue["position"] == 1 for issue in _issues(result, "INVALID_DATE"))


def test_empty_data_cannot_pass(prices):
    result = validate_price_data(prices.iloc[:0])

    assert result["status"] == "failed"
    assert "EMPTY_DATA" in _codes(result)


def test_missing_price_column_is_a_recorded_error(prices):
    result = validate_price_data(prices.drop(columns="Close"))

    assert result["status"] == "failed"
    assert "MISSING_COLUMN" in _codes(result)


def test_duplicate_columns_are_not_accepted(prices):
    data = pd.concat([prices[["Close"]], prices[["Close"]]], axis=1)

    result = validate_price_data(data)

    assert result["status"] == "failed"
    assert "DUPLICATE_COLUMN" in _codes(result)


def test_explicit_market_timezone_defines_trading_day():
    dates = pd.to_datetime(["2025-01-01 16:00Z", "2025-01-02 01:00Z"])
    data = pd.DataFrame({"Close": [100.0, 101.0]}, index=dates)

    result = validate_price_data(data, market_timezone="Asia/Seoul")

    assert result["status"] == "failed"
    assert {issue["date"] for issue in _issues(result, "DUPLICATE_DATE")} == {"2025-01-02"}


def test_consistent_timezone_aware_prices_pass(prices):
    prices.index = prices.index.tz_localize("Asia/Seoul")

    result = validate_price_data(prices)

    assert result["status"] == "passed"


def test_naive_and_aware_dates_cannot_silently_mix():
    data = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03", tz="UTC")],
            "Close": [100.0, 101.0],
        }
    )

    result = validate_price_data(data, date_column="Date")

    assert result["status"] == "failed"
    assert "INCONSISTENT_TIMEZONE" in _codes(result)


def test_price_validation_does_not_modify_valid_or_invalid_data(prices):
    prices.loc["2025-01-03", "Close"] = np.nan
    original = prices.copy(deep=True)

    validate_price_data(prices)

    pd.testing.assert_frame_equal(prices, original, check_exact=True)


def test_missing_interval_positions_and_lengths_are_exact():
    data = pd.DataFrame(
        {"Close": [100.0, np.nan, np.nan, np.nan, 104.0, np.nan, 106.0]},
        index=pd.date_range("2025-01-02", periods=7),
    )

    result = validate_missing_intervals(data)

    column = result["metrics"]["columns"]["Close"]
    assert column["missing_count"] == 4
    assert column["total_count"] == 7
    assert column["missing_ratio"] == pytest.approx(4 / 7)
    assert column["max_consecutive_missing"] == 3
    assert column["intervals"] == [
        {
            "start_position": 1,
            "end_position": 3,
            "start_date": "2025-01-03",
            "end_date": "2025-01-05",
            "length": 3,
        },
        {
            "start_position": 5,
            "end_position": 5,
            "start_date": "2025-01-07",
            "end_date": "2025-01-07",
            "length": 1,
        },
    ]
    assert "EXCESSIVE_MISSING_RATIO" in _codes(result)


@pytest.mark.parametrize(
    ("max_ratio", "max_run", "status", "code"),
    [
        (0.5, 2, "passed", None),
        (0.499, 2, "failed", "EXCESSIVE_MISSING_RATIO"),
        (0.5, 1, "failed", "EXCESSIVE_MISSING_INTERVAL"),
    ],
)
def test_missing_threshold_is_exceeded_only_above_boundary(max_ratio, max_run, status, code):
    dates = pd.date_range("2025-01-02", periods=4)
    data = pd.DataFrame({"Close": [100.0, np.nan, np.nan, 103.0]}, index=dates)

    result = validate_missing_intervals(
        data,
        trading_calendar=dates,
        max_missing_ratio=max_ratio,
        max_consecutive_missing=max_run,
    )

    assert result["status"] == status
    assert result["valid"] is (status == "passed")
    if code:
        assert code in _codes(result)
    else:
        assert not result["issues"]


def test_calendar_absent_trading_row_counts_as_missing():
    dates = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])
    data = pd.DataFrame({"Close": [100.0, 102.0]}, index=dates[[0, 2]])
    original = data.copy(deep=True)

    result = validate_missing_intervals(data, trading_calendar=dates)

    column = result["metrics"]["columns"]["Close"]
    assert result["status"] == "failed"
    assert result["metrics"]["calendar_status"] == "verified"
    assert column["missing_count"] == 1
    assert column["missing_ratio"] == pytest.approx(1 / 3)
    assert column["intervals"][0]["start_date"] == "2025-01-03"
    assert column["intervals"][0]["length"] == 1
    pd.testing.assert_frame_equal(data, original)


def test_explicit_calendar_excludes_holidays_and_weekends():
    # 1월 3일은 이 합성 캘린더에서 휴장일이며, 4~5일은 주말이다.
    dates = pd.to_datetime(["2025-01-02", "2025-01-06"])
    data = pd.DataFrame({"Close": [100.0, 101.0]}, index=dates)

    result = validate_missing_intervals(data, trading_calendar=dates)

    assert result["status"] == "passed"
    assert result["metrics"]["columns"]["Close"]["missing_count"] == 0


def test_no_calendar_does_not_invent_missing_trading_days():
    dates = pd.to_datetime(["2025-01-02", "2025-01-06"])
    data = pd.DataFrame({"Close": [100.0, 101.0]}, index=dates)

    result = validate_missing_intervals(data)

    assert result["valid"] is False
    assert result["status"] == "unverified"
    assert result["metrics"]["calendar_status"] == "unverified"
    assert result["metrics"]["columns"]["Close"]["missing_count"] == 0
    assert "CALENDAR_UNVERIFIED" in _codes(result)


def test_excessive_observed_missing_values_fail_even_without_calendar(prices):
    prices["Close"] = np.nan

    result = validate_missing_intervals(prices)

    assert result["status"] == "failed"
    assert {"CALENDAR_UNVERIFIED", "EXCESSIVE_MISSING_RATIO"}.issubset(_codes(result))
    assert result["metrics"]["columns"]["Close"]["max_consecutive_missing"] == 4


def test_infinity_is_not_reported_as_a_missing_value(prices):
    prices.loc["2025-01-03", "Close"] = np.inf

    result = validate_missing_intervals(prices, trading_calendar=prices.index)

    assert result["metrics"]["columns"]["Close"]["missing_count"] == 0
    assert "EXCESSIVE_MISSING_RATIO" not in _codes(result)


def test_missing_columns_are_measured_independently(prices):
    prices["Open"] = [np.nan, np.nan, 102.0, 103.0]
    prices["Close"] = [100.0, 101.0, np.nan, 103.0]

    result = validate_missing_intervals(
        prices, columns=("Open", "Close"), trading_calendar=prices.index
    )

    assert result["metrics"]["columns"]["Open"]["missing_count"] == 2
    assert result["metrics"]["columns"]["Close"]["missing_count"] == 1


def test_missing_validation_supports_date_column_and_preserves_input(prices):
    prices.loc["2025-01-02", "Close"] = np.nan
    dates = prices.index.copy()
    data = prices.reset_index()
    original = data.copy(deep=True)

    result = validate_missing_intervals(data, date_column="Date", trading_calendar=dates)

    assert result["metrics"]["columns"]["Close"]["missing_count"] == 1
    assert result["metrics"]["columns"]["Close"]["intervals"][0]["start_position"] == 0
    pd.testing.assert_frame_equal(data, original, check_exact=True)


@pytest.mark.parametrize("values", [[np.nan], [np.nan, np.nan, np.nan]])
def test_missing_run_covering_entire_input_has_inclusive_endpoints(values):
    dates = pd.date_range("2025-01-02", periods=len(values))
    data = pd.DataFrame({"Close": values}, index=dates)

    result = validate_missing_intervals(data, trading_calendar=dates)

    column = result["metrics"]["columns"]["Close"]
    assert column["missing_ratio"] == 1.0
    assert column["max_consecutive_missing"] == len(values)
    assert column["intervals"][0]["start_position"] == 0
    assert column["intervals"][0]["end_position"] == len(values) - 1
    assert column["intervals"][0]["length"] == len(values)


def test_observation_outside_calendar_is_not_silently_discarded(prices):
    result = validate_missing_intervals(prices, trading_calendar=prices.index[:-1])

    assert result["status"] == "failed"
    outside = _issues(result, "OUTSIDE_CALENDAR")
    assert any(issue["date"] == "2025-01-05" for issue in outside)


@pytest.mark.parametrize("market_timezone", [None, "Asia/Seoul"])
def test_data_and_calendar_timezone_mismatch_requires_explicit_market_timezone(
    prices, market_timezone
):
    prices.index = prices.index.tz_localize("Asia/Seoul")
    calendar = prices.index.tz_localize(None).tz_localize("UTC")

    result = validate_missing_intervals(
        prices, trading_calendar=calendar, market_timezone=market_timezone
    )

    if market_timezone is None:
        assert result["status"] == "failed"
        assert "INCONSISTENT_TIMEZONE" in _codes(result)
    else:
        assert result["status"] == "passed"
        assert result["metrics"]["columns"]["Close"]["missing_count"] == 0


@pytest.mark.parametrize(
    "calendar",
    [
        pd.DatetimeIndex([]),
        pd.to_datetime(["2025-01-03", "2025-01-02"]),
        pd.to_datetime(["2025-01-02", "2025-01-02"]),
        pd.DatetimeIndex(["2025-01-02", pd.NaT]),
    ],
)
def test_invalid_trading_calendar_raises(prices, calendar):
    with pytest.raises(ValueError):
        validate_missing_intervals(prices, trading_calendar=calendar)


@pytest.mark.parametrize(
    "options",
    [
        {"max_missing_ratio": -0.1},
        {"max_missing_ratio": 1.1},
        {"max_missing_ratio": np.nan},
        {"max_missing_ratio": np.inf},
        {"max_consecutive_missing": -1},
        {"max_consecutive_missing": 1.5},
    ],
)
def test_invalid_missing_threshold_raises(prices, options):
    with pytest.raises(ValueError):
        validate_missing_intervals(prices, **options)


@pytest.mark.parametrize("listing_date", ["2019-12-31", "2020-01-01"])
def test_listing_date_covering_collection_start_passes(listing_date):
    result = validate_listing_date("2020-01-01", listing_date)

    assert result["status"] == "passed"
    assert result["metrics"]["collection_start"] == "2020-01-01"
    assert result["metrics"]["listing_date"] == listing_date


def test_late_listing_is_recorded_without_declaring_data_invalid():
    result = validate_listing_date("2020-01-01", "2022-06-01")

    assert result["valid"] is True
    assert result["status"] == "passed"
    issue = _issues(result, "LATE_LISTING")[0]
    assert issue["severity"] == "info"
    assert issue["message"]
    assert result["metrics"]["listing_date"] == "2022-06-01"


def test_late_listing_can_be_a_recorded_exclusion_for_full_period():
    result = validate_listing_date("2020-01-01", "2022-06-01", require_full_period=True)

    assert result["valid"] is False
    assert result["status"] == "failed"
    assert "INSUFFICIENT_HISTORY" in _codes(result)
    assert _issues(result, "INSUFFICIENT_HISTORY")[0]["severity"] == "error"


def test_unknown_listing_date_is_not_inferred():
    result = validate_listing_date("2020-01-01")

    assert result["valid"] is False
    assert result["status"] == "unverified"
    assert result["metrics"]["listing_date"] is None
    assert _issues(result, "LISTING_DATE_UNKNOWN")[0]["severity"] == "unverified"


def test_listing_date_uses_explicit_market_timezone():
    result = validate_listing_date(
        pd.Timestamp("2025-01-02", tz="Asia/Seoul"),
        pd.Timestamp("2025-01-01 16:00", tz="UTC"),
        market_timezone="Asia/Seoul",
    )

    assert result["status"] == "passed"
    assert result["metrics"]["listing_date"] == "2025-01-02"
    assert "LATE_LISTING" not in _codes(result)


def test_listing_dates_in_different_timezones_require_explicit_market_timezone():
    start = pd.Timestamp("2025-01-01 16:00", tz="UTC")
    listing = pd.Timestamp("2025-01-02", tz="Asia/Seoul")

    with pytest.raises(ValueError, match="timezone"):
        validate_listing_date(start, listing)

    result = validate_listing_date(start, listing, market_timezone="Asia/Seoul")

    assert result["status"] == "passed"
    assert result["metrics"]["collection_start"] == "2025-01-02"
    assert result["metrics"]["listing_date"] == "2025-01-02"


def test_dividend_adjusted_series_matches_total_return_evidence(dividend_inputs):
    result = check_dividend_adjustment(**dividend_inputs)

    assert result["valid"] is True
    assert result["status"] == "passed"
    assert result["metrics"]["sources"] == dividend_inputs["sources"]
    events = result["metrics"]["events"]
    assert len(events) == 1
    assert events[0]["date"] == "2025-01-06"
    assert events[0]["previous_date"] == "2025-01-03"
    assert events[0]["dividend"] == 3.0
    assert events[0]["raw_previous"] == 101.0
    assert events[0]["raw_current"] == 98.0
    assert events[0]["expected_return"] == pytest.approx(0.0)
    assert events[0]["raw_return"] == pytest.approx(-3 / 101)
    assert events[0]["reference_return"] == pytest.approx(0.0)
    assert events[0]["price_return"] == pytest.approx(0.0)
    assert events[0]["status"] == "passed"


def test_unadjusted_raw_series_is_detected_at_ex_dividend_date(dividend_inputs):
    dividend_inputs["prices"] = dividend_inputs["raw_close"].copy()

    result = check_dividend_adjustment(**dividend_inputs)

    assert result["valid"] is False
    assert result["status"] == "failed"
    assert "DIVIDEND_NOT_REFLECTED" in _codes(result)
    assert _issues(result, "DIVIDEND_NOT_REFLECTED")[0]["date"] == "2025-01-06"


def test_unexplained_price_difference_is_not_called_unadjusted(dividend_inputs):
    dividend_inputs["prices"] = dividend_inputs["reinvested_close"].copy()
    dividend_inputs["prices"].loc["2025-01-06"] = 105.0

    result = check_dividend_adjustment(**dividend_inputs)

    assert result["status"] == "failed"
    assert "DIVIDEND_COMPARISON_MISMATCH" in _codes(result)
    assert "DIVIDEND_NOT_REFLECTED" not in _codes(result)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("raw_close", None),
        ("reinvested_close", None),
        ("dividends", None),
        ("trading_calendar", None),
        ("sources", None),
        ("comparison_basis", None),
        ("comparison_basis", "adjusted_close"),
        ("split_free", None),
        ("split_free", False),
        ("event_date_kind", "payment_date"),
    ],
)
def test_missing_dividend_evidence_cannot_produce_verified_success(dividend_inputs, key, value):
    dividend_inputs[key] = value

    result = check_dividend_adjustment(**dividend_inputs)

    assert result["valid"] is False
    assert result["status"] == "unverified"
    assert any(issue["severity"] == "unverified" and issue["message"] for issue in result["issues"])


def test_prices_alone_cannot_verify_dividend_adjustment(dividend_inputs):
    result = check_dividend_adjustment(dividend_inputs["prices"])

    assert result["status"] == "unverified"
    assert result["valid"] is False


def test_no_cash_dividend_events_remain_unverified(dividend_inputs):
    dividend_inputs["dividends"] = pd.Series(
        [0.0], index=pd.to_datetime(["2025-01-06"]), dtype=float
    )

    result = check_dividend_adjustment(**dividend_inputs)

    assert result["status"] == "unverified"
    assert result["valid"] is False


def test_empty_dividend_event_series_is_unverified_instead_of_success(dividend_inputs):
    dividend_inputs["dividends"] = pd.Series([], index=pd.DatetimeIndex([]), dtype=float)

    result = check_dividend_adjustment(**dividend_inputs)

    assert result["status"] == "unverified"
    assert result["valid"] is False
    assert "NO_DIVIDEND_EVENTS" in _codes(result)


def test_indistinguishable_dividend_effect_is_unverified(dividend_inputs):
    result = check_dividend_adjustment(**dividend_inputs, rtol=0.0, atol=0.05)

    assert result["status"] == "unverified"
    assert "DIVIDEND_SIGNAL_UNVERIFIED" in _codes(result)


@pytest.mark.parametrize(
    ("checked_current", "reference_current"),
    [
        (100.0, 100.0075),
        (100.0075, 100.015),
    ],
)
def test_dividend_tolerance_cannot_verify_transitive_or_ambiguous_matches(
    dividend_inputs, checked_current, reference_current
):
    dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
    dividend_inputs["trading_calendar"] = dates
    dividend_inputs["raw_close"] = pd.Series([100.0, 100.0], index=dates)
    dividend_inputs["prices"] = pd.Series([100.0, checked_current], index=dates)
    dividend_inputs["reinvested_close"] = pd.Series([100.0, reference_current], index=dates)
    dividend_inputs["dividends"] = pd.Series([0.015], index=dates[1:])

    result = check_dividend_adjustment(**dividend_inputs, rtol=0.0, atol=1e-4)

    assert result["status"] == "unverified"
    assert result["valid"] is False
    assert result["metrics"]["events"][0]["status"] == "unverified"


@pytest.mark.parametrize("market_timezone", [None, "Asia/Seoul"])
def test_dividend_series_timezone_mismatch_requires_explicit_market_timezone(
    dividend_inputs, market_timezone
):
    for key in ("prices", "raw_close", "reinvested_close", "dividends"):
        timezone = "Asia/Seoul" if key == "raw_close" else "UTC"
        dividend_inputs[key].index = dividend_inputs[key].index.tz_localize(timezone)
    dividend_inputs["trading_calendar"] = dividend_inputs["trading_calendar"].tz_localize("UTC")

    result = check_dividend_adjustment(**dividend_inputs, market_timezone=market_timezone)

    if market_timezone is None:
        assert result["status"] == "failed"
        assert "INCONSISTENT_TIMEZONE" in _codes(result)
    else:
        assert result["status"] == "passed"


def test_cash_dividend_on_first_calendar_day_has_no_comparable_previous_day(dividend_inputs):
    dividend_inputs["dividends"] = pd.Series([3.0], index=pd.to_datetime(["2025-01-02"]))

    result = check_dividend_adjustment(**dividend_inputs)

    assert result["status"] == "unverified"
    assert "DIVIDEND_SESSION_UNVERIFIED" in _codes(result)


def test_negative_cash_dividend_is_invalid_supplied_evidence(dividend_inputs):
    dividend_inputs["dividends"] = pd.Series([-3.0], index=pd.to_datetime(["2025-01-06"]))

    result = check_dividend_adjustment(**dividend_inputs)

    assert result["status"] == "failed"
    issue = _issues(result, "NON_POSITIVE_PRICE")[0]
    assert issue["date"] == "2025-01-06"
    assert issue["column"] == "dividends"


def test_one_failed_dividend_event_cannot_be_hidden_by_a_passed_event(dividend_inputs):
    dates = dividend_inputs["trading_calendar"]
    dividend_inputs["raw_close"] = pd.Series([100.0, 99.0, 98.0, 97.0], index=dates)
    dividend_inputs["reinvested_close"] = pd.Series([100.0] * 4, index=dates)
    dividend_inputs["prices"] = pd.Series([100.0, 100.0, 100.0 * 98.0 / 99.0, 100.0], index=dates)
    dividend_inputs["dividends"] = pd.Series([1.0, 1.0], index=dates[[1, 2]])

    result = check_dividend_adjustment(**dividend_inputs)

    assert result["status"] == "failed"
    assert [event["status"] for event in result["metrics"]["events"]] == ["passed", "failed"]
    assert _issues(result, "DIVIDEND_NOT_REFLECTED")[0]["date"] == "2025-01-06"


def test_incomplete_source_identifiers_remain_unverified(dividend_inputs):
    del dividend_inputs["sources"]["dividends"]

    result = check_dividend_adjustment(**dividend_inputs)

    assert result["status"] == "unverified"


def test_missing_previous_trading_day_is_not_replaced_by_earlier_observation(dividend_inputs):
    dividend_inputs["raw_close"] = dividend_inputs["raw_close"].drop(pd.Timestamp("2025-01-03"))

    result = check_dividend_adjustment(**dividend_inputs)

    assert result["status"] == "unverified"
    assert result["valid"] is False


def test_inconsistent_total_return_reference_cannot_verify_adjustment(dividend_inputs):
    dividend_inputs["reinvested_close"].loc["2025-01-06"] = 120.0

    result = check_dividend_adjustment(**dividend_inputs)

    assert result["status"] == "unverified"
    assert "DIVIDEND_NOT_REFLECTED" not in _codes(result)


def test_configured_tolerance_controls_dividend_comparison(dividend_inputs):
    dividend_inputs["prices"] = dividend_inputs["reinvested_close"].copy()
    dividend_inputs["prices"].loc["2025-01-06"] += 0.001

    strict = check_dividend_adjustment(**dividend_inputs, rtol=0.0, atol=1e-8)
    relaxed = check_dividend_adjustment(**dividend_inputs, rtol=0.0, atol=1e-4)

    assert strict["status"] == "failed"
    assert relaxed["status"] == "passed"


@pytest.mark.parametrize(
    "options",
    [
        {"rtol": -0.1},
        {"atol": -0.1},
        {"rtol": np.nan},
        {"atol": np.inf},
    ],
)
def test_invalid_dividend_tolerance_raises(dividend_inputs, options):
    with pytest.raises(ValueError):
        check_dividend_adjustment(**dividend_inputs, **options)


@pytest.mark.parametrize("key", ["prices", "raw_close", "reinvested_close"])
def test_invalid_supplied_dividend_prices_are_explicit_failures(dividend_inputs, key):
    dividend_inputs[key].loc["2025-01-06"] = np.inf

    result = check_dividend_adjustment(**dividend_inputs)

    assert result["status"] == "failed"
    assert result["valid"] is False
    assert any(issue["severity"] == "error" for issue in result["issues"])


def test_dividend_validation_preserves_all_input_series(dividend_inputs):
    originals = {
        key: dividend_inputs[key].copy(deep=True)
        for key in ("prices", "raw_close", "reinvested_close", "dividends")
    }

    check_dividend_adjustment(**dividend_inputs)

    for key, original in originals.items():
        pd.testing.assert_series_equal(dividend_inputs[key], original, check_exact=True)


def test_invalid_date_does_not_hide_other_price_issue_dates(prices):
    data = prices.reset_index()
    data["Date"] = data["Date"].astype(object)
    data.loc[1, "Date"] = "not-a-date"
    data.loc[2, "Close"] = 0.0

    report = validate_price_data(data, date_column="Date")

    assert report["status"] == "failed"
    assert _issues(report, "INVALID_DATE")[0]["value"] == "not-a-date"
    assert _issues(report, "NON_POSITIVE_PRICE")[0]["date"] == "2025-01-04"


def test_unverified_dividend_report_retains_known_event_and_tolerances(dividend_inputs):
    dividend_inputs["reinvested_close"] = None

    report = check_dividend_adjustment(**dividend_inputs, rtol=0.01, atol=0.001)

    assert report["status"] == "unverified"
    assert report["metrics"]["rtol"] == 0.01
    assert report["metrics"]["atol"] == 0.001
    assert report["metrics"]["events"][0]["date"] == "2025-01-06"
    assert report["metrics"]["events"][0]["status"] == "unverified"
