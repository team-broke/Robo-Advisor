"""수집 계약, 원천 보존 및 검증 근거를 외부 API 없이 확인한다."""

from datetime import datetime, timezone
from pathlib import Path
import socket

import numpy as np
import pandas as pd
import pytest

from robo_advisor.data.collectors import (
    DatasetSpec,
    collect_fred,
    collect_yahoo,
    load_asset_config,
)


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Collector tests must not access the network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def _spec(**overrides):
    fields = {
        "dataset_id": "spy_candidate",
        "symbol": "SPY",
        "source": "yahoo",
        "role": "etf",
        "market": "US",
        "currency": "USD",
        "timezone": "America/New_York",
        "asset_class": "equity",
        "rationale": "Source found; team selection remains open.",
        "selection_status": "candidate",
        "unit": "USD",
    }
    fields.update(overrides)
    return DatasetSpec(**fields)


def _fred_spec():
    return _spec(
        dataset_id="dgs3mo_candidate",
        symbol="DGS3MO",
        source="fred",
        role="risk_free",
        market="US_RATE",
        asset_class="interest_rate",
        unit="percent_per_annum",
    )


def _fx_spec():
    return _spec(
        dataset_id="usd_krw",
        symbol="KRW=X",
        role="fx",
        market="FX",
        currency="KRW",
        timezone="Asia/Seoul",
        asset_class="foreign_exchange",
        selection_status="confirmed",
        unit="KRW_per_USD",
    )


@pytest.fixture
def prices():
    frame = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [102.0, 103.0, 104.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [101.0, 102.0, 103.0],
            "Volume": [0, 20, 30],
        },
        index=pd.to_datetime(["2019-09-02", "2019-09-03", "2019-09-04"]),
    )
    frame.index.name = "Date"
    frame.attrs["source_note"] = "fixture preserved verbatim"
    return frame


def _yahoo(frame, spec=None, **kwargs):
    return collect_yahoo(
        _spec() if spec is None else spec,
        end="2019-09-05",
        download=lambda **request: frame,
        **kwargs,
    )


def _codes(result):
    return {issue["code"] for issue in result.metadata["issues"]}


class FakeResponse:
    def __init__(self, text, error=None):
        self.text = text
        self.error = error

    def raise_for_status(self):
        if self.error is not None:
            raise self.error


def _fred(text, **kwargs):
    return collect_fred(
        _fred_spec(),
        end="2019-09-05",
        http_get=lambda *args, **request: FakeResponse(text),
        **kwargs,
    )


def test_yahoo_contract_metadata_and_source_preservation(prices):
    calls = []
    original = prices.copy(deep=True)

    def download(**request):
        calls.append(request)
        return prices

    before = datetime.now(timezone.utc)
    result = collect_yahoo(_spec(), end="2019-09-05", download=download)
    after = datetime.now(timezone.utc)
    assert calls == [
        {
            "tickers": "SPY",
            "start": "2019-09-01",
            "end": "2019-09-05",
            "interval": "1d",
            "auto_adjust": True,
            "progress": False,
            "threads": False,
        }
    ]
    assert result.status == "unverified"
    assert result.validation["price"]["status"] == "passed"
    expected = {
        "dataset_id": "spy_candidate",
        "symbol": "SPY",
        "source": "yahoo",
        "role": "etf",
        "market": "US",
        "currency": "USD",
        "unit": "USD",
        "selection_status": "candidate",
        "requested_start": "2019-09-01",
        "requested_end_exclusive": "2019-09-05",
        "actual_start": "2019-09-02",
        "actual_end": "2019-09-04",
        "row_count": 3,
        "auto_adjust": True,
    }
    assert {key: result.metadata[key] for key in expected} == expected
    collected = datetime.fromisoformat(result.metadata["collected_at"].replace("Z", "+00:00"))
    assert collected.utcoffset().total_seconds() == 0
    assert before <= collected <= after
    assert result.metadata["price_basis"]
    pd.testing.assert_frame_equal(result.data, original)
    pd.testing.assert_frame_equal(result.raw_data, original)
    pd.testing.assert_frame_equal(prices, original)
    result.raw_data.iloc[0, 0] = 777.0
    assert prices.iloc[0, 0] == 100.0


def test_explicit_collection_start_overrides_warmup_default(prices):
    calls = []

    def download(**request):
        calls.append(request)
        return prices.iloc[1:].copy(deep=True)

    result = collect_yahoo(_spec(), start="2019-09-03", end="2019-09-05", download=download)
    assert calls[0]["start"] == "2019-09-03"
    assert result.metadata["requested_start"] == "2019-09-03"
    assert result.metadata["actual_start"] == "2019-09-03"
    assert result.metadata["row_count"] == 2


@pytest.mark.parametrize(
    "start,end",
    [
        ("2019-09-05", "2019-09-05"),
        ("2019-09-06", "2019-09-05"),
        ("not-a-date", "2019-09-05"),
        ("2019-09-01", "bad-end"),
        ("2019-09-01T12:00:00", "2019-09-05"),
        ("2019-09-01T00:00:00Z", "2019-09-05"),
        ("2019-09-01", "2019-09-05T00:00:00+09:00"),
    ],
)
def test_invalid_request_dates_fail_before_download(start, end):
    calls = []
    with pytest.raises(ValueError):
        collect_yahoo(_spec(), start=start, end=end, download=lambda **kw: calls.append(kw))
    assert calls == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"source": "unknown"},
        {"role": "unknown"},
        {"currency": "KRW"},
        {"selection_status": "approved"},
        {"symbol": "069500", "market": "KRX"},
        {"symbol": "069500.KS", "market": "US"},
        {"timezone": "invalid/timezone"},
        {"dataset_id": ""},
        {"rationale": ""},
    ],
)
def test_bad_dataset_contract_is_rejected_before_fetch(overrides):
    calls = []
    with pytest.raises(ValueError):
        collect_yahoo(_spec(**overrides), end="2019-09-05", download=lambda **kw: calls.append(kw))
    assert calls == []


@pytest.mark.parametrize("collector,spec", [(collect_yahoo, _spec()), (collect_fred, _fred_spec())])
def test_live_client_requires_explicit_network_authorization(collector, spec):
    result = collector(spec, end="2019-09-05")
    assert result.status == "failed"
    assert "NETWORK_NOT_AUTHORIZED" in _codes(result)


@pytest.mark.parametrize("layout", ["price_first", "ticker_first"])
def test_single_ticker_yahoo_multiindex_is_supported(prices, layout):
    source = prices.copy(deep=True)
    tuples = [(column, "SPY") for column in source.columns]
    if layout == "ticker_first":
        tuples = [tuple(reversed(pair)) for pair in tuples]
    source.columns = pd.MultiIndex.from_tuples(tuples)
    result = _yahoo(source)
    pd.testing.assert_frame_equal(result.data, prices)
    pd.testing.assert_frame_equal(result.raw_data, source)


@pytest.mark.parametrize("multiple", [False, True])
def test_wrong_or_multiple_tickers_are_not_silently_selected(prices, multiple):
    source = prices.copy(deep=True)
    source.columns = pd.MultiIndex.from_tuples([(column, "QQQ") for column in prices.columns])
    if multiple:
        second = prices.copy(deep=True)
        second.columns = pd.MultiIndex.from_tuples([(column, "SPY") for column in prices.columns])
        source = pd.concat([source, second], axis=1)
    result = _yahoo(source)
    assert result.status == "failed"
    pd.testing.assert_frame_equal(result.raw_data, source)


@pytest.mark.parametrize("column", ["Open", "High", "Low", "Close", "Volume"])
def test_missing_ohlcv_columns_fail_with_evidence(prices, column):
    source = prices.drop(columns=column)
    result = _yahoo(source)
    assert result.status == "failed"
    assert result.metadata["issues"]
    pd.testing.assert_frame_equal(result.raw_data, source)


@pytest.mark.parametrize("kind", ["low_above_high", "open_above_high"])
def test_ohlc_relationship_anomalies_are_reported_without_repair(prices, kind):
    source = prices.copy(deep=True)
    if kind == "low_above_high":
        source.loc[source.index[1], "Low"] = 104.0
    else:
        source.loc[source.index[1], "Open"] = 104.0
    result = _yahoo(source)
    assert result.status == "failed"
    assert result.metadata["issues"]
    pd.testing.assert_frame_equal(result.raw_data, source)
    pd.testing.assert_frame_equal(result.data, source)


@pytest.mark.parametrize("outside", ["2019-08-30", "2019-09-05"])
def test_provider_rows_outside_requested_period_are_failed_and_preserved(prices, outside):
    source = prices.copy(deep=True)
    dates = list(source.index)
    dates[0 if outside == "2019-08-30" else -1] = pd.Timestamp(outside)
    source.index = pd.DatetimeIndex(dates, name="Date")
    result = _yahoo(source)
    assert result.status == "failed"
    assert "OUTSIDE_REQUESTED_PERIOD" in _codes(result)
    assert len(result.data) == 3
    pd.testing.assert_frame_equal(result.raw_data, source)
    pd.testing.assert_frame_equal(result.data, source)


@pytest.mark.parametrize("bad_value", [0.0, -1.0, np.nan, np.inf, -np.inf, "101.0"])
def test_invalid_price_is_retained_and_failed(prices, bad_value):
    source = prices.copy(deep=True)
    if isinstance(bad_value, str):
        source["Close"] = source.Close.astype(object)
    source.loc[source.index[1], "Close"] = bad_value
    result = _yahoo(source)
    assert result.status == "failed"
    assert result.validation["price"]["status"] == "failed"
    assert len(result.data) == 3
    pd.testing.assert_frame_equal(result.raw_data, source)


@pytest.mark.parametrize("bad_volume", [-1.0, np.nan, np.inf, "20"])
def test_invalid_volume_is_retained_and_failed(prices, bad_volume):
    source = prices.copy(deep=True)
    source["Volume"] = source.Volume.astype(object if isinstance(bad_volume, str) else float)
    source.loc[source.index[1], "Volume"] = bad_volume
    result = _yahoo(source)
    assert result.status == "failed"
    assert result.metadata["issues"]
    pd.testing.assert_frame_equal(result.raw_data, source)


@pytest.mark.parametrize("kind", ["duplicate", "reversed", "nat"])
def test_invalid_date_rows_are_neither_sorted_nor_removed(prices, kind):
    source = prices.copy(deep=True)
    dates = list(source.index)
    if kind == "duplicate":
        dates[1] = dates[0]
    elif kind == "reversed":
        dates[1], dates[2] = dates[2], dates[1]
    else:
        dates[1] = pd.NaT
    source.index = pd.DatetimeIndex(dates, name="Date")
    result = _yahoo(source)
    assert result.status == "failed"
    assert result.validation["price"]["status"] == "failed"
    assert len(result.data) == 3
    pd.testing.assert_frame_equal(result.raw_data, source)


@pytest.mark.parametrize("source", [None, pd.DataFrame()])
def test_empty_provider_response_is_an_explicit_failure(source):
    result = _yahoo(source)
    assert result.status == "failed"
    assert result.metadata["row_count"] == 0
    assert result.metadata["issues"]


def test_provider_exception_is_recorded_without_synthetic_data():
    def download(**kwargs):
        raise RuntimeError("fixture provider outage")

    result = collect_yahoo(_spec(), end="2019-09-05", download=download)
    assert result.status == "failed"
    assert result.metadata["row_count"] == 0
    assert result.error["code"] == "COLLECTION_FAILED"
    assert result.error["exception_type"] == "RuntimeError"


def test_market_timezone_preserves_korean_native_session_dates(prices):
    source = prices.copy(deep=True)
    source.index = source.index.tz_localize("Asia/Seoul").tz_convert("UTC")
    spec = _spec(
        symbol="069500.KS", market="KRX", currency="KRW", unit="KRW", timezone="Asia/Seoul"
    )
    result = _yahoo(source, spec)
    assert list(result.data.index) == list(prices.index)
    assert result.metadata["actual_start"] == "2019-09-02"
    pd.testing.assert_frame_equal(result.raw_data, source)


def test_calendar_is_required_to_claim_session_completeness(prices):
    unchecked = _yahoo(prices)
    checked = _yahoo(prices, trading_calendar=prices.index)
    assert unchecked.validation["missing"]["status"] == "unverified"
    assert checked.validation["missing"]["status"] == "passed"
    assert checked.status == "unverified"


def test_missing_session_is_reported_without_inserting_a_row(prices):
    source = prices.iloc[[0, 2]].copy(deep=True)
    result = _yahoo(source, trading_calendar=prices.index)
    assert result.status == "failed"
    assert result.validation["missing"]["status"] == "failed"
    assert len(result.data) == 2
    pd.testing.assert_frame_equal(result.raw_data, source)


@pytest.mark.parametrize(
    "symbol,market,currency,tz",
    [
        ("^KS11", "KRX", "KRW", "Asia/Seoul"),
        ("^GSPC", "US", "USD", "America/New_York"),
    ],
)
def test_required_benchmarks_share_the_ohlcv_contract(prices, symbol, market, currency, tz):
    spec = _spec(
        symbol=symbol,
        market=market,
        currency=currency,
        unit=currency,
        timezone=tz,
        role="benchmark",
        selection_status="confirmed",
    )
    result = _yahoo(prices, spec)
    assert result.validation["price"]["status"] == "passed"
    assert result.metadata["symbol"] == symbol
    assert result.metadata["auto_adjust"] is True


def test_fx_quote_contract_maps_to_issue_12_direction(prices):
    result = _yahoo(
        prices,
        _fx_spec(),
        quote_metadata={"symbol": "KRW=X", "base_currency": "USD", "quote_currency": "KRW"},
    )
    assert result.metadata["unit"] == "KRW_per_USD"
    assert result.metadata["fx_direction"] == "USD/KRW"
    assert result.metadata["fx_direction_status"] == "verified"
    pd.testing.assert_frame_equal(result.data, prices)


def test_fx_without_provider_unit_evidence_remains_unverified(prices):
    result = _yahoo(prices, _fx_spec())
    assert result.status == "unverified"
    assert result.metadata["fx_direction_status"] == "unverified"


@pytest.mark.parametrize(
    "evidence",
    [
        {"symbol": "USD=X", "base_currency": "USD", "quote_currency": "KRW"},
        {"symbol": "KRW=X", "base_currency": "KRW", "quote_currency": "USD"},
    ],
)
def test_fx_conflicting_provider_evidence_fails(prices, evidence):
    result = _yahoo(prices, _fx_spec(), quote_metadata=evidence)
    assert result.status == "failed"
    assert result.metadata["issues"]


def test_fred_uses_exclusive_end_and_retains_annual_percent_units():
    calls = []
    text = "DATE,DGS3MO\n2019-09-02,0.0\n2019-09-03,-0.1\n2019-09-04,2.0\n"

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse(text)

    result = collect_fred(_fred_spec(), end="2019-09-05", http_get=get)
    assert calls == [
        (
            "https://fred.stlouisfed.org/graph/fredgraph.csv",
            {"params": {"id": "DGS3MO", "cosd": "2019-09-01", "coed": "2019-09-04"}, "timeout": 60},
        )
    ]
    assert result.validation["rate"]["status"] == "passed"
    assert result.status == "unverified"
    assert result.metadata["unit"] == "percent_per_annum"
    assert list(result.data["Value"]) == [0.0, -0.1, 2.0]
    assert result.metadata["requested_end_exclusive"] == "2019-09-05"
    assert result.metadata["actual_end"] == "2019-09-04"
    assert result.metadata["available_at"] is None


@pytest.mark.parametrize("date_column", ["DATE", "observation_date"])
def test_fred_date_column_variants(date_column):
    result = _fred(f"{date_column},DGS3MO\n2019-09-02,2.0\n2019-09-03,2.1\n")
    assert result.validation["rate"]["status"] == "passed"
    assert list(result.data.index) == list(pd.to_datetime(["2019-09-02", "2019-09-03"]))
    assert date_column in result.raw_data.columns


@pytest.mark.parametrize("value", [".", "unexpected", "inf"])
def test_fred_missing_or_invalid_observation_is_not_dropped(value):
    result = _fred(f"DATE,DGS3MO\n2019-09-02,2.0\n2019-09-03,{value}\n2019-09-04,2.1\n")
    assert result.status == "failed"
    assert len(result.raw_data) == 3
    assert len(result.data) == 3
    assert result.metadata["issues"]
    if value == ".":
        assert result.raw_data.loc[1, "DGS3MO"] == "."
        assert pd.isna(result.data.loc[pd.Timestamp("2019-09-03"), "Value"])


@pytest.mark.parametrize(
    "text",
    [
        "DATE,OTHER\n2019-09-02,2.0\n",
        "WRONG,DGS3MO\n2019-09-02,2.0\n",
        "DATE,DGS3MO\n",
    ],
)
def test_fred_response_schema_or_empty_data_failure_is_explicit(text):
    result = _fred(text)
    assert result.status == "failed"
    assert result.metadata["issues"]


def test_malformed_fred_csv_retains_original_response():
    text = 'DATE,DGS3MO\n"unterminated,2.0\n'
    result = _fred(text)
    assert result.status == "failed"
    assert result.raw_text == text
    assert "INVALID_FRED_CSV" in _codes(result)


@pytest.mark.parametrize(
    "dates",
    [
        ["2019-09-02", "2019-09-02", "2019-09-04"],
        ["2019-09-02", "2019-09-04", "2019-09-03"],
    ],
)
def test_fred_duplicate_or_reversed_dates_are_preserved(dates):
    text = "DATE,DGS3MO\n" + "".join(f"{date},{rate}\n" for date, rate in zip(dates, [2, 3, 4]))
    result = _fred(text)
    assert result.status == "failed"
    assert result.validation["missing"]["status"] == "failed"
    assert list(result.raw_data["DATE"]) == dates
    assert list(result.raw_data["DGS3MO"]) == ["2", "3", "4"]
    assert list(result.data.index) == list(pd.to_datetime(dates))
    assert list(result.data["Value"]) == [2.0, 3.0, 4.0]


def test_fred_http_failure_is_not_reported_as_verified():
    result = collect_fred(
        _fred_spec(),
        end="2019-09-05",
        http_get=lambda *args, **kwargs: FakeResponse("", RuntimeError("HTTP fixture failure")),
    )
    assert result.status == "failed"
    assert result.error["code"] == "COLLECTION_FAILED"
    assert result.error["exception_type"] == "RuntimeError"


def test_genuine_supplied_availability_is_preserved(prices):
    availability = pd.Series(
        pd.to_datetime(["2019-09-03T00:00Z", "2019-09-04T00:00Z", "2019-09-05T00:00Z"]),
        index=prices.index,
    )
    result = _yahoo(prices, available_at=availability)
    assert result.metadata["availability_status"] == "verified"
    pd.testing.assert_series_equal(result.metadata["available_at"], availability, check_names=False)
    assert result.status == "unverified"


@pytest.mark.parametrize("kind", ["naive", "wrong_dates", "before_observation", "nat"])
def test_bad_availability_evidence_fails(prices, kind):
    values = pd.to_datetime(["2019-09-03T00:00Z", "2019-09-04T00:00Z", "2019-09-05T00:00Z"])
    availability = pd.Series(values, index=prices.index)
    if kind == "naive":
        availability = availability.dt.tz_localize(None)
    elif kind == "wrong_dates":
        availability.index = availability.index + pd.Timedelta(days=1)
    elif kind == "before_observation":
        availability.iloc[1] = pd.Timestamp("2019-09-01T00:00Z")
    else:
        availability.iloc[1] = pd.NaT
    result = _yahoo(prices, available_at=availability)
    assert result.status == "failed"
    assert result.metadata["issues"]


@pytest.mark.leakage
def test_publication_instants_must_not_reverse_observation_order(prices):
    availability = pd.Series(
        pd.to_datetime(["2019-09-03T00:00Z", "2019-09-05T01:00Z", "2019-09-05T00:00Z"]),
        index=prices.index,
    )
    result = _yahoo(prices, available_at=availability)
    assert result.status == "failed"
    assert result.metadata["availability_status"] == "failed"
    pd.testing.assert_series_equal(result.metadata["available_at"], availability, check_names=False)


@pytest.mark.leakage
def test_collection_time_does_not_fabricate_historical_availability(prices):
    result = _yahoo(prices)
    assert result.metadata["collected_at"]
    assert result.metadata["available_at"] is None
    assert result.metadata["availability_status"] == "unverified"
    assert "available_at" not in result.data.columns


@pytest.mark.leakage
def test_missing_leading_observation_is_not_backfilled(prices):
    source = prices.copy(deep=True)
    source.loc[source.index[0], ["Open", "High", "Low", "Close"]] = np.nan
    result = _yahoo(source)
    assert result.status == "failed"
    assert result.data.iloc[0][["Open", "High", "Low", "Close"]].isna().all()
    assert len(result.data) == 3
    pd.testing.assert_frame_equal(result.raw_data, source)


@pytest.mark.leakage
def test_changing_future_prices_cannot_modify_past_collected_prices(prices):
    future_changed = prices.copy(deep=True)
    future_changed.iloc[-1, :4] = [200.0, 220.0, 190.0, 210.0]
    original = _yahoo(prices)
    changed = _yahoo(future_changed)
    pd.testing.assert_frame_equal(original.data.iloc[:2], changed.data.iloc[:2])
    assert list(original.data["Close"]) == [101.0, 102.0, 103.0]
    assert "log_return" not in original.data.columns


def test_real_config_keeps_candidates_out_of_selected_universe():
    pytest.importorskip("yaml")
    config = load_asset_config(Path(__file__).resolve().parents[1] / "config" / "assets.yaml")
    assert config["universe"]["target_etf_count"] == 12
    assert config["universe"]["status"] == "unconfirmed"
    assert config["universe"]["selected_etfs"] == []
    datasets = config["datasets"]
    assert all(isinstance(dataset, DatasetSpec) for dataset in datasets)
    etfs = [dataset for dataset in datasets if dataset.role == "etf"]
    assert len(etfs) == 16
    assert {dataset.selection_status for dataset in etfs} == {"candidate"}
    assert {dataset.symbol for dataset in datasets if dataset.role == "benchmark"} == {
        "^KS11",
        "^GSPC",
    }
    fx = next(dataset for dataset in datasets if dataset.role == "fx")
    assert fx.symbol == "KRW=X" and fx.unit == "KRW_per_USD"
    rf = next(dataset for dataset in datasets if dataset.role == "risk_free")
    assert rf.symbol == "DGS3MO" and rf.selection_status == "candidate"


@pytest.mark.parametrize(
    "problem", ["premature_etf_selection", "premature_rf_policy", "duplicate_id"]
)
def test_config_rejects_unconfirmed_decisions_and_ambiguous_dataset_ids(tmp_path, problem):
    yaml = pytest.importorskip("yaml")
    original = Path(__file__).resolve().parents[1] / "config" / "assets.yaml"
    document = yaml.safe_load(original.read_text(encoding="utf-8"))
    if problem == "premature_etf_selection":
        document["universe"]["selected_etfs"] = [document["datasets"][0]["dataset_id"]]
    elif problem == "premature_rf_policy":
        document["risk_free_policy"]["currency"] = "USD"
    else:
        document["datasets"][-1]["dataset_id"] = document["datasets"][0]["dataset_id"]
    invalid_config = tmp_path / "invalid_assets.yaml"
    invalid_config.write_text(yaml.safe_dump(document, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError):
        load_asset_config(invalid_config)


@pytest.mark.parametrize(
    "collector,spec,argument",
    [
        (collect_yahoo, _fred_spec(), "download"),
        (collect_fred, _spec(), "http_get"),
    ],
)
def test_wrong_source_adapter_is_rejected_before_fetch(collector, spec, argument):
    calls = []
    with pytest.raises(ValueError):
        collector(spec, end="2019-09-05", **{argument: lambda **kw: calls.append(kw)})
    assert calls == []
