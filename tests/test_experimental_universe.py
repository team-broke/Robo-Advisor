"""15개 실험 자산과 보조 데이터의 경계를 mock으로 확인한다."""

from pathlib import Path
import socket

import numpy as np
import pandas as pd
import pytest

from robo_advisor.data.collectors import load_asset_config
from robo_advisor.experiments.universe import (
    audit_universe,
    collect_universe,
    load_experimental_universe,
    make_portfolio_env,
)

ASSET_IDS = (
    "SPY",
    "IWM",
    "EFA",
    "EEM",
    "IEF",
    "TLT",
    "TIP",
    "LQD",
    "GLD",
    "VNQ",
    "QQQ",
    "AGG",
    "DBC",
    "KODEX_200TR",
    "KTB10Y",
)
END = "2019-09-05"


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Experimental universe tests must not access the network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.fixture
def config_path():
    return Path(__file__).resolve().parents[1] / "config" / "experiment-v5-15.yaml"


@pytest.fixture
def universe(config_path):
    return load_experimental_universe(config_path)


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
    frame.attrs["source_note"] = "synthetic; not an official trading calendar"
    return frame


class FakeResponse:
    text = "DATE,DGS3MO\n2019-09-02,2.0\n2019-09-03,0.0\n2019-09-04,-0.1\n"

    def raise_for_status(self):
        pass


def _calendars(universe, prices):
    return {spec.dataset_id: prices.index for spec in universe.collection_datasets}


def _availability(universe, prices):
    values = pd.to_datetime(["2019-09-02T22:00Z", "2019-09-03T22:00Z", "2019-09-04T22:00Z"])
    return {
        spec.dataset_id: pd.Series(values, index=prices.index)
        for spec in universe.collection_datasets
    }


def _collect(universe, prices, *, overrides=None, with_availability=False):
    by_symbol = {
        spec.symbol: prices.copy(deep=True)
        for spec in universe.collection_datasets
        if spec.source == "yahoo"
    }
    if overrides:
        by_symbol.update(overrides)
    kwargs = {}
    if with_availability:
        kwargs["available_at"] = _availability(universe, prices)
    return collect_universe(
        universe,
        end=END,
        yahoo_download=lambda **request: by_symbol[request["tickers"]],
        fred_get=lambda *args, **request: FakeResponse(),
        trading_calendars=_calendars(universe, prices),
        fx_quote_metadata={"symbol": "KRW=X", "base_currency": "USD", "quote_currency": "KRW"},
        **kwargs,
    )


def _row(report, dataset_id):
    rows = report["investment_assets"] + report["supporting_datasets"]["benchmarks"]
    rows += [report["supporting_datasets"]["fx"]]
    rows += report["supporting_datasets"]["risk_free_candidates"]
    return next(row for row in rows if row["dataset_id"] == dataset_id)


def _frames(universe, lookback=2):
    index = pd.date_range("2020-01-02", periods=lookback + 3, tz="UTC", name="timestamp")
    returns = pd.DataFrame(
        np.tile(np.arange(15) * 0.0001, (len(index), 1)), index=index, columns=universe.asset_ids
    )
    rsi = pd.DataFrame(50.0, index=index, columns=universe.asset_ids)
    macd = pd.DataFrame(0.0, index=index, columns=universe.asset_ids)
    return returns, rsi, macd


def test_experimental_universe_keeps_exact_order_and_supporting_roles(universe):
    assert universe.asset_ids == ASSET_IDS
    assert tuple(spec.dataset_id for spec in universe.investment_assets) == ASSET_IDS
    assert all(spec.role == "etf" for spec in universe.investment_assets)
    assert len(universe.investment_assets) == 15
    assert len(universe.collection_datasets) == 19
    assert tuple(spec.dataset_id for spec in universe.benchmarks) == ("KOSPI", "SP500")
    assert universe.fx.dataset_id == "USD_KRW" and universe.fx.symbol == "KRW=X"
    assert tuple(spec.dataset_id for spec in universe.risk_free_candidates) == ("DGS3MO",)
    assert "KODEX_200" not in universe.asset_ids
    assert not {"KOSPI", "SP500", "USD_KRW", "DGS3MO", "cash"}.intersection(universe.asset_ids)
    assert universe.warmup_start == "2019-09-01"
    assert universe.alignment_policy["base_currency"] == "KRW"
    assert universe.alignment_policy["fx_direction"] == "USD/KRW"
    assert universe.alignment_policy["overseas_close_lag_sessions"] == 1
    assert universe.alignment_policy["require_available_at"] is True


def test_experiment_does_not_promote_team_candidate_or_rf_policy(universe, config_path):
    config = load_asset_config(config_path.parent / "assets.yaml")
    assert config["universe"]["target_etf_count"] == 12
    assert config["universe"]["status"] == "unconfirmed"
    assert config["universe"]["selected_etfs"] == []
    assert config["risk_free_policy"]["status"] == "unconfirmed"
    assert {spec.selection_status for spec in universe.investment_assets} == {"candidate"}
    assert universe.risk_free_candidates[0].selection_status == "candidate"


@pytest.mark.parametrize("lookback,observation_dim", [(2, 91), (20, 361)])
def test_dimension_formula_includes_only_fifteen_assets_and_cash(
    universe, lookback, observation_dim
):
    assert universe.dimensions(lookback) == {
        "n_assets": 15,
        "action_dim": 16,
        "observation_dim": observation_dim,
    }


@pytest.mark.parametrize("lookback", [0, -1, True, 2.5])
def test_dimension_formula_rejects_invalid_lookback(universe, lookback):
    with pytest.raises(ValueError):
        universe.dimensions(lookback)


@pytest.mark.parametrize(
    "problem",
    [
        "confirmed_status",
        "duplicate_asset",
        "unknown_asset",
        "supporting_as_investment",
        "cash_as_investment",
        "short_universe",
        "wrong_fx",
    ],
)
def test_invalid_experiment_contract_is_rejected(config_path, tmp_path, problem):
    import yaml

    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    document["asset_config"] = "assets.yaml"
    (tmp_path / "assets.yaml").write_text(
        (config_path.parent / "assets.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    if problem == "confirmed_status":
        document["status"] = "confirmed"
    elif problem == "duplicate_asset":
        document["investment_dataset_ids"][-1] = document["investment_dataset_ids"][0]
    elif problem == "unknown_asset":
        document["investment_dataset_ids"][-1] = "UNKNOWN"
    elif problem == "supporting_as_investment":
        document["investment_dataset_ids"][-1] = "KOSPI"
    elif problem == "cash_as_investment":
        document["investment_dataset_ids"][-1] = "cash"
    elif problem == "short_universe":
        document["investment_dataset_ids"].pop()
    else:
        document["supporting_dataset_ids"]["fx"] = "SP500"
    invalid = tmp_path / "invalid-experiment.yaml"
    invalid.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError):
        load_experimental_universe(invalid)


def test_collect_dispatches_eighteen_yahoo_and_one_fred_requests(universe, prices):
    yahoo_calls, fred_calls = [], []

    def download(**request):
        yahoo_calls.append(request)
        return prices.copy(deep=True)

    def http_get(url, **request):
        fred_calls.append((url, request))
        return FakeResponse()

    results = collect_universe(
        universe,
        end=END,
        yahoo_download=download,
        fred_get=http_get,
        trading_calendars=_calendars(universe, prices),
        available_at=_availability(universe, prices),
        fx_quote_metadata={"symbol": "KRW=X", "base_currency": "USD", "quote_currency": "KRW"},
    )
    assert tuple(results) == tuple(spec.dataset_id for spec in universe.collection_datasets)
    assert len(yahoo_calls) == 18 and len(fred_calls) == 1
    assert {call["tickers"] for call in yahoo_calls} == {
        spec.symbol for spec in universe.collection_datasets if spec.source == "yahoo"
    }
    assert all(call["start"] == "2019-09-01" and call["end"] == END for call in yahoo_calls)
    assert all(call["auto_adjust"] is True for call in yahoo_calls)
    assert fred_calls[0][1]["params"] == {
        "id": "DGS3MO",
        "cosd": "2019-09-01",
        "coed": "2019-09-04",
    }
    assert results["USD_KRW"].metadata["fx_direction"] == "USD/KRW"
    assert results["USD_KRW"].metadata["fx_direction_status"] == "verified"
    assert all(result.metadata["availability_status"] == "verified" for result in results.values())
    assert list(results["DGS3MO"].data["Value"]) == [2.0, 0.0, -0.1]
    assert results["SPY"].metadata["selection_status"] == "candidate"


def test_collect_without_injected_clients_never_opts_into_network(universe):
    results = collect_universe(universe, end=END)
    assert len(results) == 19
    for result in results.values():
        assert result.status == "failed"
        assert any(issue["code"] == "NETWORK_NOT_AUTHORIZED" for issue in result.metadata["issues"])


def test_audit_without_collection_is_unverified_for_every_requested_dataset(universe):
    report = audit_universe(universe, {}, end=END)
    assert report["status"] == "unverified"
    assert len(report["investment_assets"]) == 15
    assert len(report["supporting_datasets"]["benchmarks"]) == 2
    assert len(report["supporting_datasets"]["risk_free_candidates"]) == 1
    assert tuple(row["dataset_id"] for row in report["investment_assets"]) == ASSET_IDS
    for spec in universe.collection_datasets:
        row = _row(report, spec.dataset_id)
        assert row["status"] == "unverified"
        assert any(issue["code"] == "NO_COLLECTION_RESULT" for issue in row["issues"])


def test_complete_mock_observations_do_not_imply_verified_etf_provenance(universe, prices):
    results = _collect(universe, prices, with_availability=True)
    report = audit_universe(
        universe, results, end=END, trading_calendars=_calendars(universe, prices)
    )
    row = _row(report, "SPY")
    assert report["status"] == "unverified"
    assert row["source_status"] == "unverified"
    assert row["checks"]["coverage"]["status"] == "passed"
    assert row["checks"]["listing"]["status"] == "unverified"
    assert row["checks"]["adjustment"]["status"] == "unverified"
    assert row["period"]["actual_start"] == "2019-09-02"
    assert row["period"]["actual_end"] == "2019-09-04"
    assert row["period"]["row_count"] == 3


def test_audit_does_not_infer_a_trading_calendar_from_observations(universe, prices):
    results = _collect(universe, prices)
    report = audit_universe(universe, results, end=END)
    row = _row(report, "SPY")
    assert row["checks"]["coverage"]["status"] == "unverified"
    assert row["status"] == "unverified"


@pytest.mark.parametrize("kind", ["late_start", "missing_middle", "duplicate", "reversed"])
def test_audit_preserves_partial_or_invalid_source_without_filling(universe, prices, kind):
    if kind == "late_start":
        source = prices.iloc[1:].copy(deep=True)
    elif kind == "missing_middle":
        source = prices.iloc[[0, 2]].copy(deep=True)
    else:
        source = prices.copy(deep=True)
        indices = [0, 0, 2] if kind == "duplicate" else [0, 2, 1]
        source.index = prices.index[indices]
    results = _collect(universe, prices, overrides={"SPY": source})
    report = audit_universe(
        universe, results, end=END, trading_calendars=_calendars(universe, prices)
    )
    row = _row(report, "SPY")
    assert report["status"] == "failed"
    assert row["status"] == "failed"
    assert row["checks"]["coverage"]["status"] == "failed"
    assert row["period"]["row_count"] == len(source)
    pd.testing.assert_frame_equal(results["SPY"].raw_data, source)
    assert len(results["SPY"].data) == len(source)


def test_actual_listing_date_is_recorded_as_evidence_without_invented_history(universe, prices):
    source = prices.iloc[-1:].copy(deep=True)
    results = _collect(universe, prices, overrides={"SPY": source})
    report = audit_universe(
        universe,
        results,
        end=END,
        trading_calendars=_calendars(universe, prices),
        listing_dates={"SPY": "2019-09-04"},
    )
    row = _row(report, "SPY")
    assert any(issue["code"] == "LATE_LISTING" for issue in row["checks"]["listing"]["issues"])
    assert row["status"] != "passed"
    assert row["period"]["actual_start"] == "2019-09-04"
    assert row["period"]["row_count"] == 1
    pd.testing.assert_frame_equal(results["SPY"].raw_data, source)


@pytest.mark.leakage
def test_experiment_collection_leaves_unknown_historical_availability_unknown(universe, prices):
    results = _collect(universe, prices)
    assert all(result.metadata["available_at"] is None for result in results.values())
    assert all(
        result.metadata["availability_status"] == "unverified" for result in results.values()
    )
    pd.testing.assert_frame_equal(results["SPY"].data, prices)
    assert "log_return" not in results["SPY"].data.columns


@pytest.mark.parametrize("lookback,observation_dim", [(2, 91), (20, 361)])
def test_portfolio_env_derives_fifteen_assets_and_runs_reset_step(
    universe, lookback, observation_dim
):
    frames = _frames(universe, lookback)
    env = make_portfolio_env(
        universe, *frames, lookback=lookback, commission_rate=0.0, slippage_rate=0.0
    )
    assert env.n_assets == 15
    assert env.action_space.shape == (16,)
    assert env.observation_space.shape == (observation_dim,)
    observation, info = env.reset(seed=7)
    assert observation.shape == (observation_dim,)
    assert env.observation_space.contains(observation)
    assert info["weights"].shape == (16,)
    assert info["weights"][-1] == 1.0
    next_observation, reward, terminated, truncated, next_info = env.step(np.zeros(16))
    assert next_observation.shape == (observation_dim,)
    assert np.isfinite(reward)
    assert not terminated and not truncated
    assert np.isclose(next_info["weights"].sum(), 1.0)
    assert next_info["timestamp"] > info["timestamp"]


@pytest.mark.parametrize("problem", ["permutation", "missing", "cash", "benchmark", "raw_symbol"])
def test_portfolio_env_rejects_noncanonical_investment_columns(universe, problem):
    frames = list(_frames(universe))
    for i, frame in enumerate(frames):
        if problem == "permutation":
            frames[i] = frame.iloc[:, ::-1]
        elif problem == "missing":
            frames[i] = frame.iloc[:, :-1]
        elif problem in {"cash", "benchmark"}:
            frames[i] = frame.assign(**{problem: 0.0})
        else:
            frames[i] = frame.rename(columns={"KODEX_200TR": "278530.KS"})
    with pytest.raises(ValueError):
        make_portfolio_env(universe, *frames, lookback=2)


@pytest.mark.parametrize("problem", ["naive", "duplicate", "reversed", "feature_index_mismatch"])
def test_portfolio_env_retains_existing_time_index_validation(universe, problem):
    frames = list(_frames(universe))
    if problem == "feature_index_mismatch":
        frames[1] = frames[1].copy(deep=True)
        frames[1].index += pd.Timedelta(hours=1)
    else:
        for frame in frames:
            if problem == "naive":
                frame.index = frame.index.tz_localize(None)
            elif problem == "duplicate":
                frame.index = frame.index[[0, 0, 2, 3, 4]]
            else:
                frame.index = frame.index[::-1]
    with pytest.raises(ValueError):
        make_portfolio_env(universe, *frames, lookback=2)


@pytest.mark.parametrize("problem", ["invalid_simple_return", "invalid_rsi", "nonfinite_macd"])
def test_portfolio_env_retains_existing_numeric_validation(universe, problem):
    frames = list(_frames(universe))
    if problem == "invalid_simple_return":
        frames[0].iloc[2, 0] = -1.01
    elif problem == "invalid_rsi":
        frames[1].iloc[2, 0] = 101.0
    else:
        frames[2].iloc[2, 0] = np.inf
    with pytest.raises(ValueError):
        make_portfolio_env(universe, *frames, lookback=2)


@pytest.mark.parametrize("problem", ["currency", "fx_direction", "lag", "availability", "excluded"])
def test_experiment_rejects_policy_changes_and_excluded_investments(config_path, tmp_path, problem):
    import yaml

    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    (tmp_path / "assets.yaml").write_text(
        (config_path.parent / "assets.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    if problem == "excluded":
        document["investment_dataset_ids"][-1] = "KODEX_200"
    else:
        key, value = {
            "currency": ("base_currency", "USD"),
            "fx_direction": ("fx_direction", "KRW/USD"),
            "lag": ("overseas_close_lag_sessions", 0),
            "availability": ("require_available_at", False),
        }[problem]
        document["alignment_policy"][key] = value
    changed = tmp_path / "experiment.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError):
        load_experimental_universe(changed)


@pytest.mark.parametrize(
    "key,value",
    [
        ("currency", "KRW"),
        ("timezone", "Asia/Seoul"),
        ("unit", "index_points"),
        ("requested_end_exclusive", "2019-09-06"),
    ],
)
def test_audit_detects_wrong_source_contract(universe, prices, key, value):
    results = _collect(universe, prices)
    results["SPY"].metadata[key] = value
    report = audit_universe(universe, results, end=END)
    row = _row(report, "SPY")
    assert row["status"] == "failed"
    assert any(issue["code"] == "COLLECTION_CONTRACT_MISMATCH" for issue in row["issues"])
    pd.testing.assert_frame_equal(results["SPY"].raw_data, prices)


def test_audit_preserves_invalid_multiindex_response_as_failure(universe, prices):
    source = prices.copy(deep=True)
    source.columns = pd.MultiIndex.from_tuples([(column, "OTHER") for column in source.columns])
    results = _collect(universe, prices, overrides={"SPY": source})
    report = audit_universe(universe, results, end=END)
    assert _row(report, "SPY")["status"] == "failed"
    pd.testing.assert_frame_equal(results["SPY"].raw_data, source)


def test_audit_accepts_independent_adjustment_evidence_without_promoting_source(universe, prices):
    source = prices.copy(deep=True)
    source["Close"] = [100.0, 100.0, 100.0 * 99.0 / 98.0]
    results = _collect(universe, prices, overrides={"SPY": source})
    evidence = {
        "raw_close": pd.Series([100.0, 98.0, 99.0], index=prices.index),
        "reinvested_close": source["Close"].copy(),
        "dividends": pd.Series([0.0, 2.0, 0.0], index=prices.index),
        "trading_calendar": prices.index,
        "comparison_basis": "gross_total_return",
        "split_free": True,
        "sources": {
            name: f"fixture:{name}"
            for name in ("prices", "raw_close", "reinvested_close", "dividends")
        },
    }
    report = audit_universe(universe, results, end=END, dividend_evidence={"SPY": evidence})
    row = _row(report, "SPY")
    assert row["checks"]["adjustment"]["status"] == "passed"
    assert row["source_status"] == row["status"] == "unverified"
    pd.testing.assert_frame_equal(results["SPY"].raw_data, source)


def test_portfolio_asset_count_cannot_be_overridden_by_team_twelve_target(universe):
    with pytest.raises(ValueError, match="n_assets"):
        make_portfolio_env(universe, *_frames(universe), lookback=2, n_assets=12)
