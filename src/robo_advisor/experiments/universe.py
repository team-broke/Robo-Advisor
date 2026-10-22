"""실험 유니버스의 원천 명세·검증·환경 입력 연결. 정합과 수익률은 후행 단계다."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import pandas as pd
import yaml

from robo_advisor.data.collectors import (
    CollectionResult,
    DatasetSpec,
    _bounds,
    collect_fred,
    collect_yahoo,
    load_asset_config,
)
from robo_advisor.data.validation import (
    check_dividend_adjustment,
    validate_listing_date,
    validate_missing_intervals,
)

ALIGNMENT_POLICY = MappingProxyType(
    {
        "base_currency": "KRW",
        "fx_direction": "USD/KRW",
        "overseas_close_lag_sessions": 1,
        "require_available_at": True,
    }
)


@dataclass(frozen=True)
class ExperimentalUniverse:
    universe_id: str
    asset_config_path: Path
    investment_assets: tuple[DatasetSpec, ...]
    benchmarks: tuple[DatasetSpec, ...]
    fx: DatasetSpec
    risk_free_candidates: tuple[DatasetSpec, ...]
    warmup_start: str
    alignment_policy: Mapping

    @property
    def asset_ids(self):
        return tuple(asset.dataset_id for asset in self.investment_assets)

    @property
    def collection_datasets(self):
        return (*self.investment_assets, *self.benchmarks, self.fx, *self.risk_free_candidates)

    def dimensions(self, lookback):
        if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < 1:
            raise ValueError("lookback must be a positive integer")
        count = len(self.investment_assets)
        return {
            "n_assets": count,
            "action_dim": count + 1,
            "observation_dim": count * (lookback + 4) + 1,
        }


def _mapping(value, name):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _selected(ids, registry, role):
    if (
        not isinstance(ids, list)
        or not ids
        or any(not isinstance(item, str) or not item for item in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError(f"{role} IDs must be a nonempty unique list")
    if any(item not in registry or registry[item].role != role for item in ids):
        raise ValueError(f"Unknown ID or wrong dataset role in {role}")
    return tuple(registry[item] for item in ids)


def load_experimental_universe(path) -> ExperimentalUniverse:
    """별도 설정의 ID를 #9 후보 풀에 연결하며 선정 상태와 원본 설정은 보존한다."""
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        config = _mapping(yaml.safe_load(handle), "experiment")
    if config.get("schema_version") != 1 or config.get("status") != "experimental":
        raise ValueError("Expected schema 1 and experimental status")
    universe_id = config.get("universe_id")
    if not isinstance(universe_id, str) or not universe_id.strip():
        raise ValueError("universe_id must be a nonempty string")
    reference = config.get("asset_config")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("asset_config must reference the #9 asset registry")
    registry_path = (path.parent / reference).resolve()
    registry_config = load_asset_config(registry_path)
    registry = {spec.dataset_id: spec for spec in registry_config["datasets"]}
    investments = _selected(config.get("investment_dataset_ids"), registry, "etf")
    expected = config.get("expected_asset_count")
    if isinstance(expected, bool) or not isinstance(expected, int) or expected != len(investments):
        raise ValueError("Investment count must match expected_asset_count")
    excluded = config.get("excluded_dataset_ids", [])
    if not isinstance(excluded, list) or any(not isinstance(item, str) for item in excluded):
        raise ValueError("excluded_dataset_ids must be a list of IDs")
    if len(set(excluded)) != len(excluded) or any(item not in registry for item in excluded):
        raise ValueError("Excluded IDs must be known and unique")
    if set(excluded) & {spec.dataset_id for spec in investments}:
        raise ValueError("Excluded dataset cannot be an investment")
    supporting = _mapping(config.get("supporting_dataset_ids"), "supporting_dataset_ids")
    benchmarks = _selected(supporting.get("benchmarks"), registry, "benchmark")
    fx_id = supporting.get("fx")
    fx = _selected([fx_id], registry, "fx")[0]
    rates = _selected(supporting.get("risk_free_candidates"), registry, "risk_free")
    policy = _mapping(config.get("alignment_policy"), "alignment_policy")
    if (
        dict(policy) != dict(ALIGNMENT_POLICY)
        or type(policy["overseas_close_lag_sessions"]) is not int
        or policy["require_available_at"] is not True
    ):
        raise ValueError("Experimental config must preserve the #12 alignment policy")
    warmup = registry_config["collection"]["start_date"]
    first, _ = _bounds(warmup, "2260-01-01")
    if first.date().isoformat() != "2019-09-01":
        raise ValueError("v5 comparison requires the 2019-09-01 warmup start")
    return ExperimentalUniverse(
        universe_id,
        registry_path,
        investments,
        benchmarks,
        fx,
        rates,
        first.date().isoformat(),
        MappingProxyType(dict(policy)),
    )


def _inputs(mapping, universe, name, *, investment_only=False):
    if mapping is None:
        return {}
    mapping = _mapping(mapping, name)
    known = (
        set(universe.asset_ids)
        if investment_only
        else {spec.dataset_id for spec in universe.collection_datasets}
    )
    if set(mapping) - known:
        raise ValueError(f"{name} contains IDs outside the experiment")
    return mapping


def collect_universe(
    universe: ExperimentalUniverse,
    *,
    end,
    yahoo_download=None,
    fred_get=None,
    trading_calendars=None,
    available_at=None,
    fx_quote_metadata=None,
) -> dict[str, CollectionResult]:
    """투자자산 15개와 보조 자료를 구분해 반환하며 실제 기본 클라이언트를 호출하지 않는다.

    주입 함수의 오프라인 실행은 호출자가 보장한다. 실제 수집은 별도 승인이 필요하다.
    캘린더/공개시각은 dataset_id별 근거를 받고 관측 행에서 생성하지 않는다.
    """
    _bounds(universe.warmup_start, end)
    calendars = _inputs(trading_calendars, universe, "trading_calendars")
    availability = _inputs(available_at, universe, "available_at")
    results = {}
    for spec in universe.collection_datasets:
        common = {
            "start": universe.warmup_start,
            "end": end,
            "trading_calendar": calendars.get(spec.dataset_id),
            "available_at": availability.get(spec.dataset_id),
            "allow_network": False,
        }
        if spec.source == "yahoo":
            result = collect_yahoo(
                spec,
                download=yahoo_download,
                quote_metadata=fx_quote_metadata if spec.role == "fx" else None,
                **common,
            )
        else:
            result = collect_fred(spec, http_get=fred_get, **common)
        results[spec.dataset_id] = result
    return results


def _issue(code, message, *, severity="error"):
    return {"code": code, "message": message, "severity": severity}


def _status(issues):
    if any(issue["severity"] == "error" for issue in issues):
        return "failed"
    return "unverified" if any(issue["severity"] == "unverified" for issue in issues) else "passed"


def audit_universe(
    universe: ExperimentalUniverse,
    results,
    *,
    end,
    trading_calendars=None,
    listing_dates=None,
    dividend_evidence=None,
) -> dict:
    """2019년 워밍업 이후 원천 기간/누락/조정 근거를 검사하되 수집 결과를 수정하지 않는다.

    결과가 없으면 unverified이며 실제 공급자 수집 가능성을 인증하지 않는다.
    주어진 캘린더는 요청 전체 [start,end)의 실제 기대 세션이어야 한다.
    """
    first, last = _bounds(universe.warmup_start, end)
    results = _inputs(results, universe, "results")
    calendars = _inputs(trading_calendars, universe, "trading_calendars")
    listings = _inputs(listing_dates, universe, "listing_dates", investment_only=True)
    evidence = _inputs(dividend_evidence, universe, "dividend_evidence", investment_only=True)
    rows = {}
    for spec in universe.collection_datasets:
        result = results.get(spec.dataset_id)
        row = {
            "dataset_id": spec.dataset_id,
            "symbol": spec.symbol,
            "role": spec.role,
            "source_status": None,
            "period": {
                "requested_start": first.date().isoformat(),
                "requested_end_exclusive": last.date().isoformat(),
                "actual_start": None,
                "actual_end": None,
                "row_count": 0,
            },
            "checks": {},
            "issues": [],
        }
        issues = row["issues"]
        if result is None:
            issues.append(
                _issue(
                    "NO_COLLECTION_RESULT", "No source result is available", severity="unverified"
                )
            )
        else:
            if not isinstance(result, CollectionResult):
                raise ValueError("results must contain CollectionResult objects")
            row["source_status"] = result.status
            issues.extend(dict(issue) for issue in result.metadata["issues"])
            if any(
                result.metadata.get(key) != value
                for key, value in {
                    "dataset_id": spec.dataset_id,
                    "symbol": spec.symbol,
                    "source": spec.source,
                    "role": spec.role,
                    "market": spec.market,
                    "currency": spec.currency,
                    "timezone": spec.timezone,
                    "unit": spec.unit,
                    "requested_start": first.date().isoformat(),
                    "requested_end_exclusive": last.date().isoformat(),
                }.items()
            ):
                issues.append(
                    _issue("COLLECTION_CONTRACT_MISMATCH", "Source metadata differs from plan")
                )
            row["period"].update(
                {
                    key: result.metadata.get(key)
                    for key in ("actual_start", "actual_end", "row_count")
                }
            )
            row.update(
                price_basis=result.metadata.get("price_basis"),
                availability_status=result.metadata.get("availability_status"),
                price_status=result.validation.get("price", {}).get("status", "unverified"),
            )
            if not result.data.empty and not isinstance(result.data.columns, pd.MultiIndex):
                row["checks"]["coverage"] = validate_missing_intervals(
                    result.data,
                    columns=("Value",) if spec.role == "risk_free" else ("Close",),
                    trading_calendar=calendars.get(spec.dataset_id),
                    market_timezone=spec.timezone,
                )
                if spec.role == "etf" and result.data.columns.is_unique and "Close" in result.data:
                    row["checks"]["adjustment"] = check_dividend_adjustment(
                        result.data["Close"],
                        market_timezone=spec.timezone,
                        **_mapping(evidence.get(spec.dataset_id, {}), "dividend evidence"),
                    )
        if spec.role == "etf":
            row["checks"]["listing"] = validate_listing_date(
                first,
                listings.get(spec.dataset_id),
                require_full_period=True,
                market_timezone=spec.timezone,
            )
        for report in row["checks"].values():
            issues.extend(dict(issue) for issue in report["issues"])
        row["status"] = _status(issues)
        rows[spec.dataset_id] = row
    return {
        "universe_id": universe.universe_id,
        "status": _status([issue for row in rows.values() for issue in row["issues"]]),
        "investment_assets": [rows[spec.dataset_id] for spec in universe.investment_assets],
        "supporting_datasets": {
            "benchmarks": [rows[spec.dataset_id] for spec in universe.benchmarks],
            "fx": rows[universe.fx.dataset_id],
            "risk_free_candidates": [
                rows[spec.dataset_id] for spec in universe.risk_free_candidates
            ],
        },
    }


def make_portfolio_env(universe, simple_returns, rsi, macd, *, lookback, **env_options):
    """정합된 simple return 입력을 기존 환경에 연결하며 가격/로그수익률을 변환하지 않는다."""
    from robo_advisor.rl.environment import PortfolioEnv

    universe.dimensions(lookback)
    if "n_assets" in env_options:
        raise ValueError("n_assets is derived from the ordered experiment investment IDs")
    for name, frame in (("simple_returns", simple_returns), ("rsi", rsi), ("macd", macd)):
        if not isinstance(frame, pd.DataFrame) or tuple(frame.columns) != universe.asset_ids:
            raise ValueError(f"{name} columns must match ordered investment dataset IDs exactly")
    return PortfolioEnv(
        simple_returns,
        rsi,
        macd,
        lookback=lookback,
        n_assets=len(universe.investment_assets),
        **env_options,
    )
