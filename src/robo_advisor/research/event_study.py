"""Point-in-time DART event candidates and offline event-study outcomes."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal
from uuid import NAMESPACE_URL, uuid5

import numpy as np
import pandas as pd
from pydantic import Field, HttpUrl, model_validator
from scipy.stats import t

from robo_advisor.backtest.metrics import calculate_metrics
from robo_advisor.research.contracts import Citation, ResearchModel
from robo_advisor.research.dart_body import BodyChunk, iter_body_blocks, sha256
from robo_advisor.research.ingestion import KST
from robo_advisor.research.trace import redact
from robo_advisor.schemas.common import NonEmptyString, UTCDateTime
from robo_advisor.schemas.risk import RiskEventType

HORIZONS = (1, 3, 5, 10)
_PATTERNS = {
    "earnings_shock": r"실적\s*쇼크|어닝\s*쇼크|earnings\s+(?:shock|miss)|profit\s+warning",
    "earnings_deterioration": (
        r"실적\s*악화|적자\s*전환|" r"(?:매출(?:액)?|영업이익)(?:이|가|은|는)?\s*(?:감소|급감|하락)"
    ),
    "contract_termination": (
        r"계약(?:이|을|의)?\s*(?:해지|해제)|(?:해지|해제)\s*된\s*(?:공급|판매)?\s*계약"
    ),
    "management_change": (r"(?:대표이사|최대주주|경영권)(?:가|이|의)?\s*(?:변경|교체|사임|양도)"),
    "regulation_change": r"규제\s*(?:변경|강화|완화|신설)|제재\s*(?:부과|결정)|법령\s*개정",
    "sharp_move": r"(?:주가|주식)(?:가|이)?\s*(?:급등|급락|폭락|급변)",
    "geopolitical": r"전쟁\s*(?:발발|개시)|군사\s*(?:공격|충돌)|지정학적\s*충돌",
}
_SPECULATIVE = re.compile(
    r"우려|가능성|위험이\s*있|될\s*수\s*있|예상|전망|예정|가정|"
    r"변경\s*없|해지\s*없|해당\s*없|발생하지\s*않"
)
_ISSUER_CODE = re.compile(r"[0-9]{6}")
_EXPLICIT_DATE = re.compile(
    r"(?<![0-9])(20[0-9]{2})\s*(?:년\s*|[-./])([0-9]{1,2})\s*(?:월\s*|[-./])([0-9]{1,2})(?![0-9])"
)
_RELATED_ENTITY_SECTION = re.compile(r"주주에\s*관한|계열회사|종속기업|관계기업")


def _categories(text):
    if _SPECULATIVE.search(text):
        return ()
    return tuple(kind for kind, pattern in _PATTERNS.items() if re.search(pattern, text, re.I))


def iter_event_excerpts(root):
    """Select verbatim candidate passages without asking a model to assert an event."""
    for section, locator, text in iter_body_blocks(root):
        if _categories(text) and redact(text) == text and redact(section) == section:
            yield section, locator, text


class EventCandidate(ResearchModel):
    event_id: NonEmptyString
    event_type: NonEmptyString
    risk_tag_type: RiskEventType | None
    asset: str | None
    asset_basis: Literal["dart_issuer_stock_code", "unverified"]
    receipt: str = Field(pattern=r"^[0-9]{14}$")
    receipt_date: date
    published_at: UTCDateTime
    published_precision: Literal["date"] = "date"
    available_at: UTCDateTime
    source_url: HttpUrl
    trace_id: NonEmptyString
    evidence: tuple[Citation, ...] = Field(min_length=1)
    review_status: Literal["candidate_only"] = "candidate_only"

    @model_validator(mode="after")
    def validate_provenance(self):
        if self.asset is None:
            valid_asset = self.asset_basis == "unverified"
        else:
            valid_asset = self.asset_basis == "dart_issuer_stock_code" and _ISSUER_CODE.fullmatch(
                self.asset
            )
        if not valid_asset:
            raise ValueError("event asset must be the issuer's DART stock code")
        if any(
            item.body is None
            or item.body.receipt != self.receipt
            or item.body.receipt_date != self.receipt_date
            or item.published_at != self.published_at
            or item.available_at != self.available_at
            or item.url != self.source_url
            for item in self.evidence
        ):
            raise ValueError("event citations contradict filing provenance")
        return self


def _screening_reason(citation, kind, issuer):
    body = citation.body
    if kind in {"contract_termination", "management_change"} and _RELATED_ENTITY_SECTION.search(
        body.section
    ):
        affected = re.compile(
            rf"{re.escape(issuer)}(?:주식회사|㈜)?(?:\s*의)?\s+(?:대표이사|최대주주|경영권|계약)"
        )
        if not issuer or not affected.search(body.quote):
            return "related_entity_not_linked_to_issuer"
    dates = []
    for parts in _EXPLICIT_DATE.findall(body.quote):
        try:
            dates.append(date(*map(int, parts)))
        except ValueError:
            continue
    if dates:
        if min(dates) > body.receipt_date:
            return "future_underlying_event"
        if (body.receipt_date - max(dates)).days > 30:
            return "historical_reference_needs_first_disclosure"
    return None


def collect_candidates(
    chunks: tuple[BodyChunk, ...], *, stock_code, issuer_name, corpus_sha256: str
) -> tuple[tuple[EventCandidate, ...], dict[str, int]]:
    """Group wording from the same filing by category without multiplying sample count."""
    asset = (
        stock_code if isinstance(stock_code, str) and _ISSUER_CODE.fullmatch(stock_code) else None
    )
    grouped = defaultdict(dict)
    screened = Counter()
    for item in chunks:
        citation = item.citation
        for kind in _categories(citation.body.quote):
            if reason := _screening_reason(citation, kind, issuer_name):
                screened[reason] += 1
                continue
            key = citation.body.receipt, kind
            grouped[key].setdefault(citation.body.quote_sha256, citation)

    candidates = []
    for (receipt, kind), citations in sorted(grouped.items()):
        evidence = tuple(citations.values())
        first = evidence[0]
        key = f"{receipt}:{asset}:{kind}"
        risk_tag_type = next((tag for tag in RiskEventType if tag.value == kind), None)
        candidates.append(
            EventCandidate(
                event_id=sha256(key.encode()),
                event_type=kind,
                risk_tag_type=risk_tag_type,
                asset=asset,
                asset_basis="dart_issuer_stock_code" if asset else "unverified",
                receipt=receipt,
                receipt_date=first.body.receipt_date,
                published_at=first.published_at,
                available_at=first.available_at,
                source_url=first.url,
                trace_id=str(uuid5(NAMESPACE_URL, f"dart-event-study:{corpus_sha256}:{key}")),
                evidence=evidence,
            )
        )
    return tuple(candidates), dict(screened)


class StudySplit(ResearchModel):
    name: Literal["train", "validation", "test"]
    start: date
    end_exclusive: date

    @model_validator(mode="after")
    def increasing(self):
        if self.start >= self.end_exclusive:
            raise ValueError("event-study split must be non-empty")
        return self


class StudyPolicy(ResearchModel):
    splits: tuple[StudySplit, ...] = ()
    estimation_sessions: int = Field(default=120, ge=3, strict=True)
    pre_event_gap_sessions: int = Field(default=10, ge=1, strict=True)
    min_ci_events: int = Field(default=10, ge=3, strict=True)

    @model_validator(mode="after")
    def split_order(self):
        names = tuple(item.name for item in self.splits)
        if self.splits and names != ("train", "validation", "test"):
            raise ValueError("provide train, validation and test splits in order")
        if any(
            left.end_exclusive > right.start for left, right in zip(self.splits, self.splits[1:])
        ):
            raise ValueError("event-study splits overlap")
        return self


@dataclass(frozen=True)
class StudyMarket:
    """Validated KRX calendar and daily prices indexed by local session date."""

    sessions: pd.DataFrame
    prices: dict[str, pd.DataFrame]
    benchmark: pd.DataFrame
    synthetic: bool


def _price_window(market, asset, start, end, cutoff, *, strict=False):
    index = market.sessions.index[start:end]
    paired = []
    for price in (market.prices[asset], market.benchmark):
        aligned = price.reindex(index)
        if aligned.isna().to_numpy().any():
            return None, "missing_price_observation"
        latest = aligned["available_at"].max()
        if latest > cutoff or (strict and latest == cutoff):
            return None, "price_not_available_at_cutoff"
        paired.append(aligned["close"])
    return pd.concat(paired, axis=1, keys=("asset", "market")), None


def _market_model(market, asset, first, available, policy):
    closes = pd.DatetimeIndex(market.sessions["close_at"])
    stop = min(first - policy.pre_event_gap_sessions, closes.searchsorted(available, side="left"))
    start = stop - policy.estimation_sessions - 1
    if start < 0:
        return {}, "insufficient_estimation_history"
    data, problem = _price_window(market, asset, start, stop, available, strict=True)
    if problem:
        return {}, f"estimation_{problem}"
    returns = data.pct_change(fill_method=None).iloc[1:]
    reference = returns["market"].to_numpy()
    if np.ptp(reference) <= 1e-12:
        return {}, "constant_estimation_market_return"
    x = np.column_stack((np.ones(len(returns)), reference))
    alpha, beta = np.linalg.lstsq(x, returns["asset"].to_numpy(), rcond=None)[0]
    return {
        "alpha_daily": float(alpha),
        "beta": float(beta),
        "estimation_start": returns.index[0].date(),
        "estimation_end": returns.index[-1].date(),
        "estimation_n": len(returns),
    }, "completed"


def _outcomes(data, model):
    returns = data.pct_change(fill_method=None).iloc[1:]
    asset, market = returns["asset"], returns["market"]
    active = asset - market
    if len(asset) > 1:
        metrics = calculate_metrics(asset, periods_per_year=1)
        volatility, drawdown = metrics["volatility"], metrics["mdd"]
    else:
        volatility, drawdown = None, max(0.0, -float(asset.iloc[0]))
    compound = data.iloc[-1] / data.iloc[0] - 1.0
    results = {
        "asset_compound_return": float(compound["asset"]),
        "market_compound_return": float(compound["market"]),
        "market_adjusted_horizon_return": float(compound["asset"] - compound["market"]),
        "market_adjusted_last_day_ar": float(active.iloc[-1]),
        "market_adjusted_car": float(active.sum()),
        "daily_realized_volatility": volatility,
        "max_drawdown": drawdown,
    }
    if model:
        residuals = asset - (model["alpha_daily"] + model["beta"] * market)
        results.update(
            market_model_last_day_ar=float(residuals.iloc[-1]),
            market_model_car=float(residuals.sum()),
        )
    if any(value is not None and not np.isfinite(value) for value in results.values()):
        raise ValueError("event-study arithmetic produced nonfinite values")
    return results


def _excluded(rows, candidate, reason, split=None, *, duplicate_of=None):
    rows.extend(
        {
            **_identity(candidate, split, horizon),
            "status": reason,
            "duplicate_of": duplicate_of,
        }
        for horizon in HORIZONS
    )


def _identity(candidate, split, horizon):
    return {
        "event_id": candidate.event_id,
        "receipt": candidate.receipt,
        "event_type": candidate.event_type,
        "risk_tag_type": candidate.risk_tag_type.value if candidate.risk_tag_type else None,
        "asset": candidate.asset,
        "split": split.name if split else None,
        "horizon_sessions": horizon,
    }


def study_events(
    candidates: tuple[EventCandidate, ...],
    market: StudyMarket | None,
    policy: StudyPolicy,
    *,
    as_of: datetime,
) -> list[dict]:
    """Calculate outcomes only from complete, explicitly bounded, observable windows."""
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("study cutoff must include a timezone")
    rows, previous = [], {}
    seen_ids = set()
    ordered = sorted(candidates, key=lambda item: (item.available_at, item.receipt, item.event_id))
    for item in ordered:
        event_date = item.available_at.astimezone(KST).date()
        split = next((s for s in policy.splits if s.start <= event_date < s.end_exclusive), None)
        if item.available_at > as_of:
            _excluded(rows, item, "future_disclosure", split)
            continue
        if item.asset is None:
            _excluded(rows, item, "asset_mapping_unverified", split)
            continue
        quote_hashes = tuple(sorted({citation.body.quote_sha256 for citation in item.evidence}))
        signature = (item.asset, item.event_type, quote_hashes)
        duplicate = item.event_id if item.event_id in seen_ids else previous.get(signature)
        if duplicate:
            _excluded(rows, item, "duplicate_disclosure", split, duplicate_of=duplicate)
            continue
        seen_ids.add(item.event_id)
        previous[signature] = item.event_id
        if market is None:
            _excluded(rows, item, "market_data_unavailable", split)
            continue
        if split is None:
            _excluded(rows, item, "event_outside_study_splits")
            continue
        if item.asset not in market.prices:
            _excluded(rows, item, "issuer_price_source_unavailable", split)
            continue
        sessions = market.sessions
        first = pd.DatetimeIndex(sessions["open_at"]).searchsorted(item.available_at, side="left")
        if not 0 < first < len(sessions):
            _excluded(rows, item, "calendar_or_baseline_unavailable", split)
            continue
        baseline = sessions.index[first - 1].date()
        day_one = sessions.index[first].date()
        if not split.start <= baseline < split.end_exclusive:
            _excluded(rows, item, "baseline_outside_split", split)
            continue
        _, baseline_problem = _price_window(
            market, item.asset, first - 1, first, sessions["open_at"].iloc[first]
        )
        if baseline_problem:
            _excluded(rows, item, f"baseline_{baseline_problem}", split)
            continue
        model, model_status = _market_model(market, item.asset, first, item.available_at, policy)
        for horizon in HORIZONS:
            row = {
                **_identity(item, split, horizon),
                "baseline_date": baseline,
                "first_session": day_one,
            }
            rows.append(row)
            stop = first + horizon
            if stop > len(sessions):
                row["status"] = "followup_calendar_unavailable"
                continue
            last = sessions.index[stop - 1].date()
            row["last_session"] = last
            if last >= split.end_exclusive:
                row["status"] = "cross_split_horizon"
                continue
            if sessions["close_at"].iloc[stop - 1] > as_of:
                row["status"] = "followup_not_observed"
                continue
            data, problem = _price_window(market, item.asset, first - 1, stop, as_of)
            if problem:
                row["status"] = problem
                continue
            row.update(status="completed", market_model_status=model_status)
            row.update(model)
            row.update(_outcomes(data, model))
    market_kind = None if market is None else ("synthetic" if market.synthetic else "real_krx")
    for row in rows:
        row["market_data_kind"] = market_kind
    _mark_overlaps(rows)
    return rows


def _mark_overlaps(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["status"] == "completed":
            groups[row["split"], row["horizon_sessions"]].append(row)
    for group in groups.values():
        group.sort(key=lambda row: (row["baseline_date"], row["last_session"], row["event_id"]))
        cluster, last = [], None
        for row in group:
            if cluster and row["baseline_date"] > last:
                _save_cluster(cluster)
                cluster, last = [], None
            cluster.append(row)
            last = max(last, row["last_session"]) if last else row["last_session"]
        _save_cluster(cluster)


def _save_cluster(cluster):
    if cluster:
        key = sha256("\n".join(row["event_id"] for row in cluster).encode())
        for row in cluster:
            row.update(overlap_cluster=key, overlap_cluster_size=len(cluster))


def summarize_events(rows, minimum):
    """Compute descriptive distributions and t intervals for isolated event windows only."""
    groups = defaultdict(list)
    for row in rows:
        key = row["event_type"], row["split"], row["horizon_sessions"]
        groups[key].append(row)
    summaries = []
    for (kind, split, horizon), group in groups.items():
        for approach, column in (
            ("market_adjusted", "market_adjusted_car"),
            ("market_model", "market_model_car"),
        ):
            complete = [row for row in group if row.get(column) is not None]
            values = np.array([row[column] for row in complete])
            isolated = np.array(
                [row[column] for row in complete if row.get("overlap_cluster_size") == 1]
            )
            n = len(isolated)
            summary = {
                "event_type": kind,
                "split": split,
                "horizon_sessions": horizon,
                "method": approach,
                "candidate_count": len(group),
                "completed_count": len(values),
                "isolated_count": n,
                "status": "descriptive_ci" if n >= minimum else "insufficient_evidence",
            }
            if len(values):
                q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
                summary.update(mean_car=float(values.mean()), q25=q25, median=median, q75=q75)
            if n >= minimum:
                mean = float(isolated.mean())
                margin = float(t.ppf(0.975, n - 1) * isolated.std(ddof=1) / np.sqrt(n))
                summary.update(
                    isolated_mean_car=mean, ci95_lower=mean - margin, ci95_upper=mean + margin
                )
            summaries.append(summary)
    return summaries
