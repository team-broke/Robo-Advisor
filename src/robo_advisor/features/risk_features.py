"""리스크 태그를 평가 시각별 RL 관측 피처와 출처 추적표로 만든다.

점수 규칙은 rl/environment.py와 같다. 각 평가 시각에는 그 시각까지 가용한 태그 중
가장 최근 available_at의 최대 risk_score가 보이고, 태그가 없으면 0이다. 과거 뉴스
아카이브가 없으므로 과거 구간은 DART 공시 태그만 쓴다. 뉴스 태그는 수집을 시작한
news_start 이후 가용한 것만 forward 평가에 쓰고, news_start가 없으면 모두 제외한다.
제외한 태그는 리서치 trace ID와 사유만 기록한다.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

from robo_advisor.data.snapshot import write_frame
from robo_advisor.research.risk_tags import RiskTagConversion
from robo_advisor.schemas.risk import RiskTag

DART_HOSTS = frozenset({"dart.fss.or.kr", "opendart.fss.or.kr"})
TAG_COLUMNS = tuple(RiskTag.model_fields)
PROVENANCE_COLUMNS = (
    "evaluation_at",
    "asset",
    "risk_score",
    "event_type",
    "published_at",
    "available_at",
    "source_kind",
    "source_url",
    "trace_id",
)
SourceKind = Literal["dart", "news"]
ExclusionReason = Literal[
    "trace_mismatch",
    "after_research_cutoff",
    "unknown_asset",
    "news_before_forward_start",
    "after_last_evaluation",
    "duplicate_tag",
]


@dataclass(frozen=True)
class TagExclusion:
    """원문 없이 변환 결과 위치·trace·사유만 남긴다."""

    conversion: int
    tag: int
    trace_id: str
    reason: ExclusionReason


@dataclass(frozen=True)
class RiskFeatures:
    scores: pd.DataFrame
    tags: tuple[RiskTag, ...]
    provenance: pd.DataFrame
    excluded: tuple[TagExclusion, ...]
    rules: dict = field(default_factory=dict)


def tag_source(tag: RiskTag) -> SourceKind:
    """원문 URL 호스트로 DART 공시와 뉴스를 구분한다."""
    host = (urlsplit(str(tag.source_url)).hostname or "").lower()
    return "dart" if host in DART_HOSTS else "news"


def _evaluation_index(index):
    if (
        not isinstance(index, pd.DatetimeIndex)
        or index.tz is None
        or index.empty
        or index.hasnans
        or not index.is_unique
        or not index.is_monotonic_increasing
    ):
        raise ValueError("index must be a non-empty, ordered, unique, timezone-aware DatetimeIndex")
    return index


def _aware(value, label):
    if value is None:
        return None
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError(f"{label} must include timezone information")
    return stamp.tz_convert(timezone.utc)


def build_risk_features(
    conversions: Iterable[RiskTagConversion],
    index: pd.DatetimeIndex,
    assets: Sequence[str],
    *,
    news_start: datetime | None = None,
) -> RiskFeatures:
    """검증된 변환 결과의 태그를 평가 시각 × 자산 점수로 집계한다."""
    index = _evaluation_index(index)
    assets = tuple(assets)
    if not assets or len(set(assets)) != len(assets):
        raise ValueError("assets must be non-empty and unique")
    news_start = _aware(news_start, "news_start")
    last = index[-1]
    tags, excluded, seen = [], [], set()
    for c, conversion in enumerate(conversions):
        conversion = RiskTagConversion.model_validate(conversion)
        for t, tag in enumerate(conversion.tags):
            available = pd.Timestamp(tag.available_at)
            if tag.trace_id != conversion.trace_id:
                reason = "trace_mismatch"
            elif available > pd.Timestamp(conversion.as_of):
                reason = "after_research_cutoff"
            elif tag.asset not in assets:
                reason = "unknown_asset"
            elif tag_source(tag) == "news" and (news_start is None or available < news_start):
                reason = "news_before_forward_start"
            elif available > last:
                reason = "after_last_evaluation"
            elif tag.model_dump_json() in seen:
                reason = "duplicate_tag"
            else:
                seen.add(tag.model_dump_json())
                tags.append(tag)
                continue
            excluded.append(TagExclusion(c, t, conversion.trace_id, reason))
    tags.sort(key=lambda tag: (tag.available_at, tag.asset, tag.model_dump_json()))

    scores = np.zeros((len(index), len(assets)))
    rows = []
    times = index.as_unit("ns").asi8
    for i, asset in enumerate(assets):
        groups = {}
        for tag in tags:
            if tag.asset == asset:
                groups.setdefault(pd.Timestamp(tag.available_at).value, []).append(tag)
        if not groups:
            continue
        available = np.array(sorted(groups))
        positions = np.searchsorted(available, times, side="right") - 1
        for row, position in enumerate(positions):
            if position < 0:
                continue
            group = groups[available[position]]
            best = max(tag.risk_score for tag in group)
            scores[row, i] = best
            for tag in group:
                if tag.risk_score == best:
                    rows.append(
                        (
                            index[row],
                            asset,
                            best,
                            tag.event_type.value,
                            pd.Timestamp(tag.published_at),
                            pd.Timestamp(tag.available_at),
                            tag_source(tag),
                            str(tag.source_url),
                            tag.trace_id,
                        )
                    )
    provenance = pd.DataFrame(rows, columns=list(PROVENANCE_COLUMNS))
    if provenance.empty:
        provenance = provenance.astype(
            {
                "evaluation_at": f"datetime64[ns, {index.tz}]",
                "risk_score": "float64",
                "published_at": "datetime64[ns, UTC]",
                "available_at": "datetime64[ns, UTC]",
            }
        )
    rules = {
        "score": "latest_available_max_risk_score",
        "missing": 0.0,
        "historical_source": "dart",
        "news_start": None if news_start is None else news_start.isoformat(),
        "evaluation_start": index[0].isoformat(),
        "evaluation_end": last.isoformat(),
    }
    return RiskFeatures(
        scores=pd.DataFrame(scores, index=index, columns=list(assets)),
        tags=tuple(tags),
        provenance=provenance,
        excluded=tuple(excluded),
        rules=rules,
    )


def _tag_frame(tags):
    records = [
        {
            **tag.model_dump(),
            "event_type": tag.event_type.value,
            "source_url": str(tag.source_url),
            "published_at": pd.Timestamp(tag.published_at),
            "available_at": pd.Timestamp(tag.available_at),
        }
        for tag in tags
    ]
    frame = pd.DataFrame(records, columns=list(TAG_COLUMNS))
    if frame.empty:
        frame = frame.astype(
            {
                "severity": "float64",
                "confidence": "float64",
                "published_at": "datetime64[ns, UTC]",
                "available_at": "datetime64[ns, UTC]",
            }
        )
    return frame


def save_risk_features(features: RiskFeatures, directory) -> Path:
    """새 디렉터리에 피처·태그·추적표·manifest를 쓴다. 기존 경로는 거부한다.

    risk_tags.parquet는 rl/training.py가 읽는 RiskTag 필드 열과 같아 PPO 입력
    디렉터리에 그대로 둘 수 있다.
    """
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"{directory} already exists")
    directory.mkdir(parents=True)
    frames = {
        "risk_features.parquet": features.scores,
        "risk_tags.parquet": _tag_frame(features.tags),
        "risk_provenance.parquet": features.provenance,
    }
    files = {name: write_frame(frame, directory / name) for name, frame in frames.items()}
    sources = {"dart": 0, "news": 0}
    for tag in features.tags:
        sources[tag_source(tag)] += 1
    reasons = {}
    for item in features.excluded:
        reasons[item.reason] = reasons.get(item.reason, 0) + 1
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rules": features.rules,
        "assets": list(features.scores.columns),
        "tag_count": len(features.tags),
        "tag_sources": sources,
        "trace_ids": sorted({tag.trace_id for tag in features.tags}),
        "excluded_counts": reasons,
        "excluded": [item.__dict__ for item in features.excluded],
        "files": files,
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description="리스크 태그를 RL 관측 피처로 집계한다.")
    parser.add_argument("--conversions", type=Path, nargs="+", required=True)
    parser.add_argument("--returns", type=Path, required=True, help="평가 시각·자산 순서 기준")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--news-start", help="뉴스 수집 시작 시각(ISO 8601, timezone 포함)")
    args = parser.parse_args(argv)
    returns = pd.read_parquet(args.returns)
    conversions = [
        RiskTagConversion.model_validate_json(path.read_text(encoding="utf-8"))
        for path in args.conversions
    ]
    features = build_risk_features(
        conversions, returns.index, list(returns.columns), news_start=args.news_start
    )
    print(save_risk_features(features, args.output))


if __name__ == "__main__":
    main()
