import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from robo_advisor.features.risk_features import (
    PROVENANCE_COLUMNS,
    build_risk_features,
    main,
    save_risk_features,
    tag_source,
)
from robo_advisor.research.risk_tags import RiskTagConversion
from robo_advisor.rl.environment import PortfolioEnv
from robo_advisor.rl.training import DataConfig, load_training_dataset
from robo_advisor.schemas.risk import RiskTag

INDEX = pd.date_range("2024-01-02 15:30", periods=6, freq="D", tz="Asia/Seoul")
ASSETS = ("A", "B")
DART = "https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20240103000001"
NEWS = "https://finance.naver.com/news/news_read.naver?article_id=1"


def tag(
    asset="A", available="2024-01-03 15:30", *, severity=0.8, confidence=0.9, url=DART, trace="t1"
):
    at = pd.Timestamp(available, tz="Asia/Seoul")
    return RiskTag(
        event_type="regulation_change",
        severity=severity,
        confidence=confidence,
        asset=asset,
        published_at=at.to_pydatetime(),
        available_at=at.to_pydatetime(),
        source_url=url,
        trace_id=trace,
    )


def conversion(*tags, trace="t1", as_of="2024-01-31"):
    return RiskTagConversion(
        trace_id=trace,
        as_of=pd.Timestamp(as_of, tz="UTC").to_pydatetime(),
        tags=tags,
        excluded=(),
    )


def test_scores_match_environment_rule():
    tags = (
        tag(available="2024-01-03 15:30", severity=0.5),
        tag(available="2024-01-03 15:30", severity=0.9),
        tag(available="2024-01-05 09:00", severity=0.2),
        tag("B", available="2024-01-06 15:30", severity=1.0, confidence=0.7),
    )
    features = build_risk_features([conversion(*tags)], INDEX, ASSETS)
    expected = PortfolioEnv._risk_scores(SimpleNamespace(_index=INDEX, _assets=ASSETS, _tags=tags))
    np.testing.assert_allclose(features.scores.to_numpy(), expected)
    assert features.scores.loc[INDEX[0]].tolist() == [0.0, 0.0]
    assert features.scores.loc[INDEX[1], "A"] == pytest.approx(0.81)
    assert features.scores.loc[INDEX[3], "A"] == pytest.approx(0.18)


def test_provenance_links_each_visible_score_to_source():
    features = build_risk_features(
        [conversion(tag(severity=0.5), tag(severity=0.9))], INDEX, ASSETS
    )
    provenance = features.provenance
    assert tuple(provenance.columns) == PROVENANCE_COLUMNS
    assert len(provenance) == 5
    assert set(provenance["evaluation_at"]) == set(INDEX[1:])
    assert provenance["risk_score"].tolist() == pytest.approx([0.81] * 5)
    assert set(provenance["trace_id"]) == {"t1"}
    assert set(provenance["source_kind"]) == {"dart"}


def test_news_tags_need_forward_start():
    news_early = tag(available="2024-01-03 15:30", url=NEWS)
    news_late = tag("B", available="2024-01-05 15:30", url=NEWS)
    assert tag_source(news_early) == "news"
    without = build_risk_features([conversion(news_early, news_late)], INDEX, ASSETS)
    assert without.tags == ()
    assert {item.reason for item in without.excluded} == {"news_before_forward_start"}

    start = pd.Timestamp("2024-01-04", tz="Asia/Seoul").to_pydatetime()
    forward = build_risk_features(
        [conversion(news_early, news_late)], INDEX, ASSETS, news_start=start
    )
    assert forward.tags == (news_late,)
    assert [(item.tag, item.reason) for item in forward.excluded] == [
        (0, "news_before_forward_start")
    ]
    assert forward.rules["news_start"] == pd.Timestamp(start).tz_convert("UTC").isoformat()


def test_excludes_unlinked_unknown_late_and_duplicate_tags():
    features = build_risk_features(
        [
            conversion(
                tag(),
                tag(),
                tag(trace="other"),
                tag("C"),
                tag(available="2024-01-20 15:30"),
                as_of="2024-01-31",
            ),
            conversion(
                tag(available="2024-01-04 15:30", trace="t2"), trace="t2", as_of="2024-01-04"
            ),
        ],
        INDEX,
        ASSETS,
    )
    assert len(features.tags) == 1
    assert [(item.conversion, item.tag, item.reason) for item in features.excluded] == [
        (0, 1, "duplicate_tag"),
        (0, 2, "trace_mismatch"),
        (0, 3, "unknown_asset"),
        (0, 4, "after_last_evaluation"),
        (1, 0, "after_research_cutoff"),
    ]


@pytest.mark.parametrize(
    "index",
    [
        pd.DatetimeIndex([]),
        INDEX.tz_localize(None),
        INDEX[::-1],
        INDEX.append(INDEX[-1:]),
    ],
)
def test_rejects_invalid_index(index):
    with pytest.raises(ValueError):
        build_risk_features([], index, ASSETS)


def test_rejects_invalid_assets_and_naive_news_start():
    with pytest.raises(ValueError):
        build_risk_features([], INDEX, ("A", "A"))
    with pytest.raises(ValueError):
        build_risk_features(
            [], INDEX, ASSETS, news_start=pd.Timestamp("2024-01-01").to_pydatetime()
        )


def test_save_writes_training_compatible_tags(tmp_path):
    features = build_risk_features([conversion(tag(), tag("B", url=NEWS))], INDEX, ASSETS)
    out = save_risk_features(features, tmp_path / "risk")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["files"]) == {
        "risk_features.parquet",
        "risk_tags.parquet",
        "risk_provenance.parquet",
    }
    assert manifest["tag_sources"] == {"dart": 1, "news": 0}
    assert manifest["excluded_counts"] == {"news_before_forward_start": 1}
    assert manifest["trace_ids"] == ["t1"]
    with pytest.raises(FileExistsError):
        save_risk_features(features, out)

    frame = pd.DataFrame(0.0, index=INDEX, columns=list(ASSETS))
    for name in ("returns", "rsi", "macd"):
        frame.to_parquet(out / f"{name}.parquet")
    dataset = load_training_dataset(
        DataConfig(dataset_dir=out, train_end=INDEX[-1].to_pydatetime())
    )
    assert dataset.risk_tags == features.tags


def test_save_empty_features(tmp_path):
    features = build_risk_features([], INDEX, ASSETS)
    out = save_risk_features(features, tmp_path / "risk")
    assert pd.read_parquet(out / "risk_tags.parquet").empty
    assert pd.read_parquet(out / "risk_provenance.parquet").empty


def test_cli(tmp_path):
    pd.DataFrame(0.0, index=INDEX, columns=list(ASSETS)).to_parquet(tmp_path / "returns.parquet")
    path = tmp_path / "conversion.json"
    path.write_text(conversion(tag()).model_dump_json(), encoding="utf-8")
    main(
        [
            "--conversions",
            str(path),
            "--returns",
            str(tmp_path / "returns.parquet"),
            "--output",
            str(tmp_path / "out"),
        ]
    )
    scores = pd.read_parquet(tmp_path / "out" / "risk_features.parquet")
    assert scores.loc[INDEX[-1], "A"] == pytest.approx(0.72)
