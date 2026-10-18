"""리서치 문서 색인의 출처 메타데이터·멱등성·시점 검색 검증."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from robo_advisor.research.documents import RawDocument
from robo_advisor.research.ingestion import ingest, normalize, save_documents
from robo_advisor.research.vector_store import (
    REQUIRED_METADATA,
    HashingEmbeddingFunction,
    ResearchVectorStore,
    main,
)

KST = timezone(timedelta(hours=9))


def raw(url, title, summary, published, source="rss", precision="datetime"):
    return RawDocument(
        source=source,
        source_name="dart" if source == "dart" else "naver-finance",
        source_id="20260105000123" if source == "dart" else None,
        title=title,
        summary=summary,
        url=url,
        published_at=published,
        published_precision=precision,
        fetched_at=published + timedelta(minutes=5),
        related_assets=("KODEX200",),
    )


def documents():
    return ingest(
        [
            raw(
                "https://news.example.com/fx",
                "환율 급등에 증시 하락",
                "원달러 환율이 급등하며 외국인 매도가 이어졌다",
                datetime(2026, 1, 5, 10, tzinfo=KST),
            ),
            raw(
                "https://news.example.com/chip",
                "반도체 수출 호조",
                "메모리 반도체 수출이 전년 대비 증가했다",
                datetime(2026, 1, 7, 10, tzinfo=KST),
            ),
            raw(
                "https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20260105000123",
                "주요사항보고서",
                "삼성전자 자기주식 취득 결정",
                datetime(2026, 1, 5, tzinfo=KST),
                source="dart",
                precision="date",
            ),
        ]
    ).documents


@pytest.fixture
def store(tmp_path):
    return ResearchVectorStore(tmp_path / "vectorstore")


def test_hashing_embedding_is_deterministic_normalized_and_handles_korean():
    embed = HashingEmbeddingFunction(dimension=64)
    first, second, empty = embed(["환율 급등", "환율 급등", ""])
    assert len(first) == 64
    np.testing.assert_array_equal(first, second)
    assert np.linalg.norm(first) == pytest.approx(1.0)
    assert np.linalg.norm(empty) == pytest.approx(1.0)
    assert HashingEmbeddingFunction.build_from_config(embed.get_config()).dimension == 64
    with pytest.raises(ValueError):
        HashingEmbeddingFunction(dimension=4)


def test_every_record_has_url_publication_and_source(store):
    docs = documents()
    store.index(docs)
    found = store.collection.get(include=["metadatas"])
    assert sorted(found["ids"]) == sorted(item.doc_id for item in docs)
    for metadata in found["metadatas"]:
        for key in REQUIRED_METADATA:
            assert metadata[key]
        assert metadata["url"].startswith("https://")
    assert store.missing_provenance() == ()
    by_id = dict(zip(found["ids"], found["metadatas"]))
    dart = next(item for item in docs if item.source.value == "dart")
    assert by_id[dart.doc_id]["source_id"] == "20260105000123"
    assert by_id[dart.doc_id]["related_assets"] == "KODEX200"


def test_reindexing_is_idempotent_and_persists(tmp_path):
    path = tmp_path / "vectorstore"
    docs = documents()
    first = ResearchVectorStore(path).index(docs)
    assert len(first.added) == 3 and first.unchanged == ()
    reopened = ResearchVectorStore(path)
    second = reopened.index(docs)
    assert second.added == () and len(second.unchanged) == 3
    assert reopened.count() == 3


@pytest.mark.leakage
def test_retroactive_changes_to_indexed_documents_are_rejected(store):
    docs = documents()
    store.index(docs)
    original = docs[0]
    earlier = original.model_copy(update={"available_at": original.published_at})
    if earlier.available_at == original.available_at:
        earlier = original.model_copy(
            update={"available_at": original.available_at + timedelta(seconds=1)}
        )
    with pytest.raises(ValueError, match="available_at"):
        store.index([earlier])
    edited = original.model_copy(update={"content_hash": "0" * 64})
    with pytest.raises(ValueError, match="content_hash"):
        store.index([edited])
    assert store.count() == 3


def test_duplicate_ids_in_one_batch_are_rejected(store):
    document = documents()[0]
    with pytest.raises(ValueError, match="duplicate"):
        store.index([document, document])


@pytest.mark.leakage
def test_search_only_returns_documents_available_at_as_of(store):
    store.index(documents())
    as_of = datetime(2026, 1, 5, 15, 30, tzinfo=KST)
    hits = store.search("환율 급등", as_of, n_results=5)
    assert [hit.title for hit in hits] == ["환율 급등에 증시 하락"]
    assert hits[0].url == "https://news.example.com/fx"
    assert hits[0].summary.startswith("원달러")
    later = store.search("환율 급등", datetime(2026, 1, 8, tzinfo=KST), n_results=5)
    assert len(later) == 3
    assert later[0].title == "환율 급등에 증시 하락"
    assert store.search("환율", datetime(2026, 1, 1, tzinfo=KST)) == ()
    with pytest.raises(ValueError, match="timezone"):
        store.search("환율", datetime(2026, 1, 8))
    with pytest.raises(ValueError, match="n_results"):
        store.search("환율", as_of, n_results=0)


def test_collection_refuses_a_different_embedding(tmp_path):
    path = tmp_path / "vectorstore"
    ResearchVectorStore(path).index(documents())

    class OtherEmbedding(HashingEmbeddingFunction):
        @staticmethod
        def name():
            return "other-embedding"

    with pytest.raises(ValueError, match="embedding"):
        ResearchVectorStore(path, embedding_function=OtherEmbedding())


def test_cli_indexes_saved_documents(tmp_path, capsys):
    path = save_documents(documents(), tmp_path / "documents.jsonl")
    main([str(path), "--path", str(tmp_path / "vectorstore")])
    assert "added=3 unchanged=0 total=3" in capsys.readouterr().out
    main([str(path), "--path", str(tmp_path / "vectorstore")])
    assert "added=0 unchanged=3 total=3" in capsys.readouterr().out


def test_normalized_document_metadata_roundtrip(store):
    document = normalize(
        raw(
            "https://news.example.com/x",
            "제목",
            "",
            datetime(2026, 1, 9, 9, tzinfo=KST),
        )
    )
    store.index([document])
    metadata = store.collection.get(ids=[document.doc_id], include=["metadatas"])["metadatas"][0]
    assert metadata["available_at"] == document.available_at.isoformat()
    assert "source_id" not in metadata
