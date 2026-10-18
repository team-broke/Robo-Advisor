"""정제된 리서치 문서의 로컬 ChromaDB 색인과 시점 검색.

모든 레코드는 원문 URL·게시 시각·출처·사용 가능 시각을 메타데이터로 가진다.
문서 ID는 정규화 URL 해시(doc_id)이므로 같은 문서를 다시 색인해도 레코드가
늘지 않는다. 이미 색인된 문서의 available_at이나 내용 해시가 바뀌어 들어오면
과거 관측을 바꾸는 소급 수정이므로 거부한다. 검색은 as_of 시점에 사용 가능했던
문서만 대상으로 한다.

기본 임베딩은 외부 모델 다운로드 없이 결정적으로 동작하는 문자 n-gram 해싱이다.
한국어 제목·요약에도 동작하지만 의미 기반 모델보다 검색 품질은 낮을 수 있다.
다른 임베딩 함수를 주입할 수 있으며, 컬렉션에는 사용한 임베딩 이름을 기록해
다른 임베딩으로 같은 컬렉션을 여는 것을 막는다.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.config import Settings as ChromaSettings

from robo_advisor.research.ingestion import ResearchDocument, load_documents

COLLECTION_NAME = "research-documents"
REQUIRED_METADATA = ("url", "published_at", "source", "source_name", "available_at")


class HashingEmbeddingFunction(EmbeddingFunction[Documents]):
    """문자 2·3-gram을 고정 차원에 해싱하고 L2 정규화한다."""

    def __init__(self, dimension: int = 512):
        if dimension < 8:
            raise ValueError("dimension must be at least 8")
        self.dimension = dimension

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed(text) for text in input]

    def _embed(self, text: str) -> np.ndarray:
        normalized = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()
        vector = np.zeros(self.dimension, dtype=np.float32)
        for size in (2, 3):
            for start in range(max(len(normalized) - size + 1, 0)):
                digest = hashlib.blake2b(
                    normalized[start : start + size].encode("utf-8"), digest_size=8
                ).digest()
                value = int.from_bytes(digest, "little")
                vector[value % self.dimension] += 1.0 if value >> 63 else -1.0
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            vector[0] = 1.0
            return vector
        return vector / norm

    @staticmethod
    def name() -> str:
        return "robo-advisor-char-ngram-hash"

    def get_config(self) -> dict[str, Any]:
        return {"dimension": self.dimension}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "HashingEmbeddingFunction":
        return HashingEmbeddingFunction(**config)

    def default_space(self):
        return "cosine"


def _timestamp(value: datetime) -> int:
    return int(value.timestamp() * 1_000_000)


def document_metadata(document: ResearchDocument) -> dict[str, str | int]:
    """출처 추적과 시점 필터에 필요한 메타데이터. None 값은 넣지 않는다."""
    metadata: dict[str, str | int] = {
        "url": str(document.url),
        "canonical_url": document.canonical_url,
        "title": document.title,
        "source": document.source.value,
        "source_name": document.source_name,
        "published_at": document.published_at.isoformat(),
        "published_at_us": _timestamp(document.published_at),
        "published_precision": document.published_precision,
        "available_at": document.available_at.isoformat(),
        "available_at_us": _timestamp(document.available_at),
        "availability_basis": document.availability_basis,
        "content_hash": document.content_hash,
        "related_assets": ",".join(document.related_assets),
    }
    if document.source_id is not None:
        metadata["source_id"] = document.source_id
    return metadata


@dataclass(frozen=True)
class IndexResult:
    added: tuple[str, ...]
    unchanged: tuple[str, ...]


@dataclass(frozen=True)
class SearchHit:
    doc_id: str
    distance: float
    url: str
    title: str
    summary: str
    source: str
    published_at: str
    available_at: str


class ResearchVectorStore:
    """리서치 문서 색인. path 대신 client를 넘기면 테스트용 메모리 저장소도 쓸 수 있다."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        collection_name: str = COLLECTION_NAME,
        embedding_function: EmbeddingFunction | None = None,
        client=None,
    ):
        if client is None:
            if path is None:
                from robo_advisor.config import get_settings

                path = get_settings().data_vectorstore_dir
            Path(path).mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(
                path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
            )
        self.embedding_function = embedding_function or HashingEmbeddingFunction()
        try:
            self.collection = client.get_or_create_collection(
                collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine", "embedding": self.embedding_function.name()},
            )
        except Exception as error:
            raise ValueError(
                f"cannot open collection {collection_name!r} with embedding "
                f"{self.embedding_function.name()!r}: {error}"
            ) from error
        recorded = (self.collection.metadata or {}).get("embedding")
        if recorded != self.embedding_function.name():
            raise ValueError(
                f"collection was built with embedding {recorded!r}, "
                f"not {self.embedding_function.name()!r}"
            )

    def count(self) -> int:
        return self.collection.count()

    def index(self, documents: Iterable[ResearchDocument]) -> IndexResult:
        """새 문서만 추가하고 같은 문서는 건너뛰며 소급 변경은 거부한다."""
        documents = list(documents)
        ids = [document.doc_id for document in documents]
        if len(set(ids)) != len(ids):
            raise ValueError("documents contain duplicate doc_id values")
        stored = {}
        if ids:
            found = self.collection.get(ids=ids, include=["metadatas"])
            stored = dict(zip(found["ids"], found["metadatas"]))
        new, unchanged = [], []
        for document in documents:
            metadata = document_metadata(document)
            previous = stored.get(document.doc_id)
            if previous is None:
                new.append((document, metadata))
                continue
            for key in ("available_at_us", "content_hash"):
                if previous.get(key) != metadata[key]:
                    raise ValueError(
                        f"document {document.doc_id} is already indexed with a different {key}"
                    )
            unchanged.append(document.doc_id)
        if new:
            self.collection.add(
                ids=[document.doc_id for document, _ in new],
                documents=[f"{document.title}\n{document.summary}" for document, _ in new],
                metadatas=[metadata for _, metadata in new],
            )
        return IndexResult(tuple(document.doc_id for document, _ in new), tuple(unchanged))

    def missing_provenance(self) -> tuple[str, ...]:
        """원문 URL 등 필수 메타데이터가 빠진 레코드 ID."""
        found = self.collection.get(include=["metadatas"])
        return tuple(
            record_id
            for record_id, metadata in zip(found["ids"], found["metadatas"])
            if any(not (metadata or {}).get(key) for key in REQUIRED_METADATA)
        )

    def search(self, query: str, as_of: datetime, n_results: int = 5) -> tuple[SearchHit, ...]:
        """as_of 시점에 사용 가능했던 문서에서만 검색한다."""
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError("as_of must include timezone information")
        if n_results < 1:
            raise ValueError("n_results must be positive")
        result = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where={"available_at_us": {"$lte": _timestamp(as_of)}},
            include=["metadatas", "documents", "distances"],
        )
        hits = []
        for doc_id, text, metadata, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            summary = text.split("\n", 1)[1] if "\n" in text else ""
            hits.append(
                SearchHit(
                    doc_id=doc_id,
                    distance=float(distance),
                    url=metadata["url"],
                    title=metadata["title"],
                    summary=summary,
                    source=metadata["source"],
                    published_at=metadata["published_at"],
                    available_at=metadata["available_at"],
                )
            )
        return tuple(hits)


def main(argv=None):
    parser = argparse.ArgumentParser(description="정제된 리서치 문서를 ChromaDB에 색인")
    parser.add_argument("documents", help="ingestion.save_documents로 만든 JSON Lines 파일")
    parser.add_argument("--path", help="기본값: 설정의 data_vectorstore_dir")
    args = parser.parse_args(argv)
    store = ResearchVectorStore(args.path)
    result = store.index(load_documents(args.documents))
    missing = store.missing_provenance()
    print(f"added={len(result.added)} unchanged={len(result.unchanged)} total={store.count()}")
    if missing:
        raise SystemExit(f"records without provenance: {len(missing)}")


if __name__ == "__main__":
    main()
