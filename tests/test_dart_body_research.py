import io
import json
import stat
from datetime import datetime, timedelta, timezone
from uuid import uuid4
from zipfile import ZipFile, ZipInfo

import chromadb
import httpx
import pytest
from chromadb.config import Settings

from robo_advisor.api.lifecycle import _load_dart_body
from robo_advisor.research.contracts import ResearchOptions
from robo_advisor.research.dart_body import Acquisition, build_filing, filing_url, sha256
from robo_advisor.research.documents import RawDocument, SourceKind
from robo_advisor.research.graph import run_research
from robo_advisor.research.ingestion import normalize
from robo_advisor.research.risk_tags import convert_risk_events
from robo_advisor.research.sources.http import PoliteClient
from robo_advisor.research.vector_store import ResearchVectorStore
from scripts.prepare_dart_body import acquire, build_manifest, write_json

RECEIPT = "20241115001234"
FIRST_FETCH = datetime(2024, 11, 25, 4, 6, 50, 123456, tzinfo=timezone.utc)
XML = """<?xml version="1.0" encoding="utf-8"?>
<DOCUMENT><BODY><SECTION-1>
<TITLE>연결 손익계산서</TITLE>
<P>제58기 3분기 (단위 : 백만원)</P>
<TABLE><THEAD><TR><TH>항목</TH><TH>당분기</TH><TH>전분기</TH></TR></THEAD>
<TBODY><TR><TD>매출액</TD><TD>100</TD><TD>90</TD></TR>
<TR><TD>영업이익</TD><TD>15</TD><TD>10</TD></TR></TBODY></TABLE>
<P>이번 분기에 공지된 규제 변경으로 신고 서류 제출 절차의 일부가 변경되었습니다.</P>
</SECTION-1></BODY></DOCUMENT>""".encode()


def _document(title="분기보고서", *, remarks="", receipt=RECEIPT):
    local_day = datetime.strptime(receipt[:8], "%Y%m%d").replace(tzinfo=timezone.utc)
    return normalize(
        RawDocument(
            source=SourceKind.DART,
            source_name="dart",
            source_id=receipt,
            title=title,
            summary="목록 요약",
            url=filing_url(receipt),
            published_at=local_day - timedelta(hours=9),
            published_precision="date",
            fetched_at=FIRST_FETCH - timedelta(days=1),
            metadata={"rm": remarks},
        )
    )


def _archive(xml=XML, *, receipt=RECEIPT, other=()):
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as bundle:
        bundle.writestr(f"{receipt}.xml", xml)
        for name, data in other:
            bundle.writestr(name, data)
    return buffer.getvalue()


def _filing(xml=XML, *, document=None, archive=None):
    document = document or _document()
    archive = archive if archive is not None else _archive(xml, receipt=document.source_id)
    fetched = Acquisition(
        receipt=document.source_id, received_at=FIRST_FETCH, sha256=sha256(archive)
    )
    return build_filing(document, archive, fetched)


def _store(chunks):
    store = ResearchVectorStore(
        client=chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False)),
        collection_name=f"test-dart-body-{uuid4().hex}",
    )
    store.index_body(chunks)
    return store


class _LLM:
    def __init__(self, *, failure=None):
        self.stages = []
        self.failure = failure

    def generate(self, stage, payload):
        self.stages.append(stage)
        sources = payload["sources"]
        if stage == "plan":
            return json.dumps({"query": "연결 손익계산서, 매출액, 영업이익, 규제 변경"})
        assert all("body" in source and source["summary"] == "목록 요약" for source in sources)
        if stage == "verify":
            response = {
                "evidence": [{"doc_id": item["doc_id"], "relevance": 1.0} for item in sources],
                "conflicts": [],
            }
            if self.failure == "unknown_id":
                response["evidence"][0]["doc_id"] = "unknown"
            if self.failure == "invented_fact":
                response["text"] = "매출액 999999"
            return json.dumps(response)
        raise AssertionError("the server synthesizes body excerpts without a model call")


def test_filing_records_source_units_periods_and_hashes():
    archive = _archive()
    info, chunks = _filing(archive=archive)
    assert info["receipt_time"] is None
    assert info["available_at"] == FIRST_FETCH.isoformat()
    assert info["document_sha256"] == sha256(XML)
    assert info["archive_sha256"] == sha256(archive)
    assert len(chunks) == 3
    quote = chunks[0].citation.body
    assert quote.section == "연결 손익계산서"
    assert quote.quote.splitlines() == [
        "제58기 3분기 (단위 : 백만원)",
        "항목 | 당분기 | 전분기",
        "매출액 | 100 | 90",
    ]
    assert quote.quote_sha256 == sha256(quote.quote.encode())


@pytest.mark.leakage
def test_chroma_filters_at_exact_first_body_fetch():
    _, chunks = _filing()
    store = _store(chunks)
    assert store.search("매출액", FIRST_FETCH - timedelta(microseconds=1)) == ()
    assert len(store.search("매출액", FIRST_FETCH)) == len(chunks)
    assert store.index_body(chunks).unchanged == tuple(item.citation.doc_id for item in chunks)
    llm = _LLM()
    result = run_research(
        "매출액",
        FIRST_FETCH - timedelta(microseconds=1),
        store=store,
        llm=llm,
        options=ResearchOptions(max_retries=0),
        evidence_scope="dart_body",
    )
    assert result.status == "failed"
    assert "missing_citations" in result.reasons
    assert set(llm.stages) <= {"plan", "rewrite"}


@pytest.mark.parametrize("failure", [None, "unknown_id", "invented_fact"])
def test_graph_withholds_invented_facts_and_invalid_citations(failure):
    _, chunks = _filing()
    store = _store(chunks)
    llm = _LLM(failure=failure)
    result = run_research(
        "매출액 영업이익 규제 변경",
        FIRST_FETCH,
        store=store,
        llm=llm,
        options=ResearchOptions(max_retries=0),
        evidence_scope="dart_body",
    )
    if failure:
        assert result.status == "failed" and not result.claims
        assert "invalid_verification_response" in result.reasons
        return
    assert result.status == "completed"
    assert llm.stages == ["plan", "verify"]
    assert result.trace[-1].tool == "verified_excerpts"
    assert {item.text for item in result.claims} == {item.citation.body.quote for item in chunks}
    assert len({str(item.citations[0].url) for item in result.claims}) == 1
    assert "999999" not in result.model_dump_json()
    assert len(result.risk_candidates) == 1
    assert "severity" not in result.risk_candidates[0].model_dump()
    report = convert_risk_events([{}], result, as_of=FIRST_FETCH)
    assert not report.tags
    assert report.excluded[0].reason == "unscored_body_evidence"


def test_citation_rejects_receipt_url_and_excerpt_tampering():
    _, chunks = _filing()
    data = chunks[0].model_dump(mode="json")
    data["citation"]["url"] = filing_url("20241115001235")
    with pytest.raises(ValueError, match="does not match"):
        type(chunks[0]).model_validate(data)
    data["citation"]["url"] = filing_url(RECEIPT)
    data["citation"]["body"]["quote"] = "매출액 999999"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        type(chunks[0]).model_validate(data)


@pytest.mark.leakage
@pytest.mark.parametrize(
    ("title", "remarks", "reason"),
    [
        ("[기재정정]분기보고서", "", "unlinked_correction"),
        ("[첨부추가]분기보고서", "", "unlinked_correction"),
        ("분기보고서", "유정", "later_correction_reported"),
    ],
)
def test_unlinked_correction_and_flagged_original_are_not_indexable(title, remarks, reason):
    info, chunks = _filing(document=_document(title, remarks=remarks))
    assert not chunks and info["reason"] == reason
    assert info["supersedes_receipt"] is None
    assert info["document_sha256"] == sha256(XML)


@pytest.mark.parametrize(
    "path", ["../outside.xml", "/escape.xml", "C:/escape.xml", "..\\escape.xml"]
)
def test_rejects_zip_path_escape(path):
    info, chunks = _filing(archive=_archive(other=[(path, b"")]))
    assert not chunks and info["reason"] == "unsafe_archive_member"


def test_rejects_zip_symlink():
    link = ZipInfo("link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    info, chunks = _filing(archive=_archive(other=[(link, b"outside.xml")]))
    assert not chunks and info["reason"] == "unsafe_archive_member"


@pytest.mark.parametrize(
    "prefix",
    [
        b'<!DOCTYPE DOCUMENT SYSTEM "file:///etc/passwd">',
        b'<!DOCTYPE DOCUMENT [<!ENTITY local SYSTEM "file:///etc/passwd">]>',
        b'<!DOCTYPE DOCUMENT [<!ENTITY a "ha"><!ENTITY b "&a;&a;">]>',
    ],
)
def test_rejects_dtd_and_external_or_internal_entities(prefix):
    _, _, document = XML.partition(b"?>")
    info, chunks = _filing(prefix + document)
    assert not chunks and info["reason"] == "xml_dtd_or_entity_forbidden"


def test_html_is_excluded_and_dart_literal_escapes_preserve_evidence():
    info, chunks = _filing(b'<html><head><meta charset="UTF-8"></head><body>raw</body></html>')
    assert not chunks and info["reason"] == "html_wrapper_not_supported"
    xml = XML.replace(b"<TD>100</TD>", b'<TD ENG="C&T">100 S&P <![CDATA[A&B]]> < market ></TD>')
    info, chunks = _filing(xml)
    assert info["text_literal_escapes"] == 3
    assert "100 S&P A&B < market >" in chunks[0].citation.body.quote


def test_cache_does_not_redownload_or_admit_a_fourth_receipt(tmp_path):
    requests = []

    def response(request):
        requests.append(request)
        return httpx.Response(200, content=_archive())

    receipts = tuple(f"2024111500123{end}" for end in range(4, 8))
    with httpx.Client(transport=httpx.MockTransport(response)) as http:
        with PoliteClient(http, utc_now=lambda: FIRST_FETCH) as client:
            acquire(tmp_path, receipts[:1], client, "DART-secret")
            acquire(tmp_path, receipts[:1], client, "DART-secret")
            with pytest.raises(ValueError, match="three"):
                acquire(tmp_path, receipts, client, "DART-secret")
            assert "DART-secret" not in repr(client.request_logs)
    assert len(requests) == 1
    assert requests[0].url.path == "/api/document.xml"
    assert requests[0].url.params["rcept_no"] == RECEIPT


def test_manifest_replays_original_and_rejects_tampered_archive(tmp_path):
    slot = tmp_path / "raw" / RECEIPT
    slot.mkdir(parents=True)
    archive = _archive()
    target = slot / "original.zip"
    target.write_bytes(archive)
    write_json(
        slot / "acquisition.json",
        {"receipt": RECEIPT, "received_at": FIRST_FETCH.isoformat(), "sha256": sha256(archive)},
    )
    documents = (_document(),)
    manifest = build_manifest(tmp_path, documents, "a" * 64)
    assert len(_load_dart_body(tmp_path, manifest, documents)) == 3
    with pytest.raises(ValueError, match="invalid OpenDART body manifest"):
        _load_dart_body(tmp_path, {**manifest, "training_approved": True}, documents)
    target.write_bytes(archive + b"x")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        _load_dart_body(tmp_path, manifest, documents)
