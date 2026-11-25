"""OpenDART original-file provenance and exact, point-in-time body extracts."""

from __future__ import annotations

import hashlib
import io
import re
import stat
import zlib
from pathlib import PurePosixPath, PureWindowsPath
from xml.etree.ElementTree import TreeBuilder
from xml.parsers import expat
from zipfile import ZIP_DEFLATED, ZIP_STORED, BadZipFile, ZipFile

from pydantic import Field, model_validator

from robo_advisor.research.contracts import Citation, DartBodyEvidence, ResearchModel
from robo_advisor.research.documents import SUMMARY_MAX_LENGTH, SourceKind
from robo_advisor.research.ingestion import KST, ResearchDocument, available_at
from robo_advisor.research.sources.http import DART_ARCHIVE_LIMIT
from robo_advisor.schemas.common import NonEmptyString, UTCDateTime

MAX_XML_BYTES = 40 * 1024**2
MAX_ARCHIVE_EXPANDED = 80 * 1024**2
MAX_MEMBERS = 64
MAX_RATIO = 250
MAX_XML_DEPTH = 96
MAX_XML_ELEMENTS = 250_000
MAX_CHUNKS = 180

_MARKUP = re.compile(
    r"<!--[\s\S]*?-->|<!\[CDATA\[[\s\S]*?\]\]>|<\?[\s\S]*?\?>"
    r"""|</?[A-Za-z_:](?:[^<>"']|"[^"]*"|'[^']*')*>"""
)
_BARE_AMPERSAND = re.compile(r"&(?!#(?:[0-9]+|x[0-9A-Fa-f]+);|[\w:.-]+;)")
_TEXT_ANGLE = re.compile(r"<(?=[\t ]+[^<>\r\n]{1,200}>)")
_METRIC = re.compile(r"^(?:순?매출액|수익\(매출액\)|영업이익|영업손익)")
_UNIT = re.compile(r"단위\s*[:：]")
_RISK_WORDS = re.compile(r"실적\s*쇼크|어닝\s*쇼크|규제\s*(?:변경|강화|완화|신설)")
_INDICATOR = re.compile(r"실적|매출|영업이익|주요\s*경영사항|거래\s*(?:금액|기간)")
_CORRECTION_TITLE = re.compile(r"^\s*\[[^\]]*(?:정정|첨부추가|철회)[^\]]*\]")


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class DartBodyError(ValueError):
    """A bounded, non-sensitive reason that can be recorded in the corpus manifest."""


class Acquisition(ResearchModel):
    receipt: str = Field(pattern=r"^[0-9]{14}$")
    received_at: UTCDateTime
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def filing_url(receipt: str) -> str:
    return f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={receipt}"


def validate_listing_document(document: ResearchDocument) -> str:
    receipt = document.source_id
    if (
        not isinstance(receipt, str)
        or not re.fullmatch(r"[0-9]{14}", receipt)
        or document.source is not SourceKind.DART
        or document.published_precision != "date"
        or document.available_at != available_at(document)
        or document.published_at.astimezone(KST).strftime("%Y%m%d%H%M%S%f")
        != receipt[:8] + "000000000000"
        or str(document.url) != filing_url(receipt)
        or document.related_assets
    ):
        raise ValueError("invalid pinned DART listing row")
    return receipt


def _correction_status(document):
    if _CORRECTION_TITLE.match(document.title):
        return "unlinked_correction"
    remarks = document.metadata.get("rm")
    if remarks is None:
        # Old pinned listing rows append the source remark to the summary.
        parts = document.summary.split()
        suffix = parts[-1] if parts else ""
        remarks = suffix if re.fullmatch(r"[유코넥채K정철기공연]+", suffix) else ""
    if "정" in remarks:
        return "later_correction_reported"
    if "철" in remarks:
        return "withdrawal_reported"
    return "no_correction_reported_in_listing"


def _read_xml_member(archive: bytes, receipt: str) -> tuple[str, bytes]:
    if len(archive) > DART_ARCHIVE_LIMIT or not archive.startswith(b"PK\x03\x04"):
        raise DartBodyError("invalid_archive")
    try:
        with ZipFile(io.BytesIO(archive)) as bundle:
            entries = bundle.infolist()
            if len(entries) > MAX_MEMBERS:
                raise DartBodyError("archive_member_limit")
            seen = set()
            total = 0
            for entry in entries:
                name = entry.orig_filename
                parts = PurePosixPath(name)
                file_type = stat.S_IFMT(entry.external_attr >> 16)
                if (
                    not name
                    or "\\" in name
                    or "\x00" in name
                    or ":" in name
                    or parts.is_absolute()
                    or PureWindowsPath(name).drive
                    or ".." in parts.parts
                    or "." == str(parts)
                    or str(parts).casefold() in seen
                    or file_type not in {0, stat.S_IFDIR, stat.S_IFREG}
                    or entry.flag_bits & 1
                    or entry.compress_type not in {ZIP_STORED, ZIP_DEFLATED}
                ):
                    raise DartBodyError("unsafe_archive_member")
                seen.add(str(parts).casefold())
                total += entry.file_size
                if total > MAX_ARCHIVE_EXPANDED or entry.file_size > MAX_RATIO * max(
                    entry.compress_size, 1
                ):
                    raise DartBodyError("archive_expansion_limit")
            member = f"{receipt}.xml"
            if member not in bundle.namelist():
                raise DartBodyError("receipt_xml_missing")
            if bundle.getinfo(member).file_size > MAX_XML_BYTES:
                raise DartBodyError("xml_size_limit")
            with bundle.open(member) as stream:
                content = stream.read(MAX_XML_BYTES + 1)
            if len(content) > MAX_XML_BYTES:
                raise DartBodyError("xml_size_limit")
            return member, content
    except (BadZipFile, OSError, RuntimeError, EOFError, zlib.error) as exc:
        raise DartBodyError("invalid_archive") from exc


def _normalize_text_literals(text: str) -> tuple[str, int]:
    """Escape bare literals while preserving text, comments and CDATA semantics."""
    parts, previous, count = [], 0, 0
    for marker in _MARKUP.finditer(text):
        raw, amp = _BARE_AMPERSAND.subn("&amp;", text[previous : marker.start()])
        raw, angle = _TEXT_ANGLE.subn("&lt;", raw)
        count += amp + angle
        markup = marker[0]
        if not markup.startswith(("<!", "<?")):
            markup, amp = _BARE_AMPERSAND.subn("&amp;", markup)
            count += amp
        parts.extend((raw, markup))
        previous = marker.end()
    raw, amp = _BARE_AMPERSAND.subn("&amp;", text[previous:])
    raw, angle = _TEXT_ANGLE.subn("&lt;", raw)
    parts.append(raw)
    return "".join(parts), count + amp + angle


def _parse_document(xml: bytes):
    try:
        text = xml.decode("utf-8-sig")
    except UnicodeError as exc:
        raise DartBodyError("unsupported_xml_encoding") from exc
    if re.match(r"\s*(?:<!doctype\s+html[^>]*>\s*)?<html(?:\s|>)", text, re.I):
        raise DartBodyError("html_wrapper_not_supported")
    text, repairs = _normalize_text_literals(text)
    builder = TreeBuilder()
    parser = expat.ParserCreate()
    depth = elements = 0

    def reject_declaration(*_):
        raise DartBodyError("xml_dtd_or_entity_forbidden")

    def start(tag, attributes):
        nonlocal depth, elements
        depth += 1
        elements += 1
        if depth > MAX_XML_DEPTH or elements > MAX_XML_ELEMENTS:
            raise DartBodyError("xml_structure_limit")
        builder.start(tag, attributes)

    def end(tag):
        nonlocal depth
        builder.end(tag)
        depth -= 1

    parser.StartDoctypeDeclHandler = reject_declaration
    parser.EntityDeclHandler = reject_declaration
    parser.ExternalEntityRefHandler = reject_declaration
    parser.SetParamEntityParsing(expat.XML_PARAM_ENTITY_PARSING_NEVER)
    parser.StartElementHandler = start
    parser.EndElementHandler = end
    parser.CharacterDataHandler = builder.data
    try:
        parser.Parse(text, True)
        root = builder.close()
    except expat.ExpatError as exc:
        raise DartBodyError("malformed_xml") from exc
    if root.tag != "DOCUMENT" or root.find("BODY") is None:
        raise DartBodyError("unsupported_xml_document")
    return root, repairs


def _text(element) -> str:
    return " ".join("".join(element.itertext()).split())


def _cells(row) -> tuple[str, ...]:
    return tuple(_text(cell) for cell in row if cell.tag in {"TH", "TD", "TE", "TU"})


def _table_rows(table):
    return table.findall("./THEAD/TR") + table.findall("./TBODY/TR") + table.findall("./TR")


def _table_context(table, parent):
    parts = []
    for previous in reversed(list(parent)[: list(parent).index(table)]):
        if previous.tag not in {"TABLE", "P"}:
            break
        if previous.tag == "TABLE" and (
            len(_table_rows(previous)) > 4 or not _UNIT.search(_text(previous))
        ):
            break
        text = _text(previous)
        if 0 < len(text) <= 330 and (_UNIT.search(text) or "재무정보" in text):
            parts.append(text)
        if len(parts) == 2:
            break
    return tuple(reversed(parts))


def _extract(root):
    """Keep financial row headers, source units and original values together."""
    body = root.find("BODY")
    parents = {child: element for element in body.iter() for child in element}
    numbers = {element: index for index, element in enumerate(body.iter(), 1)}

    def section(element):
        headings = []
        while element in parents:
            element = parents[element]
            if element.tag.startswith("SECTION-") or element.tag == "TABLE-GROUP":
                title = element.find("TITLE")
                if title is not None and _text(title):
                    headings.append(_text(title))
        return " / ".join(reversed(headings)) or "공시 본문"

    for table in body.iter("TABLE"):
        rows = tuple(_cells(row) for row in _table_rows(table))
        if not rows:
            continue
        name = section(table)
        context = _table_context(table, parents[table])
        header_len = len(table.findall("./THEAD/TR"))
        if header_len and "요약재무정보" in name and len(rows) > header_len:
            if any(re.search(r"20[0-9]{2}년", cell) for cell in rows[header_len]):
                header_len += 1
        financial_statement = "요약재무정보" in name or name.endswith("손익계산서")
        if financial_statement and _UNIT.search(" ".join(context)) and 0 < header_len <= 4:
            for index, cells in enumerate(rows[header_len:], header_len):
                label = cells[0].replace(" ", "") if cells else ""
                if (
                    _METRIC.match(label)
                    and len(cells) > 1
                    and any(re.search(r"[0-9]", value) for value in cells[1:])
                ):
                    lines = [*context, *(" | ".join(row) for row in rows[:header_len])]
                    yield name, f"body-element:{numbers[table]}/row:{index + 1}", (
                        "\n".join((*lines, " | ".join(cells)))
                    )
        for index, cells in enumerate(rows[:-1]):
            if cells and "거래상대방" in cells[0] and any("합계액" in cell for cell in cells):
                if not _UNIT.search(" ".join(context)):
                    continue
                labels = rows[index - 1 : index] if index else ()
                period = tuple(row for row in rows[:index] if any("거래기간" in c for c in row))
                lines = (*period, *labels, cells, rows[index + 1])
                yield name, f"body-element:{numbers[table]}/rows:{index + 1}-{index + 2}", (
                    "\n".join((*context, *(" | ".join(row) for row in lines)))
                )

    for paragraph in body.iter("P"):
        text = _text(paragraph)
        name = section(paragraph)
        if 30 <= len(text) <= 1800 and (
            _RISK_WORDS.search(text) or ("주요" in name and _INDICATOR.search(text))
        ):
            yield name, f"body-element:{numbers[paragraph]}", text


def iter_body_blocks(root):
    """Return paragraph and leaf table-row text with a stable original body location."""
    body = root.find("BODY")
    parents = {child: element for element in body.iter() for child in element}
    for number, element in enumerate(body.iter(), 1):
        if element.tag == "TR":
            if next(element.iter("TABLE"), None) is not None:
                continue
            text = " | ".join(_cells(element))
        elif element.tag == "P":
            text = _text(element)
        else:
            continue
        headings = []
        ancestor = element
        while ancestor in parents:
            ancestor = parents[ancestor]
            if element.tag == "P" and ancestor.tag == "TR":
                break
            if ancestor.tag.startswith("SECTION-") or ancestor.tag == "TABLE-GROUP":
                title = ancestor.find("TITLE")
                if title is not None and (heading := _text(title)):
                    headings.append(heading)
        else:
            if 5 <= len(text) <= 1800:
                yield " / ".join(reversed(headings)) or "공시 본문", f"body-element:{number}", text


class BodyChunk(ResearchModel):
    citation: Citation
    title: NonEmptyString
    summary: str = Field(max_length=SUMMARY_MAX_LENGTH)

    @model_validator(mode="after")
    def verify_identifier(self):
        body = self.citation.body
        if body is None or self.citation.doc_id != _chunk_id(body):
            raise ValueError("body chunk identifier mismatch")
        return self


def _chunk_id(body):
    identity = "\n".join((body.receipt, body.document_sha256, body.locator, body.quote_sha256))
    return sha256(identity.encode("utf-8"))


def build_filing(
    document: ResearchDocument, archive: bytes, acquisition: Acquisition, *, extract=_extract
) -> tuple[dict, tuple[BodyChunk, ...]]:
    receipt = validate_listing_document(document)
    if acquisition.receipt != receipt or sha256(archive) != acquisition.sha256:
        raise ValueError("OpenDART archive provenance mismatch")
    if acquisition.received_at < document.fetched_at:
        raise ValueError("OpenDART original predates the pinned listing acquisition")
    corrected = _correction_status(document)
    effective_at = max(document.available_at, acquisition.received_at)
    info = {
        "receipt": receipt,
        "receipt_date": document.published_at.astimezone(KST).date().isoformat(),
        "receipt_time": None,
        "published_precision": "date",
        "first_fetched_at": acquisition.received_at.isoformat(),
        "available_at": effective_at.isoformat(),
        "availability_basis": "later_of_first_body_fetch_and_next_seoul_midnight",
        "url": str(document.url),
        "title": document.title,
        "correction_status": corrected,
        "supersedes_receipt": None,
        "archive_sha256": acquisition.sha256,
    }
    try:
        member, xml = _read_xml_member(archive, receipt)
        doc_hash = sha256(xml)
        info.update({"member": member, "document_sha256": doc_hash})
        if corrected != "no_correction_reported_in_listing":
            return {**info, "status": "excluded", "reason": corrected, "chunk_count": 0}, ()
        root, repairs = _parse_document(xml)
    except DartBodyError as error:
        return {**info, "status": "excluded", "reason": str(error), "chunk_count": 0}, ()
    info["text_literal_escapes"] = repairs
    chunks = []
    seen = set()
    for section, locator, quote in extract(root):
        identity = (section, quote)
        if identity in seen or not 1 <= len(quote) <= 1800:
            continue
        seen.add(identity)
        body = DartBodyEvidence(
            receipt=receipt,
            receipt_date=info["receipt_date"],
            first_fetched_at=acquisition.received_at,
            member=member,
            section=section,
            locator=locator,
            document_sha256=doc_hash,
            quote=quote,
            quote_sha256=sha256(quote.encode("utf-8")),
        )
        citation = Citation(
            doc_id=_chunk_id(body),
            url=document.url,
            published_at=document.published_at,
            available_at=effective_at,
            body=body,
        )
        chunks.append(BodyChunk(citation=citation, title=document.title, summary=document.summary))
        if len(chunks) == MAX_CHUNKS:
            break
    info["status"] = "indexed" if chunks else "excluded"
    if not chunks:
        info["reason"] = "no_supported_body_evidence"
    return {**info, "chunk_count": len(chunks)}, tuple(chunks)


def is_explicit_risk_excerpt(text: str) -> bool:
    return bool(_RISK_WORDS.search(text))
