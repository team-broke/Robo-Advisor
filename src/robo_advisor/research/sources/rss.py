"""네이버 금융에서 설정한 RSS 2.0 피드만 수집한다.

시간대 없는 pubDate는 추정하지 않고 건너뛰며 사유를 기록한다. 기사 원문을
추가 요청하지 않는다. 링크 없는 항목·잘못된 게시 시각·잘못된 문서도 건너뛴다.
"""

from email.utils import parsedate_to_datetime
from xml.etree import ElementTree

from pydantic import ValidationError

from robo_advisor.research.documents import RawDocument, SourceKind
from robo_advisor.research.sources.http import (
    CollectionResult,
    FeedConfig,
    PoliteClient,
    SkippedItem,
    SourceError,
    clean_text,
    validate_source_url,
)


def _text(item, name):
    node = item.find(name)
    if node is None:
        return ""
    value = (node.text or "") + "".join(
        ElementTree.tostring(child, encoding="unicode") for child in node
    )
    return value.strip()


class RSSAdapter:
    def __init__(self, client: PoliteClient):
        self.client = client

    def fetch(self, feed: FeedConfig) -> CollectionResult:
        validate_source_url(str(feed.url), host="finance.naver.com")
        first_log = len(self.client.request_logs)
        response = self.client.get(str(feed.url))
        try:
            if b"<!DOCTYPE" in response.content.upper() or b"<!ENTITY" in response.content.upper():
                raise ValueError
            root = ElementTree.fromstring(response.content)
            if root.tag != "rss" or root.find("channel") is None:
                raise ValueError
        except (ElementTree.ParseError, ValueError):
            raise SourceError("invalid RSS 2.0 response", self.client.request_logs) from None
        documents, skipped = [], []
        for item in root.findall("./channel/item"):
            source_id = self.client.redact(clean_text(_text(item, "guid"))) or None
            link = _text(item, "link")
            if not link:
                skipped.append(SkippedItem(source_id, "missing_link"))
                continue
            try:
                published = parsedate_to_datetime(_text(item, "pubDate"))
            except (TypeError, ValueError, OverflowError):
                skipped.append(SkippedItem(source_id, "invalid_pubdate"))
                continue
            if published.tzinfo is None or published.utcoffset() is None:
                skipped.append(SkippedItem(source_id, "pubdate_timezone_missing"))
                continue
            try:
                document = RawDocument(
                    source=SourceKind.RSS,
                    source_name=self.client.redact(feed.name),
                    source_id=source_id,
                    title=self.client.redact(clean_text(_text(item, "title"))),
                    summary=self.client.safe_summary(_text(item, "description")),
                    url=self.client.public_url(link),
                    published_at=published,
                    fetched_at=response.received_at,
                    related_assets=tuple(
                        self.client.redact(asset) for asset in feed.related_assets
                    ),
                )
            except (ValidationError, ValueError, SourceError):
                skipped.append(SkippedItem(source_id, "invalid_document"))
                continue
            documents.append(document)
        return CollectionResult(
            tuple(documents), self.client.request_logs[first_log:], tuple(skipped)
        )
