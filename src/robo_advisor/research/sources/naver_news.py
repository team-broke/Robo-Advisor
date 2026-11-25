"""Request-time NAVER search results. This adapter has no ingestion or cache path."""

from datetime import datetime
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, HttpUrl, ValidationError

from robo_advisor.research.sources.http import clean_text
from robo_advisor.schemas.common import UTCDateTime

NEWS_URL = "https://openapi.naver.com/v1/search/news.json"


class NewsItem(BaseModel):
    title: str
    summary: str
    published_at: UTCDateTime
    url: HttpUrl
    original_url: HttpUrl | None = None


class NewsResults(BaseModel):
    source: str = "NAVER 검색"
    items: list[NewsItem]


class NewsUnavailable(RuntimeError):
    """An upstream status safe to expose without headers or raw response data."""

    def __init__(self, status_code):
        super().__init__("NAVER news search unavailable")
        self.status_code = status_code


def _public_url(value):
    if not isinstance(value, str) or not value:
        return None
    parts = urlsplit(value)
    if parts.scheme not in {"https", "http"} or not parts.hostname or parts.username is not None:
        return None
    return value


class NaverNewsSearch:
    def __init__(self, client_id, client_secret, *, client=None):
        self._client_id = client_id
        self._client_secret = client_secret
        self._client = client

    def search(self, query, limit=5):
        query = query.strip()
        if not 1 <= len(query) <= 100 or type(limit) is not int or not 1 <= limit <= 10:
            raise ValueError("invalid news search parameters")
        arguments = {
            "headers": {
                "X-Naver-Client-Id": self._client_id.get_secret_value(),
                "X-Naver-Client-Secret": self._client_secret.get_secret_value(),
            },
            "params": {"query": query, "display": limit, "sort": "date"},
            "timeout": 8,
            "follow_redirects": False,
        }
        try:
            if self._client is None:
                with httpx.Client(trust_env=False) as client:
                    response = client.get(NEWS_URL, **arguments)
            else:
                response = self._client.get(NEWS_URL, **arguments)
            if response.status_code != 200:
                raise NewsUnavailable(429 if response.status_code == 429 else 502)
            rows = response.json()["items"]
            if not isinstance(rows, list) or len(rows) > limit:
                raise ValueError
            items = []
            for row in rows:
                original = _public_url(row.get("originallink"))
                url = original or _public_url(row.get("link"))
                stamp: datetime = parsedate_to_datetime(row["pubDate"])
                items.append(
                    NewsItem(
                        title=clean_text(row["title"]),
                        summary=clean_text(row["description"]),
                        published_at=stamp,
                        url=url,
                        original_url=original,
                    )
                )
            return NewsResults(items=items)
        except httpx.TimeoutException:
            raise NewsUnavailable(504) from None
        except (httpx.HTTPError, ValueError, TypeError, AttributeError, KeyError, ValidationError):
            raise NewsUnavailable(502) from None
