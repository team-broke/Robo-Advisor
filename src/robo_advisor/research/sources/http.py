"""허용 호스트의 robots 정책과 최소 요청 간격을 지키는 공통 HTTP 클라이언트.

robots.txt 404는 허용, 네트워크 오류와 5xx 및 그 밖의 비정상 응답은 금지한다.
robots는 호스트별 한 번만 읽고 실패도 캐시한다. robots 조회와 실패한 요청도
간격 제한에 포함한다. 리다이렉트와 자동 재시도는 하지 않는다. crawl-delay와
request-rate가 더 엄격하면 간격을 늘린다. 원문 본문은 수집 결과에 저장하지 않는다.
"""

from __future__ import annotations

import html
import logging
import math
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Literal
from urllib.parse import parse_qsl, quote, quote_plus, urlencode, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx
import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator

from robo_advisor.research.documents import SUMMARY_MAX_LENGTH, RawDocument
from robo_advisor.schemas.common import NonEmptyString

ALLOWED_HOSTS = frozenset({"opendart.fss.or.kr", "finance.naver.com"})
SECRET_KEYS = frozenset({"crtfc_key", "api_key", "access_token", "token", "secret", "key"})


@dataclass(frozen=True)
class RequestLog:
    """UTC 요청 시작 시각과 직전 같은 호스트 요청 시작과의 간격.

    robots 조회의 판정은 None이며 blocked는 실제 요청이 없어 상태·간격이 None이다.
    """

    timestamp: datetime
    host: str
    path: str
    status_code: int | None
    robots_allowed: bool | None
    interval: float | None
    kind: Literal["robots", "data", "blocked"]


@dataclass(frozen=True)
class SkippedItem:
    source_id: str | None
    reason: str


@dataclass(frozen=True)
class CollectionResult:
    documents: tuple[RawDocument, ...]
    request_logs: tuple[RequestLog, ...]
    skipped: tuple[SkippedItem, ...] = ()


class SourceError(RuntimeError):
    """민감한 요청 정보 없이 실패 사유와 요청 기록을 제공한다."""

    def __init__(self, message: str, request_logs=()):
        super().__init__(message)
        self.request_logs = tuple(request_logs)


class RobotsDenied(SourceError):
    pass


@dataclass(frozen=True)
class HttpResult:
    content: bytes = field(repr=False)
    status_code: int
    received_at: datetime


class _PlainText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1
        self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in {"script", "style"} and self.hidden:
            self.hidden -= 1
        self.parts.append(" ")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def clean_text(value: str) -> str:
    """태그·스크립트·스타일을 제거하고 공백을 한 칸으로 정리한다."""
    parser = _PlainText()
    parser.feed(html.unescape(value))
    parser.close()
    return " ".join("".join(parser.parts).split())


def summarize(value: str) -> str:
    return clean_text(value)[:SUMMARY_MAX_LENGTH]


def validate_source_url(value: str, *, host: str | None = None) -> str:
    """HTTPS 허용 호스트만 받으며 인증 정보와 임의 포트를 금지한다."""
    try:
        parts = urlsplit(value)
        allowed = parts.hostname in ALLOWED_HOSTS and (host is None or parts.hostname == host)
        if (
            parts.scheme != "https"
            or not allowed
            or parts.port not in (None, 443)
            or parts.username is not None
            or parts.password is not None
            or parts.fragment
        ):
            raise ValueError
    except (ValueError, TypeError):
        raise SourceError("source URL requires an allowed HTTPS host without credentials") from None
    return value


class _LogRedaction(logging.Filter):
    def __init__(self, redact):
        super().__init__()
        self.redact = redact

    def filter(self, record):
        record.msg = self.redact(record.getMessage())
        record.args = ()
        if record.exc_info:
            record.exc_text = self.redact(logging.Formatter().formatException(record.exc_info))
            record.exc_info = None
        return True


class PoliteClient:
    """시계·sleep·HTTP 전송을 주입할 수 있는 동기식 수집 클라이언트.

    요청 시작 사이의 간격을 monotonic 시계로 관리하며 UTC 시각을 따로 기록한다.
    HTTP 전송 계층의 URL 로그도 필터로 비밀값을 제거한다. close는 직접 생성한
    HTTP 클라이언트만 닫는다. 동일 인스턴스의 동시 호출은 잠금으로 직렬화한다.
    주입한 HTTP 클라이언트도 자동 재시도 없는 전송 계층을 사용해야 한다.
    여러 어댑터의 요청 제한을 합산하려면 같은 PoliteClient를 공유한다.
    """

    def __init__(
        self,
        client: httpx.Client | None = None,
        *,
        interval: float = 1.0,
        user_agent: str = "RoboAdvisorResearch/0.1",
        timeout: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        utc_now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        if isinstance(interval, bool) or not math.isfinite(interval) or interval < 1:
            raise ValueError("request interval must be finite and at least one second")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if not user_agent.strip() or "\r" in user_agent or "\n" in user_agent:
            raise ValueError("User-Agent must be nonempty and single-line")
        self.interval = float(interval)
        self.user_agent = user_agent
        self.timeout = float(timeout)
        self._client = client if client is not None else httpx.Client()
        self._owns_client = client is None
        self._clock, self._sleep, self._utc_now = clock, sleep, utc_now
        self._last = {}
        self._robots = {}
        self._delays = {}
        self._logs = []
        self._secrets = ()
        self._closed = False
        self._lock = threading.RLock()
        self._filter = _LogRedaction(self.redact)
        names = (
            "httpx",
            "httpcore",
            "httpcore.connection",
            "httpcore.connection_pool",
            "httpcore.http11",
            "httpcore.http2",
            "httpcore.proxy",
            "httpcore.http_proxy",
            "httpcore.socks_proxy",
        )
        self._loggers = [logging.getLogger(name) for name in names]
        for logger in self._loggers:
            logger.addFilter(self._filter)

    @property
    def request_logs(self) -> tuple[RequestLog, ...]:
        return tuple(self._logs)

    def register_secret(self, value: str) -> None:
        if value:
            with self._lock:
                encoded = {value, quote(value, safe=""), quote_plus(value)}
                self._secrets = tuple(set(self._secrets) | encoded)

    def redact(self, value: str) -> str:
        text = str(value)
        for secret in sorted(self._secrets, key=len, reverse=True):
            text = text.replace(secret, "[redacted]")
        pattern = r"(?i)((?:crtfc_key|api_key|access_token|token|secret|key)=)[^&\s\"'<>]+"
        return re.sub(pattern, r"\1[redacted]", text)

    def safe_summary(self, value: str) -> str:
        return self.redact(clean_text(value))[:SUMMARY_MAX_LENGTH]

    def public_url(self, value: str) -> str:
        """출력 URL과 요청 기록에서 비밀 쿼리·인증 정보·fragment를 제거한다."""
        parts = urlsplit(value)
        if parts.username is not None or parts.password is not None:
            raise SourceError("URL credentials are not allowed")
        query = [
            (key, val) for key, val in parse_qsl(parts.query) if key.lower() not in SECRET_KEYS
        ]
        public = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))
        return self.redact(public)

    def _now(self):
        value = self._utc_now()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("utc_now must return a timezone-aware datetime")
        return value.astimezone(timezone.utc)

    def _path(self, url):
        parts = urlsplit(self.public_url(str(url)))
        return parts.path + ("?" + parts.query if parts.query else "")

    def _send(self, url, kind, robots_allowed):
        host = url.host
        delay = self._delays.get(host, self.interval)
        previous = self._last.get(host)
        while previous is not None and self._clock() - previous < delay:
            self._sleep(delay - (self._clock() - previous))
        start = self._clock()
        interval = None if previous is None else start - previous
        timestamp = self._now()
        self._last[host] = start
        status = None
        failed = False
        result = None
        try:
            response = self._client.get(
                url,
                headers={"User-Agent": self.user_agent},
                timeout=self.timeout,
                follow_redirects=False,
            )
            status = response.status_code
            result = HttpResult(response.content, status, self._now())
        except Exception:
            failed = True
        finally:
            self._logs.append(
                RequestLog(timestamp, host, self._path(url), status, robots_allowed, interval, kind)
            )
        if failed:
            raise SourceError("source request failed", self.request_logs) from None
        return result

    def _read_robots(self, url):
        host = url.host
        robots_url = httpx.URL(f"https://{host}/robots.txt")
        try:
            result = self._send(robots_url, "robots", None)
        except SourceError:
            self._robots[host] = False
            return
        if result.status_code == 404:
            self._robots[host] = True
        elif result.status_code == 200:
            parser = RobotFileParser(str(robots_url))
            parser.parse(result.content.decode("utf-8", errors="replace").splitlines())
            self._robots[host] = parser
            delay = parser.crawl_delay(self.user_agent) or 0
            rate = parser.request_rate(self.user_agent)
            rate_delay = rate.seconds / rate.requests if rate and rate.requests > 0 else 0
            self._delays[host] = max(self.interval, delay, rate_delay)
        else:
            self._robots[host] = False

    def get(self, url: str, *, params=None) -> HttpResult:
        validate_source_url(url)
        try:
            target = httpx.URL(url)
            if params is not None:
                target = target.copy_merge_params(params)
        except (httpx.InvalidURL, ValueError, TypeError):
            raise SourceError("invalid source request URL") from None
        with self._lock:
            if self._closed:
                raise SourceError("source client is closed")
            if target.host not in self._robots:
                self._read_robots(target)
            policy = self._robots[target.host]
            if isinstance(policy, bool):
                allowed = policy
            else:
                allowed = policy.can_fetch(self.user_agent, str(target))
            if not allowed:
                self._logs.append(
                    RequestLog(
                        self._now(), target.host, self._path(target), None, False, None, "blocked"
                    )
                )
                raise RobotsDenied("robots policy forbids this request", self.request_logs)
            result = self._send(target, "data", True)
            if not 200 <= result.status_code < 300:
                raise SourceError(f"source returned HTTP {result.status_code}", self.request_logs)
            return result

    def close(self):
        with self._lock:
            self._closed = True
            if self._owns_client:
                self._client.close()
            for logger in self._loggers:
                logger.removeFilter(self._filter)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class FeedConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    name: NonEmptyString
    url: HttpUrl
    related_assets: tuple[NonEmptyString, ...] = ()

    @field_validator("url")
    @classmethod
    def naver_finance_only(cls, value):
        try:
            validate_source_url(str(value), host="finance.naver.com")
            if any(key.lower() in SECRET_KEYS for key, _ in parse_qsl(urlsplit(str(value)).query)):
                raise ValueError
        except (SourceError, ValueError):
            raise ValueError("RSS feed must be an allowed public Naver finance URL") from None
        return value


class DartOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    page_count: int = Field(default=100, ge=1, le=100, strict=True)


class SourcesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, hide_input_in_errors=True)
    request_interval: float = Field(default=1.0, ge=1)
    user_agent: NonEmptyString = "RoboAdvisorResearch/0.1"
    timeout: float = Field(default=10.0, gt=0)
    dart: DartOptions = Field(default_factory=DartOptions)
    rss_feeds: tuple[FeedConfig, ...] = ()

    @field_validator("user_agent")
    @classmethod
    def single_line(cls, value):
        if "\r" in value or "\n" in value:
            raise ValueError("User-Agent must be single-line")
        return value


def load_sources_config(path: str | Path = "config/research_sources.yaml") -> SourcesConfig:
    with Path(path).open(encoding="utf-8") as stream:
        return SourcesConfig.model_validate(yaml.safe_load(stream))
