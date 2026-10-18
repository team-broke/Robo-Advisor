"""외부 통신 없는 소스 어댑터·robots·간격·비밀값 보호 검증."""

import logging
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from robo_advisor.config import Settings
from robo_advisor.research.documents import SourceKind
from robo_advisor.research.sources import (
    DartAdapter,
    FeedConfig,
    PoliteClient,
    RSSAdapter,
    RobotsDenied,
    SourceError,
    load_sources_config,
    summarize,
)
from robo_advisor.research.sources import dart
from robo_advisor.research.sources.dart import DART_LIST_URL

KEY = "test-sensitive-key-12345"
NOW = datetime(2026, 2, 1, tzinfo=timezone.utc)
RSS_URL = "https://finance.naver.com/fixture.xml"


class FakeClock:
    def __init__(self):
        self.value = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.value

    def sleep(self, seconds):
        assert seconds >= 0
        self.sleeps.append(seconds)
        self.value += seconds

    def utc_now(self):
        return NOW + timedelta(seconds=self.value)


@pytest.fixture
def clients():
    opened = []

    def make(handler, **kwargs):
        clock = FakeClock()
        transport = httpx.MockTransport(handler)
        raw = httpx.Client(transport=transport)
        polite = PoliteClient(
            raw,
            clock=clock.monotonic,
            sleep=clock.sleep,
            utc_now=clock.utc_now,
            **kwargs,
        )
        opened.append((polite, raw))
        return polite, clock

    yield make
    for polite, raw in opened:
        polite.close()
        raw.close()


@pytest.fixture
def configured_key(monkeypatch):
    settings = Settings(_env_file=None, dart_api_key=SecretStr(KEY))
    monkeypatch.setattr(dart, "get_settings", lambda: settings)


def disclosure(page=1):
    return {
        "corp_code": "00123456",
        "corp_name": "테스트 기업",
        "stock_code": "123456",
        "rcept_no": f"2026010500{page:04d}",
        "rcept_dt": "20260105",
        "report_nm": "주요사항보고서",
        "flr_nm": "공시 담당",
        "rm": "참고",
    }


def test_dart_pagination_dates_assets_and_redacted_request_logs(clients, configured_key, caplog):
    requests = []
    caplog.set_level(logging.DEBUG, logger="httpx")

    def handler(request):
        requests.append(request)
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        params = parse_qs(request.url.query.decode())
        assert params["crtfc_key"] == [KEY]
        assert params["page_count"] == ["100"]
        assert params["corp_code"] == ["00123456"]
        assert params["bgn_de"] == ["20260101"]
        page = int(params["page_no"][0])
        return httpx.Response(
            200, json={"status": "000", "total_page": 2, "list": [disclosure(page)]}
        )

    client, clock = clients(handler)
    result = DartAdapter(client).fetch(
        "20260101", "20260131", corp_code="00123456", related_assets={"00123456": ["A"]}
    )
    assert len(result.documents) == 2
    assert len(requests) == 3
    assert clock.sleeps == [1, 1]
    first = result.documents[0]
    assert first.source is SourceKind.DART
    assert first.source_name == "dart"
    assert first.source_id == "20260105000001"
    assert first.related_assets == ("A",)
    assert first.published_precision == "date"
    assert first.published_at == datetime(2026, 1, 4, 15, tzinfo=timezone.utc)
    assert first.fetched_at == NOW + timedelta(seconds=1)
    assert str(first.url).endswith("rcpNo=20260105000001")
    assert set(first.metadata) == {"corp_code", "corp_name", "stock_code", "rcept_no"}
    assert len(first.summary) <= 300
    assert KEY not in repr(result)
    assert KEY not in caplog.text
    for record in result.request_logs:
        assert record.timestamp.tzinfo == timezone.utc
        assert record.host == "opendart.fss.or.kr"
        assert "crtfc_key" not in record.path
        assert record.status_code in (200, 404)
    assert [log.kind for log in result.request_logs] == ["robots", "data", "data"]
    assert [log.interval for log in result.request_logs] == [None, 1, 1]
    assert result.request_logs[0].robots_allowed is None
    assert all(log.robots_allowed for log in result.request_logs[1:])
    assert all(request.headers["User-Agent"] == "RoboAdvisorResearch/0.1" for request in requests)


def test_dart_013_empty_and_optional_corp_code(clients, configured_key):
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        assert "corp_code" not in request.url.params
        return httpx.Response(200, json={"status": "013", "message": "조회 데이터 없음"})

    client, _ = clients(handler)
    result = DartAdapter(client).fetch("20260101", "20260131")
    assert result.documents == ()
    assert len(result.request_logs) == 2


def test_dart_status_error_removes_key_from_message_and_logs(clients, configured_key, caplog):
    caplog.set_level(logging.INFO, logger="httpx")

    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(200, json={"status": "100", "message": f"인증 오류 {KEY}"})

    client, _ = clients(handler)
    with pytest.raises(SourceError, match="인증 오류") as error:
        DartAdapter(client).fetch("20260101", "20260131")
    assert KEY not in str(error.value)
    assert KEY not in repr(error.value.request_logs)
    assert KEY not in caplog.text
    assert len(error.value.request_logs) == 2


def test_dart_echoed_key_is_not_saved_in_documents(clients, configured_key):
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        item = disclosure()
        item.update(corp_name=KEY, report_nm=KEY, rm=KEY)
        return httpx.Response(200, json={"status": "000", "total_page": 1, "list": [item]})

    client, _ = clients(handler)
    result = DartAdapter(client).fetch("20260101", "20260131")
    assert KEY not in result.documents[0].model_dump_json()


def test_missing_dart_key_makes_no_request(clients, monkeypatch):
    monkeypatch.setattr(dart, "get_settings", lambda: Settings(_env_file=None, dart_api_key=None))

    def forbidden(request):
        pytest.fail("missing key must not make a request")

    client, _ = clients(forbidden)
    with pytest.raises(SourceError, match="not configured"):
        DartAdapter(client).fetch("20260101", "20260131")
    assert client.request_logs == ()


def rss_item(
    *,
    link="https://finance.naver.com/article/1",
    published="Mon, 05 Jan 2026 09:30:00 +0900",
):
    return f"""<item><title>제목</title><guid>item-1</guid><link>{link}</link>
    <description><![CDATA[<p>요약 &amp; 내용</p><script>숨김</script>{'가' * 350}]]></description>
    <pubDate>{published}</pubDate></item>"""


def test_rss_html_summary_timezone_and_skipped_items(clients):
    items = [
        rss_item(),
        rss_item(link=""),
        rss_item(published="Mon, 05 Jan 2026 09:30:00"),
        rss_item(published="bad-date"),
    ]
    content = "<rss version='2.0'><channel>" + "".join(items) + "</channel></rss>"

    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(200, content=content.encode())

    client, clock = clients(handler)
    feed = FeedConfig(name="fixture", url=RSS_URL, related_assets=("A",))
    result = RSSAdapter(client).fetch(feed)
    assert len(result.documents) == 1
    document = result.documents[0]
    assert document.source is SourceKind.RSS
    assert document.source_name == "fixture"
    assert document.source_id == "item-1"
    assert document.published_at == datetime(2026, 1, 5, 0, 30, tzinfo=timezone.utc)
    assert document.published_precision == "datetime"
    assert document.fetched_at == clock.utc_now()
    assert document.related_assets == ("A",)
    assert len(document.summary) == 300
    assert "<" not in document.summary and "숨김" not in document.summary
    assert document.summary.startswith("요약 & 내용")
    assert [item.reason for item in result.skipped] == [
        "missing_link",
        "pubdate_timezone_missing",
        "invalid_pubdate",
    ]
    assert len(result.request_logs) == 2


def test_robots_disallow_never_requests_protected_path(clients):
    paths = []

    def handler(request):
        paths.append(request.url.path)
        assert request.url.path == "/robots.txt"
        return httpx.Response(200, text="User-agent: *\nDisallow: /api/\n")

    client, clock = clients(handler)
    with pytest.raises(RobotsDenied) as error:
        client.get(DART_LIST_URL)
    assert paths == ["/robots.txt"]
    assert clock.sleeps == []
    assert error.value.request_logs[-1].kind == "blocked"
    assert error.value.request_logs[-1].robots_allowed is False
    assert error.value.request_logs[-1].status_code is None


@pytest.mark.parametrize("failure", [500, 503, 403, 301, "network"])
def test_robots_failures_deny_and_are_cached(clients, failure):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if failure == "network":
            raise httpx.ConnectError("offline", request=request)
        return httpx.Response(failure)

    client, _ = clients(handler)
    for _ in range(2):
        with pytest.raises(RobotsDenied):
            client.get(DART_LIST_URL)
    assert calls == ["/robots.txt"]
    assert client.request_logs[0].status_code == (None if failure == "network" else failure)


def test_robots_404_cached_host_intervals_and_independent_hosts(clients):
    times = []
    client = clock = None

    def handler(request):
        times.append((request.url.host, request.url.path, clock.value))
        return httpx.Response(404 if request.url.path == "/robots.txt" else 200)

    client, clock = clients(handler)
    client.get(DART_LIST_URL)
    client.get(RSS_URL)
    client.get(DART_LIST_URL)
    assert clock.sleeps == [1, 1]
    assert len(times) == 5
    assert sum(path == "/robots.txt" for _, path, _ in times) == 2
    for host in ("opendart.fss.or.kr", "finance.naver.com"):
        stamps = [stamp for name, _, stamp in times if name == host]
        assert all(right - left >= 1 for left, right in zip(stamps, stamps[1:]))


def test_stricter_robots_crawl_delay_is_respected(clients):
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow:\nCrawl-delay: 3\n")
        return httpx.Response(200)

    client, clock = clients(handler)
    client.get(RSS_URL)
    client.get(RSS_URL)
    assert clock.sleeps == [3, 3]
    assert [log.interval for log in client.request_logs] == [None, 3, 3]


def test_request_failure_is_sanitized_and_still_throttled(clients, configured_key, caplog):
    caplog.set_level(logging.DEBUG, logger="httpx")

    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        raise httpx.ReadTimeout(f"failed with {KEY}", request=request)

    client, clock = clients(handler)
    for _ in range(2):
        with pytest.raises(SourceError, match="request failed") as error:
            DartAdapter(client).fetch("20260101", "20260131")
        assert KEY not in repr(error.value)
        assert error.value.request_logs[-1].status_code is None
    assert clock.sleeps == [1, 1]
    assert [log.interval for log in client.request_logs] == [None, 1, 1]
    assert KEY not in caplog.text


def test_redirect_is_not_followed(clients):
    requests = []

    def handler(request):
        requests.append(request.url)
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(302, headers={"Location": "https://example.org/private"})

    client, _ = clients(handler)
    with pytest.raises(SourceError, match="HTTP 302"):
        client.get(RSS_URL)
    assert len(requests) == 2
    assert all(url.host == "finance.naver.com" for url in requests)


@pytest.mark.parametrize("interval", [0.5, 0, float("nan"), float("inf"), True])
def test_interval_less_than_one_or_invalid_rejected(clients, interval):
    with pytest.raises(ValueError):
        clients(lambda request: httpx.Response(404), interval=interval)


@pytest.mark.parametrize(
    "url",
    [
        "https://example.org/rss",
        "http://finance.naver.com/rss",
        "https://finance.naver.com:8443/rss",
        "https://user:pass@finance.naver.com/rss",
        "https://finance.naver.com.evil.org/rss",
    ],
)
def test_unapproved_feeds_rejected_without_requests(url):
    with pytest.raises(ValidationError):
        FeedConfig(name="fixture", url=url)


def test_summary_cleanup_and_configuration(tmp_path):
    assert summarize("<p>A</p>\n <b>B &amp; C</b><style>hidden</style>") == "A B & C"
    assert len(summarize("가" * 301)) == 300
    path = tmp_path / "sources.yaml"
    path.write_text("request_interval: 1.0\nuser_agent: ResearchTest/1.0\nrss_feeds: []\n")
    config = load_sources_config(path)
    assert config.rss_feeds == ()
    assert config.dart.page_count == 100
    path.write_text("request_interval: 0.5\n")
    with pytest.raises(ValidationError):
        load_sources_config(path)


@pytest.mark.parametrize("page_count", [0, 101, True, 1.5])
def test_invalid_page_count(clients, page_count):
    client, _ = clients(lambda request: httpx.Response(404))
    with pytest.raises(ValueError):
        DartAdapter(client, page_count=page_count)


@pytest.mark.parametrize("case", ["malformed_json", "http_error", "malformed_item"])
def test_invalid_dart_response_never_exposes_secret(clients, configured_key, case):
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        if case == "malformed_json":
            return httpx.Response(200, content=KEY.encode())
        if case == "http_error":
            return httpx.Response(401, text=KEY)
        return httpx.Response(
            200, json={"status": "000", "total_page": 1, "list": [{"report_nm": KEY}]}
        )

    client, _ = clients(handler)
    with pytest.raises(SourceError) as error:
        DartAdapter(client).fetch("20260101", "20260131")
    assert KEY not in str(error.value)
    assert KEY not in repr(error.value.request_logs)


def test_fetched_timestamp_is_after_response_receipt(clients, configured_key):
    clock = None

    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        clock.value += 0.25
        return httpx.Response(200, json={"status": "000", "total_page": 1, "list": [disclosure()]})

    client, clock = clients(handler)
    result = DartAdapter(client).fetch("20260101", "20260131")
    assert result.documents[0].fetched_at == NOW + timedelta(seconds=1.25)
    assert result.request_logs[-1].timestamp == NOW + timedelta(seconds=1)


def test_robots_request_rate_can_increase_interval(clients):
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nRequest-rate: 1/4\nDisallow:\n")
        return httpx.Response(200)

    client, clock = clients(handler)
    client.get(RSS_URL)
    assert clock.sleeps == [4]


@pytest.mark.parametrize("content", ["<broken", "<html/>", "<!DOCTYPE rss><rss><channel/></rss>"])
def test_malformed_rss_rejected_without_storing_body(clients, content):
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(200, content=content.encode())

    client, _ = clients(handler)
    with pytest.raises(SourceError, match="invalid RSS") as error:
        RSSAdapter(client).fetch(FeedConfig(name="fixture", url=RSS_URL))
    assert content not in str(error.value)


def test_closed_client_and_unapproved_host_make_no_request(clients):
    def forbidden(request):
        pytest.fail("closed or unapproved request must not be sent")

    client, _ = clients(forbidden)
    with pytest.raises(SourceError):
        client.get("https://example.org/rss")
    client.close()
    with pytest.raises(SourceError, match="closed"):
        client.get(RSS_URL)
