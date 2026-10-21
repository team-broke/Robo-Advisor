"""DART 공시검색의 기간 이력과 페이지네이션 수집."""

import json
import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from robo_advisor.config import get_settings
from robo_advisor.research.documents import RawDocument, SourceKind
from robo_advisor.research.sources.http import (
    CollectionResult,
    PoliteClient,
    SourceError,
    clean_text,
)

DART_LIST_URL = "https://opendart.fss.or.kr/api/list.json"


def _date(value: date | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{8}", value):
        raise ValueError("DART dates must be dates or YYYYMMDD strings")
    return datetime.strptime(value, "%Y%m%d").date()


class DartAdapter:
    """자산 매핑은 corp_code별 related_assets 인자로만 받는다."""

    def __init__(self, client: PoliteClient, *, page_count: int = 100):
        if (
            isinstance(page_count, bool)
            or not isinstance(page_count, int)
            or not 1 <= page_count <= 100
        ):
            raise ValueError("DART page_count must be an integer in [1, 100]")
        self.client = client
        self.page_count = page_count

    def fetch(
        self,
        start_date: date | str,
        end_date: date | str,
        *,
        corp_code: str | None = None,
        related_assets: Mapping[str, Sequence[str]] | None = None,
    ) -> CollectionResult:
        secret = get_settings().dart_api_key
        if secret is None or not secret.get_secret_value().strip():
            raise SourceError("DART API key is not configured")
        key = secret.get_secret_value()
        self.client.register_secret(key)
        start, end = _date(start_date), _date(end_date)
        if start > end:
            raise ValueError("DART start date must not follow end date")
        if corp_code is not None and (
            not isinstance(corp_code, str) or not re.fullmatch(r"[0-9]{8}", corp_code)
        ):
            raise ValueError("corp_code must have eight digits")
        if related_assets is not None and not isinstance(related_assets, Mapping):
            raise ValueError("related_assets must map corp_code to asset sequences")
        mapping = related_assets or {}
        first_log = len(self.client.request_logs)
        documents = []
        page, total_pages = 1, 1
        while page <= total_pages:
            params = {
                "crtfc_key": key,
                "bgn_de": start.strftime("%Y%m%d"),
                "end_de": end.strftime("%Y%m%d"),
                "page_no": page,
                "page_count": self.page_count,
            }
            if corp_code is not None:
                params["corp_code"] = corp_code
            response = self.client.get(DART_LIST_URL, params=params)
            try:
                payload = json.loads(response.content)
                if not isinstance(payload, dict):
                    raise ValueError
                status = payload.get("status")
                if status == "013":
                    break
                if status != "000":
                    message = self.client.safe_summary(str(payload.get("message", "DART error")))
                    raise SourceError(f"DART API error: {message}", self.client.request_logs)
                count = payload["total_page"]
                if isinstance(count, bool) or not isinstance(count, int) or count < page:
                    raise ValueError
                if page > 1 and count != total_pages:
                    raise ValueError
                total_pages = count
                items = payload.get("list", [])
                if not isinstance(items, list):
                    raise ValueError
                for item in items:
                    if not isinstance(item, dict):
                        raise ValueError
                    receipt = str(item["rcept_no"])
                    if not re.fullmatch(r"[0-9]{14}", receipt):
                        raise ValueError
                    published = _date(item["rcept_dt"])
                    if not start <= published <= end:
                        raise ValueError
                    values = {
                        name: self.client.redact(clean_text(str(item.get(name, ""))))
                        for name in ("corp_code", "corp_name", "stock_code", "rcept_no")
                    }
                    report = self.client.redact(clean_text(str(item["report_nm"])))
                    summary = " ".join(
                        str(item.get(name, ""))
                        for name in ("corp_name", "report_nm", "flr_nm", "rm")
                    )
                    assets = mapping.get(str(item.get("corp_code", "")), ())
                    if isinstance(assets, str):
                        raise ValueError
                    published_at = datetime.combine(
                        published, time(), tzinfo=ZoneInfo("Asia/Seoul")
                    )
                    documents.append(
                        RawDocument(
                            source=SourceKind.DART,
                            source_name="dart",
                            source_id=self.client.redact(receipt),
                            title=report,
                            summary=self.client.safe_summary(summary),
                            url=self.client.public_url(
                                f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={receipt}"
                            ),
                            published_at=published_at,
                            published_precision="date",
                            fetched_at=response.received_at,
                            related_assets=tuple(self.client.redact(asset) for asset in assets),
                            metadata=values,
                        ),
                    )
            except SourceError:
                raise
            except (ValueError, TypeError, KeyError, OverflowError):
                raise SourceError("invalid DART response", self.client.request_logs) from None
            page += 1
        return CollectionResult(tuple(documents), self.client.request_logs[first_log:])
