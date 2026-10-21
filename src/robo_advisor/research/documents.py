"""수집기와 정제·색인 단계가 공유하는 원천 문서 계약.

원문 재배포 금지에 따라 제목, 300자 이내 요약, 게시 시각, 원문 URL과 출처만
저장한다. published_at은 출처가 밝힌 게시 시각이고 fetched_at은 우리가 실제로
받은 시각이다. 백테스트에서 쓸 수 있는 시점(available_at)은 정제 단계가 정한다.
DART 접수일처럼 시각이 없는 출처는 published_precision="date"로 표시한다.
"""

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, StringConstraints

from robo_advisor.schemas.common import NonEmptyString, UTCDateTime

SUMMARY_MAX_LENGTH = 300

Summary = Annotated[str, StringConstraints(strip_whitespace=True, max_length=SUMMARY_MAX_LENGTH)]


class SourceKind(str, Enum):
    """확정된 허용 출처 유형."""

    DART = "dart"
    RSS = "rss"


class RawDocument(BaseModel):
    """수집기 한 건의 출력. 원문 본문은 담지 않는다."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: SourceKind
    source_name: NonEmptyString
    source_id: NonEmptyString | None = None
    title: NonEmptyString
    summary: Summary = ""
    url: HttpUrl
    published_at: UTCDateTime
    published_precision: Literal["date", "datetime"] = "datetime"
    fetched_at: UTCDateTime
    related_assets: tuple[NonEmptyString, ...] = ()
    metadata: dict[str, str] = Field(default_factory=dict)
