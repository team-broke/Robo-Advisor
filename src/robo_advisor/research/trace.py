"""내부 추론 없이 공개할 실행 기록과 민감정보 제거."""

from __future__ import annotations

import re
from typing import Literal
from urllib.parse import parse_qsl, quote, quote_plus, unquote, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field

from robo_advisor.schemas.common import NonEmptyString

_SECRET_KEY = re.compile(
    r"api.?key|authorization|bearer|cookie|password|secret|credential|token|email|phone", re.I
)
_PATTERNS = (
    re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
    re.compile(r"(?<!\w)(?:\+\d{10,15}|0\d{9,10})(?!\w)"),
    re.compile(r"(?<!\w)(?:\+\d{1,3}[ .-]?)?(?:\(?\d{2,4}\)?[ .-]){2}\d{3,4}(?!\w)"),
    re.compile(r"(?i)\b(?:sk-(?:ant-)?|gh[pousr]_|github_pat_)[A-Za-z0-9_-]+"),
    re.compile(r"(?i)\bBearer\s+[^\s,;\"']+"),
    re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),
    re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|token|authorization|cookie|password|secret)"
        r"[\"']?\s*[:=]\s*[\"']?[^\s&\"']+"
    ),
)


def redact(value, *, secrets=()):
    """문자열·중첩 JSON에서 키·연락처·토큰을 제거하고 해시 ID는 보존한다."""
    if isinstance(value, dict):
        return {
            redact(str(key), secrets=secrets): (
                "[redacted]" if _SECRET_KEY.search(str(key)) else redact(item, secrets=secrets)
            )
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [redact(item, secrets=secrets) for item in value]
    if not isinstance(value, str):
        return value
    encoded = {
        form for item in secrets if item for form in (item, quote(item, safe=""), quote_plus(item))
    }
    for secret in sorted(encoded, key=len, reverse=True):
        value = value.replace(secret, "[redacted]")
    for pattern in _PATTERNS:
        value = pattern.sub("[redacted]", value)
    return re.sub(
        r"(?<![\w-])[A-Za-z0-9_-]{32,}(?![\w-])",
        lambda match: (
            match[0]
            if re.fullmatch(r"[a-f0-9]{64}|[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}", match[0])
            else "[redacted]"
        ),
        value,
    )


def public_url(value: str, *, secrets=()) -> str:
    """인증 정보·비밀 쿼리·fragment를 버린 공개 출처 URL을 만든다."""
    parts = urlsplit(value)
    if parts.username is not None or parts.password is not None:
        raise ValueError("source URL cannot contain credentials")
    query = [
        (key, redact(item, secrets=secrets))
        for key, item in parse_qsl(parts.query, keep_blank_values=True)
        if not _SECRET_KEY.search(key)
    ]
    if (
        redact(unquote(parts.path), secrets=secrets) != unquote(parts.path)
        or redact(parts.netloc, secrets=secrets) != parts.netloc
    ):
        raise ValueError("source URL path or host contains sensitive data")
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))


class TraceStep(BaseModel):
    """단계·도구·질의·출처·판정만 담는 공개 실행 메타데이터."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    step: Literal["plan", "retrieve", "verify", "rewrite", "synthesize"]
    tool: Literal["llm", "vector_store"]
    query: str
    source_ids: tuple[NonEmptyString, ...] = ()
    validation: tuple[NonEmptyString, ...]
    retries: int = Field(ge=0)
    elapsed_ms: float = Field(ge=0)
