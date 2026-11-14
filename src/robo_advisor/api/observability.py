"""요청·리서치·태그·결정의 상관 ID와 민감정보 없는 공개 실행 로그."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import json
import logging
import re
from threading import RLock
from uuid import uuid4
from urllib.parse import unquote

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from robo_advisor.api.schemas import ErrorResponse
from robo_advisor.api.services import NotFoundError, ServiceUnavailableError
from robo_advisor.config import get_settings
from robo_advisor.research.trace import redact
from robo_advisor.schemas.risk import RiskTag

_CORRELATION = ContextVar("api_correlation_id", default=None)
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_PRIVATE = re.compile(
    r"reasoning|thoughts?|chain.of.thought|prompt|completion|raw.response|messages", re.I
)
_PRIVATE_TEXT = re.compile(
    r"(?:reasoning|thoughts?|chain.of.thought|prompt|completion|raw.response)[\"']?\s*[:=]", re.I
)
LOGGER = logging.getLogger("robo_advisor.api")
LOGGER.setLevel(logging.INFO)


def get_correlation_id():
    return _CORRELATION.get()


@contextmanager
def correlation_context(value):
    """스레드 작업에서도 호출 단위의 ID를 유지하고 종료 시 복원한다."""
    token = _CORRELATION.set(value)
    try:
        yield
    finally:
        _CORRELATION.reset(token)


class SensitiveLogFilter(logging.Filter):
    """개인정보·키를 마스킹하고 내부 추론·프롬프트·예외 원문은 기록하지 않는다."""

    def __init__(self, secrets=()):
        super().__init__()
        self._secrets, self._lock = tuple(secrets), RLock()

    def register(self, secrets):
        with self._lock:
            self._secrets = tuple(set(self._secrets) | {value for value in secrets if value})

    def clean(self, value, depth=0):
        if depth > 8:
            return "[omitted]"
        if isinstance(value, dict):
            value = {
                str(key): "[omitted]" if _PRIVATE.search(str(key)) else self.clean(item, depth + 1)
                for key, item in value.items()
            }
        elif isinstance(value, (list, tuple)):
            value = [self.clean(item, depth + 1) for item in value]
        elif isinstance(value, str):
            value = unquote(value)
            if _PRIVATE_TEXT.search(value):
                return "[private content omitted]"
        elif value is not None and not isinstance(value, (bool, int, float)):
            return self.clean(str(value), depth + 1)
        return redact(value, secrets=self._secrets)

    def valid_id(self, value):
        return (
            isinstance(value, str)
            and _ID.fullmatch(value) is not None
            and self.clean(value) == value
        )

    def filter(self, record):
        access_args = (
            record.args
            if record.name == "uvicorn.access"
            and isinstance(record.args, tuple)
            and len(record.args) == 5
            else None
        )
        correlation = get_correlation_id() or getattr(record, "correlation_id", None)
        if not self.valid_id(correlation):
            correlation = None
        try:
            if isinstance(record.msg, dict):
                message = json.dumps(self.clean(record.msg), ensure_ascii=False, allow_nan=False)
            else:
                message = self.clean(record.getMessage())
            message = re.sub(r"^\[correlation_id=[^\]]*\]\s*", "", message)
            for name, value in list(record.__dict__.items()):
                if name not in {"msg", "args", "exc_info", "exc_text", "stack_info"}:
                    record.__dict__[name] = (
                        "[omitted]" if _PRIVATE.search(name) else self.clean(value)
                    )
        except Exception:
            message = "[log content omitted]"
        record.msg = f"[correlation_id={correlation or '-'}] {message}"
        record.args = ()
        if access_args is not None:
            record.msg = "[correlation_id=%s] " % (correlation or "-") + '%s - "%s %s HTTP/%s" %s'
            record.args = tuple(self.clean(value) for value in access_args)
        record.exc_info = record.exc_text = record.stack_info = None
        record.correlation_id = correlation
        return True


_FILTER = SensitiveLogFilter()


class _AccessFormatter(logging.Formatter):
    def __init__(self, original):
        super().__init__()
        self.original = original or logging.Formatter()

    def format(self, record):
        text = self.original.format(record)
        return f"[correlation_id={record.correlation_id or '-'}] {_FILTER.clean(text)}"


def sanitize_log(value):
    return _FILTER.clean(value)


def _install_filters():
    loggers = [
        logging.getLogger(),
        LOGGER,
        *(
            logging.getLogger(name)
            for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "httpx", "httpcore")
        ),
    ]
    if not LOGGER.hasHandlers():
        LOGGER.addHandler(logging.StreamHandler())
    for logger in loggers:
        if _FILTER not in logger.filters:
            logger.addFilter(_FILTER)
        for handler in logger.handlers:
            if logger.name == "uvicorn.access" and not isinstance(
                handler.formatter, _AccessFormatter
            ):
                handler.setFormatter(_AccessFormatter(handler.formatter))
            if _FILTER not in handler.filters:
                handler.addFilter(_FILTER)


def log_event(event, **fields):
    """원문 질의·모델 응답 대신 ID·상태·공개 출처 식별자만 기록한다."""
    _install_filters()
    safe = _FILTER.clean({"event": event, "correlation_id": get_correlation_id(), **fields})
    LOGGER.info(json.dumps(safe, ensure_ascii=False, allow_nan=False), extra={"audit": safe})


def log_research_result(result, *, job_id=None, model_id=None):
    log_event(
        "research_result",
        job_id=job_id,
        model_id=model_id,
        trace_id=result.trace_id,
        source_ids=sorted(
            {citation.doc_id for claim in result.claims for citation in claim.citations}
            | {source_id for step in result.trace for source_id in step.source_ids}
        ),
        status=result.status,
    )


def visible_risk_tags(tags, as_of):
    """환경과 같은 규칙으로 가장 최근 가용 시각의 최대 위험 태그만 선택한다."""
    selected = {}
    for tag in tags:
        if not isinstance(tag, RiskTag) or tag.available_at > as_of:
            continue
        previous = selected.get(tag.asset, ())
        if (
            not previous
            or tag.available_at > previous[0].available_at
            or (
                tag.available_at == previous[0].available_at
                and tag.risk_score > previous[0].risk_score
            )
        ):
            selected[tag.asset] = (tag,)
        elif (
            tag.available_at == previous[0].available_at
            and tag.risk_score == previous[0].risk_score
        ):
            selected[tag.asset] = (*previous, tag)
    return tuple(tag for group in selected.values() for tag in group)


def log_risk_tags(tags, *, model_id=None):
    for tag in tags:
        log_event(
            "risk_tag",
            model_id=model_id,
            trace_id=tag.trace_id,
            asset=tag.asset,
            as_of=tag.available_at.isoformat(),
        )


def log_policy_decision(optimizer, as_of):
    tags = visible_risk_tags(getattr(optimizer, "risk_tags", ()) or (), as_of)
    log_risk_tags(tags, model_id=optimizer.model_id)
    log_event(
        "policy_decision",
        model_id=optimizer.model_id,
        as_of=as_of.isoformat(),
        risk_tag_trace_ids=sorted({tag.trace_id for tag in tags}),
    )


def _error(status_code, correlation_id):
    code, message = {
        404: ("not_found", "Requested resource was not found"),
        422: ("validation_error", "Request validation failed"),
        503: ("service_unavailable", "Service is unavailable"),
        500: ("internal_error", "Internal server error"),
    }.get(status_code, ("http_error", "Request could not be completed"))
    return JSONResponse(
        ErrorResponse(code=code, message=message, correlation_id=correlation_id).model_dump(),
        status_code=status_code,
    )


class CorrelationMiddleware:
    """생성·검증한 ID를 ASGI 응답 헤더와 오류 응답에 일관되게 전달한다."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        _install_filters()
        values = [value for key, value in scope["headers"] if key.lower() == b"x-correlation-id"]
        try:
            supplied = values[0].decode("ascii") if len(values) == 1 else None
        except UnicodeError:
            supplied = None
        correlation_id = supplied if _FILTER.valid_id(supplied) else str(uuid4())
        scope.setdefault("state", {})["correlation_id"] = correlation_id
        started, status_code = False, 500

        async def correlated_send(message):
            nonlocal started, status_code
            if message["type"] == "http.response.start":
                started, status_code = True, message["status"]
                message = {
                    **message,
                    "headers": [
                        (key, value)
                        for key, value in message.get("headers", [])
                        if key.lower() != b"x-correlation-id"
                    ]
                    + [(b"x-correlation-id", correlation_id.encode("ascii"))],
                }
            await send(message)

        with correlation_context(correlation_id):
            try:
                await self.app(scope, receive, correlated_send)
            except Exception as error:
                log_event("request_error", status_code=500, error_type=type(error).__name__)
                if started:
                    raise
                await _error(500, correlation_id)(scope, receive, correlated_send)
            finally:
                route = getattr(scope.get("route"), "path", "unmatched")
                log_event("request_finished", route=route, status_code=status_code)


def install_observability(app):
    settings = get_settings()
    _FILTER.register(
        [
            value.get_secret_value()
            for value in (settings.llm_api_key, settings.dart_api_key)
            if value
        ]
    )
    _install_filters()
    app.add_middleware(CorrelationMiddleware)

    async def http_error(request, error):
        status_code = (
            error.status_code
            if isinstance(error, HTTPException)
            else (404 if isinstance(error, NotFoundError) else 503)
        )
        return _error(status_code, request.state.correlation_id)

    async def invalid_request(request, error):
        return _error(422, request.state.correlation_id)

    for kind in (HTTPException, NotFoundError, ServiceUnavailableError):
        app.add_exception_handler(kind, http_error)
    app.add_exception_handler(RequestValidationError, invalid_request)
