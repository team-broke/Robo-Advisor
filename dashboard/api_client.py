"""모델 패키지와 파일에 접근하지 않는 대시보드 전용 HTTP 클라이언트."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import re
import time
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

SECTIONS = ("portfolio", "rl_performance", "shap", "anova", "risk_monitor")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_CORRELATION = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")


class APIError(RuntimeError):
    """HTTP 오류 계약을 보존한다. 화면에는 user_message만 표시한다."""

    def __init__(self, code, message, correlation_id, status_code=None):
        self.code, self.message = code, message
        self.correlation_id, self.status_code = correlation_id, status_code
        super().__init__(code)

    @property
    def user_message(self):
        return {
            "connection_error": "API에 연결할 수 없습니다. 서버 주소와 실행 상태를 확인하세요.",
            "timeout": "API 응답 시간이 초과되었습니다. 잠시 후 다시 시도하세요.",
            "service_unavailable": "서비스 산출물이 아직 준비되지 않았습니다. API 상태를 확인하세요.",
            "not_found": "요청한 산출물 또는 작업을 찾을 수 없습니다.",
            "validation_error": "입력값과 평가 시각 형식을 확인하세요.",
            "poll_timeout": "작업이 아직 진행 중입니다. 같은 job ID로 다시 조회할 수 있습니다.",
            "internal_error": "API 내부 오류가 발생했습니다. 추적 ID로 서버 로그를 확인하세요.",
        }.get(self.code, "API 응답을 처리할 수 없습니다. 서버 상태를 확인하세요.")


@dataclass(frozen=True)
class ResearchReply:
    status_code: int
    data: dict
    correlation_id: str


class APIClient:
    """API_BASE_URL·타임아웃·상관 ID를 공유한다. 주입한 httpx client는 닫지 않는다."""

    def __init__(self, base_url=None, *, timeout=5.0, correlation_id=None, client=None):
        self.base_url = str(
            base_url or os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
        ).rstrip("/")
        parts = urlsplit(self.base_url)
        if (
            parts.scheme not in {"http", "https"}
            or not parts.hostname
            or parts.username
            or parts.password
            or parts.query
            or parts.fragment
        ):
            raise ValueError("API_BASE_URL must be HTTP(S) without credentials or query")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (float, int))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout must be positive and finite")
        self.timeout = float(timeout)
        self.correlation_id = (
            correlation_id
            if isinstance(correlation_id, str) and _CORRELATION.fullmatch(correlation_id)
            else str(uuid4())
        )
        self._client = client or httpx.Client(trust_env=False)
        self._owns_client = client is None

    def _request(self, method, path, **kwargs):
        try:
            response = self._client.request(
                method,
                self.base_url + path,
                timeout=self.timeout,
                headers={"X-Correlation-ID": self.correlation_id},
                follow_redirects=False,
                **kwargs,
            )
        except httpx.TimeoutException:
            raise APIError("timeout", "API request timed out", self.correlation_id) from None
        except httpx.HTTPError:
            raise APIError(
                "connection_error", "API connection failed", self.correlation_id
            ) from None
        correlation = response.headers.get("X-Correlation-ID", self.correlation_id)
        if _CORRELATION.fullmatch(correlation):
            self.correlation_id = correlation
        if not 200 <= response.status_code < 300:
            try:
                error = response.json()
                if not isinstance(error, dict) or any(
                    not isinstance(error.get(name), str)
                    for name in ("code", "message", "correlation_id")
                ):
                    raise ValueError
            except (ValueError, TypeError):
                raise APIError(
                    "protocol_error",
                    "Invalid API error response",
                    self.correlation_id,
                    response.status_code,
                ) from None
            raise APIError(
                error["code"], error["message"], error["correlation_id"], response.status_code
            )
        return response

    def _json(self, method, path, **kwargs):
        response = self._request(method, path, **kwargs)
        try:
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError
            return response.status_code, data
        except (ValueError, TypeError):
            raise APIError("protocol_error", "Expected JSON object", self.correlation_id) from None

    def health(self):
        return self._json("GET", "/health")[1]

    def news(self, query):
        return self._json("GET", "/news", params={"query": query, "limit": 5})[1]

    def section(self, section):
        if section not in SECTIONS:
            raise ValueError("unknown dashboard section")
        return self._json("GET", f"/dashboard/{section}")[1]

    def plot(self, section, plot_id):
        if section not in SECTIONS or not _ID.fullmatch(plot_id):
            raise ValueError("invalid plot identifier")
        response = self._request("GET", f"/dashboard/{section}/plots/{plot_id}")
        if response.headers.get("content-type", "").split(";", 1)[
            0
        ] != "image/png" or not response.content.startswith(b"\x89PNG\r\n\x1a\n"):
            raise APIError("protocol_error", "Expected PNG image", self.correlation_id)
        return response.content

    def research(self, question, as_of):
        code, data = self._json("POST", "/research", json={"question": question, "as_of": as_of})
        if code == 202 and (
            not isinstance(data.get("job_id"), str)
            or not _ID.fullmatch(data["job_id"])
            or data.get("status") != "queued"
        ):
            raise APIError("protocol_error", "Invalid research job response", self.correlation_id)
        if code not in (200, 202):
            raise APIError("protocol_error", "Unexpected research status", self.correlation_id)
        return ResearchReply(code, data, self.correlation_id)

    def research_job(self, job_id):
        if not isinstance(job_id, str) or not _ID.fullmatch(job_id):
            raise ValueError("invalid job ID")
        data = self._json("GET", f"/research/jobs/{job_id}")[1]
        if data.get("job_id") != job_id or data.get("status") not in {
            "queued",
            "running",
            "completed",
            "failed",
        }:
            raise APIError("protocol_error", "Invalid job status", self.correlation_id)
        if data["status"] == "completed" and not isinstance(data.get("result"), dict):
            raise APIError("protocol_error", "Missing completed result", self.correlation_id)
        return data

    def poll_job(
        self,
        job_id,
        *,
        timeout=10.0,
        interval=0.5,
        on_update=None,
        clock=time.monotonic,
        sleep=time.sleep,
    ):
        """상한 시간 안에서만 조회하며 완료되지 않아도 job을 다시 등록하지 않는다."""
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value > 0
            for value in (timeout, interval)
        ):
            raise ValueError("poll timeout and interval must be positive")
        deadline = clock() + timeout
        while True:
            data = self.research_job(job_id)
            if on_update is not None:
                on_update(data)
            if data["status"] in {"completed", "failed"}:
                return data
            remaining = deadline - clock()
            if remaining <= 0:
                raise APIError("poll_timeout", "Job is still active", self.correlation_id)
            sleep(min(interval, remaining))

    def close(self):
        if self._owns_client:
            self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
