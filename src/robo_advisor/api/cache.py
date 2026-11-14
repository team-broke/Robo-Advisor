"""모델·산출물 버전별 리서치 캐시와 보존 수가 제한된 스레드 job 큐."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from threading import RLock
import unicodedata
from uuid import uuid4

from pydantic import TypeAdapter

from robo_advisor.api.schemas import ResearchJobStatus
from robo_advisor.api.services import NotFoundError, ServiceUnavailableError
from robo_advisor.research.contracts import ResearchResult
from robo_advisor.research.trace import redact
from robo_advisor.schemas.common import UTCDateTime

_DIGEST = re.compile(r"[a-f0-9]{64}\Z")


def _versions(model_id, artifact_version):
    if not _DIGEST.fullmatch(model_id) or not _DIGEST.fullmatch(artifact_version):
        raise ValueError("cache versions must be SHA256 digests")


def normalize_question(question: str) -> str:
    """Unicode·공백을 정규화하되 의미가 달라질 수 있는 대소문자는 유지한다."""
    question = " ".join(unicodedata.normalize("NFKC", question).split())
    if not question:
        raise ValueError("question must not be empty")
    return question


def _bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class CacheKey:
    model_id: str
    artifact_version: str
    question: str
    as_of: datetime

    def __post_init__(self):
        _versions(self.model_id, self.artifact_version)
        object.__setattr__(self, "question", normalize_question(self.question))
        object.__setattr__(self, "as_of", TypeAdapter(UTCDateTime).validate_python(self.as_of))

    @property
    def digest(self):
        return hashlib.sha256(
            _bytes(
                {
                    "model_id": self.model_id,
                    "artifact_version": self.artifact_version,
                    "question": self.question,
                    "as_of": self.as_of.isoformat(),
                }
            )
        ).hexdigest()


class ResearchCache:
    """조회·저장만 수행한다. 디스크 내용은 시작 시 검증해 메모리에 올린다."""

    def __init__(self, directory=None, *, max_entries=128, secrets=()):
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.directory = Path(directory).expanduser() if directory is not None else None
        self.max_entries, self.secrets = max_entries, tuple(secrets)
        self._items, self._lock = OrderedDict(), RLock()
        if self.directory is not None:
            if any(path.is_symlink() for path in (self.directory, *self.directory.parents)):
                raise ValueError("cache cannot traverse symlinks")
            self.directory.mkdir(parents=True, exist_ok=True)
            files = sorted(self.directory.glob("*.json"), key=lambda path: path.stat().st_mtime_ns)
            for path in files:
                if path.is_symlink() or not _DIGEST.fullmatch(path.stem):
                    raise ValueError("invalid cache file")
                envelope = json.loads(path.read_text(encoding="utf-8"))
                entry = envelope["entry"]
                if (
                    envelope["sha256"] != hashlib.sha256(_bytes(entry)).hexdigest()
                    or entry["key"] != path.stem
                ):
                    raise ValueError("cache integrity mismatch")
                result = self._result(entry["result"])
                if result.as_of.isoformat() != entry["as_of"]:
                    raise ValueError("cache cutoff mismatch")
                self._items[path.stem] = entry
                if len(self._items) > max_entries:
                    self._items.popitem(last=False)

    def _result(self, value):
        result = ResearchResult.model_validate(redact(value, secrets=self.secrets))
        if result.status != "completed":
            raise ValueError("only completed research can be cached")
        return result

    def get(self, key: CacheKey):
        with self._lock:
            entry = self._items.get(key.digest)
            if entry is None:
                return None
            if (
                entry["model_id"] != key.model_id
                or entry["artifact_version"] != key.artifact_version
            ):
                raise ValueError("cache version mismatch")
            result = self._result(entry["result"])
            if result.as_of != key.as_of or normalize_question(
                result.question
            ) != normalize_question(redact(key.question, secrets=self.secrets)):
                raise ValueError("cache result does not match request")
            self._items.move_to_end(key.digest)
            return result

    def put(self, key: CacheKey, result: ResearchResult):
        with self._lock:
            result = self._result(result.model_dump(mode="json"))
            if result.as_of != key.as_of or normalize_question(
                result.question
            ) != normalize_question(redact(key.question, secrets=self.secrets)):
                raise ValueError("research result does not match cache key")
            entry = {
                "key": key.digest,
                "model_id": key.model_id,
                "artifact_version": key.artifact_version,
                "as_of": key.as_of.isoformat(),
                "result": result.model_dump(mode="json"),
            }
            if self.directory is not None:
                temporary = self.directory / f".{uuid4().hex}.tmp"
                try:
                    temporary.write_bytes(
                        _bytes(
                            {"entry": entry, "sha256": hashlib.sha256(_bytes(entry)).hexdigest()}
                        )
                    )
                    temporary.replace(self.directory / f"{key.digest}.json")
                finally:
                    temporary.unlink(missing_ok=True)
            self._items[key.digest] = entry
            self._items.move_to_end(key.digest)
            while len(self._items) > self.max_entries:
                evicted, _ = self._items.popitem(last=False)
                if self.directory is not None:
                    (self.directory / f"{evicted}.json").unlink(missing_ok=True)
            if self.directory is not None:
                files = sorted(
                    self.directory.glob("*.json"), key=lambda path: path.stat().st_mtime_ns
                )
                for path in files[: -self.max_entries]:
                    path.unlink()


class ResearchJobs:
    """캐시 미적중 리서치만 실행한다. 활성 job은 보존 상한에서 제거하지 않는다."""

    def __init__(self, runner, cache, *, model_id, artifact_version, max_workers=2, max_jobs=128):
        for value in (max_workers, max_jobs):
            if type(value) is not int or value < 1:
                raise ValueError("job limits must be positive integers")
        _versions(model_id, artifact_version)
        self.runner, self.cache = runner, cache
        self.model_id, self.artifact_version = model_id, artifact_version
        self.max_jobs = max_jobs
        self._jobs, self._by_key = OrderedDict(), {}
        self._lock, self._closed = RLock(), False
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="research")

    def _key(self, question, as_of):
        return CacheKey(self.model_id, self.artifact_version, question, as_of)

    def cached(self, question, as_of):
        return self.cache.get(self._key(question, as_of))

    def submit(self, question, as_of):
        key = self._key(question, as_of)
        with self._lock:
            if self._closed:
                raise ServiceUnavailableError("research queue is closed")
            if key.digest in self._by_key:
                return self._by_key[key.digest]
            if len(self._jobs) >= self.max_jobs:
                victim = next(
                    (
                        job_id
                        for job_id, item in self._jobs.items()
                        if item.status in {"completed", "failed"}
                    ),
                    None,
                )
                if victim is None:
                    raise ServiceUnavailableError("research queue is full")
                del self._jobs[victim]
                self._by_key = {
                    digest: job_id for digest, job_id in self._by_key.items() if job_id != victim
                }
            job_id = str(uuid4())
            self._jobs[job_id] = ResearchJobStatus(job_id=job_id, status="queued")
            self._by_key[key.digest] = job_id
            try:
                self._executor.submit(copy_context().run, self._run, job_id, key)
            except Exception:
                del self._jobs[job_id]
                del self._by_key[key.digest]
                raise ServiceUnavailableError("research queue submission failed") from None
            return job_id

    def _run(self, job_id, key):
        with self._lock:
            self._jobs[job_id] = ResearchJobStatus(job_id=job_id, status="running")
        try:
            result = self.cache.get(key)
            if result is None:
                result = ResearchResult.model_validate(self.runner(key.question, key.as_of))
            if result.status == "failed":
                reason = redact(";".join(result.reasons), secrets=self.cache.secrets)[:500]
                final = ResearchJobStatus(job_id=job_id, status="failed", failure_reason=reason)
            else:
                self.cache.put(key, result)
                final = ResearchJobStatus(
                    job_id=job_id, status="completed", result=self.cache.get(key)
                )
        except Exception as error:
            final = ResearchJobStatus(
                job_id=job_id,
                status="failed",
                failure_reason=f"research_error:{type(error).__name__}",
            )
        with self._lock:
            self._jobs[job_id] = final

    def job(self, job_id):
        with self._lock:
            if job_id not in self._jobs:
                raise NotFoundError("research job not found or expired")
            return self._jobs[job_id].model_copy(deep=True)

    def close(self):
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=True)
