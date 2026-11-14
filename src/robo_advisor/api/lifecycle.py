"""해시로 고정한 모델·관측·설명·백테스트·리서치 자료의 시작 시 로드."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from threading import Lock
from typing import Protocol
from uuid import uuid4

import numpy as np
import pandas as pd
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from robo_advisor.api.cache import ResearchCache, ResearchJobs
from robo_advisor.api.schemas import BacktestResponse
from robo_advisor.api.services import NotFoundError, Services, ShapExplainer
from robo_advisor.config import get_settings
from robo_advisor.schemas.common import AssetWeight, PortfolioWeights
from robo_advisor.schemas.risk import RiskTag

_DIGEST = r"^[a-f0-9]{64}$"


class PreloadSettings(BaseSettings):
    """API_ 환경변수로 지정한다. 설정하지 않은 서비스는 503 상태를 유지한다.

    policy_manifest는 assets/lookback과 write_frame 관측·선택적 risk_tags descriptor,
    research_manifest는 documents.jsonl descriptor를 담는다. 각 파일은 원본 해시를
    검증한 뒤 사용한다. 백테스트는 해시된 metrics.json 또는 #119 summary 형식이다.
    """

    model_config = SettingsConfigDict(
        env_prefix="API_", env_file=".env", extra="ignore", hide_input_in_errors=True
    )
    checkpoint_path: Path | None = None
    checkpoint_sha256: str | None = Field(default=None, pattern=_DIGEST)
    policy_manifest: Path | None = None
    policy_manifest_sha256: str | None = Field(default=None, pattern=_DIGEST)
    shap_dir: Path | None = None
    shap_manifest_sha256: str | None = Field(default=None, pattern=_DIGEST)
    backtest_dir: Path | None = None
    backtest_manifest_sha256: str | None = Field(default=None, pattern=_DIGEST)
    backtest_summary_sha256: str | None = Field(default=None, pattern=_DIGEST)
    research_manifest: Path | None = None
    research_manifest_sha256: str | None = Field(default=None, pattern=_DIGEST)
    research_cache_dir: Path | None = None
    research_workers: int = Field(default=2, ge=1, le=32)
    research_max_jobs: int = Field(default=128, ge=1, le=10000)
    research_cache_entries: int = Field(default=128, ge=1, le=10000)


def _manifest(path, expected):
    if path is None or expected is None:
        raise ValueError("artifact path and trusted manifest SHA256 are required")
    path = Path(path).expanduser()
    if any(candidate.is_symlink() for candidate in (path, *path.parents)):
        raise ValueError("artifact cannot traverse symlinks")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("artifact manifest SHA256 mismatch")

    def invalid(_):
        raise ValueError("nonfinite manifest JSON")

    return json.loads(payload, parse_constant=invalid)


def _payload(root, name, descriptor):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("unsafe artifact path")
    path = root / relative
    if any(
        candidate.is_symlink() for candidate in (path, *path.parents)
    ) or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("unsafe artifact path")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != descriptor["sha256"]:
        raise ValueError("artifact payload SHA256 mismatch")
    return path, content


def _checkpoint_id(settings):
    if settings.checkpoint_path is None or settings.checkpoint_sha256 is None:
        raise ValueError("checkpoint path and SHA256 are required")
    from robo_advisor.explain.shap_values import model_fingerprint

    if any(
        path.is_symlink() for path in (settings.checkpoint_path, *settings.checkpoint_path.parents)
    ):
        raise ValueError("checkpoint cannot be a symlink")
    digest = model_fingerprint(settings.checkpoint_path)
    if digest != settings.checkpoint_sha256:
        raise ValueError("checkpoint SHA256 mismatch")
    return digest


class PolicyOptimizer:
    """사전 검증된 실제 관측 한 행으로만 추론한다. 상태를 재구성하거나 학습하지 않는다."""

    def __init__(self, wrapper, observations, model_id, risk_tags=()):
        index = observations.index
        if (
            not isinstance(index, pd.DatetimeIndex)
            or index.tz is None
            or index.hasnans
            or not index.is_unique
            or not index.is_monotonic_increasing
        ):
            raise ValueError("policy observations require ordered unique aware timestamps")
        wrapper.layout.as_batch(observations)
        self.wrapper, self.observations = wrapper, observations.copy(deep=True)
        self.model_id, self.risk_tags = model_id, tuple(risk_tags)
        from robo_advisor.api.observability import visible_risk_tags

        for timestamp, row in self.observations.iterrows():
            scores = {
                tag.asset: tag.risk_score for tag in visible_risk_tags(self.risk_tags, timestamp)
            }
            if any(
                not np.isclose(
                    row[f"risk_score__{asset}"], scores.get(asset, 0.0), rtol=1e-6, atol=1e-8
                )
                for asset in wrapper.layout.assets
            ):
                raise ValueError("risk observation does not match source tag traces")
        self._lock = Lock()

    def optimize(self, as_of):
        stamp = pd.Timestamp(as_of)
        if stamp.tzinfo is None or stamp not in self.observations.index:
            raise NotFoundError("no preloaded observation at requested timestamp")
        with self._lock:
            weights = self.wrapper.predict_target_weights(self.observations.loc[[stamp]])[0]
        return PortfolioWeights(
            asset_weights=tuple(
                AssetWeight(asset=asset, weight=float(weight))
                for asset, weight in zip(self.wrapper.layout.assets, weights[:-1])
            ),
            cash_weight=float(weights[-1]),
        )


def _load_optimizer(settings):
    from stable_baselines3 import PPO
    from robo_advisor.data.snapshot import read_frame
    from robo_advisor.explain.policy import ObservationLayout, PPOPolicyWrapper

    model_id = _checkpoint_id(settings)
    manifest = _manifest(settings.policy_manifest, settings.policy_manifest_sha256)
    if manifest["model_id"] != model_id:
        raise ValueError("policy observation model_id mismatch")
    root = settings.policy_manifest.expanduser().parent
    descriptor = manifest["observations"]
    path, _ = _payload(root, descriptor["path"], descriptor)
    observations = read_frame(path, descriptor)
    tags = ()
    if "risk_tags" in manifest:
        descriptor = manifest["risk_tags"]
        path, _ = _payload(root, descriptor["path"], descriptor)
        tags = tuple(
            RiskTag.model_validate(row) for row in read_frame(path, descriptor).to_dict("records")
        )
    layout = ObservationLayout(manifest["assets"], manifest["lookback"])
    if any(tag.asset not in layout.assets for tag in tags):
        raise ValueError("risk tag asset is absent from policy layout")
    checkpoint = settings.checkpoint_path.read_bytes()
    if hashlib.sha256(checkpoint).hexdigest() != model_id:
        raise ValueError("checkpoint changed during preload")
    wrapper = PPOPolicyWrapper(PPO.load(io.BytesIO(checkpoint), device="cpu"), layout)
    return PolicyOptimizer(wrapper, observations, model_id, tags)


def _load_explainer(settings):
    from robo_advisor.explain.shap_values import load_shap_result

    if settings.shap_dir is None or settings.shap_manifest_sha256 is None:
        raise ValueError("SHAP directory and manifest SHA256 are required")
    model_id = _checkpoint_id(settings)
    manifest = _manifest(settings.shap_dir / "manifest.json", settings.shap_manifest_sha256)
    for name, descriptor in manifest["files"].items():
        _payload(settings.shap_dir, name, descriptor)
    result = load_shap_result(
        settings.shap_dir, expected_manifest_sha256=settings.shap_manifest_sha256
    )
    if result.metadata["model_id"] != model_id:
        raise ValueError("SHAP model_id mismatch")
    return ShapExplainer(result, model_id)


class BacktestLoader(Protocol):
    def load(self, directory, expected_manifest_sha256, expected_summary_sha256=None):
        """완료된 저장 지표를 검증하고 get(strategy, window) 저장소를 반환한다."""


class DirectoryBacktests:
    def __init__(self, records):
        self._records = {}
        for record in records:
            record = BacktestResponse.model_validate(record)
            key = (record.strategy, record.window)
            if key in self._records or record.start >= record.end:
                raise ValueError("ambiguous backtest key or invalid time interval")
            if any(
                value is not None and not np.isfinite(value) for value in record.metrics.values()
            ):
                raise ValueError("backtest metrics must be finite or null")
            self._records[key] = record
        if not self._records:
            raise ValueError("no completed backtest records")

    def get(self, strategy, window):
        try:
            return self._records[(strategy, window)].model_dump()
        except KeyError:
            raise NotFoundError("backtest result not found") from None


class DirectoryBacktestLoader:
    """해시된 metrics.json 또는 #119의 완료된 summary.csv/run_manifest를 읽는다.

    #119 manifest에는 summary 바이트 해시가 없으므로 별도 pin이 필수다.
    seed·dataset·variant 때문에 동일 API 키가 겹치면 집계하거나 임의 선택하지 않는다.
    """

    def load(self, directory, expected_manifest_sha256, expected_summary_sha256=None):
        root = Path(directory).expanduser()
        manifest = _manifest(root / "run_manifest.json", expected_manifest_sha256)
        if "files" in manifest:
            _, content = _payload(root, "metrics.json", manifest["files"]["metrics.json"])
            return DirectoryBacktests(json.loads(content)["records"])
        if expected_summary_sha256 is None or manifest["status"] not in {
            "completed",
            "completed_with_failures",
        }:
            raise ValueError("completed retrain manifest and summary SHA256 are required")
        _, content = _payload(root, "summary.csv", {"sha256": expected_summary_sha256})
        rows, records = pd.read_csv(io.BytesIO(content)), []
        jobs = {job["run_id"]: job for job in manifest["jobs"]}
        for row in rows.to_dict("records"):
            job = jobs[row["run_id"]]
            if (
                job["status"] != "completed"
                or job["dataset"] != row["dataset"]
                or job["window"] != row["window"]
            ):
                raise ValueError("summary contains an incomplete or mismatched job")
            window = next(
                item
                for item in manifest["windows"][row["dataset"]]
                if item["window"] == row["window"]
            )
            for strategy in ("ppo", "equal_weight"):
                metrics = {
                    name[len(strategy) + 1 :]: None if pd.isna(value) else value
                    for name, value in row.items()
                    if name.startswith(strategy + "_")
                }
                records.append(
                    {
                        "strategy": strategy,
                        "window": str(row["window"]),
                        "start": window["test_start"],
                        "end": window["test_end"],
                        "metrics": metrics,
                    }
                )
        return DirectoryBacktests(records)


def _load_backtests(settings):
    if settings.backtest_dir is None:
        raise ValueError("backtest directory is required")
    return DirectoryBacktestLoader().load(
        settings.backtest_dir, settings.backtest_manifest_sha256, settings.backtest_summary_sha256
    )


def _load_research(settings):
    if settings.research_manifest is None or settings.research_manifest_sha256 is None:
        raise ValueError("research manifest and SHA256 are required")
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from robo_advisor.research.graph import run_research
    from robo_advisor.research.ingestion import load_documents
    from robo_advisor.research.llm import HttpResearchLLM
    from robo_advisor.research.vector_store import ResearchVectorStore

    model_id = _checkpoint_id(settings)
    manifest = _manifest(settings.research_manifest, settings.research_manifest_sha256)
    descriptor = manifest["documents"]
    path, _ = _payload(
        settings.research_manifest.expanduser().parent, descriptor["path"], descriptor
    )
    llm_settings = get_settings()
    llm = HttpResearchLLM(llm_settings)
    store = ResearchVectorStore(
        client=chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False)),
        collection_name=f"api-{uuid4().hex}",
    )
    store.index(load_documents(path))
    cache = ResearchCache(
        settings.research_cache_dir,
        max_entries=settings.research_cache_entries,
        secrets=(llm_settings.llm_api_key.get_secret_value(),),
    )
    return ResearchJobs(
        lambda question, as_of: run_research(question, as_of, store=store, llm=llm),
        cache,
        model_id=model_id,
        artifact_version=settings.research_manifest_sha256,
        max_workers=settings.research_workers,
        max_jobs=settings.research_max_jobs,
    )


@dataclass(frozen=True)
class Preloaded:
    services: Services
    reasons: dict[str, str]


def preload_services(settings=None, *, loaders=None):
    """각 서비스를 독립적으로 로드한다. 경로·원문 예외·비밀값은 health에 노출하지 않는다."""
    try:
        settings = settings or PreloadSettings()
    except ValueError:
        return Preloaded(
            Services(), {name: "invalid_preload_configuration" for name in Services().loaded()}
        )
    defaults = {
        "optimizer": _load_optimizer,
        "explainer": _load_explainer,
        "research": _load_research,
        "backtests": _load_backtests,
    }
    if loaders:
        if not set(loaders).issubset(defaults):
            raise ValueError("unknown service loader")
        defaults.update(loaders)
    services, reasons = {}, {}
    for name, loader in defaults.items():
        try:
            services[name] = loader(settings)
            if services[name] is None:
                reasons[name] = "not_configured"
        except Exception as error:
            services[name] = None
            reasons[name] = f"load_failed:{type(error).__name__}"
    return Preloaded(Services(**services), reasons)


def make_lifespan(injected=None, *, settings=None, loaders=None):
    @asynccontextmanager
    async def lifespan(app):
        if injected is None:
            loaded = preload_services(settings, loaders=loaders)
            app.state.services, app.state.service_reasons = loaded.services, loaded.reasons
        try:
            yield
        finally:
            if injected is None:
                research = app.state.services.research
                if research is not None and hasattr(research, "close"):
                    research.close()

    return lifespan
