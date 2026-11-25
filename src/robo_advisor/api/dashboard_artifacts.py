"""오프라인에서 만든 화면 자료와 PNG를 검증해 메모리에서만 제공한다."""

from __future__ import annotations

import io
import json

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from robo_advisor.api.schemas import DashboardResponse
from robo_advisor.api.services import NotFoundError
from robo_advisor.schemas.common import PortfolioWeights, UTCDateTime
from robo_advisor.schemas.risk import RiskTag


class ArtifactModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class SeriesPoint(ArtifactModel):
    timestamp: UTCDateTime
    value: float | None
    series: str = "portfolio"


class LearningPoint(ArtifactModel):
    step: int = Field(ge=0)
    value: float | None
    series: str = "episode_reward"


class PortfolioData(ArtifactModel):
    weights: PortfolioWeights | None = None
    cumulative_returns: list[SeriesPoint] = Field(default_factory=list)


class StrategyMetrics(ArtifactModel):
    strategy: str
    metrics: dict[str, float | None]


class PerformanceData(ArtifactModel):
    learning_curve: list[LearningPoint] = Field(default_factory=list)
    metrics: list[StrategyMetrics] = Field(default_factory=list)
    comparison: list[SeriesPoint] = Field(default_factory=list)


class ShapSummary(ArtifactModel):
    feature: str
    mean_abs_shap: float = Field(ge=0)


class ShapForce(ArtifactModel):
    feature: str
    contribution: float
    value: float | None = None


class ShapData(ArtifactModel):
    summary: list[ShapSummary] = Field(default_factory=list)
    force: list[ShapForce] = Field(default_factory=list)
    base_value: float | None = None
    prediction: float | None = None
    output: str | None = None
    unit: str = "raw_policy_mean_logit"


class AnovaData(ArtifactModel):
    results: list[dict[str, str | float | bool | None]] = Field(default_factory=list)
    tukey: list[dict[str, str | float | bool | None]] = Field(default_factory=list)
    diagnostics: list[dict[str, str | float | bool | None]] = Field(default_factory=list)
    metric: str | None = None
    alpha: float = Field(default=0.05, gt=0, lt=1)


class SafeguardEvent(ArtifactModel):
    timestamp: UTCDateTime
    reason: str
    trace_id: str


class RiskData(ArtifactModel):
    var_95: float | None = None
    cvar_95: float | None = None
    var_limit: float | None = Field(default=None, gt=0)
    cvar_limit: float | None = Field(default=None, gt=0)
    mdd_limit: float | None = Field(default=None, gt=0)
    drawdown: list[SeriesPoint] = Field(default_factory=list)
    safeguard_events: list[SafeguardEvent] = Field(default_factory=list)
    tags: list[RiskTag] = Field(default_factory=list)


_MODELS = {
    "portfolio": PortfolioData,
    "rl_performance": PerformanceData,
    "shap": ShapData,
    "anova": AnovaData,
    "risk_monitor": RiskData,
}


class DashboardArtifacts:
    """manifest.sections의 JSON・plots descriptor를 시작 시 모두 검증한다.

    section descriptor는 path/sha256와 선택적 plots(id→path/sha256/title/role)를
    가진다. JSON에는 as_of, model_id(선택), data가 있다. 조회 시 재계산·파일 I/O는 없다.
    """

    def __init__(self):
        self._sections, self._plots = {}, {}

    @classmethod
    def load(cls, manifest_path, expected_manifest_sha256):
        from robo_advisor.api.lifecycle import _manifest, _payload

        result = cls()
        manifest = _manifest(manifest_path, expected_manifest_sha256)
        if manifest.get("schema_version") != 1 or not isinstance(manifest.get("sections"), dict):
            raise ValueError("invalid dashboard manifest")
        root = manifest_path.expanduser().parent
        for section, descriptor in manifest["sections"].items():
            if section not in _MODELS:
                raise ValueError("unknown dashboard section")
            _, content = _payload(root, descriptor["path"], descriptor)
            raw = json.loads(content)
            data = _MODELS[section].model_validate(raw["data"])
            plots = []
            for plot_id, plot in descriptor.get("plots", {}).items():
                _, png = _payload(root, plot["path"], plot)
                with Image.open(io.BytesIO(png)) as image:
                    if image.format != "PNG":
                        raise ValueError("dashboard plots must be PNG")
                    image.verify()
                plots.append({"id": plot_id, "title": plot["title"], "role": plot["role"]})
                result._plots[(section, plot_id)] = (png, plot["sha256"])
            response = DashboardResponse(
                section=section,
                as_of=raw["as_of"],
                model_id=raw.get("model_id"),
                artifact_version=expected_manifest_sha256,
                provenance=raw.get("provenance"),
                data=data.model_dump(mode="json"),
                plots=plots,
            )
            if response.provenance and response.provenance.model_id != response.model_id:
                raise ValueError("dashboard provenance model_id mismatch")
            result._validate_times(data, response.as_of)
            result._sections[section] = response
        return result

    @staticmethod
    def _validate_times(data, as_of):
        for name in ("cumulative_returns", "comparison", "drawdown", "safeguard_events"):
            values = getattr(data, name, ())
            if any(point.timestamp > as_of for point in values):
                raise ValueError("dashboard contains future observations")
            keys = [(getattr(point, "series", ""), point.timestamp) for point in values]
            if len(keys) != len(set(keys)):
                raise ValueError("duplicate dashboard series timestamp")
        if any(tag.available_at > as_of for tag in getattr(data, "tags", ())):
            raise ValueError("dashboard contains future risk tags")
        learning = getattr(data, "learning_curve", ())
        keys = [(point.series, point.step) for point in learning]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate learning curve step")

    def get(self, section):
        if section not in self._sections:
            raise NotFoundError("dashboard section is absent")
        return self._sections[section].model_dump(mode="json")

    def plot(self, section, plot_id):
        if (section, plot_id) not in self._plots:
            raise NotFoundError("dashboard plot is absent")
        return self._plots[(section, plot_id)]
