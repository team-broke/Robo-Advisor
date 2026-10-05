"""환경변수와 .env에서 불러오는 공통 설정 계약."""

from functools import lru_cache
from pathlib import Path

from pydantic import AnyHttpUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """생성 시 경로·비용·API 설정을 검증하고 비밀정보를 숨긴다."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        hide_input_in_errors=True,
    )

    data_raw_dir: Path = Path("data/raw")
    data_processed_dir: Path = Path("data/processed")
    data_vectorstore_dir: Path = Path("data/vectorstore")
    artifacts_dir: Path = Path("artifacts")
    models_dir: Path = Path("models")

    commission_rate: float = Field(default=0.00015, ge=0, lt=1, allow_inf_nan=False)
    slippage_rate: float = Field(default=0.0005, ge=0, lt=1, allow_inf_nan=False)
    safeguard_mdd_limit: float = Field(default=0.15, gt=0, le=1, allow_inf_nan=False)

    api_host: str = Field(default="127.0.0.1", min_length=1, pattern=r"\S")
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_base_url: AnyHttpUrl = AnyHttpUrl("http://localhost:8000")

    dart_api_key: SecretStr | None = Field(default=None, repr=False)
    llm_api_key: SecretStr | None = Field(default=None, repr=False)
    llm_model: str | None = None
    llm_provider: str | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """시작 시 호출하여 설정을 검증하고 이후에는 캐시된 설정을 반환한다."""
    return Settings()
