"""설정 로드·검증·비밀정보 보호 계약을 확인한다."""

import logging
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from robo_advisor.config import Settings, get_settings


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    """실제 환경변수·.env·캐시가 테스트 결과에 영향을 주지 않게 한다."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_defaults():
    settings = Settings()

    assert settings.data_raw_dir == Path("data/raw")
    assert settings.data_processed_dir == Path("data/processed")
    assert settings.data_vectorstore_dir == Path("data/vectorstore")
    assert settings.artifacts_dir == Path("artifacts")
    assert settings.models_dir == Path("models")
    assert settings.commission_rate == 0.00015
    assert settings.slippage_rate == 0.0005
    assert settings.safeguard_mdd_limit == 0.15
    assert settings.api_host == "127.0.0.1"
    assert settings.api_port == 8000
    assert str(settings.api_base_url) == "http://localhost:8000/"


@pytest.mark.parametrize(
    "name,value,expected",
    [
        ("data_raw_dir", "/tmp/raw", Path("/tmp/raw")),
        ("data_processed_dir", "/tmp/processed", Path("/tmp/processed")),
        ("data_vectorstore_dir", "/tmp/vectors", Path("/tmp/vectors")),
        ("artifacts_dir", "/tmp/artifacts", Path("/tmp/artifacts")),
        ("models_dir", "/tmp/models", Path("/tmp/models")),
        ("commission_rate", "0.001", 0.001),
        ("slippage_rate", "0.002", 0.002),
        ("safeguard_mdd_limit", "0.2", 0.2),
        ("api_host", "0.0.0.0", "0.0.0.0"),
        ("api_port", "9000", 9000),
        ("llm_model", "test-model", "test-model"),
        ("llm_provider", "test-provider", "test-provider"),
    ],
)
def test_environment_overrides(monkeypatch, name, value, expected):
    monkeypatch.setenv(name.upper(), value)

    assert getattr(Settings(), name) == expected


def test_dotenv_and_environment_priority(monkeypatch):
    Path(".env").write_text(
        "API_PORT=9000\nDATA_RAW_DIR=custom/raw\nLLM_MODEL=테스트\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("API_PORT", "9001")
    monkeypatch.setenv("API_BASE_URL", "https://example.com/api")

    settings = Settings()

    assert settings.api_port == 9001
    assert settings.data_raw_dir == Path("custom/raw")
    assert settings.llm_model == "테스트"
    assert str(settings.api_base_url) == "https://example.com/api"


@pytest.mark.parametrize(
    "name,value",
    [
        ("commission_rate", "-0.001"),
        ("commission_rate", "1"),
        ("slippage_rate", "-0.001"),
        ("slippage_rate", "1"),
        ("safeguard_mdd_limit", "0"),
        ("safeguard_mdd_limit", "-0.1"),
        ("safeguard_mdd_limit", "1.01"),
        ("commission_rate", "nan"),
        ("slippage_rate", "inf"),
        ("safeguard_mdd_limit", "nan"),
        ("api_port", "0"),
        ("api_port", "65536"),
        ("api_host", "   "),
        ("api_base_url", "ftp://example.com"),
        ("api_base_url", "invalid-url"),
    ],
)
def test_invalid_environment_fails_at_startup(monkeypatch, name, value):
    monkeypatch.setenv(name.upper(), value)

    with pytest.raises(ValidationError) as error:
        get_settings()

    assert error.value.errors()[0]["loc"] == (name,)


@pytest.mark.parametrize(
    "overrides",
    [
        {"commission_rate": 0, "slippage_rate": 0, "safeguard_mdd_limit": 1, "api_port": 1},
        {"commission_rate": 0.999, "slippage_rate": 0.999, "api_port": 65535},
    ],
)
def test_valid_boundaries(overrides):
    settings = Settings(**overrides)

    for name, value in overrides.items():
        assert getattr(settings, name) == value


def test_startup_without_optional_keys():
    settings = get_settings()

    assert settings.dart_api_key is None
    assert settings.llm_api_key is None
    assert settings.llm_model is None
    assert settings.llm_provider is None


def test_env_example_loads_without_keys():
    example = Path(__file__).resolve().parents[1] / ".env.example"
    settings = Settings(_env_file=example)

    assert settings == Settings(_env_file=None)


def test_keys_are_secret_and_hidden_in_repr_and_logs(monkeypatch, caplog):
    monkeypatch.setenv("DART_API_KEY", "test-dart-secret")
    Path(".env").write_text("LLM_API_KEY=test-llm-secret\n", encoding="utf-8")
    settings = Settings()

    assert isinstance(settings.dart_api_key, SecretStr)
    assert isinstance(settings.llm_api_key, SecretStr)
    assert settings.dart_api_key.get_secret_value() == "test-dart-secret"
    assert settings.llm_api_key.get_secret_value() == "test-llm-secret"
    with caplog.at_level(logging.INFO):
        logging.getLogger(__name__).info("settings=%r %s", settings, settings)
        logging.getLogger(__name__).info("keys=%s", settings.model_dump())

    for secret in ("test-dart-secret", "test-llm-secret"):
        assert secret not in repr(settings)
        assert secret not in str(settings)
        assert secret not in settings.model_dump_json()
        assert secret not in caplog.text


def test_validation_error_text_hides_secret_input():
    with pytest.raises(ValidationError) as error:
        Settings(dart_api_key=["test-invalid-secret"])

    assert "test-invalid-secret" not in str(error.value)


def test_settings_are_cached(monkeypatch):
    settings = get_settings()
    monkeypatch.setenv("API_PORT", "9000")

    assert get_settings() is settings
    assert get_settings().api_port == 8000
    get_settings.cache_clear()
    assert get_settings().api_port == 9000
