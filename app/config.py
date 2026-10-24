from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables or a local .env file."""

    pipeline_path: Path = BASE_DIR / "model" / "churn_pipeline.joblib"
    metadata_path: Path = BASE_DIR / "model" / "metadata.json"
    api_key: str | None = None  # when unset, auth is disabled (local development only)
    allowed_origins: str = "http://localhost:8501"  # comma-separated
    rate_limit: str = "30/minute"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def origins(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
