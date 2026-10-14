"""Runtime configuration, read from environment variables (or a local .env)."""

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_CSV = REPO_ROOT / "data" / "samples" / "demo_daily_metrics.csv"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./health.db"
    cors_origins: str = "http://localhost:8501"
    seed_demo_data: bool = True
    max_upload_mb: int = 20
    # Most data an uploaded zip (Apple Health or Fitbit) may expand to.
    max_unzipped_mb: int = 200
    # Visitor datasets are deleted after this many days without being opened or changed.
    upload_ttl_days: int = 90
    # Upper bound on visitor datasets kept at once, to protect a small free database.
    max_upload_datasets: int = 200

    @field_validator("database_url")
    @classmethod
    def _use_psycopg3(cls, value: str) -> str:
        """Hosted Postgres providers hand out postgres:// or postgresql:// URLs.

        SQLAlchemy needs the driver spelled out, so point both at psycopg 3.
        """
        for prefix in ("postgres://", "postgresql://"):
            if value.startswith(prefix):
                return "postgresql+psycopg://" + value[len(prefix) :]
        return value

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
