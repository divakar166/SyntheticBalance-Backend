from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    base_dir: Path = Path(__file__).resolve().parent
    model_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent / "models")

    ctgan_epochs_default: int = 100
    ctgan_batch_size: int = 256
    classifier_test_size: float = 0.2
    random_seed: int = 42

    supabase_url: str | None = None
    supabase_service_role_key: str | None = None
    supabase_datasets_table: str = "datasets"
    supabase_training_jobs_table: str = "training_jobs"
    supabase_models_table: str = "trained_models"

    minio_endpoint: str = "127.0.0.1:9000"
    minio_access_key: str | None = None
    minio_secret_key: str | None = None
    minio_secure: bool = False
    minio_dataset_bucket: str = "synthetic-datasets"
    minio_model_bucket: str = "synthetic-models"

    aws_region: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None
    aws_s3_endpoint_url: str | None = None
    aws_s3_dataset_bucket: str | None = None
    aws_s3_model_bucket: str | None = None

    def ensure_directories(self):
        for directory in (self.model_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    settings = AppSettings()
    settings.ensure_directories()
    return settings
