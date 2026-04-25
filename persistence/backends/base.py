from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd


class PersistenceBackend:
    def save_dataset(self, dataset_id: str, df: pd.DataFrame, schema: dict, metadata: dict) -> dict:
        raise NotImplementedError

    def get_dataset(self, dataset_id: str) -> dict | None:
        raise NotImplementedError

    def list_datasets(self, user_id: str) -> list[dict]:
        raise NotImplementedError

    def delete_dataset(self, dataset_id: str, user_id: str) -> bool:
        raise NotImplementedError

    def dataset_exists(self, dataset_id: str) -> bool:
        return self.get_dataset(dataset_id) is not None

    def save_training_job(self, job: dict) -> dict:
        raise NotImplementedError

    def update_training_job(self, job_id: str, values: dict) -> dict:
        raise NotImplementedError

    def get_training_job(self, job_or_dataset_id: str) -> dict | None:
        raise NotImplementedError

    def save_generation_job(self, job: dict) -> dict:
        raise NotImplementedError

    def update_generation_job(self, job_id: str, values: dict) -> dict:
        raise NotImplementedError

    def get_generation_job(self, job_or_dataset_id: str) -> dict | None:
        raise NotImplementedError

    def save_model(self, dataset_id: str, local_model_path: str | Path, metadata: dict | None = None) -> dict:
        raise NotImplementedError

    def get_model(self, dataset_id: str) -> dict | None:
        raise NotImplementedError

    def download_model_to_tempfile(self, dataset_id: str) -> tuple[NamedTemporaryFile, dict]:
        raise NotImplementedError

    def get_health_status(self) -> dict:
        return {
            "backend": self.__class__.__name__,
            "supabase": {"configured": False, "reachable": False},
            "minio": {"configured": False, "reachable": False},
        }
