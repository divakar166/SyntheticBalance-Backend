from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd

from .base import PersistenceBackend


class InMemoryBackend(PersistenceBackend):
    def __init__(self, datasets: dict, training_jobs: dict, models: dict, generation_jobs: dict | None = None):
        self.datasets = datasets
        self.training_jobs = training_jobs
        self.models = models
        self.generation_jobs = generation_jobs if generation_jobs is not None else {}

    def save_dataset(self, dataset_id: str, df: pd.DataFrame, schema: dict, metadata: dict) -> dict:
        record = {
            "id": dataset_id,
            "df": df.copy(),
            "schema": schema,
            "metadata": metadata,
            "target": metadata.get("target"),
            "filename": metadata.get("filename"),
        }
        self.datasets[dataset_id] = record
        return record

    def get_dataset(self, dataset_id: str) -> dict | None:
        return self.datasets.get(dataset_id)

    def save_training_job(self, job: dict) -> dict:
        self.training_jobs[job["job_id"]] = dict(job)
        return self.training_jobs[job["job_id"]]

    def update_training_job(self, job_id: str, values: dict) -> dict:
        self.training_jobs[job_id].update(values)
        return self.training_jobs[job_id]

    def get_training_job(self, job_or_dataset_id: str) -> dict | None:
        direct = self.training_jobs.get(job_or_dataset_id)
        if direct:
            return direct
        for job in reversed(list(self.training_jobs.values())):
            if job["dataset_id"] == job_or_dataset_id:
                return job
        return None

    def save_generation_job(self, job: dict) -> dict:
        self.generation_jobs[job["job_id"]] = dict(job)
        return self.generation_jobs[job["job_id"]]

    def update_generation_job(self, job_id: str, values: dict) -> dict:
        self.generation_jobs[job_id].update(values)
        return self.generation_jobs[job_id]

    def get_generation_job(self, job_or_dataset_id: str) -> dict | None:
        direct = self.generation_jobs.get(job_or_dataset_id)
        if direct:
            return direct
        for job in reversed(list(self.generation_jobs.values())):
            if job["dataset_id"] == job_or_dataset_id:
                return job
        return None

    def save_model(self, dataset_id: str, local_model_path: str | Path, metadata: dict | None = None) -> dict:
        record = {
            "id": dataset_id,
            "dataset_id": dataset_id,
            "object_key": str(local_model_path),
            "model_path": str(local_model_path),
            "metadata": metadata or {},
        }
        self.models[dataset_id] = record
        return record

    def get_model(self, dataset_id: str) -> dict | None:
        return self.models.get(dataset_id)

    def download_model_to_tempfile(self, dataset_id: str) -> tuple[NamedTemporaryFile, dict]:
        record = self.get_model(dataset_id)
        if not record:
            raise FileNotFoundError(f"Model for dataset '{dataset_id}' not found.")
        temp_file = NamedTemporaryFile(delete=False, suffix=".pkl")
        temp_file.write(Path(record["object_key"]).read_bytes())
        temp_file.flush()
        temp_file.close()
        return temp_file, record

    def get_health_status(self) -> dict:
        return {
            "backend": "InMemoryBackend",
            "supabase": {"configured": False, "reachable": False},
            "minio": {"configured": False, "reachable": False},
        }
