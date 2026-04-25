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
            "user_id": metadata.get("user_id"),
            "target": metadata.get("target"),
            "filename": metadata.get("filename"),
            "dataset_type": metadata.get("dataset_type", "real"),
            "n_rows": int(len(df)),
            "n_features": int(len(schema.get("features", {}))),
            "class_dist": schema.get("target", {}).get("class_distribution", {}),
            "created_at": metadata.get("upload_time"),
        }
        self.datasets[dataset_id] = record
        return record

    def get_dataset(self, dataset_id: str) -> dict | None:
        return self.datasets.get(dataset_id)

    def list_datasets(self, user_id: str) -> list[dict]:
        records = [
            self._dataset_summary(record)
            for record in self.datasets.values()
            if record.get("user_id") == user_id or (record.get("metadata") or {}).get("user_id") == user_id
        ]
        return sorted(records, key=lambda record: record.get("created_at") or "", reverse=True)

    def delete_dataset(self, dataset_id: str, user_id: str) -> bool:
        record = self.datasets.get(dataset_id)
        if not record:
            return False
        owner_id = record.get("user_id") or (record.get("metadata") or {}).get("user_id")
        if owner_id != user_id:
            return False
        self.datasets.pop(dataset_id, None)
        self.models.pop(dataset_id, None)
        for job_id, job in list(self.training_jobs.items()):
            if job.get("dataset_id") == dataset_id:
                self.training_jobs.pop(job_id, None)
        for job_id, job in list(self.generation_jobs.items()):
            if job.get("dataset_id") == dataset_id:
                self.generation_jobs.pop(job_id, None)
        return True

    def _dataset_summary(self, record: dict) -> dict:
        dataset_id = record["id"]
        return {
            "id": dataset_id,
            "filename": record.get("filename"),
            "dataset_type": record.get("dataset_type", "real"),
            "target": record.get("target"),
            "n_rows": record.get("n_rows", len(record.get("df", []))),
            "n_features": record.get("n_features", len((record.get("schema") or {}).get("features", {}))),
            "class_dist": record.get("class_dist", (record.get("schema") or {}).get("target", {}).get("class_distribution", {})),
            "schema": record.get("schema"),
            "created_at": record.get("created_at") or (record.get("metadata") or {}).get("upload_time"),
            "has_model": dataset_id in self.models,
            "latest_training_job": self.get_training_job(dataset_id),
            "latest_generation_job": self.get_generation_job(dataset_id),
        }

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
            "user_id": (metadata or {}).get("user_id"),
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
