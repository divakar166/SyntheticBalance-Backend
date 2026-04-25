from __future__ import annotations

import io
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd

from handlers.data_handler import normalize_dataframe
from settings import get_settings

from .base import PersistenceBackend


class SupabaseMinioBackend(PersistenceBackend):
    def __init__(self):
        settings = get_settings()
        self.supabase_url = settings.supabase_url
        self.supabase_key = settings.supabase_service_role_key
        self.minio_endpoint = settings.minio_endpoint
        self.minio_access_key = settings.minio_access_key
        self.minio_secret_key = settings.minio_secret_key
        self.minio_secure = settings.minio_secure
        self.dataset_bucket = settings.minio_dataset_bucket
        self.model_bucket = settings.minio_model_bucket
        self.datasets_table = settings.supabase_datasets_table
        self.training_jobs_table = settings.supabase_training_jobs_table
        self.models_table = settings.supabase_models_table
        self._supabase = None
        self._minio = None

    def _require_config(self):
        missing = [
            name
            for name, value in [
                ("SUPABASE_URL", self.supabase_url),
                ("SUPABASE_SERVICE_ROLE_KEY", self.supabase_key),
                ("MINIO_ACCESS_KEY", self.minio_access_key),
                ("MINIO_SECRET_KEY", self.minio_secret_key),
            ]
            if not value
        ]
        if missing:
            raise RuntimeError("Missing storage configuration: " + ", ".join(missing))

    def _format_storage_error(self, service: str, exc: Exception) -> RuntimeError:
        return RuntimeError(
            f"{service} storage operation failed: {exc}. "
            f"Check your {service} connection settings and service availability."
        )

    @property
    def supabase(self):
        if self._supabase is None:
            self._require_config()
            from supabase import create_client

            self._supabase = create_client(self.supabase_url, self.supabase_key)
        return self._supabase

    @property
    def minio(self):
        if self._minio is None:
            self._require_config()
            from minio import Minio

            self._minio = Minio(
                self.minio_endpoint,
                access_key=self.minio_access_key,
                secret_key=self.minio_secret_key,
                secure=self.minio_secure,
            )
            self._ensure_bucket(self.dataset_bucket)
            self._ensure_bucket(self.model_bucket)
        return self._minio

    def _ensure_bucket(self, bucket_name: str):
        try:
            if not self._minio.bucket_exists(bucket_name):
                self._minio.make_bucket(bucket_name)
        except Exception as exc:
            raise self._format_storage_error("MinIO", exc) from exc

    def _upload_bytes(self, bucket: str, object_key: str, content: bytes, content_type: str):
        try:
            self.minio.put_object(
                bucket,
                object_key,
                io.BytesIO(content),
                length=len(content),
                content_type=content_type,
            )
        except Exception as exc:
            raise self._format_storage_error("MinIO", exc) from exc

    def _download_bytes(self, bucket: str, object_key: str) -> bytes:
        try:
            response = self.minio.get_object(bucket, object_key)
            try:
                return response.read()
            finally:
                response.close()
                response.release_conn()
        except Exception as exc:
            raise self._format_storage_error("MinIO", exc) from exc

    def _serialize_record(self, record: dict) -> dict:
        serialized = {}
        for key, value in record.items():
            serialized[key] = value
        return serialized

    def save_dataset(self, dataset_id: str, df: pd.DataFrame, schema: dict, metadata: dict) -> dict:
        dataset_type = metadata.get("dataset_type", "real")
        object_key = f"{dataset_type}/{dataset_id}.csv"
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        self._upload_bytes(self.dataset_bucket, object_key, csv_bytes, "text/csv")
        record = {
            "id": dataset_id,
            "filename": metadata.get("filename"),
            "dataset_type": dataset_type,
            "object_key": object_key,
            "target": metadata.get("target"),
            "n_rows": int(len(df)),
            "n_features": int(len(schema.get("features", {}))),
            "class_dist": schema.get("target", {}).get("class_distribution", {}),
            "schema": schema,
            "metadata": metadata,
            "created_at": metadata.get("upload_time"),
        }
        try:
            self.supabase.table(self.datasets_table).upsert(record).execute()
        except Exception as exc:
            raise self._format_storage_error("Supabase", exc) from exc
        return record

    def get_dataset(self, dataset_id: str) -> dict | None:
        try:
            response = (
                self.supabase.table(self.datasets_table).select("*").eq("id", dataset_id).limit(1).execute()
            )
        except Exception as exc:
            raise self._format_storage_error("Supabase", exc) from exc
        if not response.data:
            return None
        record = dict(response.data[0])
        csv_bytes = self._download_bytes(self.dataset_bucket, record["object_key"])
        record["df"] = normalize_dataframe(pd.read_csv(io.BytesIO(csv_bytes)))
        return record

    def save_training_job(self, job: dict) -> dict:
        try:
            self.supabase.table(self.training_jobs_table).upsert(self._serialize_record(job)).execute()
        except Exception as exc:
            raise self._format_storage_error("Supabase", exc) from exc
        return job

    def update_training_job(self, job_id: str, values: dict) -> dict:
        try:
            self.supabase.table(self.training_jobs_table).update(values).eq("job_id", job_id).execute()
        except Exception as exc:
            raise self._format_storage_error("Supabase", exc) from exc
        return values

    def get_training_job(self, job_or_dataset_id: str) -> dict | None:
        try:
            direct = (
                self.supabase.table(self.training_jobs_table).select("*").eq("job_id", job_or_dataset_id).limit(1).execute()
            )
        except Exception as exc:
            raise self._format_storage_error("Supabase", exc) from exc
        if direct.data:
            return dict(direct.data[0])
        try:
            fallback = (
                self.supabase.table(self.training_jobs_table)
                .select("*")
                .eq("dataset_id", job_or_dataset_id)
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )
        except Exception as exc:
            raise self._format_storage_error("Supabase", exc) from exc
        if not fallback.data:
            return None
        return dict(fallback.data[0])

    def save_model(self, dataset_id: str, local_model_path: str | Path, metadata: dict | None = None) -> dict:
        local_model_path = Path(local_model_path)
        object_key = f"ctgan/{dataset_id}.pkl"
        self._upload_bytes(self.model_bucket, object_key, local_model_path.read_bytes(), "application/octet-stream")
        record = {
            "id": dataset_id,
            "dataset_id": dataset_id,
            "object_key": object_key,
            "metadata": metadata or {},
            "created_at": metadata.get("trained_at") if metadata else None,
        }
        try:
            self.supabase.table(self.models_table).upsert(record).execute()
        except Exception as exc:
            raise self._format_storage_error("Supabase", exc) from exc
        return record

    def get_model(self, dataset_id: str) -> dict | None:
        try:
            response = (
                self.supabase.table(self.models_table).select("*").eq("dataset_id", dataset_id).limit(1).execute()
            )
        except Exception as exc:
            raise self._format_storage_error("Supabase", exc) from exc
        if not response.data:
            return None
        return dict(response.data[0])

    def download_model_to_tempfile(self, dataset_id: str) -> tuple[NamedTemporaryFile, dict]:
        record = self.get_model(dataset_id)
        if not record:
            raise FileNotFoundError(f"Model for dataset '{dataset_id}' not found.")
        temp_file = NamedTemporaryFile(delete=False, suffix=".pkl")
        temp_file.write(self._download_bytes(self.model_bucket, record["object_key"]))
        temp_file.flush()
        temp_file.close()
        return temp_file, record

    def get_health_status(self) -> dict:
        supabase_status = {"configured": bool(self.supabase_url and self.supabase_key), "reachable": False}
        minio_status = {
            "configured": bool(self.minio_access_key and self.minio_secret_key and self.minio_endpoint),
            "reachable": False,
            "endpoint": self.minio_endpoint,
        }
        if supabase_status["configured"]:
            try:
                self.supabase.table(self.datasets_table).select("id").limit(1).execute()
                supabase_status["reachable"] = True
            except Exception as exc:
                supabase_status["error"] = str(exc)
        if minio_status["configured"]:
            try:
                self.minio.bucket_exists(self.dataset_bucket)
                minio_status["reachable"] = True
            except Exception as exc:
                minio_status["error"] = str(exc)
        return {"backend": "SupabaseMinioBackend", "supabase": supabase_status, "minio": minio_status}
