from __future__ import annotations

import io
from pathlib import Path
from tempfile import NamedTemporaryFile

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from handlers.data_handler import normalize_dataframe
from settings import get_settings

from .base import PersistenceBackend


def _split_bucket_prefix(value: str) -> tuple[str, str]:
    if "/" in value:
        bucket, _, rest = value.partition("/")
        prefix = rest.rstrip("/") + "/"
    else:
        bucket, prefix = value, ""
    return bucket, prefix


class SupabaseS3Backend(PersistenceBackend):
    def __init__(self):
        settings = get_settings()
        self.supabase_url = settings.supabase_url
        self.supabase_key = settings.supabase_service_role_key
        self.aws_region = settings.aws_region
        self.aws_access_key_id = settings.aws_access_key_id
        self.aws_secret_access_key = settings.aws_secret_access_key
        self.aws_session_token = settings.aws_session_token
        self.aws_s3_endpoint_url = settings.aws_s3_endpoint_url

        raw_dataset = settings.aws_s3_dataset_bucket or ""
        raw_model   = settings.aws_s3_model_bucket   or ""
        self.dataset_bucket, self.dataset_prefix = _split_bucket_prefix(raw_dataset)
        self.model_bucket,   self.model_prefix   = _split_bucket_prefix(raw_model)

        self._single_bucket = (self.dataset_bucket == self.model_bucket)

        self.datasets_table      = settings.supabase_datasets_table
        self.training_jobs_table = settings.supabase_training_jobs_table
        self.models_table        = settings.supabase_models_table
        self._supabase = None
        self._s3 = None

    def _dataset_key(self, relative_key: str) -> str:
        """Prepend the dataset folder prefix to a relative object key."""
        return f"{self.dataset_prefix}{relative_key}"

    def _model_key(self, relative_key: str) -> str:
        """Prepend the model folder prefix to a relative object key."""
        return f"{self.model_prefix}{relative_key}"

    def _require_config(self):
        missing = [
            name
            for name, value in [
                ("SUPABASE_URL", self.supabase_url),
                ("SUPABASE_SERVICE_ROLE_KEY", self.supabase_key),
                ("AWS_REGION", self.aws_region),
                ("AWS_ACCESS_KEY_ID", self.aws_access_key_id),
                ("AWS_SECRET_ACCESS_KEY", self.aws_secret_access_key),
                ("AWS_S3_DATASET_BUCKET", self.dataset_bucket),
                ("AWS_S3_MODEL_BUCKET", self.model_bucket),
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
    def s3(self):
        if self._s3 is None:
            self._require_config()
            session = boto3.session.Session(
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                aws_session_token=self.aws_session_token,
                region_name=self.aws_region,
            )
            self._s3 = session.client("s3", endpoint_url=self.aws_s3_endpoint_url or None)
            self._ensure_bucket(self.dataset_bucket)
            if not self._single_bucket:
                self._ensure_bucket(self.model_bucket)
        return self._s3

    def _ensure_bucket(self, bucket_name: str):
        try:
            self.s3.head_bucket(Bucket=bucket_name)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code not in {"404", "NoSuchBucket"}:
                raise self._format_storage_error("AWS S3", exc) from exc
            params = {"Bucket": bucket_name}
            if self.aws_region and self.aws_region != "us-east-1":
                params["CreateBucketConfiguration"] = {"LocationConstraint": self.aws_region}
            try:
                self.s3.create_bucket(**params)
            except Exception as create_exc:
                raise self._format_storage_error("AWS S3", create_exc) from create_exc
        except Exception as exc:
            raise self._format_storage_error("AWS S3", exc) from exc

    def _upload_bytes(self, bucket: str, object_key: str, content: bytes, content_type: str):
        try:
            self.s3.put_object(
                Bucket=bucket, Key=object_key, Body=content, ContentType=content_type
            )
        except Exception as exc:
            raise self._format_storage_error("AWS S3", exc) from exc

    def _download_bytes(self, bucket: str, object_key: str) -> bytes:
        try:
            response = self.s3.get_object(Bucket=bucket, Key=object_key)
            return response["Body"].read()
        except Exception as exc:
            raise self._format_storage_error("AWS S3", exc) from exc

    def _serialize_record(self, record: dict) -> dict:
        return dict(record)

    def save_dataset(self, dataset_id: str, df: pd.DataFrame, schema: dict, metadata: dict) -> dict:
        dataset_type = metadata.get("dataset_type", "real")
        relative_key = f"{dataset_type}/{dataset_id}.csv"
        object_key   = self._dataset_key(relative_key)

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
                self.supabase.table(self.datasets_table)
                .select("*")
                .eq("id", dataset_id)
                .limit(1)
                .execute()
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
                self.supabase.table(self.training_jobs_table)
                .select("*")
                .eq("job_id", job_or_dataset_id)
                .limit(1)
                .execute()
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
        relative_key = f"ctgan/{dataset_id}.pkl"
        object_key   = self._model_key(relative_key)

        self._upload_bytes(
            self.model_bucket,
            object_key,
            local_model_path.read_bytes(),
            "application/octet-stream",
        )
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
                self.supabase.table(self.models_table)
                .select("*")
                .eq("dataset_id", dataset_id)
                .limit(1)
                .execute()
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
        supabase_status = {
            "configured": bool(self.supabase_url and self.supabase_key),
            "reachable": False,
        }
        s3_status = {
            "configured": bool(
                self.aws_region
                and self.aws_access_key_id
                and self.aws_secret_access_key
                and self.dataset_bucket
                and self.model_bucket
            ),
            "reachable": False,
            "region": self.aws_region,
            "dataset_bucket": f"{self.dataset_bucket}/{self.dataset_prefix.rstrip('/')}".rstrip("/"),
            "model_bucket":   f"{self.model_bucket}/{self.model_prefix.rstrip('/')}".rstrip("/"),
        }
        if supabase_status["configured"]:
            try:
                self.supabase.table(self.datasets_table).select("id").limit(1).execute()
                supabase_status["reachable"] = True
            except Exception as exc:
                supabase_status["error"] = str(exc)
        if s3_status["configured"]:
            try:
                self.s3.head_bucket(Bucket=self.dataset_bucket)
                s3_status["reachable"] = True
            except Exception as exc:
                s3_status["error"] = str(exc)
        return {"backend": "SupabaseS3Backend", "supabase": supabase_status, "s3": s3_status}