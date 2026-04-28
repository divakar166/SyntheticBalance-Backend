from __future__ import annotations

import os

from persistence import InMemoryBackend, SupabaseS3Backend

datasets = {}
models = {}
training_jobs = {}
generation_jobs = {}

_storage_backend = None


def get_storage_backend():
    global _storage_backend
    if _storage_backend is not None:
        return _storage_backend

    has_supabase = bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_ROLE_KEY"))
    has_s3 = bool(
        os.getenv("AWS_REGION")
        and os.getenv("AWS_ACCESS_KEY_ID")
        and os.getenv("AWS_SECRET_ACCESS_KEY")
        and os.getenv("AWS_S3_DATASET_BUCKET")
        and os.getenv("AWS_S3_MODEL_BUCKET")
    )

    if has_supabase and has_s3:
        _storage_backend = SupabaseS3Backend()
    else:
        _storage_backend = InMemoryBackend(datasets, training_jobs, models, generation_jobs)

    return _storage_backend
