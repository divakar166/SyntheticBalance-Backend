from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import uuid

from settings import get_settings
from services.state import get_storage_backend
from services.uploads import create_dataset_record
from services.utils import utc_now_iso

logger = logging.getLogger(__name__)


def find_generation_job(job_or_dataset_id: str) -> dict | None:
    return get_storage_backend().get_generation_job(job_or_dataset_id)


def create_generation_job(dataset_id: str, n_samples: int, user_id: str | None = None) -> dict:
    if n_samples <= 0:
        raise ValueError("n_samples must be greater than 0.")

    job = {
        "job_id": str(uuid.uuid4()),
        "dataset_id": dataset_id,
        "user_id": user_id,
        "status": "queued",
        "n_samples": int(n_samples),
        "synthetic_id": None,
        "synthetic_path": None,
        "preview": [],
        "generation_time_seconds": None,
        "error": None,
        "modal_call_id": None,
        "last_heartbeat": None,
        "started_at": None,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
    }
    return get_storage_backend().save_generation_job(job)


def run_generation_job(job_id: str):
    """
    Called by FastAPI's BackgroundTasks. Chooses Modal or local execution.
    """
    if get_settings().use_modal:
        _dispatch_to_modal(job_id)
    else:
        _run_local(job_id)


def _dispatch_to_modal(job_id: str):
    backend = get_storage_backend()
    job = backend.get_generation_job(job_id)
    if not job:
        logger.error("Generation job %s not found - cannot dispatch to Modal", job_id)
        return

    try:
        import modal

        generate_ctgan_modal = modal.Function.from_name(
            "synthetic-data-ctgan",
            "generate_ctgan_modal",
        )

        dataset_id = job["dataset_id"]
        n_samples = int(job["n_samples"])

        logger.info(
            "Dispatching generation job %s (dataset=%s, n_samples=%d) to Modal",
            job_id,
            dataset_id,
            n_samples,
        )

        backend.update_generation_job(job_id, {
            "status": "running",
            "updated_at": utc_now_iso(),
        })

        call = generate_ctgan_modal.spawn(
            dataset_id=dataset_id,
            job_id=job_id,
            n_samples=n_samples,
        )

        backend.update_generation_job(job_id, {
            "modal_call_id": call.object_id,
            "updated_at": utc_now_iso(),
        })

        logger.info("Modal generation call spawned: %s", call.object_id)

    except Exception as exc:
        logger.exception("Failed to dispatch generation job %s to Modal", job_id)
        backend.update_generation_job(job_id, {
            "status": "failed",
            "error": f"Modal dispatch error: {exc}",
            "updated_at": utc_now_iso(),
        })


def _run_local(job_id: str):
    backend = get_storage_backend()
    job = backend.get_generation_job(job_id)
    if not job:
        logger.error("Generation job %s not found", job_id)
        return

    started_at = datetime.now(timezone.utc)
    backend.update_generation_job(job_id, {
        "status": "running",
        "started_at": started_at.isoformat(),
        "updated_at": utc_now_iso(),
    })

    try:
        result = generate_synthetic_dataset(
            backend=backend,
            dataset_id=job["dataset_id"],
            n_samples=int(job["n_samples"]),
            job_id=job_id,
            source="local",
        )
        backend.update_generation_job(job_id, {
            "status": "completed",
            "synthetic_id": result["synthetic_id"],
            "synthetic_path": result["synthetic_path"],
            "n_samples": result["n_samples"],
            "preview": result["preview"],
            "generation_time_seconds": result["generation_time_seconds"],
            "last_heartbeat": utc_now_iso(),
            "updated_at": utc_now_iso(),
        })
        logger.info(
            "Generation job %s completed in %.1fs | synthetic dataset -> %s",
            job_id,
            result["generation_time_seconds"],
            result["synthetic_path"],
        )
    except Exception as exc:
        elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
        logger.exception("Synthetic generation failed for dataset %s", job["dataset_id"])
        backend.update_generation_job(job_id, {
            "status": "failed",
            "error": str(exc),
            "generation_time_seconds": float(elapsed),
            "updated_at": utc_now_iso(),
        })


def generate_synthetic_dataset(
    *,
    backend,
    dataset_id: str,
    n_samples: int,
    job_id: str | None,
    source: str,
) -> dict:
    from generators.ctgan import CTGANWrapper

    start = perf_counter()
    model_record = backend.get_model(dataset_id)
    if not model_record:
        raise FileNotFoundError(f"Model for dataset '{dataset_id}' not found.")

    temp_file, _ = backend.download_model_to_tempfile(dataset_id)
    try:
        ctgan = CTGANWrapper.load(temp_file.name)
    finally:
        Path(temp_file.name).unlink(missing_ok=True)

    synthetic_df = ctgan.generate(n_samples)

    source_dataset = backend.get_dataset(dataset_id)
    if not source_dataset:
        raise FileNotFoundError(f"Dataset '{dataset_id}' not found.")

    target = (
        source_dataset.get("target")
        or (source_dataset.get("schema") or {}).get("target", {}).get("name")
        or (source_dataset.get("metadata") or {}).get("target")
    )
    if not target:
        raise ValueError(f"Dataset '{dataset_id}' does not have a target column configured.")

    synthetic_record = create_dataset_record(
        synthetic_df,
        f"synthetic_{dataset_id}.csv",
        target,
        user_id=source_dataset.get("user_id") or (source_dataset.get("metadata") or {}).get("user_id"),
        dataset_type="synthetic",
        extra_metadata={
            "source_dataset_id": dataset_id,
            "generation_job_id": job_id,
            "source": source,
        },
        storage_backend=backend,
    )

    elapsed = perf_counter() - start
    return {
        "synthetic_id": synthetic_record["dataset_id"],
        "synthetic_path": synthetic_record.get("object_key"),
        "n_samples": len(synthetic_df),
        "preview": synthetic_df.head(5).to_dict(orient="records"),
        "generation_time_seconds": float(elapsed),
    }


def generation_status_payload(job: dict) -> dict:
    return {
        "job_id": job["job_id"],
        "dataset_id": job["dataset_id"],
        "status": job["status"],
        "n_samples": job.get("n_samples"),
        "synthetic_id": job.get("synthetic_id"),
        "synthetic_path": job.get("synthetic_path"),
        "preview": job.get("preview") or [],
        "generation_time_seconds": job.get("generation_time_seconds"),
        "error": job.get("error"),
        "last_heartbeat": job.get("last_heartbeat"),
        "modal_call_id": job.get("modal_call_id"),
    }
