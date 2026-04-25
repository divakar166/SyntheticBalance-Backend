from __future__ import annotations

import logging
from pathlib import Path
import uuid
from datetime import datetime, timezone

from fastapi import HTTPException

from services.state import get_storage_backend
from services.utils import build_batch_history, build_loss_history, utc_now_iso

logger = logging.getLogger(__name__)
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def storage_operation_error(exc: Exception) -> HTTPException:
    return HTTPException(status_code=503, detail=str(exc))


def find_training_job(job_or_dataset_id: str) -> dict | None:
    return get_storage_backend().get_training_job(job_or_dataset_id)


def create_training_job(dataset_id: str, epochs: int) -> dict:
    job = {
        "job_id": str(uuid.uuid4()),
        "dataset_id": dataset_id,
        "status": "running",
        "current_epoch": 0,
        "total_epochs": epochs,
        "current_batch": 0,
        "total_batches": 0,
        "loss_history": [],
        "batch_history": [],
        "training_time_seconds": None,
        "final_loss": None,
        "error": None,
        "model_id": None,
        "model_path": None,
        "last_heartbeat": None,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
    }
    return get_storage_backend().save_training_job(job)


def run_training_job(job_id: str):
    from generators.ctgan import CTGANWrapper

    backend = get_storage_backend()
    job = backend.get_training_job(job_id)
    if not job:
        logger.error("Training job %s not found", job_id)
        return

    dataset_id = job["dataset_id"]
    dataset = backend.get_dataset(dataset_id)
    if not dataset:
        backend.update_training_job(
            job_id,
            {
                "status": "failed",
                "error": f"Dataset '{dataset_id}' not found.",
                "updated_at": utc_now_iso(),
            },
        )
        return

    df = dataset["df"]
    schema = dataset["schema"]
    target_col = schema["target"]["name"]
    ctgan = CTGANWrapper(schema, epochs=job["total_epochs"])
    started_at = datetime.now(timezone.utc)
    backend.update_training_job(
        job_id,
        {"started_at": started_at.isoformat(), "status": "running", "updated_at": utc_now_iso()},
    )

    def update_progress(current_epoch: int, total_epochs: int, metrics: dict):
        updated_at = utc_now_iso()
        current_job = backend.get_training_job(job_id) or job
        updates = {
            "current_epoch": current_epoch,
            "total_epochs": total_epochs,
            "updated_at": updated_at,
            "last_heartbeat": updated_at,
        }
        if metrics.get("current_batch") is not None:
            updates["current_batch"] = int(metrics["current_batch"])
        if metrics.get("total_batches") is not None:
            updates["total_batches"] = int(metrics["total_batches"])

        if metrics.get("stage", "epoch") == "batch":
            batch_history = list(current_job.get("batch_history", []))
            batch_history.append(
                {
                    "epoch": current_epoch,
                    "batch": int(metrics.get("current_batch", 0)),
                    "total_batches": int(metrics.get("total_batches", 0)),
                    "generator_loss": metrics.get("generator_loss"),
                    "discriminator_loss": metrics.get("discriminator_loss"),
                    "updated_at": updated_at,
                }
            )
            updates["batch_history"] = batch_history[-250:]
        else:
            loss_history = list(current_job.get("loss_history", []))
            loss_history.append(
                {
                    "epoch": current_epoch,
                    "generator_loss": float(metrics["generator_loss"]),
                    "discriminator_loss": float(metrics["discriminator_loss"]),
                }
            )
            updates["loss_history"] = loss_history
            updates["final_loss"] = float(metrics["generator_loss"])

        backend.update_training_job(job_id, updates)

    try:
        ctgan.train(df, target_col=target_col, progress_callback=update_progress)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        local_model_path = MODELS_DIR / f"ctgan_{dataset_id}.pkl"
        ctgan.save(local_model_path)
        model_record = backend.save_model(
            dataset_id,
            local_model_path,
            metadata={"trained_at": utc_now_iso(), "job_id": job_id},
        )
        if local_model_path.exists():
            local_model_path.unlink()

        current_job = backend.get_training_job(job_id) or job
        updates = {
            "status": "completed",
            "model_id": dataset_id,
            "model_path": model_record["object_key"],
            "training_time_seconds": float(ctgan.training_time_seconds),
            "updated_at": utc_now_iso(),
        }
        if current_job.get("loss_history"):
            updates["final_loss"] = float(current_job["loss_history"][-1]["generator_loss"])
        backend.update_training_job(job_id, updates)
    except Exception as exc:
        logger.exception("CTGAN training failed for dataset %s", dataset_id)
        backend.update_training_job(
            job_id,
            {
                "status": "failed",
                "error": str(exc),
                "training_time_seconds": float((datetime.now(timezone.utc) - started_at).total_seconds()),
                "updated_at": utc_now_iso(),
            },
        )


def training_status_payload(job: dict) -> dict:
    return {
        "job_id": job["job_id"],
        "dataset_id": job["dataset_id"],
        "status": job["status"],
        "current_epoch": job["current_epoch"],
        "total_epochs": job["total_epochs"],
        "current_batch": job["current_batch"],
        "total_batches": job["total_batches"],
        "loss_history": build_loss_history(job.get("loss_history", [])),
        "batch_history": build_batch_history(job.get("batch_history", [])),
        "training_time_seconds": job.get("training_time_seconds"),
        "final_loss": job.get("final_loss"),
        "error": job.get("error"),
        "model_id": job.get("model_id"),
        "last_heartbeat": job.get("last_heartbeat"),
    }


def training_batch_history_payload(job: dict) -> dict:
    return {
        "job_id": job["job_id"],
        "dataset_id": job["dataset_id"],
        "status": job["status"],
        "current_epoch": job["current_epoch"],
        "total_epochs": job["total_epochs"],
        "current_batch": job["current_batch"],
        "total_batches": job["total_batches"],
        "batch_history": build_batch_history(job.get("batch_history", [])),
        "last_heartbeat": job.get("last_heartbeat"),
        "error": job.get("error"),
    }
