from __future__ import annotations

import logging
from pathlib import Path
import uuid
from datetime import datetime, timezone
from settings import get_settings

from fastapi import HTTPException

from services.state import get_storage_backend
from services.utils import build_loss_history, utc_now_iso

logger = logging.getLogger(__name__)
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def storage_operation_error(exc: Exception) -> HTTPException:
    return HTTPException(status_code=503, detail=str(exc))


def find_training_job(job_or_dataset_id: str) -> dict | None:
    return get_storage_backend().get_training_job(job_or_dataset_id)


def create_training_job(dataset_id: str, epochs: int, user_id: str | None = None) -> dict:
    job = {
        "job_id": str(uuid.uuid4()),
        "dataset_id": dataset_id,
        "user_id": user_id,
        "status": "queued",
        "current_epoch": 0,
        "total_epochs": epochs,
        "loss_history": [],
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


def _use_modal() -> bool:
    settings = get_settings()
    return settings.use_modal


def run_training_job(job_id: str):
    """
    Called by FastAPI's BackgroundTasks.  Chooses Modal or local execution.
    """
    if _use_modal():
        _dispatch_to_modal(job_id)
    else:
        _run_local(job_id)


def _dispatch_to_modal(job_id: str):
    backend = get_storage_backend()
    job = backend.get_training_job(job_id)
    if not job:
        logger.error("Job %s not found - cannot dispatch to Modal", job_id)
        return

    try:
        import modal

        train_ctgan_modal = modal.Function.from_name(
            "synthetic-data-ctgan",
            "train_ctgan_modal",
        )

        dataset_id = job["dataset_id"]
        epochs     = job["total_epochs"]

        logger.info(
            "Dispatching job %s (dataset=%s, epochs=%d) to Modal",
            job_id, dataset_id, epochs,
        )

        backend.update_training_job(job_id, {
            "status": "running",
            "updated_at": utc_now_iso(),
        })

        call = train_ctgan_modal.spawn(
            dataset_id=dataset_id,
            job_id=job_id,
            epochs=epochs,
        )

        backend.update_training_job(job_id, {
            "modal_call_id": call.object_id,
            "updated_at": utc_now_iso(),
        })
 
        logger.info("Modal call spawned: %s", call.object_id)

    except Exception as exc:
        logger.exception("Failed to dispatch job %s to Modal", job_id)
        backend.update_training_job(job_id, {
            "status": "failed",
            "error": f"Modal dispatch error: {exc}",
            "updated_at": utc_now_iso(),
        })


def _run_local(job_id: str):
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
        if metrics.get("stage", "epoch") == "batch":
            return
        
        updated_at  = utc_now_iso()
        current_job = backend.get_training_job(job_id) or job
        loss_history = list(current_job.get("loss_history", []))
        loss_history.append({
            "epoch": current_epoch,
            "generator_loss": float(metrics["generator_loss"]),
            "discriminator_loss": float(metrics["discriminator_loss"]),
        })

        backend.update_training_job(job_id, {
            "current_epoch": current_epoch,
            "total_epochs": total_epochs,
            "loss_history": loss_history,
            "final_loss": float(metrics["generator_loss"]),
            "last_heartbeat": updated_at,
            "updated_at": updated_at,
        })
 
        logger.info(
            "Job %s | epoch %d/%d | G=%.4f D=%.4f",
            job_id, current_epoch, total_epochs,
            metrics["generator_loss"], metrics["discriminator_loss"],
        )

    try:
        ctgan.train(df, target_col=target_col, progress_callback=update_progress)
 
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        local_model_path = MODELS_DIR / f"ctgan_{dataset_id}.pkl"
        ctgan.save(local_model_path)
        model_record = backend.save_model(
            dataset_id,
            local_model_path,
            metadata={
                "trained_at": utc_now_iso(),
                "job_id": job_id,
                "source": "local",
                "user_id": job.get("user_id"),
            },
        )
        local_model_path.unlink(missing_ok=True)

        current_job = backend.get_training_job(job_id) or job
        final_loss  = None
        if current_job.get("loss_history"):
            final_loss = float(current_job["loss_history"][-1]["generator_loss"])
 
        backend.update_training_job(job_id, {
            "status": "completed",
            "model_id": dataset_id,
            "model_path": model_record["object_key"],
            "training_time_seconds": float(ctgan.training_time_seconds),
            "final_loss": final_loss,
            "updated_at": utc_now_iso(),
        })
 
        logger.info(
            "Job %s completed in %.1fs | model -> %s",
            job_id, ctgan.training_time_seconds, model_record["object_key"],
        )

    except Exception as exc:
        logger.exception("CTGAN training failed for dataset %s", dataset_id)
        backend.update_training_job(job_id, {
            "status": "failed",
            "error": str(exc),
            "training_time_seconds": float((datetime.now(timezone.utc) - started_at).total_seconds()),
            "updated_at": utc_now_iso(),
        })


def training_status_payload(job: dict) -> dict:
    return {
        "job_id": job["job_id"],
        "dataset_id": job["dataset_id"],
        "status": job["status"],
        "current_epoch": job["current_epoch"],
        "total_epochs": job["total_epochs"],
        "loss_history": build_loss_history(job.get("loss_history", [])),
        "training_time_seconds": job.get("training_time_seconds"),
        "final_loss": job.get("final_loss"),
        "error": job.get("error"),
        "model_id": job.get("model_id"),
        "last_heartbeat": job.get("last_heartbeat"),
        "modal_call_id": job.get("modal_call_id"),
    }
