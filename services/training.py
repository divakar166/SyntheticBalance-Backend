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


def create_training_job(
    dataset_id: str,
    epochs: int,
    user_id: str | None = None,
    extra: dict | None = None,
) -> dict:
    extra = extra or {}
    job = {
        "job_id": str(uuid.uuid4()),
        "dataset_id": dataset_id,
        "user_id": user_id,
        "status": "queued",
        "current_epoch": 0,
        "total_epochs": epochs,
        "config": {
            "epochs": epochs,
            "batch_size": extra.get("batch_size", 256),
            "embedding_dim": extra.get("embedding_dim", 128),
            "generator_dim": extra.get("generator_dim", [256, 256]),
            "discriminator_dim": extra.get("discriminator_dim", [256, 256]),
            "generator_lr": extra.get("generator_lr", 2e-4),
            "discriminator_lr": extra.get("discriminator_lr", 2e-4),
            "discriminator_steps": extra.get("discriminator_steps", 1),
            "early_stopping": extra.get("early_stopping", True),
            "early_stopping_patience": extra.get("early_stopping_patience", 20),
            "early_stopping_min_delta": extra.get("early_stopping_min_delta", 0.001),
            "run_sdmetrics": extra.get("run_sdmetrics", True),
            "sdmetrics_n_samples": extra.get("sdmetrics_n_samples", 2000),
        },
        "loss_history": [],
        "training_time_seconds": None,
        "final_loss": None,
        "final_generator_loss": None,
        "final_discriminator_loss": None,
        "final_loss_ratio": None,
        "final_mode_collapse_score": None,
        "best_generator_loss": None,
        "best_epoch": None,
        "loss_stability_std": None,
        "n_training_rows": None,
        "avg_epoch_time_seconds": None,
        "avg_samples_per_second": None,
        "steps_per_epoch": None,
        "epochs_trained": None,
        "early_stopped": None,
        "convergence_epoch": None,
        "sdmetrics": None,
        "model_id": None,
        "model_path": None,
        "modal_call_id": None,
        "last_heartbeat": None,
        "error": None,
        "source": None,
        "gpu": None,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
    }
    return get_storage_backend().save_training_job(job)


def _use_modal() -> bool:
    return get_settings().use_modal


def run_training_job(job_id: str):
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
        cfg = job.get("config") or {}

        logger.info(
            "Dispatching job %s (dataset=%s, epochs=%d) to Modal",
            job_id, dataset_id, cfg.get("epochs", job["total_epochs"]),
        )

        backend.update_training_job(job_id, {
            "status": "running",
            "updated_at": utc_now_iso(),
        })

        call = train_ctgan_modal.spawn(
            dataset_id=dataset_id,
            job_id=job_id,
            epochs=cfg.get("epochs", job["total_epochs"]),
            batch_size=cfg.get("batch_size", 256),
            generator_lr=cfg.get("generator_lr", 2e-4),
            discriminator_lr=cfg.get("discriminator_lr", 2e-4),
            discriminator_steps=cfg.get("discriminator_steps", 1),
            embedding_dim=cfg.get("embedding_dim", 128),
            generator_dim=cfg.get("generator_dim", [256, 256]),
            discriminator_dim=cfg.get("discriminator_dim", [256, 256]),
            early_stopping=cfg.get("early_stopping", True),
            early_stopping_patience=cfg.get("early_stopping_patience", 20),
            early_stopping_min_delta=cfg.get("early_stopping_min_delta", 0.001),
            run_sdmetrics=cfg.get("run_sdmetrics", True),
            sdmetrics_n_samples=cfg.get("sdmetrics_n_samples", 2000),
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
        backend.update_training_job(job_id, {
            "status": "failed",
            "error": f"Dataset '{dataset_id}' not found.",
            "updated_at": utc_now_iso(),
        })
        return

    df = dataset["df"]
    schema = dataset["schema"]
    target_col = schema["target"]["name"]
    cfg = job.get("config") or {}

    ctgan = CTGANWrapper(
        schema,
        epochs=cfg.get("epochs", job["total_epochs"]),
        batch_size=cfg.get("batch_size", 256),
        generator_lr=cfg.get("generator_lr", 2e-4),
        discriminator_lr=cfg.get("discriminator_lr", 2e-4),
        discriminator_steps=cfg.get("discriminator_steps", 1),
        embedding_dim=cfg.get("embedding_dim", 128),
        generator_dim=tuple(cfg.get("generator_dim", [256, 256])),
        discriminator_dim=tuple(cfg.get("discriminator_dim", [256, 256])),
        early_stopping=cfg.get("early_stopping", True),
        early_stopping_patience=cfg.get("early_stopping_patience", 20),
        early_stopping_min_delta=cfg.get("early_stopping_min_delta", 0.001),
    )

    started_at = datetime.now(timezone.utc)
    backend.update_training_job(job_id, {
        "started_at": started_at.isoformat(),
        "status": "running",
        "n_training_rows": int(len(df)),
        "source": "local",
        "updated_at": utc_now_iso(),
    })

    def update_progress(current_epoch: int, total_epochs: int, metrics: dict):
        if metrics.get("stage", "epoch") == "batch":
            return
        updated_at = utc_now_iso()
        current_job = backend.get_training_job(job_id) or job
        loss_history = list(current_job.get("loss_history", []))
        loss_history.append({
            "epoch": current_epoch,
            "generator_loss": float(metrics["generator_loss"]),
            "discriminator_loss": float(metrics["discriminator_loss"]),
            "loss_ratio": float(metrics.get("loss_ratio", 0)),
            "epoch_time_seconds": float(metrics.get("epoch_time_seconds", 0)),
            "samples_per_second": float(metrics.get("samples_per_second", 0)),
            **({"mode_collapse_score": float(metrics["mode_collapse_score"])}
               if "mode_collapse_score" in metrics else {}),
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
            "Job %s | epoch %d/%d | G=%.4f D=%.4f ratio=%.3f",
            job_id, current_epoch, total_epochs,
            metrics["generator_loss"], metrics["discriminator_loss"],
            metrics.get("loss_ratio", 0),
        )

    try:
        ctgan.train(df, target_col=target_col, progress_callback=update_progress)

        # Optional SDMetrics evaluation
        sdmetrics_result: dict = {}
        if cfg.get("run_sdmetrics", True):
            try:
                n = int(cfg.get("sdmetrics_n_samples", 2000))
                synthetic_sample = ctgan.generate(n)
                sdmetrics_result = ctgan.evaluate_quality(df, synthetic_sample, target_col=target_col)
                logger.info(
                    "Job %s | SDMetrics quality=%.4f diagnostic=%.4f",
                    job_id,
                    sdmetrics_result.get("quality_score", 0),
                    sdmetrics_result.get("diagnostic_score", 0),
                )
            except Exception as exc:
                logger.warning("SDMetrics evaluation failed for job %s: %s", job_id, exc)
                sdmetrics_result = {"error": str(exc)}

        # Save model
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
                "config": ctgan.get_config(),
                "sdmetrics": sdmetrics_result,
            },
        )
        local_model_path.unlink(missing_ok=True)

        # Rich completion metadata
        history = ctgan.training_history
        epochs_trained = ctgan.convergence_epoch or cfg.get("epochs", job["total_epochs"])
        early_stopped = ctgan.convergence_epoch is not None

        avg_epoch_time = (
            sum(e.get("epoch_time_seconds", 0) for e in history) / len(history)
            if history else 0.0
        )
        avg_sps = (
            sum(e.get("samples_per_second", 0) for e in history) / len(history)
            if history else 0.0
        )
        final_entry = history[-1] if history else {}
        best_entry = min(history, key=lambda e: e["generator_loss"]) if history else {}
        g_losses = [e["generator_loss"] for e in history]
        loss_std = float(
            (sum((x - (sum(g_losses) / len(g_losses))) ** 2 for x in g_losses) / len(g_losses)) ** 0.5
        ) if g_losses else 0.0

        backend.update_training_job(job_id, {
            "status": "completed",
            "model_id": dataset_id,
            "model_path": model_record["object_key"],
            "training_time_seconds": float(ctgan.training_time_seconds),
            "epochs_trained": epochs_trained,
            "early_stopped": early_stopped,
            "convergence_epoch": ctgan.convergence_epoch,
            "avg_epoch_time_seconds": round(avg_epoch_time, 3),
            "avg_samples_per_second": round(avg_sps, 1),
            "steps_per_epoch": max(len(df) // cfg.get("batch_size", 256), 1),
            "final_loss": final_entry.get("generator_loss", 0.0),
            "final_generator_loss": round(float(final_entry.get("generator_loss", 0)), 6),
            "final_discriminator_loss": round(float(final_entry.get("discriminator_loss", 0)), 6),
            "final_loss_ratio": round(float(final_entry.get("loss_ratio", 0)), 4),
            "final_mode_collapse_score": final_entry.get("mode_collapse_score", -1),
            "best_generator_loss": round(float(best_entry.get("generator_loss", 0)), 6),
            "best_epoch": best_entry.get("epoch", 0),
            "loss_stability_std": round(loss_std, 6),
            "n_training_rows": int(len(df)),
            "sdmetrics": sdmetrics_result,
            "source": "local",
            "config": ctgan.get_config(),
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
        # Core identity
        "job_id": job["job_id"],
        "dataset_id": job["dataset_id"],
        "status": job["status"],
        # Progress (live during training)
        "current_epoch": job.get("current_epoch", 0),
        "total_epochs": job.get("total_epochs"),
        "loss_history": build_loss_history(job.get("loss_history") or []),
        # Convergence & timing
        "epochs_trained": job.get("epochs_trained"),
        "early_stopped": job.get("early_stopped"),
        "convergence_epoch": job.get("convergence_epoch"),
        "training_time_seconds": job.get("training_time_seconds"),
        "avg_epoch_time_seconds": job.get("avg_epoch_time_seconds"),
        "steps_per_epoch": job.get("steps_per_epoch"),
        # Loss & stability
        "final_loss": job.get("final_loss"),
        "final_generator_loss": job.get("final_generator_loss"),
        "final_discriminator_loss": job.get("final_discriminator_loss"),
        "final_loss_ratio": job.get("final_loss_ratio"),
        "final_mode_collapse_score": job.get("final_mode_collapse_score"),
        "best_generator_loss": job.get("best_generator_loss"),
        "best_epoch": job.get("best_epoch"),
        "loss_stability_std": job.get("loss_stability_std"),
        # Throughput
        "n_training_rows": job.get("n_training_rows"),
        "avg_samples_per_second": job.get("avg_samples_per_second"),
        # SDMetrics quality report
        "sdmetrics": job.get("sdmetrics"),
        # References & infrastructure
        "model_id": job.get("model_id"),
        "model_path": job.get("model_path"),
        "config": job.get("config"),
        "source": job.get("source"),
        "gpu": job.get("gpu"),
        "error": job.get("error"),
        "last_heartbeat": job.get("last_heartbeat"),
        "modal_call_id": job.get("modal_call_id"),
    }