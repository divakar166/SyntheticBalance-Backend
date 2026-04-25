from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "ctgan==0.11.1",
        "pandas==3.0.2",
        "numpy==2.4.4",
        "torch==2.11.0",
        "scikit-learn==1.8.0",
        "supabase==2.28.3",
        "boto3==1.42.95",
        "minio==7.2.20",
        "pydantic-settings==2.14.0",
        "python-dotenv==1.2.2",
    )
    .add_local_python_source("generators")
    .add_local_python_source("handlers")
    .add_local_python_source("persistence")
    .add_local_python_source("services")
    .add_local_python_source("settings")
)

app = modal.App("synthetic-data-ctgan", image=image)

secrets = [modal.Secret.from_name("synthetic-data-secrets")]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_backend(env: dict):
    has_supabase = bool(env.get("SUPABASE_URL") and env.get("SUPABASE_SERVICE_ROLE_KEY"))
    has_s3 = bool(
        env.get("AWS_REGION")
        and env.get("AWS_ACCESS_KEY_ID")
        and env.get("AWS_SECRET_ACCESS_KEY")
        and env.get("AWS_S3_DATASET_BUCKET")
        and env.get("AWS_S3_MODEL_BUCKET")
    )

    if has_supabase and has_s3:
        from persistence.backends.supabase_s3 import SupabaseS3Backend
        return SupabaseS3Backend()
    else:
        raise RuntimeError(
            "Modal runner requires a persistent backend (Supabase + S3 or Supabase + MinIO). "
            "InMemoryBackend is not supported for remote training."
        )


@app.function(
    secrets=secrets,
    gpu="T4",
    timeout=60 * 60 * 2
)
def train_ctgan_modal(dataset_id: str, job_id: str, epochs: int = 100) -> dict:
    import sys
    if "/backend" not in sys.path:
        sys.path.insert(0, "/backend")

    from generators.ctgan import CTGANWrapper

    env = dict(os.environ)
    backend = _get_backend(env)

    job = backend.get_training_job(job_id)
    if not job:
        raise RuntimeError(f"Training job '{job_id}' not found in storage.")

    dataset = backend.get_dataset(dataset_id)
    if not dataset:
        backend.update_training_job(job_id, {
            "status": "failed",
            "error": f"Dataset '{dataset_id}' not found.",
            "updated_at": _utc_now_iso(),
        })
        raise RuntimeError(f"Dataset '{dataset_id}' not found.")

    df = dataset["df"]
    schema = dataset["schema"]
    target_col = schema["target"]["name"]

    started_at = datetime.now(timezone.utc)
    backend.update_training_job(job_id, {
        "status": "running",
        "started_at": started_at.isoformat(),
        "updated_at": _utc_now_iso(),
    })

    def update_progress(current_epoch: int, total_epochs: int, metrics: dict):
        if metrics.get("stage", "epoch") == "batch":
            return
 
        updated_at  = _utc_now_iso()
        current_job = backend.get_training_job(job_id) or job
        loss_history = list(current_job.get("loss_history") or [])
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
 
        print(
            f"[Epoch {current_epoch}/{total_epochs}] "
            f"G={metrics['generator_loss']:.4f}  D={metrics['discriminator_loss']:.4f}"
        )

    ctgan = CTGANWrapper(schema, epochs=epochs)
    try:
        ctgan.train(df, target_col=target_col, progress_callback=update_progress)
    except Exception as exc:
        elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
        backend.update_training_job(job_id, {
            "status": "failed",
            "error": str(exc),
            "training_time_seconds": float(elapsed),
            "updated_at": _utc_now_iso(),
        })
        raise

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    ctgan.save(tmp_path)

    model_record = backend.save_model(
        dataset_id,
        tmp_path,
        metadata={"trained_at": _utc_now_iso(), "job_id": job_id, "source": "modal"},
    )
    tmp_path.unlink(missing_ok=True)

    final_job = backend.get_training_job(job_id) or {}
    backend.update_training_job(job_id, {
        "status": "completed",
        "model_id": dataset_id,
        "model_path": model_record["object_key"],
        "training_time_seconds": float(ctgan.training_time_seconds),
        "final_loss": float(
            (final_job.get("loss_history") or [{}])[-1].get("generator_loss", 0)
        ),
        "updated_at": _utc_now_iso(),
    })

    summary = {
        "job_id": job_id,
        "dataset_id": dataset_id,
        "epochs_trained": epochs,
        "training_time_seconds": ctgan.training_time_seconds,
        "model_path": model_record["object_key"],
    }
    print("Training complete:", summary)
    return summary



@app.local_entrypoint()
def main(dataset_id: str, job_id: str, epochs: int = 100):
    result = train_ctgan_modal.remote(
        dataset_id=dataset_id,
        job_id=job_id,
        epochs=epochs,
    )
    print("Result:", result)