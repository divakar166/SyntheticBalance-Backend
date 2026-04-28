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
        "sdmetrics==0.25.0",
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
            "Modal runner requires a persistent backend (Supabase + S3). "
            "InMemoryBackend is not supported for remote training."
        )


@app.function(
    secrets=secrets,
    gpu="T4",
    timeout=60 * 60 * 2,
)
def train_ctgan_modal(
    dataset_id: str,
    job_id: str,
    # Basic
    epochs: int = 100,
    batch_size: int = 256,
    # Advanced
    generator_lr: float = 2e-4,
    discriminator_lr: float = 2e-4,
    discriminator_steps: int = 1,
    embedding_dim: int = 128,
    generator_dim: list[int] = (256, 256),
    discriminator_dim: list[int] = (256, 256),
    # Early stopping
    early_stopping: bool = True,
    early_stopping_patience: int = 20,
    early_stopping_min_delta: float = 0.001,
    # Post-training quality evaluation
    run_sdmetrics: bool = True,
    sdmetrics_n_samples: int = 2000,
) -> dict:
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
        "n_training_rows": int(len(df)),
        "updated_at": _utc_now_iso(),
    })


    def update_progress(current_epoch: int, total_epochs: int, metrics: dict):
        if metrics.get("stage", "epoch") == "batch":
            return

        updated_at = _utc_now_iso()
        current_job = backend.get_training_job(job_id) or job
        loss_history = list(current_job.get("loss_history") or [])
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

        print(
            f"[Epoch {current_epoch}/{total_epochs}] "
            f"G={metrics['generator_loss']:.4f}  D={metrics['discriminator_loss']:.4f}  "
            f"ratio={metrics.get('loss_ratio', 0):.3f}  "
            f"{metrics.get('samples_per_second', 0):.0f} rows/s"
        )

    ctgan = CTGANWrapper(
        schema,
        epochs=epochs,
        batch_size=batch_size,
        generator_lr=generator_lr,
        discriminator_lr=discriminator_lr,
        discriminator_steps=discriminator_steps,
        embedding_dim=embedding_dim,
        generator_dim=tuple(generator_dim),
        discriminator_dim=tuple(discriminator_dim),
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
    )

    try:
        configs = ctgan.get_config()
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

    sdmetrics_result: dict = {}
    if run_sdmetrics:
        try:
            print(f"[SDMetrics] Generating {sdmetrics_n_samples} samples for quality evaluation…")
            synthetic_sample = ctgan.generate(sdmetrics_n_samples)
            sdmetrics_result = ctgan.evaluate_quality(df, synthetic_sample, target_col=target_col)
            print(
                f"[SDMetrics] quality_score={sdmetrics_result.get('quality_score')}  "
                f"diagnostic_score={sdmetrics_result.get('diagnostic_score')}  "
                f"ml_efficacy={sdmetrics_result.get('ml_efficacy', {}).get('train_on_synthetic_test_on_real_f1')}"
            )
        except Exception as exc:
            print(f"[SDMetrics] Evaluation failed (non-fatal): {exc}")
            sdmetrics_result = {"error": str(exc)}

    ctgan.save(tmp_path)

    model_record = backend.save_model(
        dataset_id,
        tmp_path,
        metadata={
            "trained_at": _utc_now_iso(),
            "job_id": job_id,
            "source": "modal",
            "gpu": "T4",
            "user_id": job.get("user_id"),
            "config": ctgan.get_config(),
            "sdmetrics": sdmetrics_result,
        },
        config=configs,
    )
    tmp_path.unlink(missing_ok=True)

    final_job = backend.get_training_job(job_id) or {}
    loss_history = final_job.get("loss_history") or []

    history = ctgan.training_history
    epochs_trained = ctgan.convergence_epoch or epochs
    early_stopped = ctgan.convergence_epoch is not None

    avg_epoch_time = (
        sum(e.get("epoch_time_seconds", 0) for e in history) / len(history)
        if history else 0.0
    )
    avg_samples_per_second = (
        sum(e.get("samples_per_second", 0) for e in history) / len(history)
        if history else 0.0
    )
    final_mcs = history[-1].get("mode_collapse_score", -1) if history else -1
    final_loss_ratio = history[-1].get("loss_ratio", 0.0) if history else 0.0
    final_g_loss = history[-1]["generator_loss"] if history else 0.0
    final_d_loss = history[-1]["discriminator_loss"] if history else 0.0

    # Best generator loss (lowest = closest to fooling the discriminator)
    best_entry = min(history, key=lambda e: e["generator_loss"]) if history else {}
    best_g_loss = best_entry.get("generator_loss", 0.0)
    best_epoch = best_entry.get("epoch", 0)

    # Overall loss stability (std across all epochs)
    g_losses = [e["generator_loss"] for e in history]
    loss_stability_std = float(
        (sum((x - (sum(g_losses) / len(g_losses))) ** 2 for x in g_losses) / len(g_losses)) ** 0.5
    ) if g_losses else 0.0

    # Estimate steps_per_epoch from history timing (best effort)
    steps_per_epoch = max(len(df) // batch_size, 1)

    rich_metadata = {
        "status": "completed",
        "model_id": dataset_id,
        "model_path": model_record["object_key"],
        # Convergence & timing
        "epochs_trained": epochs_trained,
        "early_stopped": early_stopped,
        "convergence_epoch": ctgan.convergence_epoch,
        "training_time_seconds": float(ctgan.training_time_seconds),
        "avg_epoch_time_seconds": round(avg_epoch_time, 3),
        "steps_per_epoch": steps_per_epoch,
        # Loss & stability
        "final_loss": final_g_loss,
        "final_generator_loss": round(final_g_loss, 6),
        "final_discriminator_loss": round(final_d_loss, 6),
        "final_loss_ratio": round(final_loss_ratio, 4),
        "final_mode_collapse_score": final_mcs,
        "best_generator_loss": round(best_g_loss, 6),
        "best_epoch": best_epoch,
        "loss_stability_std": round(loss_stability_std, 6),
        # Throughput
        "n_training_rows": int(len(df)),
        "avg_samples_per_second": round(avg_samples_per_second, 1),
        # SDMetrics
        "sdmetrics": sdmetrics_result,
        # Infrastructure
        "source": "modal",
        "gpu": "T4",
        "config": ctgan.get_config(),
        "updated_at": _utc_now_iso(),
    }

    backend.update_training_job(job_id, rich_metadata)

    summary = {
        "job_id": job_id,
        "dataset_id": dataset_id,
        "epochs_trained": epochs_trained,
        "early_stopped": early_stopped,
        "convergence_epoch": ctgan.convergence_epoch,
        "training_time_seconds": ctgan.training_time_seconds,
        "avg_epoch_time_seconds": round(avg_epoch_time, 3),
        "final_generator_loss": round(final_g_loss, 6),
        "final_discriminator_loss": round(final_d_loss, 6),
        "final_loss_ratio": round(final_loss_ratio, 4),
        "final_mode_collapse_score": final_mcs,
        "best_generator_loss": round(best_g_loss, 6),
        "best_epoch": best_epoch,
        "loss_stability_std": round(loss_stability_std, 6),
        "n_training_rows": int(len(df)),
        "avg_samples_per_second": round(avg_samples_per_second, 1),
        "sdmetrics": sdmetrics_result,
        "model_path": model_record["object_key"],
        "config": ctgan.get_config(),
    }
    print("Training complete:", summary)
    return summary


@app.function(
    secrets=secrets,
    gpu="T4",
    timeout=60 * 60,
)
def generate_ctgan_modal(
    dataset_id: str,
    job_id: str,
    n_samples: int = 5000,
    run_sdmetrics: bool = True,
) -> dict:
    import sys
    if "/backend" not in sys.path:
        sys.path.insert(0, "/backend")

    from services.generation import generate_synthetic_dataset

    env = dict(os.environ)
    backend = _get_backend(env)

    job = backend.get_generation_job(job_id)
    if not job:
        raise RuntimeError(f"Generation job '{job_id}' not found in storage.")

    started_at = datetime.now(timezone.utc)
    backend.update_generation_job(job_id, {
        "status": "running",
        "started_at": started_at.isoformat(),
        "updated_at": _utc_now_iso(),
    })

    try:
        result = generate_synthetic_dataset(
            backend=backend,
            dataset_id=dataset_id,
            n_samples=n_samples,
            job_id=job_id,
            source="modal",
        )
    except Exception as exc:
        elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
        backend.update_generation_job(job_id, {
            "status": "failed",
            "error": str(exc),
            "generation_time_seconds": float(elapsed),
            "updated_at": _utc_now_iso(),
        })
        raise

    sdmetrics_result: dict = {}
    if run_sdmetrics:
        try:
            from generators.ctgan import build_sdmetrics_metadata, _sdmetrics_report

            real_dataset = backend.get_dataset(dataset_id)
            synthetic_dataset = backend.get_dataset(result["synthetic_id"])

            if real_dataset and synthetic_dataset:
                real_df = real_dataset["df"]
                synthetic_df = synthetic_dataset["df"]
                schema = real_dataset["schema"]
                target_col = (
                    real_dataset.get("target")
                    or (schema.get("target") or {}).get("name")
                )
                metadata = build_sdmetrics_metadata(schema, target_col=target_col)
                sdmetrics_result = _sdmetrics_report(real_df, synthetic_df, metadata)
                print(
                    f"[SDMetrics] quality_score={sdmetrics_result.get('quality_score')}  "
                    f"diagnostic_score={sdmetrics_result.get('diagnostic_score')}"
                )
        except Exception as exc:
            print(f"[SDMetrics] Generation evaluation failed (non-fatal): {exc}")
            sdmetrics_result = {"error": str(exc)}

    backend.update_generation_job(job_id, {
        "status": "completed",
        "synthetic_id": result["synthetic_id"],
        "synthetic_path": result["synthetic_path"],
        "n_samples": result["n_samples"],
        "preview": result["preview"],
        "generation_time_seconds": result["generation_time_seconds"],
        "sdmetrics": sdmetrics_result,
        "last_heartbeat": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
    })

    summary = {
        "job_id": job_id,
        "dataset_id": dataset_id,
        "synthetic_id": result["synthetic_id"],
        "n_samples": result["n_samples"],
        "generation_time_seconds": result["generation_time_seconds"],
        "sdmetrics": sdmetrics_result,
    }
    print("Generation complete:", summary)
    return summary


@app.local_entrypoint()
def main(
    dataset_id: str,
    job_id: str,
    epochs: int = 100,
    batch_size: int = 256,
    n_samples: int = 5000,
    mode: str = "train",
    early_stopping: bool = True,
    run_sdmetrics: bool = True,
):
    if mode == "generate":
        result = generate_ctgan_modal.remote(
            dataset_id=dataset_id,
            job_id=job_id,
            n_samples=n_samples,
            run_sdmetrics=run_sdmetrics,
        )
    else:
        result = train_ctgan_modal.remote(
            dataset_id=dataset_id,
            job_id=job_id,
            epochs=epochs,
            batch_size=batch_size,
            early_stopping=early_stopping,
            run_sdmetrics=run_sdmetrics,
        )
    print("Result:", result)