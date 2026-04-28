import logging
import os
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from services.auth import AuthenticatedUser, ensure_user_owns_record, require_user
from services.state import get_storage_backend
from services.generation import (
    create_generation_job,
    find_generation_job,
    generation_status_payload,
    run_generation_job,
)
from services.training import (
    create_training_job,
    find_training_job,
    run_training_job,
    storage_operation_error,
    training_status_payload,
)
from services.uploads import (
    UploadValidationError,
    create_dataset_record,
    load_csv,
    validate_csv_file,
    validate_target_column,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PORT = int(os.getenv("PORT", 8000))

@asynccontextmanager
async def lifespan(app):
    try:
        health = get_storage_backend().get_health_status()
        logger.info("Storage backend: %s", health["backend"])
        for service_name, status in health.items():
            if service_name == "backend":
                continue
            logger.info("%s status: %s", service_name.capitalize(), status)
    except Exception:
        logger.exception("Storage startup check failed")
    yield


app = FastAPI(title="Synthetic Data Generator", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TrainRequest(BaseModel):
    dataset_id: str

    # Basics
    epochs: int = Field(
        default=100,
        ge=1,
        le=2000,
        description="Number of training epochs (max 2 000).",
    )
    batch_size: int = Field(
        default=256,
        ge=32,
        le=4096,
        description="Mini-batch size.  Will be rounded up to a multiple of `pac`.",
    )

    # Architecture
    embedding_dim: int = Field(
        default=128,
        ge=32,
        le=512,
        description="Dimension of the random noise embedding fed to the generator.",
    )
    generator_dim: list[int] = Field(
        default=[256, 256],
        min_length=1,
        max_length=5,
        description="Hidden-layer sizes for the generator MLP (1–5 layers).",
    )
    discriminator_dim: list[int] = Field(
        default=[256, 256],
        min_length=1,
        max_length=5,
        description="Hidden-layer sizes for the discriminator MLP (1–5 layers).",
    )

    # Optimiser
    generator_lr: float = Field(
        default=2e-4,
        gt=0,
        le=1e-1,
        description="Adam learning rate for the generator.",
    )
    discriminator_lr: float = Field(
        default=2e-4,
        gt=0,
        le=1e-1,
        description="Adam learning rate for the discriminator.",
    )
    discriminator_steps: int = Field(
        default=1,
        ge=1,
        le=10,
        description=(
            "How many discriminator update steps to perform per generator step. "
            "Higher values slow training but can improve discriminator quality."
        ),
    )

    # Early stopping
    early_stopping: bool = Field(
        default=True,
        description="Stop training once the generator loss converges.",
    )
    early_stopping_patience: int = Field(
        default=20,
        ge=5,
        le=200,
        description="Number of consecutive epochs with negligible loss change before stopping.",
    )
    early_stopping_min_delta: float = Field(
        default=0.001,
        gt=0,
        le=1.0,
        description="Minimum change in generator loss range to count as progress.",
    )

    # Post-training evaluation
    run_sdmetrics: bool = Field(
        default=True,
        description=(
            "Run SDMetrics Quality + Diagnostic reports immediately after training "
            "(on a small synthetic sample).  Adds ~30 s on a T4 GPU."
        ),
    )
    sdmetrics_n_samples: int = Field(
        default=2000,
        ge=100,
        le=20000,
        description="Number of synthetic rows to generate for the SDMetrics evaluation pass.",
    )


def _json_error(message: str, status_code: int, **extra):
    return JSONResponse(
        status_code=status_code,
        content={"error": message, "status": status_code, **extra},
    )


@app.get("/health")
def health_check():
    backend = get_storage_backend()
    return {
        "status": "ok",
        "version": "0.1.0",
        "storage": backend.get_health_status(),
    }


@app.post("/api/upload")
async def upload_csv(
    file: UploadFile = File(...),
    target: str = Form("fraud"),
    current_user: AuthenticatedUser = Depends(require_user),
):
    try:
        validate_csv_file(file.filename, file.content_type)
        content = await file.read()
        df = load_csv(content)
        validate_target_column(df, target)
        return create_dataset_record(df, file.filename or "dataset.csv", target, user_id=current_user.id)
    except UploadValidationError as exc:
        extra = {}
        if exc.available_columns:
            extra["available_columns"] = exc.available_columns
        return _json_error(exc.message, 400, **extra)
    except Exception:
        logger.exception("Unexpected error while parsing upload")
        return _json_error("Unexpected parsing error while processing the uploaded CSV.", 500)


@app.post("/api/train-ctgan")
async def train_ctgan(
    request: TrainRequest,
    background_tasks: BackgroundTasks,
    current_user: AuthenticatedUser = Depends(require_user),
):
    backend = get_storage_backend()
    try:
        dataset = backend.get_dataset(request.dataset_id)
    except Exception as exc:
        raise storage_operation_error(exc) from exc

    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset '{request.dataset_id}' not found.")
    ensure_user_owns_record(dataset, current_user, f"Dataset '{request.dataset_id}'")

    job = create_training_job(
        dataset_id=request.dataset_id,
        epochs=request.epochs,
        user_id=current_user.id,
        extra={
            "batch_size": request.batch_size,
            "embedding_dim": request.embedding_dim,
            "generator_dim": request.generator_dim,
            "discriminator_dim": request.discriminator_dim,
            "generator_lr": request.generator_lr,
            "discriminator_lr": request.discriminator_lr,
            "discriminator_steps": request.discriminator_steps,
            "early_stopping": request.early_stopping,
            "early_stopping_patience": request.early_stopping_patience,
            "early_stopping_min_delta": request.early_stopping_min_delta,
            "run_sdmetrics": request.run_sdmetrics,
            "sdmetrics_n_samples": request.sdmetrics_n_samples,
        },
    )
    background_tasks.add_task(run_training_job, job["job_id"])
    return {
        "job_id": job["job_id"],
        "dataset_id": request.dataset_id,
        "status": job["status"],
        "config": {
            "epochs": request.epochs,
            "batch_size": request.batch_size,
            "embedding_dim": request.embedding_dim,
            "generator_dim": request.generator_dim,
            "discriminator_dim": request.discriminator_dim,
            "generator_lr": request.generator_lr,
            "discriminator_lr": request.discriminator_lr,
            "discriminator_steps": request.discriminator_steps,
            "early_stopping": request.early_stopping,
            "early_stopping_patience": request.early_stopping_patience,
            "early_stopping_min_delta": request.early_stopping_min_delta,
            "run_sdmetrics": request.run_sdmetrics,
            "sdmetrics_n_samples": request.sdmetrics_n_samples,
        },
    }


@app.get("/api/train-status/{job_id}")
async def get_train_status(
    job_id: str,
    current_user: AuthenticatedUser = Depends(require_user),
):
    try:
        job = find_training_job(job_id)
    except Exception as exc:
        raise storage_operation_error(exc) from exc

    if not job:
        raise HTTPException(status_code=404, detail=f"Training job '{job_id}' not found.")
    ensure_user_owns_record(job, current_user, f"Training job '{job_id}'")

    return training_status_payload(job)


@app.post("/api/generate")
async def generate_synthetic(
    background_tasks: BackgroundTasks,
    dataset_id: str,
    n_samples: int = 5000,
    run_sdmetrics: bool = True,
    current_user: AuthenticatedUser = Depends(require_user),
):
    if n_samples <= 0:
        raise HTTPException(status_code=400, detail="n_samples must be greater than 0.")

    backend = get_storage_backend()
    try:
        dataset = backend.get_dataset(dataset_id)
    except Exception as exc:
        raise storage_operation_error(exc) from exc

    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    ensure_user_owns_record(dataset, current_user, f"Dataset '{dataset_id}'")

    try:
        model_record = backend.get_model(dataset_id)
    except Exception as exc:
        raise storage_operation_error(exc) from exc

    if not model_record:
        raise HTTPException(status_code=404, detail=f"Model for dataset '{dataset_id}' not found.")

    try:
        job = create_generation_job(dataset_id, n_samples, current_user.id)
    except Exception as exc:
        raise storage_operation_error(exc) from exc

    background_tasks.add_task(run_generation_job, job["job_id"])
    return {
        "job_id": job["job_id"],
        "dataset_id": dataset_id,
        "status": job["status"],
        "n_samples": job["n_samples"],
        "run_sdmetrics": run_sdmetrics,
    }


@app.get("/api/generate-status/{job_id}")
async def get_generate_status(
    job_id: str,
    current_user: AuthenticatedUser = Depends(require_user),
):
    try:
        job = find_generation_job(job_id)
    except Exception as exc:
        raise storage_operation_error(exc) from exc

    if not job:
        raise HTTPException(status_code=404, detail=f"Generation job '{job_id}' not found.")
    ensure_user_owns_record(job, current_user, f"Generation job '{job_id}'")

    return generation_status_payload(job)


@app.get("/api/datasets")
async def list_user_datasets(current_user: AuthenticatedUser = Depends(require_user)):
    try:
        datasets = get_storage_backend().list_datasets(current_user.id)
    except Exception as exc:
        raise storage_operation_error(exc) from exc
    return {"datasets": datasets}


@app.delete("/api/datasets/{dataset_id}")
async def delete_user_dataset(
    dataset_id: str,
    current_user: AuthenticatedUser = Depends(require_user),
):
    try:
        deleted = get_storage_backend().delete_dataset(dataset_id, current_user.id)
    except Exception as exc:
        raise storage_operation_error(exc) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    return {"deleted": True, "dataset_id": dataset_id}


class EvaluationRequest(BaseModel):
    dataset_id: str
    synthetic_id: str
    run_sdmetrics: bool = Field(
        default=True,
        description="Include SDMetrics Quality + Diagnostic reports in the response.",
    )


@app.post("/api/evaluate")
async def evaluate(
    request: EvaluationRequest,
    current_user: AuthenticatedUser = Depends(require_user),
):
    from downstream.classifier import ClassifierPipeline
    from evaluation.privacy import PrivacyMetrics
    from evaluation.quality import QualityMetrics

    backend = get_storage_backend()
    try:
        real_dataset = backend.get_dataset(request.dataset_id)
        synthetic_dataset = backend.get_dataset(request.synthetic_id)
    except Exception as exc:
        raise storage_operation_error(exc) from exc

    if not real_dataset or not synthetic_dataset:
        raise HTTPException(status_code=404, detail="One or more datasets were not found.")
    ensure_user_owns_record(real_dataset, current_user, "Real dataset")
    ensure_user_owns_record(synthetic_dataset, current_user, "Synthetic dataset")

    real_df = real_dataset["df"]
    syn_df = synthetic_dataset["df"]

    quality = {
        "kl_divergence": QualityMetrics.kl_divergence(real_df, syn_df),
        "wasserstein": QualityMetrics.wasserstein_distance(real_df, syn_df),
        "correlation_diff": QualityMetrics.correlation_difference(real_df, syn_df),
        "pca_variance": QualityMetrics.pca_variance_retained(real_df, syn_df),
    }
    privacy = {
        "k_anonymity": PrivacyMetrics.k_anonymity(real_df, syn_df),
        "mia_auc": PrivacyMetrics.membership_inference_attack(real_df, syn_df),
        "dp_estimate": PrivacyMetrics.dp_budget_estimate(real_df, syn_df),
    }

    classifier = ClassifierPipeline(
        real_dataset["schema"],
        target_col=real_dataset["schema"]["target"]["name"],
    )
    _, real_metrics = classifier.train_real_only(real_df)
    _, mixed_metrics = classifier.train_synthetic_mixed(real_df, syn_df, synthetic_weight=0.5)

    # SDMetrics
    sdmetrics_result: dict = {}
    if request.run_sdmetrics:
        try:
            from generators.ctgan import build_sdmetrics_metadata, _sdmetrics_report

            schema = real_dataset["schema"]
            target_col = schema.get("target", {}).get("name")
            metadata = build_sdmetrics_metadata(schema, target_col=target_col)
            sdmetrics_result = _sdmetrics_report(real_df, syn_df, metadata)
        except Exception as exc:
            logger.exception("SDMetrics evaluation failed")
            sdmetrics_result = {"error": str(exc)}

    return {
        "quality": quality,
        "privacy": privacy,
        "sdmetrics": sdmetrics_result,
        "real_only_metrics": real_metrics,
        "synthetic_mixed_metrics": mixed_metrics,
        "impact": {
            "auc_lift": (mixed_metrics["auc"] - real_metrics["auc"]) / real_metrics["auc"],
            "recall_lift": (
                (mixed_metrics["recall_minority"] - real_metrics["recall_minority"])
                / real_metrics["recall_minority"]
            ),
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)