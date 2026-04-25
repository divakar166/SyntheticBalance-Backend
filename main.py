import logging
from pathlib import Path

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from services.state import get_storage_backend
from services.training import (
    create_training_job,
    find_training_job,
    run_training_job,
    storage_operation_error,
    training_batch_history_payload,
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

app = FastAPI(title="Synthetic Data Generator", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TrainRequest(BaseModel):
    dataset_id: str
    epochs: int = 100


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


@app.on_event("startup")
def startup_storage_check():
    try:
        health = get_storage_backend().get_health_status()
        logger.info("Storage backend: %s", health["backend"])
        for service_name, status in health.items():
            if service_name == "backend":
                continue
            logger.info("%s status: %s", service_name.capitalize(), status)
    except Exception:
        logger.exception("Storage startup check failed")


@app.post("/api/upload")
async def upload_csv(
    file: UploadFile = File(...),
    target: str = Form("fraud"),
):
    try:
        validate_csv_file(file.filename, file.content_type)
        content = await file.read()
        df = load_csv(content)
        validate_target_column(df, target)
        return create_dataset_record(df, file.filename or "dataset.csv", target)
    except UploadValidationError as exc:
        extra = {}
        if exc.available_columns:
            extra["available_columns"] = exc.available_columns
        return _json_error(exc.message, 400, **extra)
    except Exception:
        logger.exception("Unexpected error while parsing upload")
        return _json_error("Unexpected parsing error while processing the uploaded CSV.", 500)


@app.post("/api/train-ctgan")
async def train_ctgan(request: TrainRequest, background_tasks: BackgroundTasks):
    try:
        exists = get_storage_backend().dataset_exists(request.dataset_id)
    except Exception as exc:
        raise storage_operation_error(exc) from exc

    if not exists:
        raise HTTPException(status_code=404, detail=f"Dataset '{request.dataset_id}' not found.")

    job = create_training_job(request.dataset_id, request.epochs)
    background_tasks.add_task(run_training_job, job["job_id"])
    return {
        "job_id": job["job_id"],
        "dataset_id": request.dataset_id,
        "status": job["status"],
    }


@app.get("/api/train-status/{job_id}")
async def get_train_status(job_id: str):
    try:
        job = find_training_job(job_id)
    except Exception as exc:
        raise storage_operation_error(exc) from exc

    if not job:
        raise HTTPException(status_code=404, detail=f"Training job or dataset '{job_id}' not found.")

    return training_status_payload(job)


@app.get("/api/train-batch-history/{job_id}")
async def get_train_batch_history(job_id: str):
    try:
        job = find_training_job(job_id)
    except Exception as exc:
        raise storage_operation_error(exc) from exc

    if not job:
        raise HTTPException(status_code=404, detail=f"Training job or dataset '{job_id}' not found.")

    return training_batch_history_payload(job)


@app.post("/api/generate")
async def generate_synthetic(dataset_id: str, n_samples: int = 5000):
    from generators.ctgan import CTGANWrapper

    backend = get_storage_backend()
    try:
        model_record = backend.get_model(dataset_id)
    except Exception as exc:
        raise storage_operation_error(exc) from exc

    if not model_record:
        raise HTTPException(status_code=404, detail=f"Model for dataset '{dataset_id}' not found.")

    try:
        temp_file, _ = backend.download_model_to_tempfile(dataset_id)
    except Exception as exc:
        raise storage_operation_error(exc) from exc

    try:
        ctgan = CTGANWrapper.load(temp_file.name)
    finally:
        Path(temp_file.name).unlink(missing_ok=True)

    synthetic_df = ctgan.generate(n_samples)

    try:
        source_dataset = backend.get_dataset(dataset_id)
    except Exception as exc:
        raise storage_operation_error(exc) from exc

    synthetic_record = create_dataset_record(
        synthetic_df,
        f"synthetic_{dataset_id}.csv",
        source_dataset["target"],
        dataset_type="synthetic",
        extra_metadata={"source_dataset_id": dataset_id},
    )
    return {
        "synthetic_id": synthetic_record["dataset_id"],
        "n_samples": len(synthetic_df),
        "preview": synthetic_df.head(5).to_dict(orient="records"),
    }


@app.post("/api/evaluate")
async def evaluate(dataset_id: str, synthetic_id: str):
    from downstream.classifier import ClassifierPipeline
    from evaluation.privacy import PrivacyMetrics
    from evaluation.quality import QualityMetrics

    backend = get_storage_backend()
    try:
        real_dataset = backend.get_dataset(dataset_id)
        synthetic_dataset = backend.get_dataset(synthetic_id)
    except Exception as exc:
        raise storage_operation_error(exc) from exc

    if not real_dataset or not synthetic_dataset:
        raise HTTPException(status_code=404, detail="One or more datasets were not found.")

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

    return {
        "quality": quality,
        "privacy": privacy,
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


# Compatibility exports used by tests and older imports.
_create_training_job = create_training_job
_find_training_job = find_training_job
_run_training_job = run_training_job


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
