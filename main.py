from datetime import datetime, timezone
import io
import logging
from pathlib import Path
import uuid

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pandas.errors import EmptyDataError, ParserError
from pydantic import BaseModel

from handlers.data_handler import SchemaDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Synthetic Data Generator", version="0.1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

datasets = {}
models = {}
training_jobs = {}
MAX_DATASETS = 20
MODELS_DIR = Path(__file__).resolve().parent / "models"
ACCEPTED_CSV_CONTENT_TYPES = {
    "application/csv",
    "application/vnd.ms-excel",
    "text/csv",
    "text/plain",
}


class UploadValidationError(Exception):
    def __init__(self, message: str, *, available_columns: list[str] | None = None):
        super().__init__(message)
        self.message = message
        self.available_columns = available_columns or []


class TrainRequest(BaseModel):
    dataset_id: str
    epochs: int = 100


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "0.1.0"}


def _json_error(message: str, status_code: int, **extra):
    return JSONResponse(
        status_code=status_code,
        content={
            "error": message,
            "status": status_code,
            **extra,
        },
    )


def _cleanup_datasets():
    if len(datasets) < MAX_DATASETS:
        return

    oldest_dataset_id = min(
        datasets,
        key=lambda dataset_id: datasets[dataset_id]["metadata"]["upload_time"],
    )
    datasets.pop(oldest_dataset_id, None)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_csv_file(file: UploadFile):
    filename = file.filename or ""
    extension = filename.lower().endswith(".csv")
    content_type = (file.content_type or "").lower()

    if not filename:
        raise UploadValidationError("Uploaded file must include a filename.")

    if not extension or (content_type and content_type not in ACCEPTED_CSV_CONTENT_TYPES):
        raise UploadValidationError("File must be CSV format.")


def _load_csv(content: bytes) -> pd.DataFrame:
    if not content:
        raise UploadValidationError("Uploaded file is empty.")

    try:
        df = pd.read_csv(io.BytesIO(content))
    except EmptyDataError as exc:
        raise UploadValidationError("Uploaded CSV is empty.") from exc
    except (ParserError, UnicodeDecodeError) as exc:
        raise UploadValidationError("Uploaded CSV is corrupted or could not be parsed.") from exc

    if df.empty or len(df.columns) == 0:
        raise UploadValidationError("CSV must contain at least one row and one column.")

    return df


def _is_integer_like(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        return False
    return bool(((numeric - numeric.round()).abs() < 1e-9).all())


def _validate_target_column(df: pd.DataFrame, target: str):
    if target not in df.columns:
        raise UploadValidationError(
            f"Target column '{target}' not found. Available columns: {', '.join(df.columns)}",
            available_columns=df.columns.tolist(),
        )

    target_values = df[target]
    non_null = target_values.dropna()
    if non_null.empty:
        raise UploadValidationError(f"Target column '{target}' cannot be entirely empty or NaN.")

    unique_count = int(non_null.nunique())
    if unique_count < 2:
        raise UploadValidationError(
            f"Target column must contain at least 2 unique classes for classification. Found {unique_count}."
        )
    if unique_count > 20:
        raise UploadValidationError(
            f"Target column must have 2-20 unique values for classification. Found {unique_count} unique values."
        )

    numeric_target = pd.to_numeric(non_null, errors="coerce")
    if numeric_target.notna().all() and not _is_integer_like(non_null):
        raise UploadValidationError(
            "Target column must contain discrete class labels. Continuous float targets are not supported."
        )


def _serialize_scalar(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def _normalized_class_distribution(series: pd.Series) -> dict[str, int]:
    return {
        str(_serialize_scalar(label)): int(count)
        for label, count in series.dropna().value_counts().items()
    }


def _compute_class_imbalance(series: pd.Series) -> dict:
    class_counts = series.dropna().value_counts()
    minority_label = class_counts.idxmin()
    majority_label = class_counts.idxmax()
    minority_count = int(class_counts.loc[minority_label])
    majority_count = int(class_counts.loc[majority_label])
    total = int(class_counts.sum())
    minority_pct = float((minority_count / total) * 100)
    majority_pct = float((majority_count / total) * 100)
    ratio = max(1, round(majority_count / minority_count))
    is_severe = minority_pct < 5.0

    if is_severe:
        recommendation = (
            f"Dataset shows severe class imbalance ({minority_pct:.2f}% minority). "
            "Synthetic data generation is strongly recommended to improve minority coverage."
        )
    elif minority_pct < 20.0:
        recommendation = (
            f"Dataset shows moderate class imbalance ({minority_pct:.2f}% minority). "
            "Synthetic data generation can help balance this."
        )
    else:
        recommendation = (
            f"Dataset is relatively balanced ({minority_pct:.2f}% minority). "
            "Synthetic augmentation is optional and should be guided by downstream metrics."
        )

    return {
        "minority_class": _serialize_scalar(minority_label),
        "minority_count": minority_count,
        "minority_pct": minority_pct,
        "majority_class": _serialize_scalar(majority_label),
        "majority_count": majority_count,
        "majority_pct": majority_pct,
        "class_ratio": f"1:{ratio}",
        "is_severe": is_severe,
        "recommendation": recommendation,
    }


def _build_loss_history(records: list[dict]) -> list[dict]:
    return [
        {
            "epoch": int(record["epoch"]),
            "generator_loss": float(record["generator_loss"]),
            "discriminator_loss": float(record["discriminator_loss"]),
        }
        for record in records
    ]


def _run_training_job(job_id: str):
    from ctgan_wrapper import CTGANWrapper

    job = training_jobs[job_id]
    dataset_id = job["dataset_id"]
    dataset = datasets[dataset_id]
    df = dataset["df"]
    schema = dataset["schema"]
    target_col = schema["target"]["name"]

    ctgan = CTGANWrapper(schema, epochs=job["total_epochs"])
    started_at = datetime.now(timezone.utc)
    job["started_at"] = started_at.isoformat()
    job["status"] = "running"

    def update_progress(current_epoch: int, total_epochs: int, metrics: dict):
        job["current_epoch"] = current_epoch
        job["total_epochs"] = total_epochs
        job["loss_history"].append(
            {
                "epoch": current_epoch,
                "generator_loss": float(metrics["generator_loss"]),
                "discriminator_loss": float(metrics["discriminator_loss"]),
            }
        )
        job["final_loss"] = float(metrics["generator_loss"])
        job["updated_at"] = _utc_now_iso()

    try:
        ctgan.train(df, target_col=target_col, progress_callback=update_progress)
        model_path = MODELS_DIR / f"ctgan_{dataset_id}.pkl"
        ctgan.save(model_path)

        models[dataset_id] = {
            "model_id": dataset_id,
            "model_path": str(model_path),
            "job_id": job_id,
            "trained_at": _utc_now_iso(),
        }
        job["status"] = "completed"
        job["model_id"] = dataset_id
        job["model_path"] = str(model_path)
        job["training_time_seconds"] = float(ctgan.training_time_seconds)
        if job["loss_history"]:
            job["final_loss"] = float(job["loss_history"][-1]["generator_loss"])
    except Exception as exc:
        logger.exception("CTGAN training failed for dataset %s", dataset_id)
        elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
        job["status"] = "failed"
        job["error"] = str(exc)
        job["training_time_seconds"] = float(elapsed)
        job["updated_at"] = _utc_now_iso()
    else:
        job["updated_at"] = _utc_now_iso()


def _create_training_job(dataset_id: str, epochs: int) -> dict:
    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "dataset_id": dataset_id,
        "status": "running",
        "current_epoch": 0,
        "total_epochs": epochs,
        "loss_history": [],
        "training_time_seconds": None,
        "final_loss": None,
        "error": None,
        "model_id": None,
        "model_path": None,
        "created_at": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
    }
    training_jobs[job_id] = job
    return job


@app.post("/api/upload")
async def upload_csv(
    file: UploadFile = File(...),
    target: str = Form("fraud"),
):
    try:
        _validate_csv_file(file)
        content = await file.read()
        df = _load_csv(content)
        _validate_target_column(df, target)

        schema = SchemaDetector.detect(df, target_col=target)
        schema["target"]["class_distribution"] = _normalized_class_distribution(df[target])
        class_imbalance = _compute_class_imbalance(df[target])

        dataset_id = str(uuid.uuid4())
        _cleanup_datasets()
        datasets[dataset_id] = {
            "df": df,
            "schema": schema,
            "metadata": {
                "filename": file.filename,
                "upload_time": _utc_now_iso(),
                "target": target,
                "class_imbalance": class_imbalance,
            },
        }

        return {
            "dataset_id": dataset_id,
            "n_rows": len(df),
            "n_features": len(schema["features"]),
            "class_dist": schema["target"]["class_distribution"],
            "schema": schema,
            "class_imbalance": class_imbalance,
        }
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
    if request.dataset_id not in datasets:
        raise HTTPException(status_code=404, detail=f"Dataset '{request.dataset_id}' not found.")

    job = _create_training_job(request.dataset_id, request.epochs)
    background_tasks.add_task(_run_training_job, job["job_id"])

    return {
        "job_id": job["job_id"],
        "dataset_id": request.dataset_id,
        "status": job["status"],
    }


@app.get("/api/train-status/{job_id}")
async def get_train_status(job_id: str):
    job = training_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Training job '{job_id}' not found.")

    return {
        "job_id": job["job_id"],
        "dataset_id": job["dataset_id"],
        "status": job["status"],
        "current_epoch": job["current_epoch"],
        "total_epochs": job["total_epochs"],
        "loss_history": _build_loss_history(job["loss_history"]),
        "training_time_seconds": job["training_time_seconds"],
        "final_loss": job["final_loss"],
        "error": job["error"],
        "model_id": job["model_id"],
    }


@app.post("/api/generate")
async def generate_synthetic(dataset_id: str, n_samples: int = 5000):
    from ctgan_wrapper import CTGANWrapper

    if dataset_id not in models:
        raise HTTPException(status_code=404, detail=f"Model for dataset '{dataset_id}' not found.")

    # Load model
    ctgan = CTGANWrapper.load(models[dataset_id]["model_path"])

    # Generate
    synthetic_df = ctgan.generate(n_samples)

    # Store
    syn_id = str(uuid.uuid4())
    datasets[syn_id] = {'df': synthetic_df, 'schema': datasets[dataset_id]['schema']}

    return {
        'synthetic_id': syn_id,
        'n_samples': len(synthetic_df),
        'preview': synthetic_df.head(5).to_dict(orient='records')
    }


@app.post("/api/evaluate")
async def evaluate(dataset_id: str, synthetic_id: str):
    from downstream.classifier import ClassifierPipeline
    from evaluation.privacy import PrivacyMetrics
    from evaluation.quality import QualityMetrics

    real_df = datasets[dataset_id]['df']
    syn_df = datasets[synthetic_id]['df']

    quality = {
        'kl_divergence': QualityMetrics.kl_divergence(real_df, syn_df),
        'wasserstein': QualityMetrics.wasserstein_distance(real_df, syn_df),
        'correlation_diff': QualityMetrics.correlation_difference(real_df, syn_df),
        'pca_variance': QualityMetrics.pca_variance_retained(real_df, syn_df)
    }

    privacy = {
        'k_anonymity': PrivacyMetrics.k_anonymity(real_df, syn_df),
        'mia_auc': PrivacyMetrics.membership_inference_attack(real_df, syn_df),
        'dp_estimate': PrivacyMetrics.dp_budget_estimate(real_df, syn_df)
    }

    # Classify
    classifier = ClassifierPipeline(datasets[dataset_id]['schema'], target_col=datasets[dataset_id]['schema']['target']['name'])
    _, real_metrics = classifier.train_real_only(real_df)
    _, mixed_metrics = classifier.train_synthetic_mixed(real_df, syn_df, synthetic_weight=0.5)

    return {
        'quality': quality,
        'privacy': privacy,
        'real_only_metrics': real_metrics,
        'synthetic_mixed_metrics': mixed_metrics,
        'impact': {
            'auc_lift': (mixed_metrics['auc'] - real_metrics['auc']) / real_metrics['auc'],
            'recall_lift': (mixed_metrics['recall_minority'] - real_metrics['recall_minority']) / real_metrics['recall_minority']
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
