from __future__ import annotations

import io
import uuid

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from handlers.data_handler import SchemaDetector, normalize_dataframe
from services.state import get_storage_backend
from services.utils import compute_class_imbalance, normalized_class_distribution, utc_now_iso

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


def validate_csv_file(filename: str | None, content_type: str | None):
    safe_name = filename or ""
    extension = safe_name.lower().endswith(".csv")
    safe_content_type = (content_type or "").lower()

    if not safe_name:
        raise UploadValidationError("Uploaded file must include a filename.")

    if not extension or (safe_content_type and safe_content_type not in ACCEPTED_CSV_CONTENT_TYPES):
        raise UploadValidationError("File must be CSV format.")


def load_csv(content: bytes) -> pd.DataFrame:
    if not content:
        raise UploadValidationError("Uploaded file is empty.")

    try:
        df = pd.read_csv(io.BytesIO(content))
    except EmptyDataError as exc:
        raise UploadValidationError("Uploaded CSV is empty.") from exc
    except (ParserError, UnicodeDecodeError) as exc:
        raise UploadValidationError("Uploaded CSV is corrupted or could not be parsed.") from exc

    df = normalize_dataframe(df)
    if df.empty or len(df.columns) == 0:
        raise UploadValidationError("CSV must contain at least one row and one column.")

    return df


def is_integer_like(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        return False
    return bool(((numeric - numeric.round()).abs() < 1e-9).all())


def validate_target_column(df: pd.DataFrame, target: str):
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
    if numeric_target.notna().all() and not is_integer_like(non_null):
        raise UploadValidationError(
            "Target column must contain discrete class labels. Continuous float targets are not supported."
        )


def create_dataset_record(
    df: pd.DataFrame,
    filename: str,
    target: str,
    *,
    dataset_type: str = "real",
    extra_metadata: dict | None = None,
    storage_backend=None,
) -> dict:
    schema = SchemaDetector.detect(df, target_col=target)
    schema["target"]["class_distribution"] = normalized_class_distribution(df[target])
    class_imbalance = compute_class_imbalance(df[target])

    dataset_id = str(uuid.uuid4())
    metadata = {
        "filename": filename,
        "upload_time": utc_now_iso(),
        "target": target,
        "class_imbalance": class_imbalance,
        "dataset_type": dataset_type,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    backend = storage_backend or get_storage_backend()
    saved_record = backend.save_dataset(dataset_id, df, schema, metadata)
    response = {
        "dataset_id": dataset_id,
        "n_rows": len(df),
        "n_features": len(schema["features"]),
        "class_dist": schema["target"]["class_distribution"],
        "schema": schema,
        "class_imbalance": class_imbalance,
    }
    if saved_record.get("object_key"):
        response["object_key"] = saved_record["object_key"]
    return response
