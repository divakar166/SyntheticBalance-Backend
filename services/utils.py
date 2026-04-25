from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def serialize_scalar(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def normalized_class_distribution(series: pd.Series) -> dict[str, int]:
    return {
        str(serialize_scalar(label)): int(count)
        for label, count in series.dropna().value_counts().items()
    }


def compute_class_imbalance(series: pd.Series) -> dict:
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
        "minority_class": serialize_scalar(minority_label),
        "minority_count": minority_count,
        "minority_pct": minority_pct,
        "majority_class": serialize_scalar(majority_label),
        "majority_count": majority_count,
        "majority_pct": majority_pct,
        "class_ratio": f"1:{ratio}",
        "is_severe": is_severe,
        "recommendation": recommendation,
    }


def build_loss_history(records: list[dict]) -> list[dict]:
    return [
        {
            "epoch": int(record["epoch"]),
            "generator_loss": float(record["generator_loss"]),
            "discriminator_loss": float(record["discriminator_loss"]),
        }
        for record in records
    ]