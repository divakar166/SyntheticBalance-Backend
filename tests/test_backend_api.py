import asyncio
import json
import unittest
from io import BytesIO
from unittest.mock import patch

import pandas as pd
from fastapi import BackgroundTasks, HTTPException
from starlette.datastructures import Headers, UploadFile

import services.state as app_state
from main import (
    TrainRequest,
    generate_synthetic,
    get_generate_status,
    get_train_status,
    train_ctgan,
    upload_csv,
)
from services.auth import AuthenticatedUser
from services.generation import create_generation_job
from services.generation import _run_local as run_generation_local

TEST_USER = AuthenticatedUser(id="user-1", email="user@example.com")


class StorageBackendStub:
    def __init__(self):
        self.datasets = {}
        self.training_jobs = {}
        self.models = {}
        self.generation_jobs = {}

    def save_dataset(self, dataset_id: str, df: pd.DataFrame, schema: dict, metadata: dict) -> dict:
        record = {
            "id": dataset_id,
            "df": df,
            "schema": schema,
            "target": metadata.get("target"),
            "metadata": metadata,
            "user_id": metadata.get("user_id"),
            "filename": metadata.get("filename"),
            "dataset_type": metadata.get("dataset_type", "real"),
            "object_key": f"{metadata.get('dataset_type', 'real')}/{dataset_id}.csv",
            "n_rows": int(len(df)),
            "n_features": int(len(schema.get("features", {}))),
            "class_dist": schema.get("target", {}).get("class_distribution", {}),
            "created_at": metadata.get("upload_time"),
        }
        self.datasets[dataset_id] = record
        return record

    def get_dataset(self, dataset_id: str) -> dict | None:
        return self.datasets.get(dataset_id)

    def list_datasets(self, user_id: str) -> list[dict]:
        return [record for record in self.datasets.values() if record.get("user_id") == user_id]

    def delete_dataset(self, dataset_id: str, user_id: str) -> bool:
        record = self.datasets.get(dataset_id)
        if not record or record.get("user_id") != user_id:
            return False
        self.datasets.pop(dataset_id, None)
        self.models.pop(dataset_id, None)
        return True

    def save_training_job(self, job: dict) -> dict:
        self.training_jobs[job["job_id"]] = dict(job)
        return self.training_jobs[job["job_id"]]

    def update_training_job(self, job_id: str, values: dict) -> dict:
        self.training_jobs[job_id].update(values)
        return self.training_jobs[job_id]

    def get_training_job(self, job_or_dataset_id: str) -> dict | None:
        direct = self.training_jobs.get(job_or_dataset_id)
        if direct:
            return direct
        for job in reversed(list(self.training_jobs.values())):
            if job.get("dataset_id") == job_or_dataset_id:
                return job
        return None

    def save_generation_job(self, job: dict) -> dict:
        self.generation_jobs[job["job_id"]] = dict(job)
        return self.generation_jobs[job["job_id"]]

    def update_generation_job(self, job_id: str, values: dict) -> dict:
        self.generation_jobs[job_id].update(values)
        return self.generation_jobs[job_id]

    def get_generation_job(self, job_or_dataset_id: str) -> dict | None:
        direct = self.generation_jobs.get(job_or_dataset_id)
        if direct:
            return direct
        for job in reversed(list(self.generation_jobs.values())):
            if job.get("dataset_id") == job_or_dataset_id:
                return job
        return None

    def save_model(self, dataset_id: str, local_model_path, metadata: dict | None = None, config=None) -> dict:
        record = {
            "id": dataset_id,
            "dataset_id": dataset_id,
            "object_key": f"ctgan/{dataset_id}.pkl",
            "metadata": metadata or {},
            "config": config or {},
            "user_id": (metadata or {}).get("user_id"),
        }
        self.models[dataset_id] = record
        return record

    def get_model(self, dataset_id: str) -> dict | None:
        return self.models.get(dataset_id)

    def list_models(self, user_id: str) -> list[dict]:
        return [record for record in self.models.values() if record.get("user_id") == user_id]


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


class BackendApiTests(unittest.TestCase):
    def setUp(self):
        self.backend = StorageBackendStub()
        app_state._storage_backend = self.backend
        self.backend.datasets["dataset-1"] = {
            "id": "dataset-1",
            "df": pd.DataFrame(
                {
                    "amount": [10, 12, 18, 42],
                    "merchant": ["A", "B", "A", "C"],
                    "fraud": [0, 0, 0, 1],
                }
            ),
            "schema": {
                "features": {
                    "amount": {"type": "numeric"},
                    "merchant": {"type": "categorical"},
                },
                "target": {"name": "fraud", "type": "categorical"},
            },
            "target": "fraud",
            "user_id": TEST_USER.id,
            "metadata": {
                "target": "fraud",
                "upload_time": "2026-04-24T00:00:00+00:00",
                "user_id": TEST_USER.id,
            },
        }
        self.backend.models["dataset-1"] = {
            "id": "dataset-1",
            "dataset_id": "dataset-1",
            "object_key": "ctgan/dataset-1.pkl",
            "user_id": TEST_USER.id,
            "metadata": {"user_id": TEST_USER.id},
        }

    def tearDown(self):
        app_state._storage_backend = None

    def upload_file(self, filename: str, content: bytes, content_type: str = "text/csv"):
        upload = UploadFile(
            file=BytesIO(content),
            filename=filename,
            headers=Headers({"content-type": content_type}),
        )
        return asyncio.run(upload_csv(file=upload, target="fraud", current_user=TEST_USER))

    def json_body(self, response):
        return json.loads(response.body.decode("utf-8"))

    def test_upload_valid_csv_saves_dataset_record(self):
        df = pd.DataFrame(
            {
                "amount": [10.5, 12.0, 13.5, 99.9],
                "merchant": ["A", "B", "A", "C"],
                "fraud": [0, 0, 0, 1],
            }
        )

        response = self.upload_file("transactions.csv", csv_bytes(df))

        self.assertEqual(response["n_rows"], 4)
        self.assertEqual(response["n_features"], 2)
        self.assertEqual(response["class_dist"], {"0": 3, "1": 1})
        self.assertIn(response["dataset_id"], self.backend.datasets)

    def test_upload_rejects_missing_target_column(self):
        response = self.upload_file("transactions.csv", csv_bytes(pd.DataFrame({"amount": [1, 2]})))

        self.assertEqual(response.status_code, 400)
        self.assertIn("Target column 'fraud' not found", self.json_body(response)["error"])

    def test_train_ctgan_enqueues_background_job(self):
        background_tasks = BackgroundTasks()

        response = asyncio.run(
            train_ctgan(
                TrainRequest(dataset_id="dataset-1", epochs=3),
                background_tasks=background_tasks,
                current_user=TEST_USER,
            )
        )

        self.assertEqual(response["status"], "queued")
        self.assertEqual(response["dataset_id"], "dataset-1")
        self.assertEqual(len(background_tasks.tasks), 1)
        self.assertEqual(self.backend.training_jobs[response["job_id"]]["total_epochs"], 3)

    def test_train_ctgan_returns_404_for_missing_dataset(self):
        with self.assertRaises(HTTPException) as exc:
            asyncio.run(
                train_ctgan(
                    TrainRequest(dataset_id="missing-dataset", epochs=3),
                    background_tasks=BackgroundTasks(),
                    current_user=TEST_USER,
                )
            )

        self.assertEqual(exc.exception.status_code, 404)

    def test_get_train_status_returns_latest_job_for_dataset(self):
        first = asyncio.run(
            train_ctgan(
                TrainRequest(dataset_id="dataset-1", epochs=2),
                background_tasks=BackgroundTasks(),
                current_user=TEST_USER,
            )
        )
        second = asyncio.run(
            train_ctgan(
                TrainRequest(dataset_id="dataset-1", epochs=4),
                background_tasks=BackgroundTasks(),
                current_user=TEST_USER,
            )
        )
        self.backend.training_jobs[first["job_id"]]["status"] = "completed"

        response = asyncio.run(get_train_status("dataset-1", current_user=TEST_USER))

        self.assertEqual(response["job_id"], second["job_id"])
        self.assertEqual(response["total_epochs"], 4)

    def test_generate_synthetic_enqueues_generation_job(self):
        background_tasks = BackgroundTasks()

        response = asyncio.run(
            generate_synthetic(
                background_tasks=background_tasks,
                dataset_id="dataset-1",
                n_samples=25,
                current_user=TEST_USER,
            )
        )

        self.assertEqual(response["status"], "queued")
        self.assertEqual(response["dataset_id"], "dataset-1")
        self.assertEqual(response["n_samples"], 25)
        self.assertEqual(len(background_tasks.tasks), 1)
        self.assertIn(response["job_id"], self.backend.generation_jobs)

    def test_generate_synthetic_requires_trained_model(self):
        self.backend.models.clear()

        with self.assertRaises(HTTPException) as exc:
            asyncio.run(
                generate_synthetic(
                    background_tasks=BackgroundTasks(),
                    dataset_id="dataset-1",
                    n_samples=25,
                    current_user=TEST_USER,
                )
            )

        self.assertEqual(exc.exception.status_code, 404)
        self.assertIn("Model for dataset 'dataset-1' not found", exc.exception.detail)

    def test_get_generate_status_returns_latest_job_for_dataset(self):
        first = create_generation_job("dataset-1", 10, TEST_USER.id)
        second = create_generation_job("dataset-1", 20, TEST_USER.id)
        self.backend.generation_jobs[first["job_id"]]["status"] = "completed"

        response = asyncio.run(get_generate_status("dataset-1", current_user=TEST_USER))

        self.assertEqual(response["job_id"], second["job_id"])
        self.assertEqual(response["n_samples"], 20)

    def test_local_generation_job_marks_completed(self):
        job = create_generation_job("dataset-1", 5, TEST_USER.id)

        with patch(
            "services.generation.generate_synthetic_dataset",
            return_value={
                "synthetic_id": "synthetic-1",
                "synthetic_path": "synthetic/synthetic-1.csv",
                "n_samples": 5,
                "preview": [{"amount": 10, "merchant": "A", "fraud": 0}],
                "generation_time_seconds": 1.25,
            },
        ):
            run_generation_local(job["job_id"])

        updated = self.backend.generation_jobs[job["job_id"]]
        self.assertEqual(updated["status"], "completed")
        self.assertEqual(updated["synthetic_id"], "synthetic-1")
        self.assertEqual(updated["synthetic_path"], "synthetic/synthetic-1.csv")
        self.assertEqual(updated["generation_time_seconds"], 1.25)

    def test_local_generation_job_marks_failure(self):
        job = create_generation_job("dataset-1", 5, TEST_USER.id)

        with patch("services.generation.generate_synthetic_dataset", side_effect=RuntimeError("boom")):
            run_generation_local(job["job_id"])

        updated = self.backend.generation_jobs[job["job_id"]]
        self.assertEqual(updated["status"], "failed")
        self.assertEqual(updated["error"], "boom")


if __name__ == "__main__":
    unittest.main()
