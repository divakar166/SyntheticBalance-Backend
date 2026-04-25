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
from persistence import InMemoryBackend
from services.auth import AuthenticatedUser
from services.generation import create_generation_job
from services.generation import _run_local as run_generation_local

TEST_USER = AuthenticatedUser(id="user-1", email="user@example.com")


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


class BackendApiTests(unittest.TestCase):
    def setUp(self):
        app_state.datasets.clear()
        app_state.models.clear()
        app_state.training_jobs.clear()
        app_state.generation_jobs.clear()
        app_state._storage_backend = InMemoryBackend(
            app_state.datasets,
            app_state.training_jobs,
            app_state.models,
            app_state.generation_jobs,
        )
        app_state.datasets["dataset-1"] = {
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
        app_state.models["dataset-1"] = {
            "id": "dataset-1",
            "dataset_id": "dataset-1",
            "object_key": "ctgan/dataset-1.pkl",
            "user_id": TEST_USER.id,
            "metadata": {"user_id": TEST_USER.id},
        }

    def tearDown(self):
        app_state.datasets.clear()
        app_state.models.clear()
        app_state.training_jobs.clear()
        app_state.generation_jobs.clear()
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
        self.assertIn(response["dataset_id"], app_state.datasets)

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
        self.assertEqual(app_state.training_jobs[response["job_id"]]["total_epochs"], 3)

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
        app_state.training_jobs[first["job_id"]]["status"] = "completed"

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
        self.assertIn(response["job_id"], app_state.generation_jobs)

    def test_generate_synthetic_requires_trained_model(self):
        app_state.models.clear()

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
        app_state.generation_jobs[first["job_id"]]["status"] = "completed"

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

        updated = app_state.generation_jobs[job["job_id"]]
        self.assertEqual(updated["status"], "completed")
        self.assertEqual(updated["synthetic_id"], "synthetic-1")
        self.assertEqual(updated["synthetic_path"], "synthetic/synthetic-1.csv")
        self.assertEqual(updated["generation_time_seconds"], 1.25)

    def test_local_generation_job_marks_failure(self):
        job = create_generation_job("dataset-1", 5, TEST_USER.id)

        with patch("services.generation.generate_synthetic_dataset", side_effect=RuntimeError("boom")):
            run_generation_local(job["job_id"])

        updated = app_state.generation_jobs[job["job_id"]]
        self.assertEqual(updated["status"], "failed")
        self.assertEqual(updated["error"], "boom")


if __name__ == "__main__":
    unittest.main()
