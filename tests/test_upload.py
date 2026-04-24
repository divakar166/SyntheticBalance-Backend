import asyncio
import json
from io import BytesIO
import pandas as pd
import unittest
from unittest.mock import patch

from fastapi import BackgroundTasks, HTTPException
from starlette.datastructures import Headers, UploadFile

from handlers.data_handler import SchemaDetector
from main import (
    _create_training_job,
    _run_training_job,
    datasets,
    get_train_status,
    models,
    train_ctgan,
    training_jobs,
    upload_csv,
    TrainRequest,
)


def _csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


class UploadApiTests(unittest.TestCase):
    def setUp(self):
        datasets.clear()
        models.clear()
        training_jobs.clear()

    def tearDown(self):
        datasets.clear()
        models.clear()
        training_jobs.clear()

    def _upload_file(self, filename: str, content: bytes, content_type: str = "text/csv"):
        upload = UploadFile(
            file=BytesIO(content),
            filename=filename,
            headers=Headers({"content-type": content_type}),
        )
        return asyncio.run(upload_csv(file=upload, target="fraud"))

    def _json_body(self, response):
        return json.loads(response.body.decode("utf-8"))

    def test_upload_valid_csv_returns_schema_and_imbalance(self):
        df = pd.DataFrame(
            {
                "amount": [10.5, 12.0, 13.5, 99.9],
                "merchant": ["A", "B", "A", "C"],
                "channel": ["web", None, "web", "pos"],
                "fraud": [0, 0, 0, 1],
            }
        )

        response = self._upload_file("transactions.csv", _csv_bytes(df))

        self.assertEqual(response["n_rows"], 4)
        body = response
        self.assertEqual(body["n_rows"], 4)
        self.assertEqual(body["n_features"], 3)
        self.assertEqual(body["class_dist"], {"0": 3, "1": 1})
        self.assertEqual(body["schema"]["features"]["amount"]["type"], "numeric")
        self.assertEqual(body["schema"]["features"]["amount"]["median"], 12.75)
        self.assertEqual(body["schema"]["features"]["merchant"]["type"], "categorical")
        self.assertEqual(body["schema"]["features"]["channel"]["missing_pct"], 25.0)
        self.assertEqual(body["class_imbalance"]["minority_count"], 1)
        self.assertEqual(body["class_imbalance"]["class_ratio"], "1:3")

    def test_upload_rejects_missing_target_column(self):
        df = pd.DataFrame({"amount": [1, 2], "label": [0, 1]})

        response = self._upload_file("transactions.csv", _csv_bytes(df))

        self.assertEqual(response.status_code, 400)
        body = self._json_body(response)
        self.assertIn("Target column 'fraud' not found", body["error"])
        self.assertEqual(body["available_columns"], ["amount", "label"])

    def test_upload_rejects_non_csv_file(self):
        response = self._upload_file("notes.txt", b"hello", "text/plain")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(self._json_body(response)["error"], "File must be CSV format.")

    def test_upload_rejects_all_nan_target(self):
        df = pd.DataFrame({"amount": [10, 20], "fraud": [None, None]})

        response = self._upload_file("transactions.csv", _csv_bytes(df))

        self.assertEqual(response.status_code, 400)
        self.assertIn("cannot be entirely empty or NaN", self._json_body(response)["error"])

    def test_upload_computes_severe_class_imbalance(self):
        df = pd.DataFrame(
            {
                "amount": list(range(100)),
                "fraud": ([0] * 98) + ([1] * 2),
            }
        )

        response = self._upload_file("transactions.csv", _csv_bytes(df))

        imbalance = response["class_imbalance"]
        self.assertEqual(imbalance["minority_pct"], 2.0)
        self.assertEqual(imbalance["majority_pct"], 98.0)
        self.assertEqual(imbalance["class_ratio"], "1:49")
        self.assertTrue(imbalance["is_severe"])

    def test_upload_rejects_single_class_target(self):
        df = pd.DataFrame({"amount": [10, 20, 30], "fraud": [1, 1, 1]})

        response = self._upload_file("transactions.csv", _csv_bytes(df))

        self.assertEqual(response.status_code, 400)
        self.assertIn("at least 2 unique classes", self._json_body(response)["error"])


class SchemaDetectorTests(unittest.TestCase):
    def test_schema_detector_skips_all_nan_and_profiles_constant_columns(self):
        df = pd.DataFrame(
            {
                "all_missing": [None, None, None],
                "constant": [7, 7, 7],
                "city": ["NY", "SF", "NY"],
                "fraud": [0, 1, 0],
            }
        )

        schema = SchemaDetector.detect(df, target_col="fraud")

        self.assertNotIn("all_missing", schema["features"])
        self.assertIn("all_missing", schema["skipped_features"])
        self.assertTrue(schema["features"]["constant"]["is_constant"])
        self.assertEqual(schema["features"]["city"]["missing_pct"], 0.0)
        self.assertEqual(schema["target"]["class_distribution"], {"0": 2, "1": 1})


class TrainingApiTests(unittest.TestCase):
    def setUp(self):
        datasets.clear()
        models.clear()
        training_jobs.clear()
        datasets["dataset-1"] = {
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
            "metadata": {"upload_time": "2026-04-24T00:00:00+00:00"},
        }

    def tearDown(self):
        datasets.clear()
        models.clear()
        training_jobs.clear()

    def test_train_ctgan_starts_background_job(self):
        background_tasks = BackgroundTasks()

        response = asyncio.run(
            train_ctgan(
                TrainRequest(dataset_id="dataset-1", epochs=3),
                background_tasks=background_tasks,
            )
        )

        self.assertEqual(response["status"], "running")
        self.assertEqual(response["dataset_id"], "dataset-1")
        self.assertIn(response["job_id"], training_jobs)
        self.assertEqual(training_jobs[response["job_id"]]["total_epochs"], 3)
        self.assertEqual(len(background_tasks.tasks), 1)

    def test_train_ctgan_returns_404_for_missing_dataset(self):
        with self.assertRaises(HTTPException) as exc:
            asyncio.run(
                train_ctgan(
                    TrainRequest(dataset_id="missing-dataset", epochs=3),
                    background_tasks=BackgroundTasks(),
                )
            )

        self.assertEqual(exc.exception.status_code, 404)

    def test_training_job_updates_status_and_model_on_success(self):
        class FakeCTGANWrapper:
            def __init__(self, schema, epochs, batch_size=256):
                self.schema = schema
                self.epochs = epochs
                self.batch_size = batch_size
                self.training_time_seconds = 1.25
                self.saved_path = None

            def train(self, df, target_col=None, progress_callback=None):
                for epoch in range(1, self.epochs + 1):
                    progress_callback(
                        epoch,
                        self.epochs,
                        {
                            "epoch": epoch,
                            "generator_loss": 1.0 / epoch,
                            "discriminator_loss": 2.0 / epoch,
                        },
                    )

            def save(self, path):
                self.saved_path = str(path)

        job = _create_training_job("dataset-1", 2)

        with patch("ctgan_wrapper.CTGANWrapper", FakeCTGANWrapper):
            _run_training_job(job["job_id"])

        updated = training_jobs[job["job_id"]]
        self.assertEqual(updated["status"], "completed")
        self.assertEqual(updated["current_epoch"], 2)
        self.assertEqual(updated["final_loss"], 0.5)
        self.assertEqual(len(updated["loss_history"]), 2)
        self.assertEqual(updated["model_id"], "dataset-1")
        self.assertIn("dataset-1", models)
        self.assertTrue(models["dataset-1"]["model_path"].endswith("ctgan_dataset-1.pkl"))

    def test_training_job_marks_failure(self):
        class FailingCTGANWrapper:
            def __init__(self, schema, epochs, batch_size=256):
                self.training_time_seconds = 0.0

            def train(self, df, target_col=None, progress_callback=None):
                raise RuntimeError("boom")

            def save(self, path):
                raise AssertionError("save should not be called")

        job = _create_training_job("dataset-1", 2)

        with patch("ctgan_wrapper.CTGANWrapper", FailingCTGANWrapper):
            _run_training_job(job["job_id"])

        updated = training_jobs[job["job_id"]]
        self.assertEqual(updated["status"], "failed")
        self.assertEqual(updated["error"], "boom")
        self.assertNotIn("dataset-1", models)

    def test_get_train_status_returns_job_payload(self):
        job = _create_training_job("dataset-1", 4)
        training_jobs[job["job_id"]]["current_epoch"] = 1
        training_jobs[job["job_id"]]["loss_history"] = [
            {"epoch": 1, "generator_loss": 0.8, "discriminator_loss": 1.6}
        ]
        training_jobs[job["job_id"]]["final_loss"] = 0.8

        response = asyncio.run(get_train_status(job["job_id"]))

        self.assertEqual(response["status"], "running")
        self.assertEqual(response["current_epoch"], 1)
        self.assertEqual(response["loss_history"][0]["generator_loss"], 0.8)

    def test_get_train_status_returns_404_for_unknown_job(self):
        with self.assertRaises(HTTPException) as exc:
            asyncio.run(get_train_status("missing-job"))

        self.assertEqual(exc.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
