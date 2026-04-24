import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

CTGAN_EPOCHS_DEFAULT = 100
CTGAN_BATCH_SIZE = 256
CLASSIFIER_TEST_SIZE = 0.2
RANDOM_SEED = 42

for d in [DATA_DIR, OUTPUT_DIR, MODEL_DIR]:
    d.mkdir(exist_ok=True)