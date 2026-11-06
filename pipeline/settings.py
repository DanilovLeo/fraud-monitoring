"""Общие пути и параметры поставки данных и переобучения.

Все значения можно переопределить переменными окружения (см. .env.example).
"""

import os
from pathlib import Path

TRAINING_DIR = Path(os.getenv("TRAINING_DATA_DIR", "/opt/airflow/training_data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "/opt/airflow/models"))
FRAUD_SRC = os.getenv("FRAUD_SRC", "/opt/airflow/fraud_src")

CURRENT_DATASET = TRAINING_DIR / "current.parquet"
PENDING_WINDOW = TRAINING_DIR / "pending_window.parquet"
BATCHES_DIR = TRAINING_DIR / "batches"
SPLITS_DIR = TRAINING_DIR / "splits"
SUPPLY_BATCH = TRAINING_DIR / "supply_batch.parquet"

# Скользящее окно обучения: WINDOW_MONTHS - 1 месяцев в train, последний в test.
WINDOW_MONTHS = int(os.getenv("WINDOW_MONTHS", "4"))
# Размер месячного батча, который DAG поставки дописывает в витрину.
SAMPLE_SIZE = int(os.getenv("SUPPLY_BATCH_SIZE", "50000"))
# Глубина хранения витрины в годах.
HISTORY_YEARS = int(os.getenv("HISTORY_YEARS", "2"))
# Quality gate: кандидат публикуется, если PR-AUC >= PR-AUC прода - QUALITY_TOLERANCE.
QUALITY_TOLERANCE = float(os.getenv("QUALITY_TOLERANCE", "0.0"))
