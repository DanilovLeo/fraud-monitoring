"""Препроцессинг train и test тем же модулем, что и онлайн-скоринг."""

import logging

import pandas as pd

from pipeline.features import preprocessing
from pipeline.settings import SPLITS_DIR

logger = logging.getLogger(__name__)


def preprocess_dataset(split_info: dict) -> dict:
    """Применяет собственный run_preproc (fraud_detector/src/preprocessing.py).

    Читает: splits/train_raw.parquet и splits/test_raw.parquet из split_info.
    Пишет: splits/train_proc.parquet и splits/test_proc.parquet (перезаписываются).
    Порядок строк train сохраняется (он отсортирован по времени), это нужно
    для временного holdout при обучении.
    Повтор безопасен: выход полностью определяется входными файлами,
    витрина и указатель месяца не меняются.
    """
    train = preprocessing.run_preproc(pd.read_parquet(split_info["train_raw_path"]))
    test = preprocessing.run_preproc(pd.read_parquet(split_info["test_raw_path"]))
    train_path = SPLITS_DIR / "train_proc.parquet"
    test_path = SPLITS_DIR / "test_proc.parquet"
    train.to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)
    logger.info(
        "После препроцессинга: train %s (fraud %.4f), test %s (fraud %.4f)",
        train.shape, train[preprocessing.LABEL].mean(), test.shape, test[preprocessing.LABEL].mean(),
    )
    return {**split_info, "train_proc_path": str(train_path), "test_proc_path": str(test_path)}
