"""Обучение кандидата CatBoost на train-месяцах окна."""

import logging
from datetime import datetime, timezone

import pandas as pd

from pipeline.features import training
from pipeline.settings import MODELS_DIR

logger = logging.getLogger(__name__)


def fit_catboost(preproc_info: dict) -> dict:
    """Учит модель-кандидата и считает её метрики на свежем (test) месяце.

    Читает: train_proc.parquet и test_proc.parquet.
    Ранняя остановка идёт по последним 15% train по времени, а не по test:
    test-месяц остаётся «невиданным» и для кандидата, и для прод-модели,
    поэтому сравнение в quality gate честное.
    Пишет: черновик MODELS_DIR/.staging_<version>.cbm. latest.txt не трогает,
    сервис скоринга черновик не видит.
    Повтор безопасен: создаётся новый черновик с новой версией, старый
    черновик ни на что не влияет и не публикуется.
    """
    train = pd.read_parquet(preproc_info["train_proc_path"])
    test = pd.read_parquet(preproc_info["test_proc_path"])
    order = pd.Series(range(len(train)))
    fit_part, valid_part = training.time_holdout(train, order)
    model = training.fit_model(fit_part, valid_part)
    metrics = training.evaluate(model, test)
    logger.info("Кандидат на test %s: %s", preproc_info["test_month"], metrics)

    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    staging_model = MODELS_DIR / f".staging_{version}.cbm"
    model.save_model(str(staging_model))
    logger.info("Черновик модели сохранён: %s", staging_model)
    return {
        **{k: preproc_info[k] for k in ("dataset_path", "batch_month", "train_months", "test_month", "test_proc_path")},
        "version": version,
        "staging_model": str(staging_model),
        "train_rows": int(len(train)),
        "candidate": metrics,
    }
