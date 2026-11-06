"""Сдвиг окна обучения из WINDOW_MONTHS месяцев на один месяц вперёд."""

import logging
from datetime import date

import pandas as pd

from pipeline.settings import CURRENT_DATASET, PENDING_WINDOW, WINDOW_MONTHS

logger = logging.getLogger(__name__)


def slide_training_window(batch_info: dict) -> dict:
    """Добавляет новый месяц к текущему окну и оставляет последние WINDOW_MONTHS месяцев.

    Читает current.parquet и parquet батча из batch_info. Самый старый месяц
    выпадает, число месяцев в окне фиксировано. Новое окно пишет
    в pending_window.parquet и не заменяет current.parquet: замена происходит
    только после публикации модели, чтобы повтор задачи не сдвинул окно дважды.
    Возвращает путь к новому окну и месяц добавленного батча.
    """
    if not CURRENT_DATASET.exists():
        raise FileNotFoundError(f"Нет исходного датасета {CURRENT_DATASET}")

    batch_month = date.fromisoformat(batch_info["batch_month"])
    current = pd.read_parquet(CURRENT_DATASET)
    batch = pd.read_parquet(batch_info["batch_path"])
    current["transaction_time"] = pd.to_datetime(current["transaction_time"])
    batch["transaction_time"] = pd.to_datetime(batch["transaction_time"])

    combined = pd.concat([current, batch], ignore_index=True)
    # Строки самого батча могли уже лежать в current (повтор после частичного
    # успеха) — выкидываем этот месяц из current перед склейкой, дублей не будет.
    combined["_p"] = combined["transaction_time"].dt.to_period("M")
    batch_period = pd.Period(batch_month, freq="M")
    current_part = combined.iloc[: len(current)]
    combined = pd.concat([current_part[current_part["_p"] != batch_period], combined.iloc[len(current):]])
    periods = combined["_p"]
    first_kept = batch_period - (WINDOW_MONTHS - 1)
    oldest = sorted(periods.loc[periods < first_kept].unique().astype(str))
    updated = combined.loc[(periods >= first_kept) & (periods <= batch_period)]
    updated = updated.drop(columns="_p").sort_values("transaction_time")
    updated.to_parquet(PENDING_WINDOW, index=False)

    logger.info(
        "Окно после сдвига: %s строк, %s — %s. Удалены месяцы %s.",
        len(updated),
        updated["transaction_time"].min(),
        updated["transaction_time"].max(),
        oldest,
    )
    return {
        "dataset_path": str(PENDING_WINDOW),
        "batch_month": batch_month.isoformat(),
    }
