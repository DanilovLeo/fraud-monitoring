"""Фиксация сдвига окна обучения после того, как судьба модели решена."""

import logging
import os
from datetime import date
from pathlib import Path

from pipeline.db import advance_training_state, get_conn
from pipeline.settings import CURRENT_DATASET

logger = logging.getLogger(__name__)


def commit_training_window(publish_info: dict) -> str:
    """Переносит pending_window.parquet в current.parquet и сдвигает training_state.

    Запускается только после успешного publish_model, то есть после того, как
    модель сохранена в каталог (или отклонена gate-ом с записью отчёта).
    Если любая предыдущая задача упала, окно и указатель месяца остаются
    прежними, и следующий запуск DAG обработает тот же месяц заново.
    Отклонённый кандидат тоже сдвигает окно: месяц данных обработан, иначе
    DAG бесконечно переучивался бы на одном и том же окне.
    Повтор безопасен: если pending уже перенесён, файл не трогается, а
    UPDATE training_state срабатывает только при next_batch_month = batch_month.
    """
    pending = Path(publish_info["dataset_path"])
    if pending.exists():
        os.replace(pending, CURRENT_DATASET)
        logger.info("Окно обучения обновлено: %s", CURRENT_DATASET)
    elif not CURRENT_DATASET.exists():
        raise FileNotFoundError(f"Нет нового окна обучения: {pending}")
    else:
        logger.info("Окно уже перенесено в %s.", CURRENT_DATASET)

    conn = get_conn()
    try:
        advance_training_state(conn, date.fromisoformat(publish_info["batch_month"]))
    finally:
        conn.close()
    return publish_info.get("published_version") or f"rejected:{publish_info['version']}"
