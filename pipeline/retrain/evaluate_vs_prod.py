"""Quality gate: сравнение кандидата с текущей прод-моделью на одном и том же месяце."""

import logging

import pandas as pd

from pipeline.features import training
from pipeline.settings import MODELS_DIR, QUALITY_TOLERANCE

logger = logging.getLogger(__name__)


def _load_prod():
    """Возвращает (версия, модель) из latest.txt или (None, None), если прода нет."""
    from catboost import CatBoostClassifier

    latest = MODELS_DIR / "latest.txt"
    if not latest.exists() or not latest.read_text(encoding="utf-8").strip():
        return None, None
    version = latest.read_text(encoding="utf-8").strip()
    path = MODELS_DIR / f"my_catboost_{version}.cbm"
    if not path.exists():
        return version, None
    model = CatBoostClassifier()
    model.load_model(str(path))
    return version, model


def evaluate_vs_prod(train_info: dict) -> dict:
    """Решает, можно ли публиковать кандидата.

    Читает: test_proc.parquet (самый свежий месяц окна), latest.txt и файл
    прод-модели из общего каталога.
    Обе модели скорят ОДИН и тот же test-месяц, которого ни одна из них не
    видела при обучении. Основная метрика — PR-AUC (average precision): фрод
    — редкий класс, ROC-AUC на таком дисбалансе почти всегда близок к 1 и
    плохо различает модели.
    Правило: publish, если PR-AUC кандидата >= PR-AUC прода - QUALITY_TOLERANCE.
    Если прода нет, прод-модель не читается или в test один класс
    (метрика не определена) — публикуем кандидата.
    Ничего не пишет: решение уходит в XCom, повтор даёт то же решение.
    """
    test = pd.read_parquet(train_info["test_proc_path"])
    candidate = train_info["candidate"]
    prod_version, prod_model = _load_prod()
    prod_metrics = None
    if prod_model is not None:
        try:
            prod_metrics = training.evaluate(prod_model, test)
        except ValueError as exc:
            logger.warning("Прод-модель %s не применима к новым признакам: %s", prod_version, exc)

    cand_pr = candidate.get("pr_auc")
    prod_pr = prod_metrics.get("pr_auc") if prod_metrics else None
    if prod_pr is None or cand_pr is None:
        decision, reason = "publish", "нет сопоставимой метрики прода"
    elif cand_pr >= prod_pr - QUALITY_TOLERANCE:
        decision, reason = "publish", f"PR-AUC {cand_pr:.4f} >= {prod_pr:.4f} - {QUALITY_TOLERANCE}"
    else:
        decision, reason = "reject", f"PR-AUC {cand_pr:.4f} < {prod_pr:.4f} - {QUALITY_TOLERANCE}"
    logger.info("Quality gate на %s: %s (%s). Прод=%s", train_info["test_month"], decision, reason, prod_version)
    return {**train_info, "decision": decision, "reason": reason,
            "prod_version": prod_version, "prod": prod_metrics}
