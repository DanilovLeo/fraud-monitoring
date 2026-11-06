"""Публикация кандидата в общий каталог моделей (или фиксация отказа)."""

import json
import logging
import os
from pathlib import Path

from pipeline.settings import MODELS_DIR

logger = logging.getLogger(__name__)


def publish_model(gate_info: dict) -> dict:
    """Публикует кандидата, если quality gate разрешил, иначе только пишет отчёт.

    Читает: черновик .staging_<version>.cbm и решение gate из gate_info.
    Пишет при decision=publish:
      * my_catboost_<version>.cbm (атомарный rename черновика),
      * my_catboost_<version>.metrics.json,
      * latest.txt (через временный файл + os.replace, сервис скоринга
        никогда не увидит пустой или недописанный указатель).
    При decision=reject: rejected_<version>.metrics.json, черновик удаляется,
    latest.txt не меняется — прод остаётся на старой версии.
    Указатель месяца здесь НЕ двигается, это делает commit_training_window.
    Повтор безопасен: если черновик уже переименован, используется готовый файл;
    latest.txt просто перезаписывается той же версией.
    """
    version = gate_info["version"]
    staging = Path(gate_info["staging_model"])
    report = {k: gate_info.get(k) for k in (
        "version", "decision", "reason", "batch_month", "train_months", "test_month",
        "train_rows", "candidate", "prod_version", "prod")}

    if gate_info["decision"] != "publish":
        (MODELS_DIR / f"rejected_{version}.metrics.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        staging.unlink(missing_ok=True)
        logger.warning("Кандидат %s НЕ опубликован: %s. Прод остаётся %s",
                       version, gate_info["reason"], gate_info["prod_version"])
        return {**gate_info, "published_version": None}

    final_model = MODELS_DIR / f"my_catboost_{version}.cbm"
    if staging.exists():
        os.replace(staging, final_model)
    elif not final_model.exists():
        raise FileNotFoundError(f"Нет файла модели для публикации: {staging}")
    os.chmod(final_model, 0o666)
    (MODELS_DIR / f"my_catboost_{version}.metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    latest_tmp = MODELS_DIR / f".latest.{version}.txt"
    latest_tmp.write_text(version + "\n", encoding="utf-8")
    os.replace(latest_tmp, MODELS_DIR / "latest.txt")
    logger.info("Опубликована модель %s (%s)", version, final_model)
    return {**gate_info, "published_version": version}
