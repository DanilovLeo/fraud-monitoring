"""Одноразовое обучение стартовой модели (сервис model_bootstrap).

Читает train.csv, берёт стартовое окно (WINDOW_MONTHS месяцев начиная с
INITIAL_WINDOW_START), учит CatBoost на всех месяцах окна кроме последнего,
проверяет на последнем и публикует my_catboost_initial.cbm + latest.txt в
общий каталог моделей.

Повторный старт ничего не делает, если latest.txt уже есть: опубликованные
DAG-ом версии не перетираются.
"""

import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(os.path.abspath("./src"))  # noqa: PTH100
from preprocessing import run_preproc  # noqa: E402
from training import evaluate, fit_model, time_holdout  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("model_bootstrap")

CSV_PATH = Path(os.getenv("TRANSACTIONS_CSV_PATH", "/data/train.csv"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "/models"))
WINDOW_START = pd.Timestamp(os.getenv("INITIAL_WINDOW_START", "2019-01-01"))
WINDOW_MONTHS = int(os.getenv("WINDOW_MONTHS", "4"))


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(MODELS_DIR, 0o777)
    latest = MODELS_DIR / "latest.txt"
    if latest.exists() and latest.read_text(encoding="utf-8").strip():
        logger.info("latest.txt уже есть (%s), стартовая модель не нужна.", latest.read_text().strip())
        return

    frame = pd.read_csv(CSV_PATH)
    frame["transaction_time"] = pd.to_datetime(frame["transaction_time"])
    window_end = WINDOW_START + pd.DateOffset(months=WINDOW_MONTHS)
    frame = frame[(frame["transaction_time"] >= WINDOW_START) & (frame["transaction_time"] < window_end)]
    periods = frame["transaction_time"].dt.to_period("M")
    test_month = periods.max()
    train_raw, test_raw = frame[periods < test_month], frame[periods == test_month]
    logger.info("Стартовое окно: train %s строк, test %s (%s строк)", len(train_raw), test_month, len(test_raw))

    train_proc = run_preproc(train_raw)
    fit_part, valid_part = time_holdout(train_proc, train_raw["transaction_time"].reset_index(drop=True))
    model = fit_model(fit_part, valid_part)
    metrics = evaluate(model, run_preproc(test_raw))
    logger.info("Метрики стартовой модели на %s: %s", test_month, metrics)

    tmp = MODELS_DIR / ".staging_initial.cbm"
    model.save_model(str(tmp))
    os.replace(tmp, MODELS_DIR / "my_catboost_initial.cbm")
    (MODELS_DIR / "my_catboost_initial.metrics.json").write_text(
        json.dumps({"version": "initial", "decision": "published", "test_month": str(test_month),
                    "candidate": metrics}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_latest = MODELS_DIR / ".latest.initial.txt"
    tmp_latest.write_text("initial\n", encoding="utf-8")
    os.replace(tmp_latest, latest)
    for path in MODELS_DIR.iterdir():
        os.chmod(path, 0o666)
    logger.info("Стартовая модель опубликована: %s", MODELS_DIR / "my_catboost_initial.cbm")


if __name__ == "__main__":
    main()
