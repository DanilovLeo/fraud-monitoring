"""Доступ к общим модулям скоринга (fraud_detector/src) из задач Airflow.

Каталог fraud_detector/src смонтирован в Airflow как FRAUD_SRC. Импорт через
этот модуль гарантирует, что DAG учит модель на тех же признаках, которые
считает онлайн-скоринг.
"""

import sys

from pipeline.settings import FRAUD_SRC

if FRAUD_SRC not in sys.path:
    sys.path.insert(0, FRAUD_SRC)

import preprocessing  # noqa: E402
import training  # noqa: E402

__all__ = ["preprocessing", "training"]
