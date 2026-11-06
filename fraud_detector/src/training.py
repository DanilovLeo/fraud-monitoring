"""Общая логика обучения и оценки CatBoost.

Используется DAG переобучения (через смонтированный fraud_src) и сервисом
model_bootstrap, который учит стартовую модель. Гиперпараметры и метрики
в обоих местах одинаковые.
"""

import logging

import numpy as np
import pandas as pd

from preprocessing import CAT_FEATURES, FEATURES, LABEL

logger = logging.getLogger(__name__)

THRESHOLD = 0.5
VALID_FRACTION = 0.15

CATBOOST_PARAMS = {
    "iterations": 300,
    "depth": 6,
    "learning_rate": 0.08,
    "l2_leaf_reg": 5,
    "loss_function": "Logloss",
    "eval_metric": "PRAUC",
    "auto_class_weights": "SqrtBalanced",
    "early_stopping_rounds": 30,
    "random_seed": 42,
    "thread_count": 2,
    "allow_writing_files": False,
    "verbose": 50,
}


def split_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Возвращает признаки в порядке FEATURES и вектор target."""
    missing = [c for c in FEATURES + [LABEL] if c not in frame.columns]
    if missing:
        raise ValueError(f"После препроцессинга нет колонок: {missing}")
    x = frame[FEATURES].copy()
    for column in CAT_FEATURES:
        x[column] = x[column].astype(str)
    return x, frame[LABEL].astype(int)


def time_holdout(train_proc: pd.DataFrame, order: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Отрезает последние VALID_FRACTION строк train (по времени) под early stopping.

    Тестовый месяц в ранней остановке не участвует, иначе он «подсматривается»
    при выборе числа деревьев и сравнение с прод-моделью станет нечестным.
    """
    ordered = train_proc.iloc[np.argsort(order.to_numpy(), kind="stable")]
    cut = int(len(ordered) * (1 - VALID_FRACTION))
    return ordered.iloc[:cut], ordered.iloc[cut:]


def fit_model(fit_part: pd.DataFrame, valid_part: pd.DataFrame):
    """Обучает CatBoostClassifier с ранней остановкой по valid_part."""
    from catboost import CatBoostClassifier

    x_fit, y_fit = split_xy(fit_part)
    x_val, y_val = split_xy(valid_part)
    model = CatBoostClassifier(**CATBOOST_PARAMS)
    eval_set = (x_val, y_val) if y_val.nunique() > 1 else None
    model.fit(
        x_fit,
        y_fit,
        cat_features=CAT_FEATURES,
        eval_set=eval_set,
        use_best_model=eval_set is not None,
    )
    return model


def predict_proba(model, frame: pd.DataFrame) -> np.ndarray:
    """Скор модели в порядке её собственных feature_names_ (подходит и для старых версий)."""
    names = list(model.feature_names_)
    missing = [c for c in names if c not in frame.columns]
    if missing:
        raise ValueError(f"Модель ждёт признаки, которых нет: {missing}")
    x = frame[names].copy()
    for idx in model.get_cat_feature_indices():
        x[names[idx]] = x[names[idx]].astype(str)
    return model.predict_proba(x)[:, 1]


def evaluate(model, test_proc: pd.DataFrame) -> dict:
    """PR-AUC (основная), ROC-AUC, F1 и recall на test.

    Если в test один класс, ранговые метрики не определены и возвращаются как None.
    """
    from sklearn.metrics import average_precision_score, f1_score, recall_score, roc_auc_score

    y = test_proc[LABEL].astype(int)
    proba = predict_proba(model, test_proc)
    pred = (proba > THRESHOLD).astype(int)
    both = y.nunique() > 1
    return {
        "pr_auc": float(average_precision_score(y, proba)) if both else None,
        "roc_auc": float(roc_auc_score(y, proba)) if both else None,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "rows": int(len(y)),
        "fraud_rate": float(y.mean()) if len(y) else 0.0,
    }
