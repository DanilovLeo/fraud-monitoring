"""Собственный препроцессинг транзакций.

Один и тот же модуль используется в трёх местах:
  * онлайн-скоринг (fraud_detector/app/app.py),
  * DAG переобучения (pipeline/retrain/preprocess_dataset.py),
  * обучение стартовой модели (fraud_detector/app/bootstrap_model.py).
Поэтому признаки при обучении и инференсе совпадают по построению.

Что делается:
  1. Очистка: сумма приводится к числу и обрезается снизу нулём, пропуски в
     числовых колонках заполняются медианно-нейтральными значениями, пропуски
     в категориях — строкой "unknown".
  2. Время: час, день недели, флаги ночи и выходного, час в виде sin/cos
     (полночь и 23:00 рядом, а не на разных концах шкалы).
  3. Сумма: log1p(amount) и флаг «круглой» суммы (кратна 10).
  4. География: расстояние клиент → мерчант по формуле гаверсинуса (км) и
     его логарифм, логарифм населения города клиента.
  5. Пол кодируется в 0/1.
  6. Кросс-категория cat_id × «ночь»: ночные траты в части категорий
     заметно чаще мошеннические, CatBoost получает это сочетание явно.
  7. Удаляются персональные и избыточные поля: имена, улица, индекс,
     город (дублирует штат + координаты), сырые координаты мерчанта.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Категориальные признаки модели. Источник правды для обучения и скоринга.
CAT_FEATURES = ["merch", "cat_id", "us_state", "jobs", "cat_night"]

NUM_FEATURES = [
    "amount",
    "amount_log",
    "amount_is_round",
    "hour",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "is_night",
    "is_weekend",
    "lat",
    "lon",
    "dist_km",
    "dist_log",
    "city_pop_log",
    "is_male",
]

FEATURES = NUM_FEATURES + CAT_FEATURES
LABEL = "target"

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Расстояние по большому кругу в километрах для векторов координат."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _num(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype(float)


def _cat(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series("unknown", index=frame.index)
    return frame[column].fillna("unknown").astype(str).str.strip().replace("", "unknown")


def run_preproc(input_df: pd.DataFrame) -> pd.DataFrame:
    """Строит признаки модели из сырых колонок транзакции.

    Принимает DataFrame со схемой train.csv (одна строка в онлайне или
    месяц данных в DAG). Возвращает DataFrame ровно с колонками FEATURES
    и, если она была во входе, колонкой target. Вход не изменяется.
    """
    df = input_df.reset_index(drop=True)
    out = pd.DataFrame(index=df.index)

    amount = _num(df, "amount").clip(lower=0.0)
    out["amount"] = amount
    out["amount_log"] = np.log1p(amount)
    out["amount_is_round"] = ((amount > 0) & (np.mod(amount, 10) == 0)).astype(int)

    raw_time = df["transaction_time"] if "transaction_time" in df.columns else pd.Series(pd.NaT, index=df.index)
    times = pd.to_datetime(raw_time, errors="coerce")
    hour = times.dt.hour.fillna(12).astype(int)
    out["hour"] = hour
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["day_of_week"] = times.dt.dayofweek.fillna(0).astype(int)
    out["is_night"] = ((hour < 6) | (hour >= 22)).astype(int)
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)

    lat, lon = _num(df, "lat"), _num(df, "lon")
    out["lat"] = lat
    out["lon"] = lon
    dist = haversine_km(lat, lon, _num(df, "merchant_lat"), _num(df, "merchant_lon"))
    out["dist_km"] = dist
    out["dist_log"] = np.log1p(dist)
    out["city_pop_log"] = np.log1p(_num(df, "population_city").clip(lower=0.0))
    out["is_male"] = (_cat(df, "gender").str.upper() == "M").astype(int)

    out["merch"] = _cat(df, "merch")
    out["cat_id"] = _cat(df, "cat_id")
    out["us_state"] = _cat(df, "us_state")
    out["jobs"] = _cat(df, "jobs")
    out["cat_night"] = out["cat_id"] + "_" + np.where(out["is_night"] == 1, "night", "day")

    out = out[FEATURES]
    if LABEL in df.columns:
        out[LABEL] = pd.to_numeric(df[LABEL], errors="coerce").fillna(0).astype(int)
    logger.debug("Препроцессинг: %s -> %s", input_df.shape, out.shape)
    return out
