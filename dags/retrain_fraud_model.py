"""DAG скользящего переобучения антифрод-модели с quality gate.

Файл только собирает граф. Логика задач лежит в pipeline/retrain.
"""

from datetime import datetime, timedelta

from airflow.decorators import dag, task

from pipeline.retrain.commit_training_window import commit_training_window
from pipeline.retrain.evaluate_vs_prod import evaluate_vs_prod
from pipeline.retrain.extract_month_batch import extract_month_batch
from pipeline.retrain.fit_catboost import fit_catboost
from pipeline.retrain.preprocess_dataset import preprocess_dataset
from pipeline.retrain.publish_model import publish_model
from pipeline.retrain.slide_training_window import slide_training_window
from pipeline.retrain.time_train_test_split import time_train_test_split


@dag(
    dag_id="retrain_fraud_model",
    description="Каждые 5 минут: сдвиг 4-месячного окна, обучение, quality gate, публикация",
    schedule="*/5 * * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(minutes=15),
    default_args={"retries": 1, "retry_delay": timedelta(seconds=30)},
    tags=["mlops", "catboost", "retrain"],
    doc_md="""
### Переобучение антифрод-модели

1. `extract_month_batch` — следующий месяц из витрины по указателю `training_state`.
2. `slide_training_window` — окно из 4 месяцев: новый месяц добавляется, старый выпадает.
3. `time_train_test_split` — 3 ранних месяца в train, самый свежий в test.
4. `preprocess_dataset` — собственный препроцессинг (`fraud_detector/src/preprocessing.py`).
5. `fit_catboost` — кандидат, early stopping по хвосту train.
6. `evaluate_vs_prod` — кандидат и прод скорят один и тот же test-месяц, сравнение по PR-AUC.
7. `publish_model` — публикация в общий каталог моделей только если кандидат не хуже.
8. `commit_training_window` — сдвиг указателя месяца, только после шага 7.
    """,
)
def retrain_fraud_model():
    """Цепочка extract -> slide -> split -> preprocess -> fit -> gate -> publish -> commit."""
    batch = task(extract_month_batch, task_id="extract_month_batch")()
    window = task(slide_training_window, task_id="slide_training_window")(batch)
    split = task(time_train_test_split, task_id="time_train_test_split")(window)
    processed = task(preprocess_dataset, task_id="preprocess_dataset")(split)
    candidate = task(fit_catboost, task_id="fit_catboost")(processed)
    gate = task(evaluate_vs_prod, task_id="evaluate_vs_prod")(candidate)
    published = task(publish_model, task_id="publish_model")(gate)
    task(commit_training_window, task_id="commit_training_window")(published)


retrain_fraud_model()
