"""DAG поставки нового размеченного месяца в витрину.

Файл только собирает граф. Логика задач лежит в pipeline/supply.
"""

from datetime import datetime, timedelta

from airflow.decorators import dag, task

from pipeline.supply.append_labeled_batch import append_labeled_batch
from pipeline.supply.prune_old_months import prune_old_months
from pipeline.supply.sample_next_month import sample_next_month


@dag(
    dag_id="supply_labeled_month",
    description="Каждые 5 минут дописывает в витрину 50 тысяч строк следующего месяца",
    schedule="*/5 * * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(minutes=15),
    default_args={"retries": 1, "retry_delay": timedelta(seconds=30)},
    tags=["mlops", "etl"],
    doc_md="""
### Поставка размеченного месяца

1. Из витрины `labeled_transactions` случайно берётся `SUPPLY_BATCH_SIZE` строк (50 000).
2. `transaction_time` переносится в месяц, следующий за последним в витрине.
3. Батч дописывается в витрину.
4. Наблюдения старше `HISTORY_YEARS` (2 года) от самого нового месяца удаляются.
    """,
)
def supply_labeled_month():
    """Цепочка sample -> append -> prune."""
    sampled = task(sample_next_month, task_id="sample_next_month")()
    appended = task(append_labeled_batch, task_id="append_labeled_batch")(sampled)
    task(prune_old_months, task_id="prune_old_months")(appended)


supply_labeled_month()
