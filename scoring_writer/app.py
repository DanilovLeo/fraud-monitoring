import json
import logging
import os

import psycopg2
from confluent_kafka import Consumer
from prometheus_client import Counter, Histogram, start_http_server

# Настройка логгирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

METRICS_PORT = int(os.getenv("METRICS_PORT", "8001"))

WRITTEN = Counter("fraud_writer_scores_written_total", "Скоры, записанные в PostgreSQL")
WRITTEN_FRAUD = Counter("fraud_writer_fraud_written_total", "Записанные транзакции с флагом фрода")
WRITE_ERRORS = Counter("fraud_writer_errors_total", "Ошибки чтения или записи")
WRITE_SECONDS = Histogram(
    "fraud_writer_write_seconds",
    "Время записи одного скора в PostgreSQL",
    buckets=[0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25],
)
WRITTEN_SCORE = Histogram(
    "fraud_writer_score",
    "Распределение записанных скоров",
    buckets=[round(i * 0.05, 2) for i in range(1, 21)],
)


def get_db_config():
    return {
        "host": os.getenv("POSTGRES_HOST"),
        "port": os.getenv("POSTGRES_PORT"),
        "database": os.getenv("POSTGRES_DB"),
        "user": os.getenv("POSTGRES_USER"),
        "password": os.getenv("POSTGRES_PASSWORD"),
    }


def create_table(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS scores (
                id SERIAL PRIMARY KEY,
                transaction_id TEXT NOT NULL,
                score FLOAT NOT NULL,
                fraud_flag INT NOT NULL,
                model_version TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cur.execute("""
            ALTER TABLE scores ADD COLUMN IF NOT EXISTS model_version TEXT;
        """)
        conn.commit()
        logger.info("Таблица 'scores' проверена или создана.")


def insert_score(conn, data):
    with WRITE_SECONDS.time(), conn.cursor() as cur:
        cur.execute("""
            INSERT INTO scores (transaction_id, score, fraud_flag, model_version)
            VALUES (%s, %s, %s, %s);
        """, (
            data["transaction_id"],
            data["score"],
            data["fraud_flag"],
            data.get("model_version"),
        ))
        conn.commit()
    WRITTEN.inc()
    WRITTEN_SCORE.observe(float(data["score"]))
    if int(data["fraud_flag"]) == 1:
        WRITTEN_FRAUD.inc()


def run_consumer():
    kafka_bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
    scoring_topic = os.getenv("KAFKA_SCORING_TOPIC")

    consumer_config = {
        'bootstrap.servers': kafka_bootstrap_servers,
        'group.id': 'scoring-writer',
        'auto.offset.reset': 'earliest',
    }

    logger.info(f"Подключение к Kafka: {kafka_bootstrap_servers}, топик: {scoring_topic}")
    consumer = Consumer(consumer_config)
    consumer.subscribe([scoring_topic])

    start_http_server(METRICS_PORT)
    logger.info("Метрики Prometheus: порт %s", METRICS_PORT)

    db_config = get_db_config()
    conn = psycopg2.connect(**db_config)
    create_table(conn)

    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                logger.debug("Ожидание сообщений...")
                continue
            if msg.error():
                logger.error(f"Kafka error: {msg.error()}")
                continue

            try:
                value_str = msg.value().decode('utf-8')
                logger.debug(f"Получено сообщение: {value_str}")
                data_list = json.loads(value_str)

                # Проверяем, что это список
                if isinstance(data_list, list):
                    for data in data_list:
                        required_keys = ['transaction_id', 'score', 'fraud_flag']
                        if not all(k in data for k in required_keys):
                            logger.error(f"Некорректный формат элемента: {data}")
                            continue
                        insert_score(conn, data)
                        logger.debug(f"Записано в БД: {data['transaction_id']}")
                else:
                    logger.error(f"Ожидался список, получен: {type(data_list)}")

            except json.JSONDecodeError as je:
                WRITE_ERRORS.inc()
                logger.exception(f"Ошибка декодирования JSON: {je}")
            except Exception as e:
                WRITE_ERRORS.inc()
                conn.rollback()
                logger.exception(f"Ошибка обработки сообщения: {e}")

    except KeyboardInterrupt:
        logger.info("Потребитель остановлен пользователем.")
    finally:
        consumer.close()
        conn.close()


if __name__ == "__main__":
    logger.info("Запуск потребителя и писателя в БД...")
    run_consumer()
