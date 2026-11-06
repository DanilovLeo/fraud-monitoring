import json
import logging
import os
import sys
import time

import pandas as pd
from confluent_kafka import Consumer
from confluent_kafka import Producer
from prometheus_client import start_http_server

sys.path.append(os.path.abspath('./src'))  # noqa: PTH100
from preprocessing import run_preproc
from scorer import ensure_bootstrap, make_pred

from metrics import ERRORS, PROCESSING_SECONDS, STAGE_SECONDS, ScoringStats


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/service.log'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Set kafka configuration file
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TRANSACTIONS_TOPIC = os.getenv("KAFKA_TRANSACTIONS_TOPIC", "transactions")
SCORING_TOPIC = os.getenv("KAFKA_SCORING_TOPIC", "scoring")
METRICS_PORT = int(os.getenv("METRICS_PORT", "8000"))


class ProcessingService:
    def __init__(self):
        self.consumer_config = {
            'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
            'group.id': 'ml-scorer',
            'auto.offset.reset': 'earliest',
        }
        self.producer_config = {
            'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
        }
        self.consumer = Consumer(self.consumer_config)
        self.consumer.subscribe([TRANSACTIONS_TOPIC])
        self.producer = Producer(self.producer_config)
        self.stats = ScoringStats()
        start_http_server(METRICS_PORT)
        logger.info("Метрики Prometheus: порт %s", METRICS_PORT)

    def process_message(self, msg):
        started = time.perf_counter()
        try:
            data = json.loads(msg.value().decode('utf-8'))
            transaction_id = data['transaction_id']
            input_df = pd.DataFrame([data['data']])

            with STAGE_SECONDS.labels(stage="preprocess").time():
                processed_df = run_preproc(input_df)
            with STAGE_SECONDS.labels(stage="predict").time():
                submission, y_proba, model_version = make_pred(processed_df, "kafka_stream")

            submission['transaction_id'] = transaction_id
            submission['model_version'] = model_version

            with STAGE_SECONDS.labels(stage="publish").time():
                self.producer.produce(SCORING_TOPIC, value=submission.to_json(orient='records'))
                self.producer.flush()

            amount = float(pd.to_numeric(input_df.get('amount', pd.Series([0])), errors='coerce').fillna(0).iloc[0])
            self.stats.observe(float(y_proba[0]), amount, model_version)
            PROCESSING_SECONDS.observe(time.perf_counter() - started)
            return True
        except Exception as e:
            ERRORS.inc()
            logger.exception(f"Error processing message: {e}")
            return False

    def process_messages(self):
        while True:
            msg = self.consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                logger.error(f"Kafka error: {msg.error()}")
                continue
            
            self.process_message(msg)


if __name__ == "__main__":
    logger.info('Starting Kafka ML scoring service...')
    ensure_bootstrap()
    service = ProcessingService()
    try:
        service.process_messages()
    except KeyboardInterrupt:
        logger.info('Service stopped by user')
