"""Prometheus-метрики сервиса скоринга."""

from prometheus_client import Counter, Gauge, Histogram

SCORE_BUCKETS = [round(i * 0.05, 2) for i in range(1, 21)]
LATENCY_BUCKETS = [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
FRAUD_THRESHOLD = 0.5

TRANSACTIONS = Counter(
    "fraud_scoring_transactions_total", "Обработанные транзакции"
)
FLAGGED = Counter(
    "fraud_scoring_flagged_total", "Транзакции с флагом фрода (score > 0.5)"
)
FLAGGED_SHARE = Gauge(
    "fraud_scoring_flagged_share", "Доля транзакций с флагом фрода с момента старта сервиса"
)
FLAGGED_AMOUNT = Counter(
    "fraud_scoring_flagged_amount_dollars_total", "Сумма транзакций с флагом фрода, $"
)
ERRORS = Counter(
    "fraud_scoring_errors_total", "Сообщения, которые не удалось обработать"
)
PROCESSING_SECONDS = Histogram(
    "fraud_scoring_processing_seconds", "Время обработки одной транзакции", buckets=LATENCY_BUCKETS
)
STAGE_SECONDS = Histogram(
    "fraud_scoring_stage_seconds",
    "Время этапа обработки: preprocess, predict, publish",
    ["stage"],
    buckets=LATENCY_BUCKETS,
)
SCORE = Histogram("fraud_scoring_score", "Распределение скоров", buckets=SCORE_BUCKETS)
MODEL_SWITCHES = Counter(
    "fraud_scoring_model_switches_total", "Переходы на новую версию модели без перезапуска", ["version"]
)
MODEL_ACTIVE = Gauge(
    "fraud_scoring_model_active", "1 у версии модели, которая сейчас скорит", ["version"]
)


class ScoringStats:
    """Обновляет метрики по каждой обработанной транзакции."""

    def __init__(self):
        self.total = 0
        self.flagged = 0
        self.version = None

    def observe(self, score: float, amount: float, version: str) -> None:
        self.total += 1
        TRANSACTIONS.inc()
        SCORE.observe(score)
        if score > FRAUD_THRESHOLD:
            self.flagged += 1
            FLAGGED.inc()
            FLAGGED_AMOUNT.inc(max(amount, 0.0))
        FLAGGED_SHARE.set(self.flagged / self.total)
        self._observe_version(version)

    def _observe_version(self, version: str) -> None:
        if version == self.version:
            return
        if self.version is not None:
            MODEL_ACTIVE.labels(version=self.version).set(0)
            MODEL_SWITCHES.labels(version=version).inc()
        MODEL_ACTIVE.labels(version=version).set(1)
        self.version = version
