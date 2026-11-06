# Мониторинг антифрод-скоринга: Prometheus + Grafana

Сервис скоринга читает транзакции из Kafka, считает скор и флаг фрода и пишет результат в PostgreSQL. Airflow каждые 5 минут переобучает модель, и скоринг подхватывает новую версию без перезапуска. Поверх этого добавлен мониторинг:

- **Prometheus** каждую секунду снимает метрики со скоринга, с сервиса записи и с Node Exporter;
- **Grafana** поднимается с готовыми источниками данных, тремя дашбордами и алертами.

## Запуск

```bash
cp .env.example .env
# в .env укажите абсолютный путь к train.csv:
# TRAIN_DATA_CSV=/полный/путь/к/train.csv
docker compose up -d --build
```

Первый старт занимает несколько минут: загружается витрина и обучается стартовая модель. Сервисы `data_loader` и `model_bootstrap` должны завершиться с `Exited (0)` (проверка: `docker compose ps -a`), остальные — `Up`.

| Что | Адрес | Логин |
|---|---|---|
| Grafana | http://localhost:3000 | `admin` / `admin` |
| Prometheus | http://localhost:9090 | — |
| Метрики скоринга (внутри сети) | `fraud_detector:8000/metrics` | — |
| Метрики записи (внутри сети) | `scoring_writer:8001/metrics` | — |
| Airflow | http://localhost:8081 | `admin` / `admin` |

## Дашборды

Grafana → Dashboards → папка **Fraud monitoring**. Первый дашборд открывается сразу как домашний.

| Дашборд | Что на нём |
|---|---|
| **1. Скоринг транзакций** | транзакций в секунду, время обработки p50/p95/p99, доля фрода, время по этапам, распределение скоров во времени (heatmap), сумма под подозрением, текущая версия модели, отметки смены модели; внизу — таблица и график из PostgreSQL по версиям модели |
| **2. Запись результатов** | скорость и время записи в PostgreSQL, число выявленного фрода, распределение записанных скоров, ошибки |
| **3. Системные метрики** | CPU, память, сеть, диск (Node Exporter) |

## Метрики

**Сервис скоринга** (`fraud_detector`, порт 8000). Все значения обновляются на каждом сообщении.

| Метрика | Тип | Что это |
|---|---|---|
| `fraud_scoring_transactions_total` | counter | обработанные транзакции |
| `fraud_scoring_processing_seconds` | histogram | время обработки одной транзакции |
| `fraud_scoring_score` | histogram | распределение скоров |
| `fraud_scoring_flagged_total` | counter | транзакции с флагом фрода (score > 0.5) |
| `fraud_scoring_flagged_share` | gauge | доля транзакций с флагом фрода с момента старта |
| `fraud_scoring_model_switches_total{version}` | counter | +1, когда скоринг подхватил новую версию модели |
| `fraud_scoring_model_active{version}` | gauge | 1 у версии, которая сейчас скорит |
| `fraud_scoring_stage_seconds{stage}` | histogram | время этапа: `preprocess`, `predict`, `publish` |
| `fraud_scoring_flagged_amount_dollars_total` | counter | сумма транзакций с флагом фрода, $ |
| `fraud_scoring_errors_total` | counter | сообщения, которые не удалось обработать |

**Сервис записи** (`scoring_writer`, порт 8001):
- `fraud_writer_scores_written_total` — записанные скоры;
- `fraud_writer_write_seconds` — время записи;
- `fraud_writer_fraud_written_total` — записанный фрод;
- `fraud_writer_score` — гистограмма записанных скоров;
- `fraud_writer_errors_total` — ошибки.

### Собственные метрики (их нет в коде семинара)

- **`fraud_scoring_stage_seconds{stage}`** — время по этапам. Общее время обработки показывает, что сервис замедлился, а эта метрика показывает, где именно: в препроцессинге, в модели или в отправке в Kafka. Например, препроцессинг занимает около 11 мс, а сама модель — около 4 мс, поэтому ускорять стоит в первую очередь его. График: «Время по этапам (среднее)» на дашборде 1.
- **`fraud_scoring_flagged_amount_dollars_total`** — деньги в транзакциях с флагом фрода. Доля фрода говорит о числе транзакций, а бизнесу важнее, сколько денег под угрозой. Графики: «Сумма под подозрением» и «Сумма под подозрением в минуту» на дашборде 1.
- **`fraud_scoring_model_active{version}`** показывает текущую версию модели (плитка «Модель в работе»). **`fraud_scoring_errors_total`** показывает ошибки обработки, которые иначе видны только в логах.

## Как проверить

1. **Метрики растут с потоком.** Симулятор шлёт около 10 транзакций в секунду. На дашборде 1 «Транзакций в секунду» около 10, а «Обработано всего» растёт каждые 5 секунд (интервал обновления дашборда). То же видно напрямую в Prometheus, на вкладке Graph: `fraud_scoring_transactions_total`.
2. **Отметка смены модели.** DAG `retrain_fraud_model` в Airflow публикует новую модель каждые 5 минут. Чтобы не ждать, нажмите в Airflow ▶ Trigger DAG. Через несколько секунд после публикации на графиках дашборда 1 появляется красная вертикальная отметка «Модель <версия>»: она держится около 30 секунд и подписана версией. Плитка «Модель в работе» показывает новую версию, а в таблице «Скоры по версиям модели» появляется новая строка. Отметки нет, если quality gate из прошлого задания отклонил модель (`latest.txt` не изменился).

## Алерты

Grafana → Alerting → Alert rules → папка **Fraud monitoring**. Правила лежат в `monitoring/grafana/provisioning/alerting/alerts.yaml` и проверяются каждые 10 секунд.

| Правило | Условие | Как проверить |
|---|---|---|
| **Доля фрода выше 5%** | доля транзакций с флагом фрода за минуту > 5% в течение 30 с | запустить симулятор с долей фродовых транзакций 30%: `FRAUD_INJECTION_SHARE=0.3 docker compose up -d transaction_simulator`. Через 1–2 минуты правило переходит в Firing. Вернуть обычный поток: `docker compose up -d transaction_simulator` |
| **Поток транзакций остановился** | меньше 1 транзакции в секунду в течение 30 с (или нет данных) | `docker compose stop transaction_simulator`. Через ~1 минуту правило в Firing. Вернуть: `docker compose start transaction_simulator` |

В обычном потоке доля фрода меньше 1%. Сразу после старта, пока скоринг ещё не начал работу, правило «Поток остановился» может ненадолго сработать: это ожидаемо.

## Структура мониторинга

```
monitoring/
├── prometheus/prometheus.yml                        # цели и интервал 1 с
└── grafana/
    ├── dashboards/{scoring,writer,system}.json       # дашборды
    └── provisioning/
        ├── datasources/datasources.yaml              # Prometheus + PostgreSQL
        ├── dashboards/dashboards.yaml                # загрузка дашбордов
        └── alerting/alerts.yaml                      # правила алертов
fraud_detector/app/metrics.py                         # метрики скоринга
scoring_writer/app.py                                 # метрики записи
```

Остановка: `docker compose down`. Полная очистка: `docker compose down -v`.
