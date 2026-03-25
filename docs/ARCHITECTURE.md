# Архитектура системы APT Early Detection

## Обзор

Система выполняет классификацию сетевого трафика в режиме **раннего потока** — решение принимается
по первым 5 пакетам соединения, до завершения потока. Это ключевое отличие от классических
систем обнаружения вторжений, ожидающих окончания сессии.

```
         Источник трафика
               │
       ┌───────▼────────┐
       │  FeatureExtractor│   (features/feature_extractor.py)
       │  mode=early     │   14 признаков из первых N пакетов
       └───────┬────────┘
               │ X: DataFrame[14]
       ┌───────▼────────┐
       │  ML-классификатор│   (models/*.pkl)
       │  RF или XGBoost │
       └───────┬────────┘
               │ label + confidence + probabilities
       ┌───────▼────────┐
       │   FastAPI       │   (api/app.py)
       │  /predict/early │
       └───────┬────────┘
               │ JSON ответ
          Потребитель API
```

---

## Компоненты

### 1. `features/feature_extractor.py` — Экстрактор признаков

**Класс:** `FeatureExtractor(n_packets=5)`

Три режима входных данных:

| Метод | Вход | Описание |
|-------|------|----------|
| `from_synthetic_csv` | CSV нашего формата | Синтетические или обработанные данные ISCX |
| `from_cicflowmeter_csv` | CICFlowMeter CSV | Нормализация имён колонок, вычисление производных признаков |
| `from_pcap` | .pcap файл | Прямое чтение через Scapy, группировка по 5-кортежам |

**Признаки раннего потока (14 штук):**

| Признак | Описание |
|---------|----------|
| `avg_packet_size` | Среднее значение размера пакетов |
| `std_packet_size` | Стандартное отклонение размера пакетов |
| `min_packet_size` | Минимальный размер пакета |
| `max_packet_size` | Максимальный размер пакета |
| `avg_interarrival` | Среднее межпакетное время |
| `std_interarrival` | Стандартное отклонение межпакетного времени |
| `min_interarrival` | Минимальное межпакетное время |
| `max_interarrival` | Максимальное межпакетное время |
| `incoming_ratio` | Доля входящих пакетов |
| `packet_count` | Количество пакетов в сегменте |
| `total_bytes` | Суммарный объём данных |
| `flow_duration` | Длительность наблюдаемого сегмента |
| `bytes_per_second` | Пропускная способность (байт/с) |
| `pkts_per_second` | Интенсивность (пакет/с) |

Все признаки вычислимы уже после получения первых 5 пакетов, что обеспечивает
возможность раннего обнаружения без ожидания завершения потока.

---

### 2. `ml/` — Обучение и оценка

#### `train_model.py`

Пайплайн обучения:
1. Загрузка данных через `FeatureExtractor`
2. Стратифицированное разбиение 70/30 (`random_state=42`)
3. 5-кратная кросс-валидация (`StratifiedKFold`, `f1_weighted`)
4. Обучение на тренировочной выборке
5. Оценка на тестовой выборке
6. Сохранение модели (`.pkl`) и метрик (`.json`)

**Модели:**

| Модель | Гиперпараметры | Назначение |
|--------|---------------|-----------|
| `RandomForestClassifier` | `n_estimators=50`, `max_depth=8`, `class_weight=balanced` | Стабильный бейзлайн, интерпретируемые важности признаков |
| `XGBClassifier` | `n_estimators=50`, `max_depth=4`, `lr=0.1`, `tree_method=hist` | Градиентный бустинг, `scale_pos_weight` для дисбаланса |

Параметры адаптируются к размеру датасета: при `n_samples > 20 000` используются
ограничения глубины и выборки для экономии памяти.

#### `evaluate_model.py`

Генерирует:
- Матрицу ошибок (`*_cm.png`)
- Кривую ROC (`*_roc.png`)
- Важность признаков (`*_importance.png`)
- Итоговую таблицу метрик

#### `shap_explain.py`

Использует **SHAP TreeExplainer** для объяснимости:
- `shap_beeswarm.png` — глобальная важность признаков по всей выборке
- `shap_waterfall_N.png` — индивидуальное объяснение конкретного потока
- `shap_summary.csv` — mean|SHAP| по каждому признаку (для таблицы в дипломе)

---

### 3. `api/app.py` — FastAPI-сервис

**Эндпоинты:**

| Эндпоинт | Метод | Описание |
|----------|-------|----------|
| `/predict/early` | POST | Классификация по 14 early-признакам |
| `/predict/full` | POST | Классификация по 24 full-признакам |
| `/health` | GET | Liveness probe |
| `/ready` | GET | Readiness probe (проверяет загрузку модели) |
| `/metrics` | GET | Prometheus-метрики (кол-во запросов, латентность) |

**Схема ответа:**
```json
{
  "label":        "normal",
  "label_id":     0,
  "confidence":   0.92,
  "probabilities": {"normal": 0.92, "vpn": 0.07, "attack": 0.01},
  "model_used":   "rf_early",
  "latency_ms":   2.1
}
```

---

### 4. `dashboard/app.py` — Streamlit-дашборд

4 страницы:
1. **Live Monitor** — потоковая классификация входящих записей
2. **Model Metrics** — таблица метрик, confusion matrix, ROC-кривые
3. **SHAP Explanations** — интерактивные объяснения модели
4. **Dataset Explorer** — визуализация распределений признаков

---

### 5. `realtime/` — Детектор в реальном времени

Компоненты:
- `packet_capture.py` — захват пакетов через Scapy (промискуитетный режим)
- `flow_builder.py` — накопление пакетов по 5-кортежам потока
- `detector.py` — применение модели по достижении N пакетов

В демо-режиме использует синтетически сгенерированные пакеты.

---

## Поток данных

```
PCAP / CSV / Live packets
         │
         ▼
   FeatureExtractor
   (нормализация, очистка inf/NaN, median imputation)
         │
         ▼
   14 early-признаков
         │
    ┌────┴────┐
    │         │
  RF early  XGB early
    │         │
    └────┬────┘
         │  predict_proba()
         ▼
  label + confidence + probabilities
         │
    ┌────┴──────────────┐
    │                   │
  FastAPI            Streamlit
  response           dashboard
```

---

## Хранение моделей

```
models/
├── best_early_model.pkl    — лучшая early-модель (RF, F1=0.9007)
├── rf_early.pkl            — Random Forest (early)
├── xgb_early.pkl           — XGBoost (early)
├── rf_full.pkl             — Random Forest (full) — бейзлайн
├── xgb_full.pkl            — XGBoost (full) — бейзлайн
├── *_metrics.json          — метрики на тестовой выборке
├── *_cm.png                — матрицы ошибок
├── *_roc.png               — ROC-кривые
├── *_importance.png        — важность признаков
├── shap_beeswarm.png       — глобальная SHAP-диаграмма
├── shap_waterfall_*.png    — SHAP для отдельных потоков
├── shap_summary.csv        — mean|SHAP| по признакам
├── summary.csv             — сводная таблица всех моделей
└── comparison_full_vs_early.png — сравнение early vs full
```

---

## Зависимости

| Пакет | Версия | Назначение |
|-------|--------|-----------|
| scikit-learn | 1.8.0 | Random Forest, метрики, CV |
| xgboost | 3.2.0 | XGBoost классификатор |
| shap | 0.50.0 | SHAP TreeExplainer |
| fastapi | 0.129.0 | REST API |
| streamlit | 1.55.0 | Дашборд |
| scapy | 2.7.0 | PCAP-парсинг, захват пакетов |
| pandas | 2.3.3 | Обработка данных |
| numpy | 2.3.5 | Вычисления |
