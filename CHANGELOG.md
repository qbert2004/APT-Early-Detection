# Changelog

## [1.0.0] — 2024

### Added
- Двухрежимный экстрактор признаков (`early` — первые N пакетов, `full` — весь поток)
- Обучение Random Forest и XGBoost в вариантах early/full для сравнения
- 5-кратная стратифицированная кросс-валидация с отчётом F1 ± std
- SHAP TreeExplainer: beeswarm-диаграмма (глобальная), waterfall-диаграммы (per-flow)
- FastAPI-сервис: `/predict/early`, `/predict/full`, `/health`, `/ready`, `/metrics`
- Streamlit-дашборд: 4 страницы — мониторинг, метрики, SHAP, датасет
- Realtime-детектор на Scapy (демонстрационный режим)
- Docker + docker-compose для API и дашборда
- CI/CD на GitHub Actions: тесты (Python 3.11/3.12), ruff, Docker smoke-test, Codecov
- 60+ pytest-тестов: API, экстрактор признаков, модели
- Структурированное JSON-логирование через `utils/logger.py`
- Поддержка датасета ISCX VPN-nonVPN 2016 (PCAP + ARFF)

### Dataset
- Источник: ISCX VPN-nonVPN 2016 (University of New Brunswick)
- Итого: 115 446 потоков (24 000 VPN PCAP + 36 000 NonVPN PCAP + 55 446 ARFF)
- Баланс классов: ~2.4:1 (normal:vpn)
- Разбивка: 70% train / 30% test, `random_state=42`
