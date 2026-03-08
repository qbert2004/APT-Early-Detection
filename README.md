# APT Early Detection

Real-time network traffic classifier that detects **VPN tunnelling and anomalous flows** from only the **first 5 packets** of a connection — before the full flow completes.

Built for a diploma thesis on ML-based early intrusion detection.

---

## Key Features

| Feature | Detail |
|---------|--------|
| **Early-flow classification** | Decision at packet 5, not flow end |
| **Two model families** | Random Forest + XGBoost (early & full variants) |
| **SHAP interpretability** | Beeswarm + waterfall plots per flow |
| **FastAPI inference API** | `/predict/early`, `/predict/full`, Prometheus `/metrics` |
| **Streamlit dashboard** | Live monitor, model metrics, SHAP explanations, dataset explorer |
| **Docker-ready** | Multi-stage Dockerfile + docker-compose for API + dashboard |
| **CI/CD** | GitHub Actions: test matrix (py3.11/3.12), ruff lint, Docker smoke test |

---

## Dataset

Training uses the **ISCX VPN-nonVPN 2016** dataset (combined):

| Source | Flows | Label |
|--------|-------|-------|
| VPN-PCAPS-01/02 (raw pcap) | 24 000 | vpn |
| NonVPN-PCAPs-01/02/03 (raw pcap) | 36 000 | normal |
| Scenario A1/A2/B ARFF files | 55 446 | vpn / normal |
| **Total** | **115 446** | balanced ~2.4:1 |

---

## Model Performance (115K-flow dataset)

| Model | F1 (weighted) | AUC |
|-------|:---:|:---:|
| RF early  | **0.9007** | **0.9684** |
| XGB early | 0.8924 | 0.9614 |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare dataset

Place ISCX zip files in `~/Downloads/` and run:

```bash
# Extract features from raw PCAPs (all 5 ZIPs, ~22 GB)
python -X utf8 -m data.extract_from_pcap

# Convert ARFF files to CSV
python -X utf8 -m data.convert_iscx_arff

# Merge both sources
python -X utf8 -c "
import pandas as pd
a = pd.read_csv('dataset/raw_csv/iscx_pcap_early.csv', index_col='flow_id')
b = pd.read_csv('dataset/raw_csv/iscx_real.csv', index_col='flow_id')
pd.concat([a, b]).sample(frac=1, random_state=42).to_csv('dataset/raw_csv/iscx_combined.csv', index_label='flow_id')
print('Done:', len(pd.concat([a, b])), 'flows')
"
```

### 3. Train models

```bash
python -X utf8 -m ml.train_model --csv dataset/raw_csv/iscx_combined.csv
```

### 4. Evaluate & explain

```bash
python -X utf8 -m ml.evaluate_model --csv dataset/raw_csv/iscx_combined.csv
python -X utf8 -m ml.shap_explain  --csv dataset/raw_csv/iscx_combined.csv --samples 6
```

### 5. Run tests

```bash
python -X utf8 -m pytest tests/ -v
```

### 6. Start the API

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Visit: http://localhost:8000/docs

### 7. Start the dashboard

```bash
streamlit run dashboard/app.py
```

Visit: http://localhost:8501

---

## Docker

```bash
# Build and start both services
docker compose up --build

# API:       http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

---

## API Reference

### `POST /predict/early`

Classify from first-5-packet flow features.

```json
{
  "avg_packet_size": 200.0,
  "std_packet_size": 50.0,
  "min_packet_size": 100.0,
  "max_packet_size": 300.0,
  "avg_interarrival": 0.01,
  "std_interarrival": 0.002,
  "min_interarrival": 0.005,
  "max_interarrival": 0.02,
  "incoming_ratio": 0.3,
  "packet_count": 5,
  "total_bytes": 1000.0,
  "flow_duration": 0.05,
  "bytes_per_second": 20000.0,
  "pkts_per_second": 100.0,
  "model": "rf"
}
```

Response:

```json
{
  "label": "normal",
  "label_id": 0,
  "confidence": 0.92,
  "probabilities": {"normal": 0.92, "vpn": 0.07, "attack": 0.01},
  "model_used": "rf_early",
  "latency_ms": 2.1
}
```

### `GET /health` · `GET /ready` · `GET /metrics`

Standard liveness / readiness / Prometheus probes.

---

## Project Structure

```
apt_early_detection/
├── api/                  FastAPI inference service
│   └── app.py
├── dashboard/            Streamlit 4-page dashboard
│   └── app.py
├── data/                 Dataset extraction & conversion scripts
│   ├── extract_from_pcap.py
│   └── convert_iscx_arff.py
├── dataset/raw_csv/      Generated CSVs (gitignored)
├── features/             FeatureExtractor (early & full mode)
├── ml/                   Training, evaluation, SHAP explanation
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── shap_explain.py
├── models/               Saved .pkl files + plots (gitignored)
├── realtime/             Live packet-capture detector
├── tests/                60 pytest tests (API, features, models)
├── utils/                JSON structured logger
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .github/workflows/ci.yml
```

---

## Interpretation Method: SHAP

This project uses **SHAP TreeExplainer** for model interpretability:

- `models/shap_beeswarm.png` — global feature importance (all flows)
- `models/shap_waterfall_1..4.png` — per-flow explanation (why this prediction)
- `models/shap_summary.csv` — mean |SHAP| per feature (for thesis table)

---

## Requirements

- Python 3.11+ (developed on Python 3.14)
- 7-Zip (for Deflate64-compressed PCAP ZIPs): https://www.7-zip.org/
- Scapy (for PCAP extraction): `pip install scapy`

---

## License

MIT
