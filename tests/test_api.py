"""Integration tests for FastAPI inference endpoints."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.app import app, CLASSES


@pytest.fixture(scope="module")
def client():
    """TestClient that triggers lifespan (loads models)."""
    with TestClient(app) as c:
        yield c


# ── sample payloads ────────────────────────────────────────────────────────────

EARLY_PAYLOAD = {
    "avg_packet_size":  200.0,
    "std_packet_size":  50.0,
    "min_packet_size":  100.0,
    "max_packet_size":  300.0,
    "avg_interarrival": 0.01,
    "std_interarrival": 0.002,
    "min_interarrival": 0.005,
    "max_interarrival": 0.02,
    "incoming_ratio":   0.3,
    "packet_count":     5,
    "total_bytes":      1000.0,
    "flow_duration":    0.05,
    "bytes_per_second": 20000.0,
    "pkts_per_second":  100.0,
    "model":            "rf",
}

FULL_PAYLOAD = {
    **EARLY_PAYLOAD,
    "fwd_packet_length_mean": 150.0,
    "bwd_packet_length_mean": 250.0,
    "fwd_iat_mean":           0.01,
    "bwd_iat_mean":           0.01,
    "flow_iat_mean":          0.01,
    "flow_iat_std":           0.002,
    "active_mean":            0.05,
    "idle_mean":              0.1,
    "subflow_fwd_packets":    3.0,
    "subflow_bwd_packets":    2.0,
    "model":                  "rf",
}


# ── /health ────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_body(self, client):
        r = client.get("/health")
        assert r.json() == {"status": "ok"}


# ── /ready ─────────────────────────────────────────────────────────────────────

class TestReady:
    def test_returns_200_when_models_loaded(self, client):
        r = client.get("/ready")
        assert r.status_code == 200

    def test_lists_models(self, client):
        r = client.get("/ready")
        data = r.json()
        assert "models" in data
        assert len(data["models"]) > 0


# ── /predict/early ─────────────────────────────────────────────────────────────

class TestPredictEarly:
    def test_returns_200(self, client):
        r = client.post("/predict/early", json=EARLY_PAYLOAD)
        assert r.status_code == 200

    def test_response_schema(self, client):
        r = client.post("/predict/early", json=EARLY_PAYLOAD)
        data = r.json()
        assert "label" in data
        assert "label_id" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "model_used" in data
        assert "latency_ms" in data

    def test_label_is_valid_class(self, client):
        r = client.post("/predict/early", json=EARLY_PAYLOAD)
        assert r.json()["label"] in CLASSES

    def test_confidence_in_range(self, client):
        r = client.post("/predict/early", json=EARLY_PAYLOAD)
        conf = r.json()["confidence"]
        assert 0.0 <= conf <= 1.0

    def test_probabilities_sum_to_one(self, client):
        r = client.post("/predict/early", json=EARLY_PAYLOAD)
        probs = r.json()["probabilities"]
        assert abs(sum(probs.values()) - 1.0) < 1e-5

    def test_latency_is_positive(self, client):
        r = client.post("/predict/early", json=EARLY_PAYLOAD)
        assert r.json()["latency_ms"] > 0

    def test_invalid_incoming_ratio_rejected(self, client):
        bad = {**EARLY_PAYLOAD, "incoming_ratio": 1.5}
        r = client.post("/predict/early", json=bad)
        assert r.status_code == 422

    def test_negative_packet_count_rejected(self, client):
        bad = {**EARLY_PAYLOAD, "packet_count": 0}
        r = client.post("/predict/early", json=bad)
        assert r.status_code == 422

    def test_xgb_model_variant(self, client):
        payload = {**EARLY_PAYLOAD, "model": "xgb"}
        r = client.post("/predict/early", json=payload)
        assert r.status_code == 200
        assert "xgb" in r.json()["model_used"]

    def test_best_model_variant(self, client):
        payload = {**EARLY_PAYLOAD, "model": "best"}
        r = client.post("/predict/early", json=payload)
        assert r.status_code == 200


# ── /predict/full ──────────────────────────────────────────────────────────────

class TestPredictFull:
    def test_returns_200(self, client):
        r = client.post("/predict/full", json=FULL_PAYLOAD)
        assert r.status_code == 200

    def test_label_is_valid_class(self, client):
        r = client.post("/predict/full", json=FULL_PAYLOAD)
        assert r.json()["label"] in CLASSES

    def test_rf_full_model(self, client):
        r = client.post("/predict/full", json=FULL_PAYLOAD)
        assert "rf_full" in r.json()["model_used"]


# ── /metrics ───────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_returns_200(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200

    def test_content_type_text(self, client):
        r = client.get("/metrics")
        assert "text/plain" in r.headers["content-type"]

    def test_contains_prometheus_labels(self, client):
        r = client.get("/metrics")
        text = r.text
        assert "apt_requests_total" in text
        assert "apt_models_loaded" in text
        assert "apt_uptime_seconds" in text
