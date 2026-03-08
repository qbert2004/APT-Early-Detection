"""
Shared pytest fixtures for APT Early Detection test suite.

Scope strategy:
  - session  → heavy resources loaded once (FeatureExtractor, test CSV)
  - module   → FastAPI TestClient (triggers lifespan model loading once per module)
  - function → fresh mutable copies of sample dicts

Usage (automatic — pytest collects conftest.py automatically):
    pytest tests/ -v
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent


# ── synthetic CSV fixture ────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def synthetic_csv_path(tmp_path_factory) -> Path:
    """
    Create a minimal synthetic CSV (200 rows, balanced) for tests that need a
    real file on disk without depending on the full 115K dataset.
    """
    rng = np.random.default_rng(42)
    n   = 200

    df = pd.DataFrame({
        "avg_packet_size":  rng.uniform(40, 1500, n),
        "std_packet_size":  rng.uniform(0, 500,  n),
        "min_packet_size":  rng.uniform(40, 200,  n),
        "max_packet_size":  rng.uniform(200, 1500, n),
        "avg_interarrival": rng.uniform(0, 1,     n),
        "std_interarrival": rng.uniform(0, 0.5,   n),
        "min_interarrival": rng.uniform(0, 0.1,   n),
        "max_interarrival": rng.uniform(0.1, 2,   n),
        "incoming_ratio":   rng.uniform(0, 1,     n),
        "packet_count":     rng.integers(2, 50,   n).astype(float),
        "total_bytes":      rng.uniform(100, 50000, n),
        "flow_duration":    rng.uniform(0.001, 60, n),
        "bytes_per_second": rng.uniform(100, 1e6,  n),
        "pkts_per_second":  rng.uniform(0.5, 1000, n),
        "label":            (["normal"] * 100 + ["vpn"] * 100),
    })
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    p = tmp_path_factory.mktemp("data") / "test_flows.csv"
    df.to_csv(p, index_label="flow_id")
    return p


# ── FeatureExtractor fixture ─────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def feature_extractor():
    """Session-scoped FeatureExtractor with default n_packets=5."""
    from features.feature_extractor import FeatureExtractor
    return FeatureExtractor(n_packets=5)


@pytest.fixture(scope="session")
def extracted_XY(feature_extractor, synthetic_csv_path):
    """(X, y) already extracted — shared across the session."""
    return feature_extractor.from_synthetic_csv(synthetic_csv_path, mode="early")


# ── sample payload fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def early_payload() -> dict:
    """Valid EarlyFlowFeatures payload dict (function-scoped → safe to mutate)."""
    return {
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


@pytest.fixture
def full_payload(early_payload) -> dict:
    """Valid FullFlowFeatures payload dict."""
    return {
        **early_payload,
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
