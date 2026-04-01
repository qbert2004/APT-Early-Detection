"""Smoke tests: verify all serialised models load and produce valid predictions."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import joblib

from features.feature_extractor import EARLY_FEATURES, FULL_EXTRA_FEATURES

MODELS_DIR = Path(__file__).parent.parent / "models"
N_EARLY = len(EARLY_FEATURES)                    # 14
N_FULL  = N_EARLY + len(FULL_EXTRA_FEATURES)     # 24

EARLY_MODELS = ["rf_early", "xgb_early", "best_early_model"]
FULL_MODELS  = ["rf_full", "xgb_full"]


def _sample_early() -> pd.DataFrame:
    """One-row DataFrame with all 14 early-flow features (proper column names)."""
    values = [
        200.0, 50.0, 100.0, 300.0,        # avg/std/min/max packet size
        0.01, 0.002, 0.005, 0.02,         # avg/std/min/max interarrival
        0.3,                               # incoming_ratio
        5, 1000.0, 0.05, 20000.0, 100.0,  # packet_count, total_bytes, duration, bps, pps
    ]
    return pd.DataFrame([dict(zip(EARLY_FEATURES, values))])


def _sample_full() -> pd.DataFrame:
    """One-row DataFrame with all 24 features (early + full-flow extras)."""
    row = _sample_early()
    extra = {
        "fwd_packet_length_mean": 120.0,
        "bwd_packet_length_mean": 80.0,
        "fwd_iat_mean":           0.010,
        "bwd_iat_mean":           0.012,
        "flow_iat_mean":          0.010,
        "flow_iat_std":           0.002,
        "active_mean":            0.033,
        "idle_mean":              0.015,
        "subflow_fwd_packets":    3.0,
        "subflow_bwd_packets":    2.0,
    }
    for col, val in extra.items():
        row[col] = val
    return row[EARLY_FEATURES + FULL_EXTRA_FEATURES]


@pytest.mark.parametrize("name", EARLY_MODELS)
class TestEarlyModels:
    def test_model_file_exists(self, name):
        assert (MODELS_DIR / f"{name}.pkl").exists(), f"{name}.pkl not found"

    def test_model_loads(self, name):
        mdl = joblib.load(MODELS_DIR / f"{name}.pkl")
        assert mdl is not None

    def test_n_classes(self, name):
        mdl = joblib.load(MODELS_DIR / f"{name}.pkl")
        assert mdl.n_classes_ == 2  # ISCX dataset: normal vs vpn (binary)

    def test_n_features(self, name):
        mdl = joblib.load(MODELS_DIR / f"{name}.pkl")
        assert mdl.n_features_in_ == N_EARLY

    def test_predict_returns_valid_class(self, name):
        mdl = joblib.load(MODELS_DIR / f"{name}.pkl")
        pred = mdl.predict(_sample_early())
        assert pred[0] in {0, 1, 2}

    def test_predict_proba_sums_to_one(self, name):
        mdl = joblib.load(MODELS_DIR / f"{name}.pkl")
        proba = mdl.predict_proba(_sample_early())
        assert abs(proba[0].sum() - 1.0) < 1e-5

    def test_proba_in_range(self, name):
        mdl = joblib.load(MODELS_DIR / f"{name}.pkl")
        proba = mdl.predict_proba(_sample_early())
        assert (proba >= 0).all() and (proba <= 1).all()


@pytest.mark.parametrize("name", FULL_MODELS)
class TestFullModels:
    def test_model_file_exists(self, name):
        assert (MODELS_DIR / f"{name}.pkl").exists(), f"{name}.pkl not found"

    def test_model_loads(self, name):
        mdl = joblib.load(MODELS_DIR / f"{name}.pkl")
        assert mdl is not None

    def test_n_classes(self, name):
        mdl = joblib.load(MODELS_DIR / f"{name}.pkl")
        assert mdl.n_classes_ == 2

    def test_n_features(self, name):
        mdl = joblib.load(MODELS_DIR / f"{name}.pkl")
        assert mdl.n_features_in_ == N_FULL, (
            f"{name}: expected {N_FULL} features, got {mdl.n_features_in_}. "
            "Run: python -m ml.train_model --csv dataset/raw_csv/iscx_combined.csv"
        )

    def test_predict_returns_valid_class(self, name):
        mdl = joblib.load(MODELS_DIR / f"{name}.pkl")
        pred = mdl.predict(_sample_full())
        assert pred[0] in {0, 1, 2}
