"""Smoke tests: verify all serialised models load and produce valid predictions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import joblib

MODELS_DIR = Path(__file__).parent.parent / "models"
N_EARLY = 14
N_FULL  = 24  # 14 early + 10 extra

EARLY_MODELS = ["rf_early", "xgb_early", "best_early_model"]
FULL_MODELS  = ["rf_full", "xgb_full"]


def _sample_early() -> np.ndarray:
    return np.array([[
        200.0, 50.0, 100.0, 300.0,   # packet size stats
        0.01, 0.002, 0.005, 0.02,    # interarrival stats
        0.3,                          # incoming_ratio
        5, 1000.0, 0.05, 20000.0, 100.0,  # count, bytes, duration, bps, pps
    ]])


def _sample_full() -> np.ndarray:
    extra = np.zeros((1, N_FULL - N_EARLY))
    return np.hstack([_sample_early(), extra])


@pytest.mark.parametrize("name", EARLY_MODELS)
class TestEarlyModels:
    def test_model_file_exists(self, name):
        assert (MODELS_DIR / f"{name}.pkl").exists(), f"{name}.pkl not found"

    def test_model_loads(self, name):
        mdl = joblib.load(MODELS_DIR / f"{name}.pkl")
        assert mdl is not None

    def test_n_classes(self, name):
        mdl = joblib.load(MODELS_DIR / f"{name}.pkl")
        assert mdl.n_classes_ == 2  # ISCX dataset: normal vs vpn

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

    def test_predict_returns_valid_class(self, name):
        mdl = joblib.load(MODELS_DIR / f"{name}.pkl")
        X = np.zeros((1, mdl.n_features_in_))
        X[0, :N_EARLY] = _sample_early()[0]
        pred = mdl.predict(X)
        assert pred[0] in {0, 1, 2}
