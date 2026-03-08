"""
APT Early Detection — FastAPI Inference API

Endpoints:
  GET  /health          liveness probe
  GET  /ready           readiness probe (checks models loaded)
  POST /predict/early   classify from early-flow features (5 packets)
  POST /predict/full    classify from full-flow features
  GET  /metrics         Prometheus-compatible plain-text metrics

Usage:
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from utils.logger import get_logger

log = get_logger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"

EARLY_FEATURES = [
    "avg_packet_size", "std_packet_size", "min_packet_size", "max_packet_size",
    "avg_interarrival", "std_interarrival", "min_interarrival", "max_interarrival",
    "incoming_ratio", "packet_count", "total_bytes",
    "flow_duration", "bytes_per_second", "pkts_per_second",
]
FULL_FEATURES = EARLY_FEATURES + [
    "fwd_packet_length_mean", "bwd_packet_length_mean",
    "fwd_iat_mean", "bwd_iat_mean",
    "flow_iat_mean", "flow_iat_std",
    "active_mean", "idle_mean",
    "subflow_fwd_packets", "subflow_bwd_packets",
]
CLASSES = ["normal", "vpn", "attack"]

# ── model registry ─────────────────────────────────────────────────────────────
_models: dict[str, object] = {}
_startup_time: float = time.time()
_request_counts: dict[str, int] = {"early": 0, "full": 0}
_error_counts:   dict[str, int] = {"early": 0, "full": 0}


def _load_models() -> None:
    for name in ("rf_early", "rf_full", "xgb_early", "xgb_full", "best_early_model"):
        path = MODELS_DIR / f"{name}.pkl"
        if path.exists():
            _models[name] = joblib.load(path)
            log.info("model loaded", model=name, path=str(path))
        else:
            log.warning("model file not found", model=name, path=str(path))


# ── FastAPI app ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models()
    log.info("API started", models=list(_models.keys()))
    yield


app = FastAPI(
    title="APT Early Detection API",
    description="Real-time network flow classifier — early vs full flow",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class EarlyFlowFeatures(BaseModel):
    avg_packet_size:  float = Field(..., ge=0, description="Mean packet size in bytes")
    std_packet_size:  float = Field(..., ge=0)
    min_packet_size:  float = Field(..., ge=0)
    max_packet_size:  float = Field(..., ge=0)
    avg_interarrival: float = Field(..., ge=0, description="Mean inter-arrival time (s)")
    std_interarrival: float = Field(..., ge=0)
    min_interarrival: float = Field(..., ge=0)
    max_interarrival: float = Field(..., ge=0)
    incoming_ratio:   float = Field(..., ge=0, le=1, description="Fraction of incoming pkts")
    packet_count:     int   = Field(..., ge=1)
    total_bytes:      float = Field(..., ge=0)
    flow_duration:    float = Field(..., ge=0, description="Flow duration in seconds")
    bytes_per_second: float = Field(..., ge=0)
    pkts_per_second:  float = Field(..., ge=0)
    model: Literal["rf", "xgb", "best"] = Field("best", description="Model variant")


class FullFlowFeatures(EarlyFlowFeatures):
    fwd_packet_length_mean: float = Field(0.0, ge=0)
    bwd_packet_length_mean: float = Field(0.0, ge=0)
    fwd_iat_mean:           float = Field(0.0, ge=0)
    bwd_iat_mean:           float = Field(0.0, ge=0)
    flow_iat_mean:          float = Field(0.0, ge=0)
    flow_iat_std:           float = Field(0.0, ge=0)
    active_mean:            float = Field(0.0, ge=0)
    idle_mean:              float = Field(0.0, ge=0)
    subflow_fwd_packets:    float = Field(0.0, ge=0)
    subflow_bwd_packets:    float = Field(0.0, ge=0)


class PredictionResponse(BaseModel):
    label:       str
    label_id:    int
    confidence:  float
    probabilities: dict[str, float]
    model_used:  str
    latency_ms:  float


# ── helpers ────────────────────────────────────────────────────────────────────

def _select_model(model_hint: str, mode: str) -> tuple[str, object]:
    if model_hint == "best":
        key = "best_early_model" if mode == "early" else "rf_full"
    else:
        key = f"{model_hint}_{mode}"
    mdl = _models.get(key)
    if mdl is None:
        raise HTTPException(status_code=503, detail=f"Model '{key}' not loaded")
    return key, mdl


def _to_df(features_dict: dict, mdl, feature_list: list[str]) -> pd.DataFrame:
    # Prefer the model's own feature list (avoids name-mismatch errors after retraining)
    cols = list(mdl.feature_names_in_) if hasattr(mdl, "feature_names_in_") else feature_list
    row = {f: features_dict.get(f, 0.0) for f in cols}
    df = pd.DataFrame([row])
    df = df.replace([float("inf"), float("-inf")], float("nan")).fillna(0.0)
    return df


def _run_inference(mdl, df: pd.DataFrame) -> tuple[int, np.ndarray]:
    proba = mdl.predict_proba(df)[0]
    label_id = int(np.argmax(proba))
    return label_id, proba


# ── endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready", tags=["ops"])
def ready() -> dict:
    if not _models:
        raise HTTPException(status_code=503, detail="No models loaded")
    return {"status": "ready", "models": list(_models.keys())}


@app.post("/predict/early", response_model=PredictionResponse, tags=["inference"])
def predict_early(body: EarlyFlowFeatures) -> PredictionResponse:
    t0 = time.perf_counter()
    _request_counts["early"] += 1
    try:
        key, mdl = _select_model(body.model, "early")
        df = _to_df(body.model_dump(), mdl, EARLY_FEATURES)
        label_id, proba = _run_inference(mdl, df)
    except HTTPException:
        _error_counts["early"] += 1
        raise
    except Exception as exc:
        _error_counts["early"] += 1
        log.error("inference failed", mode="early", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency = (time.perf_counter() - t0) * 1000
    label = CLASSES[label_id] if label_id < len(CLASSES) else "unknown"
    resp = PredictionResponse(
        label=label,
        label_id=label_id,
        confidence=float(proba[label_id]),
        probabilities={c: float(p) for c, p in zip(CLASSES, proba)},
        model_used=key,
        latency_ms=round(latency, 3),
    )
    log.info("prediction", mode="early", label=label, confidence=resp.confidence,
             latency_ms=resp.latency_ms, model=key)
    return resp


@app.post("/predict/full", response_model=PredictionResponse, tags=["inference"])
def predict_full(body: FullFlowFeatures) -> PredictionResponse:
    t0 = time.perf_counter()
    _request_counts["full"] += 1
    try:
        key, mdl = _select_model(body.model, "full")
        df = _to_df(body.model_dump(), mdl, FULL_FEATURES)
        label_id, proba = _run_inference(mdl, df)
    except HTTPException:
        _error_counts["full"] += 1
        raise
    except Exception as exc:
        _error_counts["full"] += 1
        log.error("inference failed", mode="full", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency = (time.perf_counter() - t0) * 1000
    label = CLASSES[label_id] if label_id < len(CLASSES) else "unknown"
    resp = PredictionResponse(
        label=label,
        label_id=label_id,
        confidence=float(proba[label_id]),
        probabilities={c: float(p) for c, p in zip(CLASSES, proba)},
        model_used=key,
        latency_ms=round(latency, 3),
    )
    log.info("prediction", mode="full", label=label, confidence=resp.confidence,
             latency_ms=resp.latency_ms, model=key)
    return resp


@app.get("/metrics", response_class=PlainTextResponse, tags=["ops"])
def metrics() -> str:
    uptime = time.time() - _startup_time
    lines = [
        "# HELP apt_requests_total Total inference requests",
        "# TYPE apt_requests_total counter",
        f'apt_requests_total{{mode="early"}} {_request_counts["early"]}',
        f'apt_requests_total{{mode="full"}}  {_request_counts["full"]}',
        "",
        "# HELP apt_errors_total Total inference errors",
        "# TYPE apt_errors_total counter",
        f'apt_errors_total{{mode="early"}} {_error_counts["early"]}',
        f'apt_errors_total{{mode="full"}}  {_error_counts["full"]}',
        "",
        "# HELP apt_models_loaded Number of loaded models",
        "# TYPE apt_models_loaded gauge",
        f"apt_models_loaded {len(_models)}",
        "",
        "# HELP apt_uptime_seconds API uptime in seconds",
        "# TYPE apt_uptime_seconds gauge",
        f"apt_uptime_seconds {uptime:.1f}",
    ]
    return "\n".join(lines)
