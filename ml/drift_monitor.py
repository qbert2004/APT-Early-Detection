"""
Model Drift Monitor — Population Stability Index (PSI)
=======================================================
Compares the distribution of incoming live traffic features
against the training-set distribution.

PSI interpretation (industry standard):
  PSI < 0.10  → No significant shift (OK)
  PSI < 0.20  → Moderate shift    (MONITOR)
  PSI >= 0.20 → Major drift       (RETRAIN recommended)

Usage:
    # build reference from training data (run once after training)
    python -X utf8 -m ml.drift_monitor --build --csv dataset/raw_csv/iscx_combined.csv

    # check live alerts.log against reference
    python -X utf8 -m ml.drift_monitor --check
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)

ROOT       = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
REF_FILE   = MODELS_DIR / "drift_reference.json"
REPORT_FILE= MODELS_DIR / "drift_report.json"
ALERT_LOG  = MODELS_DIR / "alerts.log"

FEATURES = [
    "avg_packet_size", "std_packet_size", "min_packet_size", "max_packet_size",
    "avg_interarrival", "std_interarrival", "min_interarrival", "max_interarrival",
    "incoming_ratio", "packet_count", "total_bytes",
    "flow_duration", "bytes_per_second", "pkts_per_second",
]

N_BINS = 10   # number of PSI buckets


# ── PSI core ───────────────────────────────────────────────────────────────────

def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = N_BINS) -> float:
    """Compute Population Stability Index between two 1-D arrays."""
    eps = 1e-8
    # build bin edges from expected (reference) distribution
    _, edges = np.histogram(expected, bins=bins)
    edges[0]  -= 1e-9
    edges[-1] += 1e-9

    e_hist, _ = np.histogram(expected, bins=edges)
    a_hist, _ = np.histogram(actual,   bins=edges)

    e_frac = (e_hist / len(expected)).clip(min=eps)
    a_frac = (a_hist / max(len(actual), 1)).clip(min=eps)

    psi = float(np.sum((a_frac - e_frac) * np.log(a_frac / e_frac)))
    return round(psi, 6)


def _psi_label(psi: float) -> str:
    if psi < 0.10:
        return "OK"
    if psi < 0.20:
        return "MONITOR"
    return "RETRAIN"


# ── Build reference ────────────────────────────────────────────────────────────

def build_reference(csv_path: str | Path) -> dict:
    """Compute per-feature statistics from the training CSV."""
    df = pd.read_csv(csv_path)

    available = [f for f in FEATURES if f in df.columns]
    if not available:
        raise ValueError(f"None of the expected features found in {csv_path}")

    ref: dict = {"built_at": time.strftime("%Y-%m-%d %H:%M:%S"), "features": {}}
    for feat in available:
        col = df[feat].replace([np.inf, -np.inf], np.nan).dropna().values
        ref["features"][feat] = {
            "mean":   float(col.mean()),
            "std":    float(col.std()),
            "p05":    float(np.percentile(col, 5)),
            "p95":    float(np.percentile(col, 95)),
            "sample": col.tolist()[:2000],   # store up to 2K values for PSI bins
        }

    MODELS_DIR.mkdir(exist_ok=True)
    REF_FILE.write_text(json.dumps(ref, indent=2), encoding="utf-8")
    log.info("drift reference built", features=len(available), path=str(REF_FILE))
    print(f"[drift] Reference saved → {REF_FILE}  ({len(available)} features)")
    return ref


# ── Parse alerts.log for live features ────────────────────────────────────────

def _parse_alerts_to_df(max_rows: int = 2000) -> pd.DataFrame:
    """
    Re-parse alerts.log to extract numeric features for drift calculation.
    We reconstruct from bytes and pkts fields as a minimal proxy.
    """
    if not ALERT_LOG.exists():
        return pd.DataFrame()

    rows = []
    for line in ALERT_LOG.read_text(encoding="utf-8", errors="ignore").splitlines()[-max_rows:]:
        if "ALERT" in line or not line.strip():
            continue
        try:
            parts = line.split()
            # find key=value pairs
            kv = {}
            for p in parts:
                if "=" in p and not p.startswith("["):
                    k, v = p.split("=", 1)
                    try:
                        kv[k] = float(v.split("/")[0])   # handle "87/100"
                    except ValueError:
                        pass
            if "pkts" in kv and "bytes" in kv:
                rows.append(kv)
        except Exception:
            continue
    return pd.DataFrame(rows)


# ── Check drift ────────────────────────────────────────────────────────────────

def check_drift(ref: dict | None = None) -> dict:
    """Compare live traffic (from alerts.log) against reference distribution."""
    if ref is None:
        if not REF_FILE.exists():
            raise FileNotFoundError(
                "No reference file found. Run first:\n"
                "  python -m ml.drift_monitor --build --csv dataset/raw_csv/iscx_combined.csv"
            )
        ref = json.loads(REF_FILE.read_text(encoding="utf-8"))

    live_df = _parse_alerts_to_df()
    if live_df.empty:
        print("[drift] No live data yet — run the detector first.")
        return {}

    report: dict = {
        "checked_at":  time.strftime("%Y-%m-%d %H:%M:%S"),
        "live_samples": len(live_df),
        "features":     {},
        "overall_psi":  0.0,
        "overall_status": "OK",
    }

    psi_values = []
    for feat, stats in ref["features"].items():
        if feat not in live_df.columns:
            continue
        live_col = live_df[feat].replace([np.inf, -np.inf], np.nan).dropna().values
        if len(live_col) < 5:
            continue

        ref_arr = np.array(stats["sample"])
        psi_val = _psi(ref_arr, live_col)
        status  = _psi_label(psi_val)

        report["features"][feat] = {
            "psi":    psi_val,
            "status": status,
            "live_mean":   round(float(live_col.mean()), 4),
            "train_mean":  round(stats["mean"], 4),
        }
        psi_values.append(psi_val)

    if psi_values:
        overall = round(float(np.mean(psi_values)), 6)
        report["overall_psi"]    = overall
        report["overall_status"] = _psi_label(overall)

    MODELS_DIR.mkdir(exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")

    status = report["overall_status"]
    emoji  = {"OK": "✅", "MONITOR": "⚠️", "RETRAIN": "🔴"}.get(status, "")
    print(f"[drift] Overall PSI={report['overall_psi']:.4f}  Status={status} {emoji}")
    if status == "RETRAIN":
        log.warning("model drift detected — consider retraining",
                    psi=report["overall_psi"])
    else:
        log.info("drift check complete", psi=report["overall_psi"], status=status)

    return report


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Drift Monitor (PSI)")
    parser.add_argument("--build", action="store_true",
                        help="Build reference from training CSV")
    parser.add_argument("--check", action="store_true",
                        help="Check live drift against reference")
    parser.add_argument("--csv",   default="dataset/raw_csv/iscx_combined.csv",
                        help="Path to training CSV (used with --build)")
    args = parser.parse_args()

    if args.build:
        build_reference(ROOT / args.csv)
    if args.check:
        check_drift()
    if not args.build and not args.check:
        parser.print_help()
