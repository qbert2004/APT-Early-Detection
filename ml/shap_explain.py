"""
SHAP Explanations — APT Early Detection
========================================
Generates per-prediction SHAP explanations for the best early-flow model.

Produces:
  models/shap_beeswarm.png     — global feature impact (all test samples)
  models/shap_waterfall_N.png  — per-sample waterfall for N examples
  models/shap_summary.csv      — mean |SHAP| per feature

Usage:
    python -m ml.shap_explain
    python -m ml.shap_explain --csv dataset/raw_csv/iscx_pcap_early.csv --samples 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "models"

try:
    import shap
except ImportError:
    raise ImportError("pip install shap")

from features.feature_extractor import FeatureExtractor, LABEL_MAP_INV
from sklearn.model_selection import train_test_split

FEATURE_LABELS = {
    "avg_packet_size":  "Avg Packet Size",
    "std_packet_size":  "Std Packet Size",
    "min_packet_size":  "Min Packet Size",
    "max_packet_size":  "Max Packet Size",
    "avg_interarrival": "Avg Inter-Arrival",
    "std_interarrival": "Std Inter-Arrival",
    "min_interarrival": "Min Inter-Arrival",
    "max_interarrival": "Max Inter-Arrival",
    "incoming_ratio":   "Incoming Pkt Ratio",
    "packet_count":     "Packet Count",
    "total_bytes":      "Total Bytes",
    "flow_duration":    "Flow Duration",
    "bytes_per_second": "Bytes / Second",
    "pkts_per_second":  "Pkts / Second",
}


def _load_test_data(csv_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Returns (X_te_orig, X_te_renamed, y_te)."""
    fe = FeatureExtractor(n_packets=5)
    X, y = fe.from_synthetic_csv(csv_path, mode="early")
    _, X_te, _, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_te_renamed = X_te.rename(columns=FEATURE_LABELS)
    return X_te, X_te_renamed, y_te


def run(csv_path: str | Path, n_waterfall_samples: int = 6):
    print("[shap_explain] Loading model …")
    model = joblib.load(MODELS_DIR / "rf_early.pkl")

    print("[shap_explain] Loading test data …")
    X_te, X_te_r, y_te = _load_test_data(csv_path)

    print("[shap_explain] Building TreeExplainer …")
    # Use a background sample for speed (200 rows) — use ORIGINAL column names
    bg = shap.sample(X_te, 200, random_state=42)
    explainer = shap.TreeExplainer(model, bg)

    # Compute SHAP values on the full test set (renamed cols for display)
    # check_additivity=False: avoids spurious ExplainerError on large datasets
    # caused by floating-point rounding in the tree-path summation.
    print("[shap_explain] Computing SHAP values …")
    shap_values = explainer(X_te_r, check_additivity=False)

    # ── 1. Beeswarm plot (global importance, all samples) ──────────────────
    print("[shap_explain] Plotting beeswarm …")
    fig, ax = plt.subplots(figsize=(10, 6))
    # For binary classification shap_values has shape (n, features, 2)
    # Use class-1 (vpn) slice or the raw shap_values object
    if hasattr(shap_values, "values") and shap_values.values.ndim == 3:
        sv_class1 = shap.Explanation(
            values=shap_values.values[:, :, 1],
            base_values=shap_values.base_values[:, 1],
            data=shap_values.data,
            feature_names=shap_values.feature_names,
        )
    else:
        sv_class1 = shap_values

    shap.plots.beeswarm(sv_class1, max_display=14, show=False)
    plt.title("SHAP Beeswarm — Feature Impact on VPN Classification\n"
              "(positive = pushes toward VPN, negative = pushes toward Normal)",
              fontsize=11, pad=10)
    plt.tight_layout()
    out_bee = MODELS_DIR / "shap_beeswarm.png"
    plt.savefig(out_bee, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_bee}")

    # ── 2. Waterfall plots (individual predictions) ────────────────────────
    # Pick n_waterfall_samples: mix of correct/incorrect and both classes
    y_pred = model.predict(X_te)      # use original column names for sklearn
    y_te_arr = y_te.values if hasattr(y_te, "values") else np.array(y_te)

    # Sample indices: some correct, some wrong, both classes
    correct_mask   = y_pred == y_te_arr
    vpn_mask       = y_te_arr == 1
    normal_mask    = y_te_arr == 0

    candidates = []
    for mask in [vpn_mask & correct_mask,
                 normal_mask & correct_mask,
                 vpn_mask & ~correct_mask,
                 normal_mask & ~correct_mask]:
        idx = np.where(mask)[0]
        if len(idx):
            candidates.extend(idx[:max(1, n_waterfall_samples // 4)].tolist())
    candidates = candidates[:n_waterfall_samples]

    label_name = {0: "Normal", 1: "VPN"}

    for rank, i in enumerate(candidates):
        sv_i = shap.Explanation(
            values=sv_class1.values[i],
            base_values=sv_class1.base_values[i],
            data=sv_class1.data[i],
            feature_names=sv_class1.feature_names,
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        shap.plots.waterfall(sv_i, max_display=10, show=False)
        true_lbl = label_name.get(int(y_te_arr[i]), "?")
        pred_lbl = label_name.get(int(y_pred[i]), "?")
        verdict  = "✓ correct" if true_lbl == pred_lbl else "✗ wrong"
        plt.title(
            f"SHAP Waterfall — Sample #{i}   "
            f"True: {true_lbl}   Pred: {pred_lbl}   {verdict}",
            fontsize=10, pad=8,
        )
        plt.tight_layout()
        out_wf = MODELS_DIR / f"shap_waterfall_{rank+1}.png"
        plt.savefig(out_wf, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"[saved] {out_wf}  (true={true_lbl}, pred={pred_lbl})")

    # ── 3. Mean |SHAP| summary CSV ─────────────────────────────────────────
    mean_abs = np.abs(sv_class1.values).mean(axis=0)
    summary_df = pd.DataFrame({
        "feature":   sv_class1.feature_names,
        "mean_|shap|": mean_abs,
    }).sort_values("mean_|shap|", ascending=False)
    out_csv = MODELS_DIR / "shap_summary.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")

    print("\nTop features by mean |SHAP|:")
    for _, row in summary_df.head(8).iterrows():
        bar = "█" * int(row["mean_|shap|"] / mean_abs.max() * 20)
        print(f"  {row['feature']:25s}  {row['mean_|shap|']:.5f}  {bar}")

    print("\n[done] SHAP explanations complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="dataset/raw_csv/iscx_pcap_early.csv")
    parser.add_argument("--samples", type=int, default=6,
                        help="Number of waterfall plots to generate")
    args = parser.parse_args()
    run(args.csv, args.samples)
