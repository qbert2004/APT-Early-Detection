"""
Evaluate saved models and produce thesis-ready plots.

Usage:
    python -m ml.evaluate_model
    python -m ml.evaluate_model --model models/rf_early.pkl --csv dataset/...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import RocCurveDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from features.feature_extractor import FeatureExtractor, LABEL_MAP, LABEL_MAP_INV
MODELS_DIR = ROOT / "models"


def _load_data(csv_path, use_cicflowmeter=False, n_packets=5):
    fe = FeatureExtractor(n_packets=n_packets)
    if use_cicflowmeter:
        return fe.from_cicflowmeter_csv(csv_path, mode="early")
    return fe.from_synthetic_csv(csv_path, mode="early")


def roc_multiclass(model, X, y, title: str, out_path: Path):
    classes = sorted(y.unique())
    proba   = model.predict_proba(X)
    fig, ax = plt.subplots(figsize=(7, 5))

    if len(classes) == 2:
        # Binary classification: one ROC curve using positive-class prob
        mc = list(model.classes_)
        pos = classes[1]
        col = mc.index(pos) if pos in mc else 1
        fpr, tpr, _ = roc_curve((y == pos).astype(int), proba[:, col])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{LABEL_MAP_INV.get(pos, pos)} (AUC={roc_auc:.3f})")
    else:
        y_bin = label_binarize(y, classes=classes)
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{LABEL_MAP_INV.get(cls, cls)} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[saved] {out_path}")


def n_packets_sensitivity(csv_path, clf_name="rf", use_cicflowmeter=False):
    """
    Show how F1 changes as we allow more early packets.
    Key plot for diploma — proves that even 5 packets are enough.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score

    ns = [2, 3, 5, 8, 10, 15, 20]
    f1s = []

    for n in ns:
        fe = FeatureExtractor(n_packets=n)
        if use_cicflowmeter:
            X, y = fe.from_cicflowmeter_csv(csv_path, mode="early")
        else:
            X, y = fe.from_synthetic_csv(csv_path, mode="early")

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_tr, y_tr)
        f1 = f1_score(y_te, model.predict(X_te), average="weighted", zero_division=0)
        f1s.append(f1)
        print(f"  n_packets={n:3d}  F1={f1:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ns, f1s, marker="o", color="steelblue", linewidth=2)
    ax.axhline(0.8, color="green", linestyle="--", linewidth=1.2, label="Target F1=0.80")
    ax.set_xlabel("Number of Early Packets Used")
    ax.set_ylabel("Weighted F1-score")
    ax.set_title("Classification Performance vs. Number of Early Packets")
    ax.set_xticks(ns)
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    out = MODELS_DIR / "n_packets_sensitivity.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[saved] {out}")
    return dict(zip(ns, f1s))


def main(model_path=None, csv_path=None, use_cicflowmeter=False):
    if csv_path is None:
        csv_path = ROOT / "dataset" / "raw_csv" / "synthetic_flows.csv"
        if not csv_path.exists():
            from data.generate_synthetic import generate
            generate(csv_path)

    print("\n[evaluate] N-packets sensitivity curve...")
    n_packets_sensitivity(csv_path, use_cicflowmeter=use_cicflowmeter)

    if model_path is None:
        # try to find best early model
        candidates = sorted(MODELS_DIR.glob("rf_early.pkl"))
        if not candidates:
            print("[warn] No model found. Run ml/train_model.py first.")
            return
        model_path = candidates[0]

    print(f"\n[evaluate] Loading {model_path}")
    model = joblib.load(model_path)
    X, y  = _load_data(csv_path, use_cicflowmeter=use_cicflowmeter)

    from sklearn.model_selection import train_test_split
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    roc_multiclass(
        model, X_te, y_te,
        title=f"ROC Curves — {Path(model_path).stem}",
        out_path=MODELS_DIR / f"{Path(model_path).stem}_roc.png",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--csv",   default=None)
    parser.add_argument("--cicflowmeter", action="store_true")
    args = parser.parse_args()
    main(args.model, args.csv, args.cicflowmeter)
