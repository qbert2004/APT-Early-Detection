"""
ML Training Pipeline — Early vs Full Flow Comparison

Trains two sets of models:
  A) full-flow  model (baseline)
  B) early-flow model (thesis contribution)

Classifiers:
  - RandomForest
  - XGBoost

Outputs per model:
  models/<name>.pkl           serialised model
  models/<name>_metrics.json  accuracy / precision / recall / f1 / roc_auc
  models/<name>_cm.png        confusion matrix
  models/<name>_importance.png feature importance

Usage:
    python -m ml.train_model                        # synthetic data
    python -m ml.train_model --csv path/to/file.csv # CICFlowMeter CSV
    python -m ml.train_model --n 10                 # early = first 10 packets
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

# structured logging (set up before any module imports that might print)
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import get_logger
log = get_logger(__name__)

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[warn] xgboost not installed — only RandomForest will be trained")

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from features.feature_extractor import FeatureExtractor, LABEL_MAP_INV

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ── helpers ────────────────────────────────────────────────────────────────────

def _multiclass_roc_auc(model, X_test, y_test) -> float:
    try:
        proba  = model.predict_proba(X_test)
        classes = sorted(np.unique(y_test))
        if len(classes) < 2:
            return 0.0
        if len(classes) == 2:
            # Binary: use positive-class column matching model.classes_
            mc = list(model.classes_)
            pos_idx = mc.index(classes[1]) if classes[1] in mc else 1
            return float(roc_auc_score(y_test, proba[:, pos_idx]))
        return float(roc_auc_score(
            y_test, proba,
            multi_class="ovr",
            average="weighted",
            labels=classes,
        ))
    except Exception:
        return 0.0


def _evaluate(name: str, model, X_test: pd.DataFrame, y_test: pd.Series, label_names: list[str]):
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "roc_auc":   round(_multiclass_roc_auc(model, X_test, y_test), 4),
    }

    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v}")
    print()
    print(classification_report(y_test, y_pred, target_names=label_names, zero_division=0))

    # confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=label_names,
        cmap="Blues",
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {name}")
    fig.tight_layout()
    fig.savefig(MODELS_DIR / f"{name}_cm.png", dpi=120)
    plt.close(fig)

    # feature importance
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=X_test.columns)
        importances = importances.sort_values(ascending=True).tail(15)
        fig, ax = plt.subplots(figsize=(7, 5))
        importances.plot.barh(ax=ax, color="steelblue")
        ax.set_xlabel("Importance")
        ax.set_title(f"Feature Importances — {name}")
        fig.tight_layout()
        fig.savefig(MODELS_DIR / f"{name}_importance.png", dpi=120)
        plt.close(fig)

    return metrics


def _save(name: str, model, metrics: dict):
    model_path = MODELS_DIR / f"{name}.pkl"
    joblib.dump(model, model_path)
    metrics_path = MODELS_DIR / f"{name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"[saved] {model_path}  |  {metrics_path}")
    log.info("model saved", model=name, f1=metrics.get("f1"), roc_auc=metrics.get("roc_auc"),
             path=str(model_path))


# ── cross-validation report ────────────────────────────────────────────────────

def _cv_report(name: str, model, X: pd.DataFrame, y: pd.Series):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1)
    print(f"  [CV] {name}  F1 = {scores.mean():.4f} ± {scores.std():.4f}")


# ── main ────────────────────────────────────────────────────────────────────────

def train(
    csv_path: str | Path | None = None,
    n_packets: int = 5,
    use_cicflowmeter: bool = False,
):
    fe = FeatureExtractor(n_packets=n_packets)

    if csv_path is None:
        # generate synthetic data
        from data.generate_synthetic import generate
        csv_path = ROOT / "dataset" / "raw_csv" / "synthetic_flows.csv"
        generate(csv_path)
        X_full, y = fe.from_synthetic_csv(csv_path, mode="full")
        X_early,_ = fe.from_synthetic_csv(csv_path, mode="early")
    elif use_cicflowmeter:
        X_full,  y = fe.from_cicflowmeter_csv(csv_path, mode="full")
        X_early, _ = fe.from_cicflowmeter_csv(csv_path, mode="early")
    else:
        X_full,  y = fe.from_synthetic_csv(csv_path, mode="full")
        X_early, _ = fe.from_synthetic_csv(csv_path, mode="early")

    label_names = [LABEL_MAP_INV[i] for i in sorted(y.unique())]

    # ── train / test split
    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(idx, test_size=0.30, stratify=y, random_state=42)

    datasets = {
        "early": (X_early.iloc[tr_idx], X_early.iloc[te_idx]),
        "full":  (X_full.iloc[tr_idx],  X_full.iloc[te_idx]),
    }
    y_train, y_test = y.iloc[tr_idx], y.iloc[te_idx]

    all_metrics: dict[str, dict] = {}

    # Memory-safe settings: depth-limited trees, single-threaded RF
    n_samples = len(y_train)
    large     = n_samples > 20_000
    n_trees   = 50  if large else 150
    max_depth = 8   if large else None

    classifiers = {
        "rf": RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=max_depth,
            max_leaf_nodes=500 if large else None,
            max_samples=0.7   if large else None,
            class_weight="balanced",   # handles class imbalance
            random_state=42,
            n_jobs=1,
        )
    }
    if HAS_XGB:
        # compute scale_pos_weight for binary imbalance
        counts = y_train.value_counts()
        if len(counts) == 2:
            majority, minority = counts.iloc[0], counts.iloc[1]
            spw = float(majority) / float(minority)
        else:
            spw = 1.0
        classifiers["xgb"] = XGBClassifier(
            n_estimators=n_trees,
            max_depth=4,
            learning_rate=0.1,
            scale_pos_weight=spw,      # handles class imbalance for XGBoost
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            tree_method="hist",    # memory-efficient histogram method
            device="cpu",
        )

    print("\n" + "="*60)
    print("  CROSS-VALIDATION (5-fold, F1-weighted)")
    print("="*60)

    for flow_mode, (X_tr, _) in datasets.items():
        for clf_name, clf in classifiers.items():
            tag = f"{clf_name}_{flow_mode}"
            _cv_report(tag, clf, X_tr, y_train)

    print("\n" + "="*60)
    print("  HELD-OUT TEST SET EVALUATION")
    print("="*60)

    best_early_f1 = 0.0
    best_early_model = None

    for flow_mode, (X_tr, X_te) in datasets.items():
        for clf_name, clf in classifiers.items():
            tag = f"{clf_name}_{flow_mode}"
            clf.fit(X_tr, y_train)
            m = _evaluate(tag, clf, X_te, y_test, label_names)
            _save(tag, clf, m)
            all_metrics[tag] = m

            if flow_mode == "early" and m["f1"] > best_early_f1:
                best_early_f1 = m["f1"]
                best_early_model = clf
                joblib.dump(clf, MODELS_DIR / "best_early_model.pkl")

    # ── summary comparison table
    print("\n" + "="*60)
    print("  SUMMARY — Early vs Full Flow")
    print("="*60)
    summary_rows = []
    for tag, m in all_metrics.items():
        parts = tag.split("_")
        clf_name  = parts[0]
        flow_mode = "_".join(parts[1:])
        summary_rows.append({
            "classifier": clf_name,
            "flow_mode":  flow_mode,
            **m,
        })
    summary = pd.DataFrame(summary_rows).set_index(["classifier", "flow_mode"])
    print(summary.to_string())

    # save summary
    summary.reset_index().to_csv(MODELS_DIR / "summary.csv", index=False)
    print(f"\n[saved] {MODELS_DIR / 'summary.csv'}")

    # ── delta plot: full vs early
    _plot_delta(all_metrics, classifiers.keys())

    print(f"\n[done] Best early-flow model: F1 = {best_early_f1:.4f}")
    log.info("training complete", best_early_f1=round(best_early_f1, 4),
             n_models=len(all_metrics), n_samples=int(len(y)))
    return all_metrics


def _plot_delta(all_metrics: dict, clf_names):
    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    fig, axes = plt.subplots(1, len(clf_names), figsize=(7 * len(clf_names), 5), squeeze=False)

    for ax, clf in zip(axes[0], clf_names):
        early_vals = [all_metrics.get(f"{clf}_early", {}).get(m, 0) for m in metrics_to_plot]
        full_vals  = [all_metrics.get(f"{clf}_full",  {}).get(m, 0) for m in metrics_to_plot]

        x = np.arange(len(metrics_to_plot))
        w = 0.35
        ax.bar(x - w/2, full_vals,  w, label="Full flow",  color="steelblue",  alpha=0.85)
        ax.bar(x + w/2, early_vals, w, label=f"Early (first N pkts)", color="tomato", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_to_plot, rotation=15, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title(f"{clf.upper()} — Full vs Early Flow")
        ax.legend()
        ax.axhline(0.8, color="green", linestyle="--", linewidth=1, label="Target F1=0.8")

    fig.tight_layout()
    fig.savefig(MODELS_DIR / "comparison_full_vs_early.png", dpi=130)
    plt.close(fig)
    print(f"[saved] {MODELS_DIR / 'comparison_full_vs_early.png'}")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train APT early-detection models")
    parser.add_argument("--csv",  default=None,  help="Path to dataset CSV")
    parser.add_argument("--n",    default=5,    type=int, help="Early packet count")
    parser.add_argument("--cicflowmeter", action="store_true",
                        help="Treat CSV as CICFlowMeter output")
    args = parser.parse_args()

    train(
        csv_path=args.csv,
        n_packets=args.n,
        use_cicflowmeter=args.cicflowmeter,
    )
