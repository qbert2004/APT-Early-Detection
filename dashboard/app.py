"""
Streamlit Dashboard — APT Early Detection System

Shows:
  - Live flow classification feed (from alert log or live detector)
  - Traffic distribution pie/bar chart
  - Attack probability timeline
  - Model metrics comparison (full vs early)
  - N-packet sensitivity curve

Run:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

MODELS_DIR  = ROOT / "models"
ALERT_LOG   = MODELS_DIR / "alerts.log"
SUMMARY_CSV = MODELS_DIR / "summary.csv"

st.set_page_config(
    page_title="APT Early Detection",
    page_icon=":shield:",
    layout="wide",
)

# ── sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("APT Early Detection")
    st.caption("Diploma project — Early Flow Classification")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Live Monitor", "Model Metrics", "SHAP Explanations", "Dataset Explorer"],
        index=0,
    )
    st.divider()
    refresh_interval = st.slider("Auto-refresh (sec)", 1, 10, 2)
    st.caption("Dashboard refreshes automatically while Live Monitor is open.")


# ── helpers ────────────────────────────────────────────────────────────────────

LABEL_COLOR = {
    "NORMAL":  "#2ecc71",
    "VPN":     "#f39c12",
    "ATTACK":  "#e74c3c",
    "UNKNOWN": "#95a5a6",
}

def _parse_alert_log(max_rows: int = 500) -> pd.DataFrame:
    if not ALERT_LOG.exists():
        return pd.DataFrame(columns=["ts", "label", "prob", "src", "dst", "pkts", "bytes"])

    rows = []
    for line in ALERT_LOG.read_text(errors="ignore").splitlines()[-max_rows:]:
        line = line.strip()
        if not line or "ALERT" in line:
            continue
        try:
            # [2024-01-01 12:00:00] NORMAL    prob=0.923  10.0.0.1:12345 → 192.168.1.1:80  pkts=5  bytes=1234
            ts    = line[1:20]
            rest  = line[22:]
            parts = rest.split()
            label = parts[0].upper()
            prob  = float(parts[1].split("=")[1])
            src   = parts[2]
            dst   = parts[4]
            pkts  = int(parts[5].split("=")[1])
            byt   = int(parts[6].split("=")[1])
            rows.append(dict(ts=ts, label=label, prob=prob,
                             src=src, dst=dst, pkts=pkts, bytes=byt))
        except Exception:
            continue

    return pd.DataFrame(rows)


def _load_summary() -> pd.DataFrame | None:
    if not SUMMARY_CSV.exists():
        return None
    return pd.read_csv(SUMMARY_CSV)


def _load_sensitivity() -> Path | None:
    img = MODELS_DIR / "n_packets_sensitivity.png"
    return img if img.exists() else None


# ── page: Live Monitor ─────────────────────────────────────────────────────────

if page == "Live Monitor":
    st.header("Live Traffic Monitor")

    placeholder = st.empty()

    for _ in range(9999):
        df = _parse_alert_log()

        with placeholder.container():
            if df.empty:
                st.info("No traffic recorded yet.\n\n"
                        "Run: `python -m realtime.detector --demo`\n\n"
                        "Or train first: `python -m ml.train_model`")
            else:
                total  = len(df)
                counts = df["label"].value_counts()

                # top KPI row
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Flows",   total)
                col2.metric("Normal",  counts.get("NORMAL",  0), delta=None)
                col3.metric("VPN",     counts.get("VPN",     0), delta=None)
                col4.metric("Attacks", counts.get("ATTACK",  0),
                            delta=None,
                            delta_color="inverse" if counts.get("ATTACK", 0) > 0 else "off")

                st.divider()
                left, right = st.columns([1, 2])

                # pie chart
                with left:
                    st.subheader("Traffic Distribution")
                    pie_df = counts.reset_index()
                    pie_df.columns = ["label", "count"]
                    pie_df["color"] = pie_df["label"].map(LABEL_COLOR).fillna("#95a5a6")
                    try:
                        import plotly.express as px
                        fig = px.pie(
                            pie_df, values="count", names="label",
                            color="label",
                            color_discrete_map=LABEL_COLOR,
                            hole=0.3,
                        )
                        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=280)
                        st.plotly_chart(fig, use_container_width=True)
                    except ImportError:
                        st.bar_chart(pie_df.set_index("label")["count"])

                # probability timeline
                with right:
                    st.subheader("Attack Probability — Last 50 Flows")
                    tail = df.tail(50).copy()
                    tail["idx"] = range(len(tail))
                    if "ATTACK" in tail["label"].values:
                        try:
                            import plotly.graph_objects as go
                            fig2 = go.Figure()
                            for lbl, color in LABEL_COLOR.items():
                                mask = tail["label"] == lbl
                                fig2.add_trace(go.Scatter(
                                    x=tail[mask]["idx"],
                                    y=tail[mask]["prob"],
                                    mode="markers",
                                    marker=dict(color=color, size=8),
                                    name=lbl,
                                ))
                            fig2.add_hline(y=0.65, line_dash="dash",
                                           line_color="red", annotation_text="Alert threshold")
                            fig2.update_layout(
                                yaxis_title="Confidence",
                                xaxis_title="Flow #",
                                height=280,
                                margin=dict(t=10, b=10),
                            )
                            st.plotly_chart(fig2, use_container_width=True)
                        except ImportError:
                            st.line_chart(tail.set_index("idx")["prob"])

                st.divider()
                st.subheader("Recent Flows")
                display = df.tail(20).iloc[::-1].copy()
                display["label"] = display["label"].apply(
                    lambda x: f"[!] {x}" if x == "ATTACK"
                              else (f"[~] {x}" if x == "VPN" else f"[+] {x}")
                )
                st.dataframe(
                    display[["ts", "label", "prob", "src", "dst", "pkts", "bytes"]],
                    use_container_width=True,
                    height=380,
                )

        time.sleep(refresh_interval)

# ── page: Model Metrics ────────────────────────────────────────────────────────

elif page == "Model Metrics":
    st.header("Model Performance — Full vs Early Flow")

    summary = _load_summary()
    if summary is None:
        st.warning("No model trained yet.\n\nRun: `python -m ml.train_model`")
    else:
        st.subheader("Summary Table")
        st.dataframe(summary, use_container_width=True)

        st.divider()

        # grouped bar chart
        st.subheader("Metric Comparison")
        metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        available   = [c for c in metric_cols if c in summary.columns]
        if available:
            try:
                import plotly.express as px
                melt = summary.melt(
                    id_vars=["classifier", "flow_mode"],
                    value_vars=available,
                    var_name="metric",
                    value_name="score",
                )
                melt["model"] = melt["classifier"] + " / " + melt["flow_mode"]
                fig = px.bar(
                    melt, x="metric", y="score", color="model",
                    barmode="group",
                    title="Full-flow vs Early-flow Performance",
                    height=420,
                )
                fig.add_hline(y=0.8, line_dash="dash",
                              line_color="green", annotation_text="Target F1=0.80")
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.bar_chart(summary.set_index(["classifier", "flow_mode"])[available])

    st.divider()
    st.subheader("N-Packets Sensitivity")
    sens_img = _load_sensitivity()
    if sens_img and sens_img.exists():
        st.image(str(sens_img), use_container_width=True)
    else:
        st.info("Run `python -m ml.evaluate_model` to generate this plot.")

    roc_all = MODELS_DIR / "roc_all_models.png"
    if roc_all.exists():
        st.subheader("ROC Curves — All Models")
        st.image(str(roc_all), use_container_width=True)

    st.subheader("ROC Curves (per model)")
    for roc_img in sorted(MODELS_DIR.glob("*_roc.png")):
        st.image(str(roc_img), caption=roc_img.stem, width=550)

    st.subheader("Confusion Matrices")
    for cm_img in sorted(MODELS_DIR.glob("*_cm.png")):
        st.image(str(cm_img), caption=cm_img.stem, width=500)

    st.subheader("Feature Importances")
    for fi_img in sorted(MODELS_DIR.glob("*_importance.png")):
        st.image(str(fi_img), caption=fi_img.stem, width=600)

    comp = MODELS_DIR / "comparison_full_vs_early.png"
    if comp.exists():
        st.subheader("Full vs Early — Comparison Chart")
        st.image(str(comp), use_container_width=True)

# ── page: SHAP Explanations ────────────────────────────────────────────────────

elif page == "SHAP Explanations":
    st.header("SHAP — Per-Prediction Model Explanations")
    st.caption(
        "SHAP (SHapley Additive exPlanations) shows **why** the model made each "
        "individual decision — not just which features are globally important, "
        "but how much each feature pushed a specific flow toward VPN or Normal."
    )

    bee = MODELS_DIR / "shap_beeswarm.png"
    if not bee.exists():
        st.warning("SHAP plots not generated yet.")
        st.code("python -m ml.shap_explain")
        st.info(
            "This will analyse the test set with TreeExplainer and produce:\n"
            "- **Beeswarm plot** — global impact across all test flows\n"
            "- **Waterfall plots** — per-flow explanation (correct & wrong)\n"
            "- **mean |SHAP| CSV** — ranked feature contribution table"
        )
    else:
        st.subheader("Global Feature Impact (Beeswarm)")
        st.image(str(bee), use_container_width=True)
        st.caption(
            "Each dot = one test flow. Color = feature value (red = high, blue = low). "
            "X-axis = SHAP value: how much the feature shifted the prediction."
        )

        st.divider()
        st.subheader("Per-Flow Waterfall Explanations")
        st.caption(
            "Waterfall plots show the reasoning chain for individual flows. "
            "Blue bars push toward Normal, red bars push toward VPN."
        )
        wf_imgs = sorted(MODELS_DIR.glob("shap_waterfall_*.png"))
        if wf_imgs:
            cols = st.columns(2)
            for k, img in enumerate(wf_imgs):
                with cols[k % 2]:
                    st.image(str(img), caption=img.stem, use_container_width=True)
        else:
            st.info("No waterfall plots found.")

        st.divider()
        shap_csv = MODELS_DIR / "shap_summary.csv"
        if shap_csv.exists():
            st.subheader("Feature Ranking by Mean |SHAP|")
            df_shap = pd.read_csv(shap_csv)
            try:
                import plotly.express as px
                fig = px.bar(
                    df_shap, x="mean_|shap|", y="feature",
                    orientation="h",
                    color="mean_|shap|",
                    color_continuous_scale="Blues",
                    title="Mean Absolute SHAP Value per Feature",
                    height=420,
                )
                fig.update_layout(yaxis={"categoryorder": "total ascending"},
                                  coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.dataframe(df_shap, use_container_width=True)

# ── page: Dataset Explorer ─────────────────────────────────────────────────────

elif page == "Dataset Explorer":
    st.header("Dataset Explorer")

    csv_options = sorted((ROOT / "dataset" / "raw_csv").glob("*.csv"))
    if not csv_options:
        st.warning("No CSV files found in dataset/raw_csv/")
        st.code("python -m data.generate_synthetic")
    else:
        selected = st.selectbox("Select dataset", [str(p) for p in csv_options])
        df = pd.read_csv(selected, index_col=0, nrows=5000)

        st.metric("Rows", len(df))
        if "label" in df.columns:
            counts = df["label"].value_counts()
            st.bar_chart(counts)

        st.subheader("Feature Distributions")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        feat = st.selectbox("Feature", num_cols)
        if feat and "label" in df.columns:
            try:
                import plotly.express as px
                fig = px.histogram(
                    df, x=feat, color="label", barmode="overlay",
                    opacity=0.7, nbins=60,
                    color_discrete_map={
                        "normal": "#2ecc71",
                        "vpn":    "#f39c12",
                        "attack": "#e74c3c",
                    },
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.bar_chart(df.groupby("label")[feat].mean())

        st.subheader("Raw Sample (first 50 rows)")
        st.dataframe(df.head(50), use_container_width=True)
