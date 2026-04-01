"""
Streamlit Dashboard — APT Early Detection System
=================================================
Pages:
  1. Live Monitor      — real-time flow feed from alerts.log
  2. Model Metrics     — full vs early comparison, ROC, confusion matrices
  3. SHAP Explanations — beeswarm, waterfall, feature ranking
  4. Dataset Explorer  — raw CSV browser with feature distributions
  5. Analyst Feedback  — mark flows as TP / FP, view feedback history
  6. Drift & Health    — PSI drift monitor + adversarial robustness report

Run:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

MODELS_DIR   = ROOT / "models"
ALERT_LOG    = MODELS_DIR / "alerts.log"
SUMMARY_CSV  = MODELS_DIR / "summary.csv"
FEEDBACK_CSV = MODELS_DIR / "feedback.csv"
DRIFT_REPORT = MODELS_DIR / "drift_report.json"
ADV_REPORT   = MODELS_DIR / "adversarial_report.json"

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="APT Early Detection",
    page_icon="🛡",
    layout="wide",
)

# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛡 APT Early Detection")
    st.caption("Diploma project — Early Flow Classification")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🔴 Live Monitor",
            "📊 Model Metrics",
            "🧠 SHAP Explanations",
            "🗂 Dataset Explorer",
            "✅ Analyst Feedback",
            "📡 Drift & Health",
        ],
        index=0,
    )
    st.divider()
    if page == "🔴 Live Monitor":
        refresh_interval = st.slider("Auto-refresh (sec)", 1, 10, 2)
    st.caption(f"API: `{API_BASE}`")


# ── colour palette ─────────────────────────────────────────────────────────────
LABEL_COLOR = {
    "NORMAL": "#2ecc71",
    "VPN":    "#f39c12",
    "ATTACK": "#e74c3c",
    "UNKNOWN":"#95a5a6",
}


# ── robust log parser ──────────────────────────────────────────────────────────
# Regex handles both original and enriched (ip_risk/hybrid) log formats.
_LOG_RE = re.compile(
    r"\[(?P<ts>[^\]]+)\]\s+"
    r"(?P<label>\w+)\s+"
    r"prob=(?P<prob>[\d.]+)\s+"
    r"(?P<src>\S+)\s+[→>]\s+(?P<dst>\S+)\s+"
    r"pkts=(?P<pkts>\d+)\s+"
    r"bytes=(?P<bytes>\d+)"
    r"(?:.*?ip_risk=(?P<ip_risk>[\d]+))?"
    r"(?:.*?hybrid=(?P<hybrid>[\d.]+))?",
    re.IGNORECASE,
)

def _parse_alert_log(max_rows: int = 1000) -> pd.DataFrame:
    if not ALERT_LOG.exists():
        return pd.DataFrame(
            columns=["ts", "label", "prob", "src", "dst",
                     "pkts", "bytes", "ip_risk", "hybrid"]
        )

    rows = []
    lines = ALERT_LOG.read_text(encoding="utf-8", errors="ignore").splitlines()
    for raw in lines[-max_rows:]:
        # strip ANSI escape codes
        line = re.sub(r"\x1b\[[0-9;]*m", "", raw).strip()
        if not line or "[!]" in line:
            continue
        m = _LOG_RE.search(line)
        if not m:
            continue
        try:
            rows.append({
                "ts":      m.group("ts"),
                "label":   m.group("label").upper(),
                "prob":    float(m.group("prob")),
                "src":     m.group("src"),
                "dst":     m.group("dst"),
                "pkts":    int(m.group("pkts")),
                "bytes":   int(m.group("bytes")),
                "ip_risk": int(m.group("ip_risk")) if m.group("ip_risk") else None,
                "hybrid":  float(m.group("hybrid")) if m.group("hybrid") else None,
            })
        except Exception:
            continue

    return pd.DataFrame(rows)


def _load_summary() -> pd.DataFrame | None:
    return pd.read_csv(SUMMARY_CSV) if SUMMARY_CSV.exists() else None


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Live Monitor
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🔴 Live Monitor":
    st.header("🔴 Live Traffic Monitor")

    df = _parse_alert_log()

    if df.empty:
        st.info(
            "**No traffic recorded yet.**\n\n"
            "Start the detector:\n"
            "```\npython -X utf8 -m realtime.detector --demo\n```\n\n"
            f"Watching: `{ALERT_LOG}`"
        )
    else:
        total  = len(df)
        counts = df["label"].value_counts()
        n_att  = int(counts.get("ATTACK", 0))

        # ── KPI row ────────────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Flows",  total)
        c2.metric("Normal",       int(counts.get("NORMAL", 0)))
        c3.metric("VPN",          int(counts.get("VPN",    0)))
        c4.metric("🚨 Attacks",   n_att,
                  delta=f"+{n_att}" if n_att else None,
                  delta_color="inverse" if n_att else "off")

        st.divider()

        # ── charts ─────────────────────────────────────────────────────────────
        left, right = st.columns([1, 2])

        with left:
            st.subheader("Traffic Distribution")
            pie_df = counts.reset_index()
            pie_df.columns = ["label", "count"]
            try:
                import plotly.express as px
                fig = px.pie(pie_df, values="count", names="label",
                             color="label", color_discrete_map=LABEL_COLOR, hole=0.3)
                fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=280)
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.bar_chart(pie_df.set_index("label")["count"])

        with right:
            st.subheader("Attack Probability — Last 60 Flows")
            tail = df.tail(60).copy().reset_index(drop=True)
            tail["idx"] = tail.index
            try:
                import plotly.graph_objects as go
                fig2 = go.Figure()
                for lbl, color in LABEL_COLOR.items():
                    mask = tail["label"] == lbl
                    if mask.any():
                        # use hybrid score if available, else prob
                        y_col = tail.loc[mask, "hybrid"].fillna(tail.loc[mask, "prob"])
                        fig2.add_trace(go.Scatter(
                            x=tail[mask]["idx"], y=y_col,
                            mode="markers",
                            marker=dict(color=color, size=9,
                                        line=dict(width=1, color="white")),
                            name=lbl,
                            hovertemplate=(
                                "<b>%{customdata[0]}</b><br>"
                                "src: %{customdata[1]}<br>"
                                "score: %{y:.3f}<extra></extra>"
                            ),
                            customdata=tail[mask][["label", "src"]].values,
                        ))
                fig2.add_hline(y=0.65, line_dash="dash", line_color="red",
                               annotation_text="Alert threshold (0.65)")
                fig2.update_layout(
                    yaxis=dict(title="Score", range=[0, 1.05]),
                    xaxis_title="Flow #",
                    height=280,
                    margin=dict(t=10, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig2, use_container_width=True)
            except ImportError:
                st.line_chart(tail.set_index("idx")["prob"])

        st.divider()

        # ── Recent flows table ─────────────────────────────────────────────────
        st.subheader("Recent Flows (last 25)")
        display = df.tail(25).iloc[::-1].copy()

        # label badge
        display["status"] = display["label"].map({
            "ATTACK": "🚨 ATTACK",
            "VPN":    "🔒 VPN",
            "NORMAL": "✅ NORMAL",
        }).fillna(display["label"])

        show_cols = ["ts", "status", "prob", "src", "dst", "pkts", "bytes"]
        if display["hybrid"].notna().any():
            show_cols.insert(4, "hybrid")
        if display["ip_risk"].notna().any():
            show_cols.insert(5, "ip_risk")

        st.dataframe(
            display[show_cols].rename(columns={
                "prob":    "ML score",
                "hybrid":  "Hybrid score",
                "ip_risk": "IP risk",
            }),
            use_container_width=True,
            height=450,
        )

    # auto-refresh
    time.sleep(refresh_interval)
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Model Metrics
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Metrics":
    st.header("📊 Model Performance — Full vs Early Flow")

    summary = _load_summary()
    if summary is None:
        st.warning("No model trained yet.\n\nRun: `python -m ml.train_model`")
    else:
        st.subheader("Summary Table")
        st.dataframe(summary, use_container_width=True)
        st.divider()

        metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        available   = [c for c in metric_cols if c in summary.columns]
        if available:
            try:
                import plotly.express as px
                melt = summary.melt(
                    id_vars=["classifier", "flow_mode"],
                    value_vars=available, var_name="metric", value_name="score",
                )
                melt["model"] = melt["classifier"] + " / " + melt["flow_mode"]
                fig = px.bar(melt, x="metric", y="score", color="model",
                             barmode="group",
                             title="Full-flow vs Early-flow Performance",
                             height=420)
                fig.add_hline(y=0.8, line_dash="dash", line_color="green",
                              annotation_text="Target F1 = 0.80")
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.bar_chart(summary.set_index(["classifier", "flow_mode"])[available])

    st.divider()
    st.subheader("N-Packets Sensitivity")
    sens_img = MODELS_DIR / "n_packets_sensitivity.png"
    if sens_img.exists():
        st.image(str(sens_img), use_container_width=True)
    else:
        st.info("Run `python -m ml.evaluate_model` to generate this plot.")

    roc_all = MODELS_DIR / "roc_all_models.png"
    if roc_all.exists():
        st.subheader("ROC Curves — All Models")
        st.image(str(roc_all), use_container_width=True)

    st.subheader("ROC Curves (per model)")
    for img in sorted(MODELS_DIR.glob("*_roc.png")):
        st.image(str(img), caption=img.stem, width=550)

    st.subheader("Confusion Matrices")
    for img in sorted(MODELS_DIR.glob("*_cm.png")):
        st.image(str(img), caption=img.stem, width=500)

    st.subheader("Feature Importances")
    for img in sorted(MODELS_DIR.glob("*_importance.png")):
        st.image(str(img), caption=img.stem, width=600)

    comp = MODELS_DIR / "comparison_full_vs_early.png"
    if comp.exists():
        st.subheader("Full vs Early — Comparison Chart")
        st.image(str(comp), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SHAP Explanations
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 SHAP Explanations":
    st.header("🧠 SHAP — Per-Prediction Model Explanations")
    st.caption(
        "SHAP (SHapley Additive exPlanations) shows **why** the model made each "
        "individual decision — not just which features are globally important, "
        "but how much each feature pushed a specific flow toward VPN or Normal."
    )

    bee = MODELS_DIR / "shap_beeswarm.png"
    if not bee.exists():
        st.warning("SHAP plots not generated yet.")
        st.code("python -X utf8 -m ml.shap_explain --csv dataset/raw_csv/iscx_combined.csv")
    else:
        st.subheader("Global Feature Impact (Beeswarm)")
        st.image(str(bee), use_container_width=True)
        st.caption(
            "Each dot = one test flow. Red = high feature value, blue = low. "
            "X-axis = SHAP value: how much this feature shifted the prediction."
        )
        st.divider()
        st.subheader("Per-Flow Waterfall Explanations")
        wf_imgs = sorted(MODELS_DIR.glob("shap_waterfall_*.png"))
        if wf_imgs:
            cols = st.columns(2)
            for k, img in enumerate(wf_imgs):
                with cols[k % 2]:
                    st.image(str(img), caption=img.stem, use_container_width=True)
        st.divider()
        shap_csv = MODELS_DIR / "shap_summary.csv"
        if shap_csv.exists():
            st.subheader("Feature Ranking by Mean |SHAP|")
            df_shap = pd.read_csv(shap_csv)
            try:
                import plotly.express as px
                fig = px.bar(df_shap, x="mean_|shap|", y="feature",
                             orientation="h", color="mean_|shap|",
                             color_continuous_scale="Blues",
                             title="Mean Absolute SHAP Value per Feature",
                             height=420)
                fig.update_layout(yaxis={"categoryorder": "total ascending"},
                                  coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.dataframe(df_shap, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Dataset Explorer
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🗂 Dataset Explorer":
    st.header("🗂 Dataset Explorer")

    csv_options = sorted((ROOT / "dataset" / "raw_csv").glob("*.csv"))
    if not csv_options:
        st.warning("No CSV files found in `dataset/raw_csv/`")
        st.code("python -m data.generate_synthetic")
    else:
        selected = st.selectbox("Select dataset", [str(p) for p in csv_options])
        df = pd.read_csv(selected, index_col=0, nrows=5000)
        st.metric("Rows (preview)", len(df))

        if "label" in df.columns:
            st.subheader("Class Distribution")
            counts = df["label"].value_counts()
            try:
                import plotly.express as px
                fig = px.bar(counts.reset_index(), x="label", y="count",
                             color="label",
                             color_discrete_map={"normal":"#2ecc71",
                                                 "vpn":"#f39c12",
                                                 "attack":"#e74c3c"},
                             height=300)
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.bar_chart(counts)

        st.subheader("Feature Distributions")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        feat = st.selectbox("Feature", num_cols)
        if feat and "label" in df.columns:
            try:
                import plotly.express as px
                fig = px.histogram(df, x=feat, color="label", barmode="overlay",
                                   opacity=0.7, nbins=60,
                                   color_discrete_map={"normal":"#2ecc71",
                                                       "vpn":"#f39c12",
                                                       "attack":"#e74c3c"},
                                   height=350)
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.bar_chart(df.groupby("label")[feat].mean())

        st.subheader("Raw Sample (first 50 rows)")
        st.dataframe(df.head(50), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Analyst Feedback
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "✅ Analyst Feedback":
    st.header("✅ Analyst Feedback — False Positive Management")
    st.caption(
        "Mark detected flows as **True Positive** or **False Positive**. "
        "Feedback is saved and can be used to retrain the model to reduce noise."
    )

    # ── Submit feedback form ───────────────────────────────────────────────────
    with st.expander("➕ Submit Feedback for a Flow", expanded=True):
        df_log = _parse_alert_log()
        attack_flows = df_log[df_log["label"] == "ATTACK"] if not df_log.empty else pd.DataFrame()

        col1, col2 = st.columns(2)
        with col1:
            if not attack_flows.empty:
                options = [
                    f"{r['ts']} | {r['src']} → {r['dst']} (prob={r['prob']:.2f})"
                    for _, r in attack_flows.tail(20).iloc[::-1].iterrows()
                ]
                selected_flow = st.selectbox("Select ATTACK flow from log", options)
                idx = options.index(selected_flow)
                row = attack_flows.tail(20).iloc[::-1].iloc[idx]
                src_val  = row["src"]
                dst_val  = row["dst"]
                prob_val = float(row["prob"])
            else:
                st.info("No ATTACK flows in log yet. Run the detector first.")
                src_val  = st.text_input("Source IP:port", "10.0.0.1:12345")
                dst_val  = st.text_input("Destination IP:port", "192.168.1.1:80")
                prob_val = st.slider("ML Confidence", 0.0, 1.0, 0.85)

        with col2:
            verdict = st.radio(
                "Analyst verdict",
                ["true_positive", "false_positive", "unknown"],
                format_func=lambda x: {
                    "true_positive":  "✅ True Positive — real attack",
                    "false_positive": "❌ False Positive — benign traffic",
                    "unknown":        "❓ Unknown / needs investigation",
                }[x],
            )
            note = st.text_area("Note (optional)", placeholder="e.g. This is our backup server")

        if st.button("📨 Submit Feedback", type="primary"):
            try:
                resp = requests.post(
                    f"{API_BASE}/feedback",
                    json={
                        "src":             src_val,
                        "dst":             dst_val,
                        "model_label":     "attack",
                        "confidence":      prob_val,
                        "analyst_verdict": verdict,
                        "analyst_note":    note,
                    },
                    timeout=5,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"✅ {data['message']}  (ID #{data['feedback_id']})")
                else:
                    st.error(f"API error {resp.status_code}: {resp.text}")
            except requests.exceptions.ConnectionError:
                # Fallback: write directly if API is down
                import csv as _csv
                row_data = {
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "src": src_val, "dst": dst_val,
                    "model_label": "attack", "confidence": round(prob_val, 4),
                    "analyst_verdict": verdict, "analyst_note": note,
                }
                write_header = not FEEDBACK_CSV.exists()
                with open(FEEDBACK_CSV, "a", newline="", encoding="utf-8") as f:
                    w = _csv.DictWriter(f, fieldnames=list(row_data.keys()))
                    if write_header:
                        w.writeheader()
                    w.writerow(row_data)
                st.success("✅ Feedback saved locally (API offline).")

    st.divider()

    # ── Feedback history ───────────────────────────────────────────────────────
    st.subheader("📋 Feedback History")
    if FEEDBACK_CSV.exists():
        fb_df = pd.read_csv(FEEDBACK_CSV)
        total_fb = len(fb_df)
        tp = int((fb_df["analyst_verdict"] == "true_positive").sum())
        fp = int((fb_df["analyst_verdict"] == "false_positive").sum())
        un = int((fb_df["analyst_verdict"] == "unknown").sum())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Feedback", total_fb)
        m2.metric("✅ True Positives",  tp)
        m3.metric("❌ False Positives", fp)
        m4.metric("❓ Unknown",         un)

        if total_fb > 0:
            fp_rate = fp / total_fb
            if fp_rate > 0.3:
                st.warning(
                    f"⚠ False positive rate is **{fp_rate:.0%}** — "
                    "consider retraining the model."
                )

        st.dataframe(
            fb_df.tail(50).iloc[::-1],
            use_container_width=True, height=350,
        )

        # ── Retrain suggestion ─────────────────────────────────────────────────
        if fp > 5:
            st.info(
                f"**{fp} false positives** recorded. "
                "When ready to retrain with analyst labels:\n"
                "```\npython -X utf8 -m ml.train_model "
                "--csv dataset/raw_csv/iscx_combined.csv\n```"
            )
    else:
        st.info("No feedback submitted yet.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Drift & Health
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📡 Drift & Health":
    st.header("📡 Model Drift & System Health")

    # ── Drift Monitor ──────────────────────────────────────────────────────────
    st.subheader("Population Stability Index (PSI) — Drift Monitor")
    st.caption(
        "PSI measures how much the live traffic distribution has shifted "
        "compared to the training data.\n\n"
        "**< 0.10** → ✅ No drift | **0.10–0.20** → ⚠️ Monitor | **> 0.20** → 🔴 Retrain"
    )

    col_run, col_build = st.columns(2)
    with col_run:
        if st.button("🔄 Run Drift Check Now"):
            with st.spinner("Computing PSI…"):
                try:
                    import subprocess
                    result = subprocess.run(
                        ["python", "-X", "utf8", "-m", "ml.drift_monitor", "--check"],
                        capture_output=True, text=True, cwd=str(ROOT)
                    )
                    st.code(result.stdout or result.stderr)
                except Exception as e:
                    st.error(str(e))
    with col_build:
        if st.button("🏗 Build Reference (from training CSV)"):
            with st.spinner("Building reference distribution…"):
                try:
                    import subprocess
                    result = subprocess.run(
                        ["python", "-X", "utf8", "-m", "ml.drift_monitor",
                         "--build", "--csv", "dataset/raw_csv/iscx_combined.csv"],
                        capture_output=True, text=True, cwd=str(ROOT)
                    )
                    st.code(result.stdout or result.stderr)
                except Exception as e:
                    st.error(str(e))

    if DRIFT_REPORT.exists():
        report = json.loads(DRIFT_REPORT.read_text(encoding="utf-8"))
        overall = report.get("overall_psi", 0)
        status  = report.get("overall_status", "OK")

        emoji = {"OK": "✅", "MONITOR": "⚠️", "RETRAIN": "🔴"}.get(status, "")
        st.metric(f"Overall PSI {emoji}", f"{overall:.4f}",
                  delta=status,
                  delta_color="normal" if status == "OK" else "inverse")

        feat_data = report.get("features", {})
        if feat_data:
            rows = [
                {
                    "Feature":    k,
                    "PSI":        v["psi"],
                    "Status":     v["status"],
                    "Live Mean":  v.get("live_mean", "—"),
                    "Train Mean": v.get("train_mean", "—"),
                }
                for k, v in feat_data.items()
            ]
            drift_df = pd.DataFrame(rows).sort_values("PSI", ascending=False)

            try:
                import plotly.express as px
                color_map = {"OK": "#2ecc71", "MONITOR": "#f39c12", "RETRAIN": "#e74c3c"}
                drift_df["color"] = drift_df["Status"].map(color_map)
                fig = px.bar(drift_df, x="PSI", y="Feature", orientation="h",
                             color="Status",
                             color_discrete_map=color_map,
                             title="PSI per Feature", height=420)
                fig.add_vline(x=0.10, line_dash="dash", line_color="orange",
                              annotation_text="0.10")
                fig.add_vline(x=0.20, line_dash="dash", line_color="red",
                              annotation_text="0.20")
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.dataframe(drift_df, use_container_width=True)

        st.caption(f"Last checked: {report.get('checked_at', '—')} | "
                   f"Live samples: {report.get('live_samples', 0)}")
    else:
        st.info(
            "No drift report yet.\n\n"
            "1. First build reference: `python -m ml.drift_monitor --build --csv dataset/raw_csv/iscx_combined.csv`\n"
            "2. Run detector to collect live data\n"
            "3. Then check: `python -m ml.drift_monitor --check`"
        )

    st.divider()

    # ── Adversarial Robustness ─────────────────────────────────────────────────
    st.subheader("🛡 Adversarial Robustness Report")
    st.caption(
        "Shows how much detection rate drops under three evasion strategies. "
        "**This is not a bug — it is an honest documentation of model limitations.**"
    )

    if st.button("▶ Run Adversarial Tests"):
        with st.spinner("Running tests/test_adversarial.py …"):
            try:
                import subprocess
                result = subprocess.run(
                    ["python", "-X", "utf8", "-m", "pytest",
                     "tests/test_adversarial.py", "-v", "-s", "--tb=short"],
                    capture_output=True, text=True, cwd=str(ROOT)
                )
                st.code(result.stdout[-3000:] if len(result.stdout) > 3000
                        else result.stdout)
            except Exception as e:
                st.error(str(e))

    if ADV_REPORT.exists():
        adv = json.loads(ADV_REPORT.read_text(encoding="utf-8"))
        rows = [
            {
                "Evasion Strategy": k.replace("_", " ").title(),
                "Baseline Recall":  f"{v['baseline_recall']:.1%}",
                "Evaded Recall":    f"{v['evaded_recall']:.1%}",
                "Drop":             f"{v['drop']:.1%}",
            }
            for k, v in adv.items()
        ]
        adv_df = pd.DataFrame(rows)
        st.dataframe(adv_df, use_container_width=True)

        try:
            import plotly.graph_objects as go
            fig = go.Figure()
            for _, r in adv_df.iterrows():
                strategy = r["Evasion Strategy"]
                baseline = float(r["Baseline Recall"].strip("%")) / 100
                evaded   = float(r["Evaded Recall"].strip("%")) / 100
                fig.add_trace(go.Bar(name="Baseline", x=[strategy], y=[baseline],
                                     marker_color="#2ecc71"))
                fig.add_trace(go.Bar(name="Evaded",   x=[strategy], y=[evaded],
                                     marker_color="#e74c3c"))
            fig.update_layout(barmode="group",
                              yaxis=dict(title="Detection Rate", tickformat=".0%"),
                              title="Detection Rate: Baseline vs Evaded",
                              height=380,
                              showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            pass

        st.info(
            "**Mitigation for evasion attacks:**\n"
            "- Combine with Threat Intel (AbuseIPDB)\n"
            "- Add network baseline (unknown host detection)\n"
            "- Retrain with adversarial examples\n"
            "- Use ensemble: ML + rule-based + behavioral"
        )
    else:
        st.info("No adversarial report yet. Click **Run Adversarial Tests** above.")

    st.divider()

    # ── API Health ─────────────────────────────────────────────────────────────
    st.subheader("🔌 API Health")
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        if r.status_code == 200:
            st.success(f"✅ API is **online** — `{API_BASE}`")
        else:
            st.error(f"API returned {r.status_code}")
    except Exception:
        st.error(f"❌ API offline — `{API_BASE}`")

    try:
        r2 = requests.get(f"{API_BASE}/ready", timeout=3)
        if r2.status_code == 200:
            data = r2.json()
            st.success(f"✅ Models loaded: `{', '.join(data.get('models', []))}`")
    except Exception:
        pass

    try:
        r3 = requests.get(f"{API_BASE}/feedback/stats", timeout=3)
        if r3.status_code == 200:
            s = r3.json()
            st.info(
                f"Feedback stats: Total={s.get('total',0)} | "
                f"TP={s.get('true_positive',0)} | "
                f"FP={s.get('false_positive',0)}"
            )
    except Exception:
        pass
