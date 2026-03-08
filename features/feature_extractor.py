"""
Feature Extractor — two modes:

1. from_pcap(pcap_file, n_packets)
   Reads a .pcap file with Scapy, groups packets into flows,
   takes only the first n_packets per flow, computes features.

2. from_cicflowmeter_csv(csv_file, n_packets_mode)
   Reads a CICFlowMeter output CSV.  Because CICFlowMeter already
   aggregated the whole flow we simulate "early" extraction by
   selecting a feature subset that is computable from few packets
   (packet-size stats, IAT stats, direction ratio, short duration).
   When n_packets_mode='full' all features are used.

3. from_synthetic_csv(csv_file)
   Reads our generated synthetic CSV (same schema as CICFlowMeter
   early-feature subset).

Usage:
    from features.feature_extractor import FeatureExtractor
    fe = FeatureExtractor(n_packets=5)
    X, y = fe.from_synthetic_csv("dataset/raw_csv/synthetic_flows.csv")
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Features available after only N early packets
EARLY_FEATURES = [
    "avg_packet_size",
    "std_packet_size",
    "min_packet_size",
    "max_packet_size",
    "avg_interarrival",
    "std_interarrival",
    "min_interarrival",
    "max_interarrival",
    "incoming_ratio",
    "packet_count",
    "total_bytes",
    "flow_duration",
    "bytes_per_second",
    "pkts_per_second",
]

# Additional features only available from full-flow CICFlowMeter export
FULL_EXTRA_FEATURES = [
    "fwd_packet_length_mean",
    "bwd_packet_length_mean",
    "fwd_iat_mean",
    "bwd_iat_mean",
    "flow_iat_mean",
    "flow_iat_std",
    "active_mean",
    "idle_mean",
    "subflow_fwd_packets",
    "subflow_bwd_packets",
]

LABEL_MAP = {"normal": 0, "vpn": 1, "attack": 2}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}


class FeatureExtractor:
    def __init__(self, n_packets: int = 5):
        self.n_packets = n_packets

    # ── Synthetic / pre-built CSV ──────────────────────────────────────────────

    def from_synthetic_csv(
        self,
        csv_path: str | Path,
        mode: Literal["early", "full"] = "early",
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Return (X, y) ready for sklearn."""
        df = pd.read_csv(csv_path, index_col="flow_id")

        y_raw = df["label"]
        y = y_raw.map(LABEL_MAP)
        if y.isna().any():
            unknown = y_raw[y.isna()].unique()
            raise ValueError(f"Unknown labels in CSV: {unknown}")

        features = EARLY_FEATURES if mode == "early" else EARLY_FEATURES + FULL_EXTRA_FEATURES
        available = [f for f in features if f in df.columns]
        X = df[available].copy()

        # basic sanity cleaning
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median(numeric_only=True))

        print(f"[extractor] mode={mode}  samples={len(X)}  features={len(available)}")
        return X, y

    # ── CICFlowMeter CSV ──────────────────────────────────────────────────────

    def from_cicflowmeter_csv(
        self,
        csv_path: str | Path,
        mode: Literal["early", "full"] = "early",
        label_col: str = "Label",
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Parse a CICFlowMeter CSV.
        Column names are normalised (lowercase + underscores).
        """
        df = pd.read_csv(csv_path, low_memory=False)
        df.columns = (
            df.columns.str.strip()
                      .str.lower()
                      .str.replace(r"[\s/]+", "_", regex=True)
        )

        lc = label_col.lower()
        if lc not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found. "
                             f"Available: {list(df.columns)}")

        # normalise label strings
        y_raw = df[lc].str.strip().str.lower()
        y_raw = y_raw.replace({
            "benign": "normal",
            "normal": "normal",
            "vpn":    "vpn",
        })
        # anything with 'attack','dos','ddos','scan','brute' → attack
        y_raw = y_raw.where(
            y_raw.isin(LABEL_MAP),
            other=y_raw.apply(
                lambda v: "attack" if any(
                    k in str(v) for k in ["attack", "dos", "ddos", "scan", "brute", "syn"]
                ) else "normal"
            ),
        )
        y = y_raw.map(LABEL_MAP)

        # build feature mapping from CICFlowMeter column names → our names
        col_map = {
            "flow_duration":              "flow_duration",
            "total_length_of_fwd_packets": "total_bytes",
            "packet_length_mean":         "avg_packet_size",
            "packet_length_std":          "std_packet_size",
            "packet_length_min":          "min_packet_size",
            "packet_length_max":          "max_packet_size",
            "flow_iat_mean":              "avg_interarrival",
            "flow_iat_std":               "std_interarrival",
            "flow_iat_min":               "min_interarrival",
            "flow_iat_max":               "max_interarrival",
            "total_fwd_packets":          "packet_count",
        }

        X = pd.DataFrame()
        for src, dst in col_map.items():
            if src in df.columns:
                X[dst] = pd.to_numeric(df[src], errors="coerce")

        # derived
        if "flow_duration" in X and "total_bytes" in X:
            dur = X["flow_duration"].replace(0, np.nan)
            X["bytes_per_second"] = X["total_bytes"] / dur
        if "flow_duration" in X and "packet_count" in X:
            dur = X["flow_duration"].replace(0, np.nan)
            X["pkts_per_second"] = X["packet_count"] / dur

        if "total_fwd_packets" in df.columns and "total_backward_packets" in df.columns:
            fwd = pd.to_numeric(df["total_fwd_packets"], errors="coerce").fillna(0)
            bwd = pd.to_numeric(df["total_backward_packets"], errors="coerce").fillna(0)
            total = (fwd + bwd).replace(0, np.nan)
            X["incoming_ratio"] = bwd / total

        if mode == "full":
            for col in FULL_EXTRA_FEATURES:
                if col in df.columns:
                    X[col] = pd.to_numeric(df[col], errors="coerce")

        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median(numeric_only=True))

        mask = y.notna()
        print(f"[extractor] CICFlowMeter mode={mode}  samples={mask.sum()}  features={len(X.columns)}")
        return X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)

    # ── PCAP (Scapy) ──────────────────────────────────────────────────────────

    def from_pcap(
        self,
        pcap_path: str | Path,
        label: str = "unknown",
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Extract early-flow features directly from a .pcap file.
        Requires scapy installed.
        Returns (X, y) where every row is one flow.
        """
        try:
            from scapy.all import rdpcap, IP, TCP, UDP
        except ImportError:
            raise ImportError("scapy not installed. Run: pip install scapy")

        packets = rdpcap(str(pcap_path))

        # group packets into flows by 5-tuple
        flows: dict[tuple, list] = {}
        for pkt in packets:
            if not pkt.haslayer(IP):
                continue
            ip = pkt[IP]
            proto = ip.proto

            if pkt.haslayer(TCP):
                sport, dport = pkt[TCP].sport, pkt[TCP].dport
            elif pkt.haslayer(UDP):
                sport, dport = pkt[UDP].sport, pkt[UDP].dport
            else:
                continue

            # bidirectional key (smaller tuple first)
            key = tuple(sorted([
                (ip.src, sport),
                (ip.dst, dport),
            ])) + (proto,)

            if key not in flows:
                flows[key] = []
            if len(flows[key]) < self.n_packets:
                flows[key].append({
                    "size": len(pkt),
                    "time": float(pkt.time),
                    "direction": 1 if (ip.src, sport) == key[0] else 0,
                })

        rows = []
        for key, pkts in flows.items():
            if len(pkts) < 2:
                continue
            sizes = np.array([p["size"] for p in pkts], dtype=float)
            times = np.array([p["time"] for p in pkts], dtype=float)
            dirs  = np.array([p["direction"] for p in pkts], dtype=float)

            iats = np.diff(times)
            rows.append({
                "avg_packet_size":  sizes.mean(),
                "std_packet_size":  sizes.std(),
                "min_packet_size":  sizes.min(),
                "max_packet_size":  sizes.max(),
                "avg_interarrival": iats.mean() if len(iats) else 0,
                "std_interarrival": iats.std()  if len(iats) else 0,
                "min_interarrival": iats.min()  if len(iats) else 0,
                "max_interarrival": iats.max()  if len(iats) else 0,
                "incoming_ratio":   dirs.mean(),
                "packet_count":     len(pkts),
                "total_bytes":      sizes.sum(),
                "flow_duration":    times[-1] - times[0],
                "bytes_per_second": sizes.sum() / max(times[-1] - times[0], 1e-9),
                "pkts_per_second":  len(pkts) / max(times[-1] - times[0], 1e-9),
            })

        X = pd.DataFrame(rows)
        y = pd.Series([LABEL_MAP.get(label, 0)] * len(X), name="label")
        print(f"[extractor] pcap={pcap_path}  flows={len(X)}")
        return X, y


# ── CLI quick-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data.generate_synthetic import generate
    generate("dataset/raw_csv/synthetic_flows.csv")

    fe = FeatureExtractor(n_packets=5)
    X, y = fe.from_synthetic_csv("dataset/raw_csv/synthetic_flows.csv", mode="early")
    print(X.describe())
    print("\nLabel distribution:")
    print(y.map(LABEL_MAP_INV).value_counts())
