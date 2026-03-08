"""
PCAP-based N-packets sensitivity analysis.

Reads PCAPs once, stores up to MAX_N=20 packets per flow,
then computes features for n = 2,3,5,8,10,15,20 without re-reading.
Trains RF for each n and plots F1 vs n_packets.

Usage:
    python -m data.pcap_sensitivity
"""

from __future__ import annotations

import io
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DOWNLOADS = Path("C:/Users/Nitro V15/Downloads")
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

SOURCES = [
    ("VPN-PCAPS-01.zip",    "vpn"),
    ("VPN-PCAPs-02.zip",    "vpn"),
    ("NonVPN-PCAPs-01.zip", "normal"),
]
MAX_FILES_PER_ZIP = 6
MAX_N = 20          # store up to 20 packets per flow
MAX_FLOWS_PER_PCAP = 3000
LABEL_MAP = {"normal": 0, "vpn": 1}
N_VALUES  = [2, 3, 5, 8, 10, 15, 20]


def _read_raw_flows(pcap_bytes: bytes, label: str, max_flows: int) -> list[list[dict]]:
    """Return list of flow-buffers (each buffer = list of dicts with size/time/dir)."""
    try:
        from scapy.all import rdpcap
        from scapy.layers.inet import IP, TCP, UDP
    except ImportError:
        raise ImportError("pip install scapy")

    try:
        pkts = rdpcap(io.BytesIO(pcap_bytes))
    except Exception as e:
        print(f"    [warn] rdpcap failed: {e}")
        return []

    flows: dict[tuple, list[dict]] = defaultdict(list)
    for pkt in pkts:
        try:
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
            a, b = (ip.src, sport), (ip.dst, dport)
            lo, hi = (a, b) if a <= b else (b, a)
            key = (*lo, *hi, proto)
            buf = flows[key]
            if len(buf) < MAX_N:
                buf.append({
                    "size": len(pkt),
                    "time": float(pkt.time),
                    "dir":  1 if a == lo else 0,
                    "label": LABEL_MAP.get(label, 0),
                })
        except Exception:
            continue

    # keep only flows that have at least MAX_N packets
    full_flows = [buf for buf in flows.values() if len(buf) >= MAX_N]
    return full_flows[:max_flows]


def _features_from_buf(buf: list[dict], n: int) -> dict:
    """Compute features from first n packets of a flow buffer."""
    subset = buf[:n]
    sizes = np.array([p["size"] for p in subset], dtype=float)
    times = np.array([p["time"] for p in subset], dtype=float)
    dirs  = np.array([p["dir"]  for p in subset], dtype=float)
    iats  = np.diff(times)
    dur   = times[-1] - times[0]
    if dur <= 0:
        dur = 1e-9
    return {
        "avg_packet_size":  sizes.mean(),
        "std_packet_size":  sizes.std(),
        "min_packet_size":  sizes.min(),
        "max_packet_size":  sizes.max(),
        "avg_interarrival": iats.mean() if len(iats) else 0.0,
        "std_interarrival": iats.std()  if len(iats) else 0.0,
        "min_interarrival": iats.min()  if len(iats) else 0.0,
        "max_interarrival": iats.max()  if len(iats) else 0.0,
        "incoming_ratio":   dirs.mean(),
        "packet_count":     float(n),
        "total_bytes":      sizes.sum(),
        "flow_duration":    dur,
        "bytes_per_second": sizes.sum() / dur,
        "pkts_per_second":  n / dur,
    }


def run():
    # ── Step 1: read all raw flows from PCAPs ─────────────────────────────────
    print("[pcap_sensitivity] Reading PCAPs (storing up to 20 pkts/flow) ...")
    all_bufs  = []   # list of list[dict]
    all_labels = []  # parallel list of int labels

    for zip_name, label in SOURCES:
        zip_path = DOWNLOADS / zip_name
        if not zip_path.exists():
            print(f"  [skip] {zip_name}")
            continue

        with zipfile.ZipFile(zip_path) as zf:
            pcap_files = [n for n in zf.namelist()
                          if n.lower().endswith((".pcap", ".pcapng"))]
            selected = pcap_files[:MAX_FILES_PER_ZIP]
            print(f"  {zip_name}  ({len(selected)} files)")

            for fname in selected:
                info = zf.getinfo(fname)
                if info.file_size / 1024**2 > 800:
                    print(f"    [skip] {fname} too large")
                    continue
                print(f"    {fname} ...", end=" ", flush=True)
                raw   = zf.read(fname)
                bufs  = _read_raw_flows(raw, label, MAX_FLOWS_PER_PCAP)
                del raw
                all_bufs.extend(bufs)
                all_labels.extend([LABEL_MAP[label]] * len(bufs))
                print(f"{len(bufs)} flows")

    print(f"\nTotal flows with >= {MAX_N} packets: {len(all_bufs)}")
    if len(all_bufs) < 50:
        print("[error] Not enough flows. Try increasing MAX_FLOWS_PER_PCAP or adding more pcap files.")
        return

    y = np.array(all_labels)

    # ── Step 2: compute F1 for each n ────────────────────────────────────────
    print(f"\n[pcap_sensitivity] Training RF for n in {N_VALUES} ...")
    f1s = []

    for n in N_VALUES:
        rows = [_features_from_buf(b, n) for b in all_bufs]
        X = pd.DataFrame(rows)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=10,
            class_weight="balanced", random_state=42, n_jobs=1
        )
        rf.fit(X_tr, y_tr)
        f1 = f1_score(y_te, rf.predict(X_te), average="weighted", zero_division=0)
        f1s.append(f1)
        print(f"  n={n:3d}  F1={f1:.4f}")

    # ── Step 3: plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(N_VALUES, f1s, marker="o", color="steelblue", linewidth=2, markersize=7)
    ax.fill_between(N_VALUES, [f - 0.02 for f in f1s], [f + 0.02 for f in f1s],
                    alpha=0.15, color="steelblue")
    ax.axhline(0.80, color="green",  linestyle="--", linewidth=1.2, label="Target F1=0.80")
    ax.axvline(5,    color="orange", linestyle=":",  linewidth=1.5, label="n=5 (thesis)")
    ax.set_xlabel("Number of Early Packets Used")
    ax.set_ylabel("Weighted F1-score")
    ax.set_title("Real PCAP — Classification Performance vs. Number of Early Packets")
    ax.set_xticks(N_VALUES)
    ax.set_ylim(0.5, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.35)
    fig.tight_layout()

    out = MODELS_DIR / "n_packets_sensitivity.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\n[saved] {out}")

    # print summary table
    print("\nSummary:")
    for n, f in zip(N_VALUES, f1s):
        bar = "#" * int(f * 30)
        print(f"  n={n:3d}  F1={f:.4f}  {bar}")


if __name__ == "__main__":
    run()
