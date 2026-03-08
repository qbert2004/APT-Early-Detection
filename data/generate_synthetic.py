"""
Synthetic network flow generator for testing the pipeline
without the real ISCX dataset.

Generates realistic statistical features for:
  - normal   : web browsing, file download, DNS
  - vpn      : tunnelled traffic with distinct timing/size patterns
  - attack   : port scan, brute-force, SYN flood

Output: dataset/raw_csv/synthetic_flows.csv  (same schema as CICFlowMeter)
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

# ── per-class parameter ranges ─────────────────────────────────────────────────
PROFILES = {
    "normal": dict(
        n=4000,
        pkt_size_mean=(200, 1400),   # (low, high) uniform range for mean
        pkt_size_std=(30, 300),
        interarrival_mean=(0.005, 0.5),
        interarrival_std=(0.001, 0.1),
        incoming_ratio=(0.3, 0.7),
        pkt_count=(8, 200),
    ),
    "vpn": dict(
        n=3000,
        pkt_size_mean=(400, 1400),   # VPN tends to use larger, fixed-size cells
        pkt_size_std=(10, 80),       # more uniform (encryption overhead)
        interarrival_mean=(0.001, 0.05),
        interarrival_std=(0.0005, 0.02),
        incoming_ratio=(0.4, 0.6),
        pkt_count=(10, 300),
    ),
    "attack": dict(
        n=3000,
        pkt_size_mean=(40, 120),     # small probes / SYN packets
        pkt_size_std=(5, 30),
        interarrival_mean=(0.00001, 0.005),   # very fast
        interarrival_std=(0.000001, 0.001),
        incoming_ratio=(0.0, 0.15),  # almost all outgoing (scan)
        pkt_count=(2, 20),           # short abortive flows
    ),
}


def _sample(profile: dict, n: int | None = None) -> pd.DataFrame:
    n = n or profile["n"]
    p = profile

    pkt_mean = RNG.uniform(*p["pkt_size_mean"], n)
    pkt_std  = RNG.uniform(*p["pkt_size_std"],  n)
    iat_mean = RNG.uniform(*p["interarrival_mean"], n)
    iat_std  = RNG.uniform(*p["interarrival_std"],  n)
    pkt_cnt  = RNG.integers(*p["pkt_count"],    n)
    inc_rat  = RNG.uniform(*p["incoming_ratio"], n)

    # derived columns
    pkt_min  = np.clip(pkt_mean - 2 * pkt_std, 20, None)
    pkt_max  = pkt_mean + 2 * pkt_std
    iat_min  = np.clip(iat_mean - 2 * iat_std, 1e-7, None)
    iat_max  = iat_mean + 2 * iat_std
    duration = pkt_cnt * iat_mean + RNG.normal(0, 0.001, n)
    duration = np.clip(duration, 1e-6, None)
    total_bytes = pkt_mean * pkt_cnt + RNG.normal(0, 50, n)
    total_bytes = np.clip(total_bytes, 0, None)

    # add small gaussian noise to prevent perfect separation
    noise = lambda arr, s=0.05: arr * (1 + RNG.normal(0, s, len(arr)))

    return pd.DataFrame({
        "avg_packet_size":      noise(pkt_mean),
        "std_packet_size":      noise(pkt_std),
        "min_packet_size":      noise(pkt_min),
        "max_packet_size":      noise(pkt_max),
        "avg_interarrival":     noise(iat_mean, 0.1),
        "std_interarrival":     noise(iat_std,  0.1),
        "min_interarrival":     noise(iat_min,  0.1),
        "max_interarrival":     noise(iat_max,  0.1),
        "incoming_ratio":       np.clip(noise(inc_rat, 0.05), 0, 1),
        "packet_count":         pkt_cnt,
        "total_bytes":          noise(total_bytes),
        "flow_duration":        noise(duration, 0.15),
        "bytes_per_second":     noise(total_bytes / duration),
        "pkts_per_second":      noise(pkt_cnt   / duration),
    })


def generate(out_path: str | Path = "dataset/raw_csv/synthetic_flows.csv") -> pd.DataFrame:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    for label, profile in PROFILES.items():
        df = _sample(profile)
        df["label"] = label
        frames.append(df)

    data = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=42)
    data.index = [f"flow_{i}" for i in range(len(data))]
    data.index.name = "flow_id"

    data.to_csv(out_path)
    print(f"[synthetic] {len(data)} flows saved → {out_path}")
    print(data["label"].value_counts().to_string())
    return data


if __name__ == "__main__":
    generate()
