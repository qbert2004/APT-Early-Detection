"""
ISCX VPN-nonVPN ARFF → project CSV converter

Reads from:
  Scenario A1-ARFF.zip  (VPN vs Non-VPN)
  Scenario A2-ARFF.zip  (VPN vs Non-VPN, both dirs)
  Scenario B-ARFF.zip   (app types: BROWSING, CHAT, STREAMING, …)

Produces:
  dataset/raw_csv/iscx_real.csv   (merged, mapped to our feature schema)

Label mapping:
  VPN          → vpn
  Non-VPN      → normal
  BROWSING,
  CHAT, MAIL,
  STREAMING,
  VOIP, P2P,
  FT           → normal   (legitimate app traffic = normal)

Usage:
    python -m data.convert_iscx_arff
    python -m data.convert_iscx_arff --downloads "C:/Users/You/Downloads"
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
OUT_PATH = ROOT / "dataset" / "raw_csv" / "iscx_real.csv"

DOWNLOADS = Path("C:/Users/Nitro V15/Downloads")

ZIPS = [
    "Scenario A1-ARFF.zip",
    "Scenario A2-ARFF.zip",
    "Scenario B-ARFF.zip",
]

# ISCX IAT values are in MICROSECONDS — convert to seconds
US = 1e6

LABEL_MAP = {
    "vpn":       "vpn",
    "non-vpn":   "normal",
    "browsing":  "normal",
    "chat":      "normal",
    "streaming": "normal",
    "mail":      "normal",
    "voip":      "normal",
    "p2p":       "normal",
    "ft":        "normal",
}


def _parse_arff(raw: bytes) -> pd.DataFrame:
    """Parse a (malformed) ISCX ARFF file into a DataFrame."""
    text = raw.decode("utf-8", errors="ignore")
    lines = text.splitlines()

    attrs = []
    data_start = False
    rows = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith("%"):
            continue

        upper = line.upper()

        if upper.startswith("@ATTRIBUTE"):
            # @ATTRIBUTE duration NUMERIC,,,,  (trailing commas — strip them)
            parts = line.rstrip(",").split()
            if len(parts) >= 2:
                attrs.append(parts[1].lower())
            continue

        if upper.startswith("@DATA"):
            data_start = True
            continue

        if data_start and line:
            # strip trailing commas, split
            values = line.rstrip(",").split(",")
            rows.append(values)

    if not attrs or not rows:
        return pd.DataFrame()

    # make sure row lengths match
    n = len(attrs)
    clean_rows = [r[:n] if len(r) >= n else r + [""] * (n - len(r)) for r in rows]
    df = pd.DataFrame(clean_rows, columns=attrs)
    return df


def _map_features(df: pd.DataFrame, raw_label: str) -> pd.DataFrame:
    """Convert ISCX columns → our feature schema."""
    out = pd.DataFrame()

    def col(name):
        return pd.to_numeric(df[name], errors="coerce") if name in df.columns \
               else pd.Series(np.nan, index=df.index)

    # Duration: microseconds → seconds
    dur_us  = col("duration")
    dur_sec = (dur_us / US).replace(0, np.nan)

    # IAT: microseconds → seconds
    out["avg_interarrival"] = col("mean_flowiat") / US
    out["std_interarrival"] = col("std_flowiat")  / US
    out["min_interarrival"] = col("min_flowiat")  / US
    out["max_interarrival"] = col("max_flowiat")  / US

    # Rates
    out["pkts_per_second"]  = col("flowpktspersecond")
    out["bytes_per_second"] = col("flowbytespersecond")

    # Duration
    out["flow_duration"] = dur_sec

    # Packet count and total bytes (derived)
    out["packet_count"] = (out["pkts_per_second"] * dur_sec).clip(lower=0)
    out["total_bytes"]  = (out["bytes_per_second"] * dur_sec).clip(lower=0)

    # Packet size (derived from bytes / packets)
    pkt = out["packet_count"].replace(0, np.nan)
    out["avg_packet_size"] = out["total_bytes"] / pkt
    out["std_packet_size"] = (col("mean_fiat") / US) * 1000   # proxy: scaled fwd IAT
    out["min_packet_size"] = out["avg_packet_size"] * 0.5
    out["max_packet_size"] = out["avg_packet_size"] * 1.5

    # Direction ratio: forward IAT share (proxy for incoming_ratio)
    fwd = col("total_fiat").clip(lower=0)
    bwd = col("total_biat").clip(lower=0)
    total_iat = (fwd + bwd).replace(0, np.nan)
    out["incoming_ratio"] = (bwd / total_iat).clip(0, 1)

    # Label
    lc = df[raw_label].str.strip().str.lower()
    out["label"] = lc.map(LABEL_MAP)

    # Drop rows with unknown label or all-NaN features
    out = out[out["label"].notna()].copy()
    out = out.replace([np.inf, -np.inf], np.nan)

    return out


def convert(downloads_dir: str | Path | None = None) -> pd.DataFrame:
    dl = Path(downloads_dir) if downloads_dir else DOWNLOADS
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_frames = []
    total_files = 0

    for zip_name in ZIPS:
        zip_path = dl / zip_name
        if not zip_path.exists():
            print(f"[skip] {zip_name} not found in {dl}")
            continue

        with zipfile.ZipFile(zip_path) as zf:
            arff_files = [n for n in zf.namelist() if n.endswith(".arff")]
            # prefer 15s time-window files for richer stats; use all if not found
            preferred = [n for n in arff_files if "15s" in n]
            selected  = preferred if preferred else arff_files

            for arff_name in selected:
                raw = zf.read(arff_name)
                df  = _parse_arff(raw)
                if df.empty:
                    print(f"  [warn] empty parse: {arff_name}")
                    continue

                # find label column (last @ATTRIBUTE that's nominal)
                label_col = df.columns[-1]
                mapped = _map_features(df, label_col)
                if mapped.empty:
                    continue

                all_frames.append(mapped)
                total_files += 1
                print(f"  [ok] {arff_name}  "
                      f"rows={len(mapped)}  "
                      f"labels={mapped['label'].value_counts().to_dict()}")

    if not all_frames:
        raise RuntimeError("No ARFF files found or parsed. Check your Downloads folder.")

    merged = pd.concat(all_frames, ignore_index=True)
    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)
    merged.index.name = "flow_id"

    # final cleanup
    num_cols = [c for c in merged.columns if c != "label"]
    merged[num_cols] = merged[num_cols].apply(pd.to_numeric, errors="coerce")
    merged[num_cols] = merged[num_cols].fillna(merged[num_cols].median())

    merged.to_csv(OUT_PATH)
    print(f"\n[done] {total_files} files  {len(merged)} rows → {OUT_PATH}")
    print("\nLabel distribution:")
    print(merged["label"].value_counts().to_string())
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--downloads", default=None,
                        help="Path to Downloads folder with ISCX zip files")
    args = parser.parse_args()
    convert(args.downloads)
