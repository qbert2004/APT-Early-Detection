"""
ISCX PCAP → Early-Flow Feature Extractor

Reads pcap files directly from zip archives (no full extraction needed).
Uses streaming PcapReader — works even for 5 GB+ individual files.
Groups packets into flows by 5-tuple, takes the FIRST N packets of each flow,
computes statistical features — this is the TRUE early-flow classification.

Sources (all 5 zips):
  VPN-PCAPS-01.zip       → label = vpn
  VPN-PCAPs-02.zip       → label = vpn
  NonVPN-PCAPs-01.zip    → label = normal
  NonVPN-PCAPs-02.zip    → label = normal
  NonVPN-PCAPs-03.zip    → label = normal

Output:
  dataset/raw_csv/iscx_pcap_early.csv

Usage:
    python -m data.extract_from_pcap
    python -m data.extract_from_pcap --n 10 --max-flows 5000
    python -m data.extract_from_pcap --max-flows-per-zip 15000
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

# 7-Zip executable — used as fallback for Deflate64 (compress_type=9)
SEVENZIP = r"C:\Program Files\7-Zip\7z.exe"

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "dataset" / "raw_csv" / "iscx_pcap_early.csv"
DOWNLOADS = Path("C:/Users/Nitro V15/Downloads")

# All 5 zip sources with their class labels
SOURCES = [
    ("VPN-PCAPS-01.zip",    "vpn"),
    ("VPN-PCAPs-02.zip",    "vpn"),
    ("NonVPN-PCAPs-01.zip", "normal"),
    ("NonVPN-PCAPs-02.zip", "normal"),
    ("NonVPN-PCAPs-03.zip", "normal"),
]

# Skip individual pcap files larger than this (uncompressed bytes)
# FTP captures like ftps_down_1a.pcap (5.5 GB) = one big file transfer = not useful
MAX_FILE_MB = 900

# Max total flows to extract per zip (keeps dataset balanced)
MAX_FLOWS_PER_ZIP = 12000


def _stream_flows(zf_stream, label: str, n_packets: int, max_flows: int) -> list[dict]:
    """Stream packets from a file-like object, extract first-N-packet flow features.
    Uses PcapReader so memory usage stays low regardless of file size.
    """
    try:
        from scapy.all import PcapReader
        from scapy.layers.inet import IP, TCP, UDP
    except ImportError:
        raise ImportError("pip install scapy")

    flows: dict[tuple, list[dict]] = defaultdict(list)
    completed: list[dict] = []

    try:
        reader = PcapReader(zf_stream)
        for pkt in reader:
            if len(completed) >= max_flows:
                break
            try:
                if not pkt.haslayer(IP):
                    continue
                ip    = pkt[IP]
                proto = ip.proto
                if pkt.haslayer(TCP):
                    sport, dport = pkt[TCP].sport, pkt[TCP].dport
                elif pkt.haslayer(UDP):
                    sport, dport = pkt[UDP].sport, pkt[UDP].dport
                else:
                    continue

                a, b   = (ip.src, sport), (ip.dst, dport)
                lo, hi = (a, b) if a <= b else (b, a)
                key    = (*lo, *hi, proto)

                buf = flows[key]
                if len(buf) >= n_packets:
                    continue

                buf.append({
                    "size":      len(pkt),
                    "time":      float(pkt.time),
                    "direction": 1 if a == lo else 0,
                })

                # flow complete → compute features immediately (frees buffer)
                if len(buf) == n_packets:
                    sizes = np.array([p["size"]      for p in buf], dtype=float)
                    times = np.array([p["time"]      for p in buf], dtype=float)
                    dirs  = np.array([p["direction"] for p in buf], dtype=float)
                    iats  = np.diff(times)
                    dur   = max(times[-1] - times[0], 1e-9)

                    completed.append({
                        "avg_packet_size":  sizes.mean(),
                        "std_packet_size":  sizes.std(),
                        "min_packet_size":  sizes.min(),
                        "max_packet_size":  sizes.max(),
                        "avg_interarrival": iats.mean() if len(iats) else 0.0,
                        "std_interarrival": iats.std()  if len(iats) else 0.0,
                        "min_interarrival": iats.min()  if len(iats) else 0.0,
                        "max_interarrival": iats.max()  if len(iats) else 0.0,
                        "incoming_ratio":   dirs.mean(),
                        "packet_count":     float(len(buf)),
                        "total_bytes":      sizes.sum(),
                        "flow_duration":    dur,
                        "bytes_per_second": sizes.sum() / dur,
                        "pkts_per_second":  len(buf) / dur,
                        "label":            label,
                    })
                    del flows[key]  # free memory

            except Exception:
                continue
    except Exception as e:
        print(f"    [warn] reader error: {e}")

    # also include flows that accumulated < n_packets but >= 2
    for key, buf in flows.items():
        if len(completed) >= max_flows:
            break
        if len(buf) < 2:
            continue
        sizes = np.array([p["size"]      for p in buf], dtype=float)
        times = np.array([p["time"]      for p in buf], dtype=float)
        dirs  = np.array([p["direction"] for p in buf], dtype=float)
        iats  = np.diff(times)
        dur   = max(times[-1] - times[0], 1e-9)
        completed.append({
            "avg_packet_size":  sizes.mean(),
            "std_packet_size":  sizes.std(),
            "min_packet_size":  sizes.min(),
            "max_packet_size":  sizes.max(),
            "avg_interarrival": iats.mean() if len(iats) else 0.0,
            "std_interarrival": iats.std()  if len(iats) else 0.0,
            "min_interarrival": iats.min()  if len(iats) else 0.0,
            "max_interarrival": iats.max()  if len(iats) else 0.0,
            "incoming_ratio":   dirs.mean(),
            "packet_count":     float(len(buf)),
            "total_bytes":      sizes.sum(),
            "flow_duration":    dur,
            "bytes_per_second": sizes.sum() / dur,
            "pkts_per_second":  len(buf) / dur,
            "label":            label,
        })

    return completed


def _extract_with_7zip(
    zip_path: Path,
    entry_name: str,
    label: str,
    n_packets: int,
    max_flows: int,
) -> list[dict] | None:
    """Extract a single file from a Deflate64 ZIP using 7-Zip, stream flows, clean up."""
    if not os.path.exists(SEVENZIP):
        print(f"SKIP (7-Zip not found at {SEVENZIP})")
        return None

    tmp_dir = tempfile.mkdtemp(prefix="iscx_pcap_")
    try:
        result = subprocess.run(
            [SEVENZIP, "e", str(zip_path), entry_name, f"-o{tmp_dir}", "-y"],
            capture_output=True, timeout=300,
        )
        if result.returncode != 0:
            print(f"SKIP (7z error: {result.stderr[:120]})")
            return None

        # find extracted file
        fname = os.path.basename(entry_name)
        extracted = os.path.join(tmp_dir, fname)
        if not os.path.exists(extracted):
            print(f"SKIP (7z: file not found after extraction)")
            return None

        print(f"[7z→stream]", end=" ", flush=True)
        with open(extracted, "rb") as fh:
            rows = _stream_flows(fh, label, n_packets, max_flows)
        return rows
    except subprocess.TimeoutExpired:
        print(f"SKIP (7z timeout)")
        return None
    except Exception as e:
        print(f"SKIP (7z exception: {e})")
        return None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def extract(
    downloads_dir: str | Path | None = None,
    n_packets: int = 5,
    max_flows_per_zip: int = MAX_FLOWS_PER_ZIP,
) -> pd.DataFrame:
    dl = Path(downloads_dir) if downloads_dir else DOWNLOADS
    OUT.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    for zip_name, label in SOURCES:
        zip_path = dl / zip_name
        if not zip_path.exists():
            print(f"[skip] {zip_name} — not found in {dl}")
            continue

        zip_size_mb = zip_path.stat().st_size // 1024 // 1024
        print(f"\n[zip] {zip_name}  ({zip_size_mb:,} MB)  label={label}")

        zip_rows: list[dict] = []

        with zipfile.ZipFile(zip_path) as zf:
            pcap_infos = [i for i in zf.infolist()
                          if i.filename.lower().endswith((".pcap", ".pcapng"))]
            # process smallest files first for faster early results
            pcap_infos.sort(key=lambda i: i.file_size)

            for info in pcap_infos:
                if len(zip_rows) >= max_flows_per_zip:
                    break

                file_mb = info.file_size // 1024 // 1024
                if file_mb > MAX_FILE_MB:
                    print(f"  [skip] {info.filename}  ({file_mb:,} MB) — over {MAX_FILE_MB} MB limit")
                    continue

                remaining = max_flows_per_zip - len(zip_rows)
                print(f"  [read] {info.filename}  ({file_mb:,} MB)", end=" ... ", flush=True)

                try:
                    with zf.open(info.filename) as stream:
                        rows = _stream_flows(stream, label, n_packets, remaining)
                except NotImplementedError:
                    # Deflate64 (compress_type=9) — Python zipfile unsupported.
                    # Extract to temp dir with 7-Zip, then stream from disk.
                    rows = _extract_with_7zip(
                        zip_path, info.filename, label, n_packets, remaining
                    )
                    if rows is None:
                        continue
                except Exception as e:
                    print(f"SKIP ({e})")
                    continue

                zip_rows.extend(rows)
                print(f"{len(rows)} flows  (zip total: {len(zip_rows)})")

        all_rows.extend(zip_rows)
        print(f"  => {len(zip_rows)} flows from {zip_name}")

    if not all_rows:
        raise RuntimeError("No flows extracted. Check zip paths and scapy installation.")

    df = pd.DataFrame(all_rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.index.name = "flow_id"

    num_cols = [c for c in df.columns if c != "label"]
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    df.to_csv(OUT)
    print(f"\n[done]  {len(df):,} flows  →  {OUT}")
    print("Label distribution:")
    print(df["label"].value_counts().to_string())
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--downloads",        default=None)
    parser.add_argument("--n",                default=5,             type=int,
                        help="Early packet count (default: 5)")
    parser.add_argument("--max-flows-per-zip", default=MAX_FLOWS_PER_ZIP, type=int,
                        help=f"Max flows per zip file (default: {MAX_FLOWS_PER_ZIP})")
    args = parser.parse_args()

    extract(
        downloads_dir=args.downloads,
        n_packets=args.n,
        max_flows_per_zip=args.max_flows_per_zip,
    )
