"""
Standalone packet capture utilities.

  capture_to_pcap(iface, duration, out_file)  — save traffic to pcap
  replay_pcap(pcap_file, callback)            — feed pcap packets to callback

These are thin wrappers used by detector.py and attack_sim.py.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable


def capture_to_pcap(
    iface: str,
    duration: int,
    out_file: str | Path = "captures/capture.pcap",
):
    """Capture live traffic and save to pcap (requires root/npcap)."""
    try:
        from scapy.all import sniff, wrpcap
    except ImportError:
        raise ImportError("scapy not installed: pip install scapy")

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"[capture] capturing on {iface} for {duration}s → {out_file}")
    packets = sniff(iface=iface, timeout=duration, store=True)
    wrpcap(str(out_file), packets)
    print(f"[capture] {len(packets)} packets saved")
    return out_file


def replay_pcap(pcap_file: str | Path, callback: Callable):
    """Read a pcap and call callback(pkt) for each packet."""
    try:
        from scapy.all import rdpcap
    except ImportError:
        raise ImportError("scapy not installed: pip install scapy")

    pkts = rdpcap(str(pcap_file))
    print(f"[replay] {len(pkts)} packets from {pcap_file}")
    for pkt in pkts:
        callback(pkt)
    time.sleep(0.2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Packet capture helper")
    parser.add_argument("--iface",    required=True)
    parser.add_argument("--duration", default=10, type=int)
    parser.add_argument("--out",      default="captures/capture.pcap")
    args = parser.parse_args()

    capture_to_pcap(args.iface, args.duration, args.out)
