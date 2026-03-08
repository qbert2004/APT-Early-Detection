"""
Real-time APT Early-Detection Engine

Architecture:
  PacketCapture (Scapy sniff thread)
      ↓
  FlowBuilder   (accumulates first N packets)
      ↓
  MLClassifier  (loads saved .pkl, predicts class)
      ↓
  AlertEngine   (logs + emits alert events)

Usage:
    # live capture (requires root / npcap on Windows)
    python -m realtime.detector --iface eth0 --n 5

    # replay a pcap file
    python -m realtime.detector --pcap captures/test.pcap --n 5

    # demo mode (generates synthetic packets internally)
    python -m realtime.detector --demo
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from realtime.flow_builder import FlowBuilder, FlowKey
from features.feature_extractor import LABEL_MAP_INV, EARLY_FEATURES
from utils.logger import get_logger

log = get_logger(__name__)

MODELS_DIR = ROOT / "models"
ALERT_LOG   = ROOT / "models" / "alerts.log"

CLASS_COLOR = {
    "normal": "\033[92m",   # green
    "vpn":    "\033[93m",   # yellow
    "attack": "\033[91m",   # red
}
RESET = "\033[0m"


# ── Alert Engine ───────────────────────────────────────────────────────────────

class AlertEngine:
    def __init__(self, threshold: float = 0.65):
        self.threshold   = threshold
        self._callbacks: list[Callable] = []
        self._log_fh     = open(ALERT_LOG, "a", buffering=1)

    def register(self, fn: Callable):
        self._callbacks.append(fn)

    def emit(self, flow_key: FlowKey, label: str, proba: float, features: dict):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        src = f"{flow_key[0]}:{flow_key[1]}"
        dst = f"{flow_key[2]}:{flow_key[3]}"
        color = CLASS_COLOR.get(label, "")

        line = (f"[{ts}] {color}{label.upper():8s}{RESET}"
                f"  prob={proba:.3f}"
                f"  {src} → {dst}"
                f"  pkts={int(features['packet_count'])}"
                f"  bytes={int(features['total_bytes'])}")
        print(line)
        try:
            clean = line.replace("\033[92m","").replace("\033[93m","") \
                        .replace("\033[91m","").replace("\033[0m","")
            self._log_fh.write(clean + "\n")

            if label == "attack" and proba >= self.threshold:
                alert = f"  [!] ALERT: suspicious flow from {src} (confidence {proba:.1%})"
                print(f"\033[91m{alert}{RESET}")
                self._log_fh.write(alert + "\n")
                log.warning("attack detected", src=src, dst=dst, confidence=round(proba, 4))
        except ValueError:
            pass  # file closed during shutdown

        for cb in self._callbacks:
            cb(label=label, proba=proba, flow_key=flow_key, features=features, ts=ts)

    def close(self):
        self._log_fh.close()


# ── ML Classifier wrapper ──────────────────────────────────────────────────────

class MLClassifier:
    def __init__(self, model_path: str | Path | None = None):
        if model_path is None:
            candidates = sorted(MODELS_DIR.glob("best_early_model.pkl"))
            if not candidates:
                candidates = sorted(MODELS_DIR.glob("rf_early.pkl"))
            if not candidates:
                raise FileNotFoundError(
                    "No model found in models/. Run: python -m ml.train_model"
                )
            model_path = candidates[0]

        self.model = joblib.load(model_path)
        print(f"[detector] model loaded: {model_path}")
        log.info("detector model loaded", path=str(model_path))

    def predict(self, features: dict) -> tuple[str, float]:
        """Return (label_str, probability)."""
        row = [features.get(f, 0.0) for f in EARLY_FEATURES]
        X = pd.DataFrame([row], columns=EARLY_FEATURES)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        pred  = int(self.model.predict(X)[0])
        proba = float(self.model.predict_proba(X)[0][pred])
        return LABEL_MAP_INV.get(pred, "unknown"), proba


# ── Packet Capture ─────────────────────────────────────────────────────────────

class PacketCapture:
    """Wraps Scapy sniff for live or pcap replay."""

    def __init__(self, flow_builder: FlowBuilder):
        self._fb = flow_builder

    def _process(self, pkt):
        try:
            from scapy.layers.inet import IP, TCP, UDP
            if not pkt.haslayer(IP):
                return
            ip    = pkt[IP]
            size  = len(pkt)
            ts    = float(pkt.time)
            proto = ip.proto
            if pkt.haslayer(TCP):
                sport, dport = pkt[TCP].sport, pkt[TCP].dport
            elif pkt.haslayer(UDP):
                sport, dport = pkt[UDP].sport, pkt[UDP].dport
            else:
                return
            self._fb.add_packet(ip.src, ip.dst, sport, dport, proto, size, ts)
        except Exception:
            pass

    def live(self, iface: str | None = None, count: int = 0, timeout: int | None = None):
        from scapy.all import sniff
        kwargs = dict(prn=self._process, store=False, count=count)
        if iface:
            kwargs["iface"] = iface
        if timeout:
            kwargs["timeout"] = timeout
        print(f"[capture] sniffing on {iface or 'default'} …  (Ctrl-C to stop)")
        sniff(**kwargs)

    def from_pcap(self, path: str | Path):
        from scapy.all import rdpcap
        pkts = rdpcap(str(path))
        print(f"[capture] replaying {len(pkts)} packets from {path}")
        for pkt in pkts:
            self._process(pkt)
        time.sleep(0.5)   # let flow_builder flush callbacks


# ── Demo Mode ──────────────────────────────────────────────────────────────────

def _demo_loop(flow_builder: FlowBuilder, stop: threading.Event):
    """
    Inject synthetic packets to simulate traffic without needing root or pcap.
    Pattern: 5 s normal → 3 s attack → 5 s normal → ...
    """
    import random
    rng = random.Random(1)
    base_ts = time.time()
    flow_counter = 0

    PROFILES = {
        "normal": dict(size=(200, 1400), iat=(0.05, 0.5),  incoming=0.5),
        "attack": dict(size=(40,  120),  iat=(0.0001, 0.005), incoming=0.05),
        "vpn":    dict(size=(800, 1400), iat=(0.01, 0.05),   incoming=0.45),
    }
    schedule = [
        ("normal", 6), ("attack", 4), ("normal", 5),
        ("vpn",    5), ("attack", 3), ("normal", 6),
    ]

    ts = base_ts
    for profile_name, duration_sec in schedule:
        if stop.is_set():
            break
        p = PROFILES[profile_name]
        end_ts = ts + duration_sec
        while ts < end_ts:
            if stop.is_set():
                return
            flow_counter += 1
            src_ip  = f"10.0.0.{rng.randint(1, 50)}"
            dst_ip  = f"192.168.1.{rng.randint(1, 10)}"
            src_prt = rng.randint(10000, 60000)
            dst_prt = rng.choice([80, 443, 22, 8080, 3389])
            for _ in range(5):   # 5 packets per mini-flow
                size = rng.randint(*p["size"])
                flow_builder.add_packet(
                    src_ip, dst_ip, src_prt, dst_prt, 6,
                    size, ts
                )
                ts += rng.uniform(*p["iat"])
            time.sleep(0.05)   # slow down so output is readable


# ── Main Detector ──────────────────────────────────────────────────────────────

class Detector:
    def __init__(
        self,
        n_packets: int = 5,
        model_path: str | Path | None = None,
        alert_threshold: float = 0.65,
    ):
        self.clf    = MLClassifier(model_path)
        self.alerts = AlertEngine(threshold=alert_threshold)

        self.fb = FlowBuilder(
            n_packets=n_packets,
            flow_timeout_sec=30.0,
            on_ready=self._on_flow_ready,
        )
        self.capture = PacketCapture(self.fb)

        # shared queue so on_ready callback (capture thread) hands off to
        # main thread for inference (avoids GIL contention on heavy models)
        self._q: queue.Queue = queue.Queue(maxsize=500)
        self._inf_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._inf_thread.start()

        # stats
        self.counts: dict[str, int] = {"normal": 0, "vpn": 0, "attack": 0}

    def _on_flow_ready(self, flow_key: FlowKey, features: dict):
        try:
            self._q.put_nowait((flow_key, features))
        except queue.Full:
            pass   # drop if overwhelmed

    def _inference_loop(self):
        while True:
            flow_key, features = self._q.get()
            label, proba = self.clf.predict(features)
            self.counts[label] = self.counts.get(label, 0) + 1
            self.alerts.emit(flow_key, label, proba, features)

    def run_demo(self, duration: int = 30):
        print(f"[detector] DEMO MODE — running for {duration}s")
        stop = threading.Event()
        t = threading.Thread(target=_demo_loop, args=(self.fb, stop), daemon=True)
        t.start()
        time.sleep(duration)
        stop.set()
        t.join(timeout=2)
        self._print_summary()

    def run_live(self, iface: str | None = None, duration: int | None = None):
        print("[detector] LIVE CAPTURE — press Ctrl-C to stop")
        try:
            self.capture.live(iface=iface, timeout=duration)
        except KeyboardInterrupt:
            pass
        self._print_summary()

    def run_pcap(self, pcap_path: str | Path):
        self.capture.from_pcap(pcap_path)
        time.sleep(1)
        self._print_summary()

    def _print_summary(self):
        total = sum(self.counts.values())
        print(f"\n{'─'*40}")
        print(f"  Flows classified: {total}")
        for label, n in self.counts.items():
            pct = 100 * n / max(total, 1)
            print(f"  {label:10s}: {n:5d}  ({pct:.1f}%)")
        print(f"{'─'*40}")
        print(f"  Alert log: {ALERT_LOG}")

    def stop(self):
        self.fb.stop()
        self.alerts.close()


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APT Early-Detection Detector")
    parser.add_argument("--iface",  default=None,  help="Network interface for live capture")
    parser.add_argument("--pcap",   default=None,  help="Replay a .pcap file")
    parser.add_argument("--demo",   action="store_true", help="Run synthetic demo")
    parser.add_argument("--n",      default=5,  type=int,   help="Early packet count")
    parser.add_argument("--model",  default=None, help="Path to .pkl model")
    parser.add_argument("--duration", default=30, type=int, help="Demo/live duration (sec)")
    parser.add_argument("--threshold", default=0.65, type=float, help="Alert confidence threshold")
    args = parser.parse_args()

    det = Detector(
        n_packets=args.n,
        model_path=args.model,
        alert_threshold=args.threshold,
    )

    if args.demo:
        det.run_demo(duration=args.duration)
    elif args.pcap:
        det.run_pcap(args.pcap)
    elif args.iface:
        det.run_live(iface=args.iface, duration=args.duration)
    else:
        # default: demo mode
        det.run_demo(duration=args.duration)

    det.stop()
