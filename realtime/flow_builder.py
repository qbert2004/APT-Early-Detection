"""
Online Flow Builder

Accumulates raw packets into flows keyed by 5-tuple.
When a flow reaches `n_packets` it is "ready" and its features
are computed by FlowBuilder.extract_features().

Thread-safe: uses a lock so the detector thread and the capture
callback can coexist safely.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Callable

import numpy as np

# flow key: (src_ip, dst_ip, src_port, dst_port, proto)
FlowKey = tuple


class RawPacket:
    __slots__ = ("size", "timestamp", "direction")

    def __init__(self, size: int, timestamp: float, direction: int):
        self.size      = size       # bytes
        self.timestamp = timestamp  # epoch float
        self.direction = direction  # 1 = forward (src→dst), 0 = reverse


class FlowBuilder:
    """
    Collects up to `n_packets` raw packets per flow,
    then fires `on_ready(flow_key, features_dict)`.
    """

    def __init__(
        self,
        n_packets: int = 5,
        flow_timeout_sec: float = 30.0,
        on_ready: Callable | None = None,
    ):
        self.n_packets       = n_packets
        self.flow_timeout    = flow_timeout_sec
        self.on_ready        = on_ready or (lambda k, f: None)

        self._lock: threading.Lock = threading.Lock()
        self._flows: dict[FlowKey, list[RawPacket]] = defaultdict(list)
        self._last_seen: dict[FlowKey, float] = {}

        # start background reaper
        self._stop_event = threading.Event()
        self._reaper = threading.Thread(target=self._reap_old_flows, daemon=True)
        self._reaper.start()

    # ── public ────────────────────────────────────────────────────────────────

    def add_packet(
        self,
        src_ip: str, dst_ip: str,
        src_port: int, dst_port: int,
        proto: int,
        size: int,
        timestamp: float | None = None,
    ):
        timestamp = timestamp or time.time()
        # canonical bidirectional key
        key = self._make_key(src_ip, dst_ip, src_port, dst_port, proto)
        direction = 1 if (src_ip, src_port) < (dst_ip, dst_port) else 0
        pkt = RawPacket(size=size, timestamp=timestamp, direction=direction)

        ready_features = None
        with self._lock:
            self._last_seen[key] = timestamp
            buf = self._flows[key]
            if len(buf) >= self.n_packets:
                return  # already processed

            buf.append(pkt)
            if len(buf) == self.n_packets:
                ready_features = self._extract(buf)
                del self._flows[key]
                del self._last_seen[key]

        # fire callback outside lock to avoid deadlock
        if ready_features is not None:
            self.on_ready(key, ready_features)

    def stop(self):
        self._stop_event.set()

    # ── internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _make_key(src_ip, dst_ip, src_port, dst_port, proto) -> FlowKey:
        a = (src_ip, src_port)
        b = (dst_ip, dst_port)
        lo, hi = (a, b) if a <= b else (b, a)
        return (*lo, *hi, proto)

    @staticmethod
    def _extract(packets: list[RawPacket]) -> dict:
        sizes = np.array([p.size      for p in packets], dtype=float)
        times = np.array([p.timestamp for p in packets], dtype=float)
        dirs  = np.array([p.direction for p in packets], dtype=float)

        iats = np.diff(times)
        dur  = times[-1] - times[0] if len(times) > 1 else 1e-9

        return {
            "avg_packet_size":  float(sizes.mean()),
            "std_packet_size":  float(sizes.std()),
            "min_packet_size":  float(sizes.min()),
            "max_packet_size":  float(sizes.max()),
            "avg_interarrival": float(iats.mean()) if len(iats) else 0.0,
            "std_interarrival": float(iats.std())  if len(iats) else 0.0,
            "min_interarrival": float(iats.min())  if len(iats) else 0.0,
            "max_interarrival": float(iats.max())  if len(iats) else 0.0,
            "incoming_ratio":   float(dirs.mean()),
            "packet_count":     len(packets),
            "total_bytes":      float(sizes.sum()),
            "flow_duration":    float(dur),
            "bytes_per_second": float(sizes.sum() / max(dur, 1e-9)),
            "pkts_per_second":  float(len(packets) / max(dur, 1e-9)),
        }

    def _reap_old_flows(self):
        """Remove stale flows that never reached n_packets."""
        while not self._stop_event.wait(timeout=5.0):
            now = time.time()
            with self._lock:
                stale = [k for k, t in self._last_seen.items()
                         if now - t > self.flow_timeout]
                for k in stale:
                    self._flows.pop(k, None)
                    self._last_seen.pop(k, None)
