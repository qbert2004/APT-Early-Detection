"""
Threat Intelligence — IP Reputation via AbuseIPDB
==================================================
Enriches detected flows with an external IP reputation score so the
system can combine ML confidence + threat-intel into a hybrid risk score:

    final_risk = 0.7 * ml_score + 0.3 * (abuse_score / 100)

Free tier: 1 000 checks/day — enough for a demo or low-volume deployment.

Setup:
  1. Register at https://www.abuseipdb.com  (free)
  2. API → Create key → copy
  3. Add to .env:
       ABUSEIPDB_API_KEY=your_key_here

The module is silently disabled when the key is absent — all lookups
return None and the rest of the pipeline is unaffected.

Local IP ranges (RFC 1918 / loopback) are skipped automatically.
Results are cached in-memory (TTL = 1 h) to save quota.
"""
from __future__ import annotations

import ipaddress
import os
import time
from functools import lru_cache
from typing import Optional

import requests

from utils.logger import get_logger

log = get_logger(__name__)

_ABUSEIPDB_URL = "https://api.abuseipdb.com/api/v2/check"
_CACHE_TTL_SEC = 3600          # 1 hour per IP
_TIMEOUT_SEC   = 4             # don't block detection loop


# ── private ranges to skip ─────────────────────────────────────────────────────

_PRIVATE_NETS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
]


def _is_private(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
        return any(addr in net for net in _PRIVATE_NETS)
    except ValueError:
        return True   # malformed → treat as private, skip lookup


# ── in-memory TTL cache ─────────────────────────────────────────────────────────

_cache: dict[str, tuple[Optional[int], float]] = {}   # ip → (score, expiry_ts)


class ThreatIntel:
    """
    IP reputation checker using AbuseIPDB.

    Usage::

        ti = ThreatIntel()
        score = ti.check("1.2.3.4")   # 0-100 or None
        risk  = ti.hybrid_risk(ml_prob=0.85, src_ip="1.2.3.4")
    """

    # weights for the hybrid score
    ML_WEIGHT    = 0.70
    INTEL_WEIGHT = 0.30

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ABUSEIPDB_API_KEY", "")
        self._enabled = bool(self.api_key)
        if self._enabled:
            log.info("ThreatIntel enabled (AbuseIPDB)")
        else:
            log.warning(
                "ThreatIntel disabled — set ABUSEIPDB_API_KEY in .env to enable"
            )

    # ── public API ──────────────────────────────────────────────────────────

    def check(self, ip: str) -> Optional[int]:
        """
        Return AbuseIPDB confidence score (0–100) or None.

        None is returned when:
        - API key not configured
        - IP is a private/loopback address
        - Network error or API limit reached
        """
        if not self._enabled:
            return None
        if _is_private(ip):
            return None

        # strip port if present (e.g. "1.2.3.4:48000")
        ip = ip.split(":")[0]

        # check cache
        if ip in _cache:
            score, expiry = _cache[ip]
            if time.time() < expiry:
                return score

        score = self._fetch(ip)
        _cache[ip] = (score, time.time() + _CACHE_TTL_SEC)
        return score

    def hybrid_risk(self, ml_prob: float, src_ip: str) -> dict:
        """
        Combine ML probability with IP reputation into a single risk score.

        Returns a dict::

            {
              "ml_score":      0.85,
              "ip_risk":       72,          # AbuseIPDB score, or None
              "hybrid_score":  0.811,       # weighted combination
              "enriched":      True,        # False if TI unavailable
            }
        """
        ip_score = self.check(src_ip)

        if ip_score is not None:
            hybrid = (
                self.ML_WEIGHT    * ml_prob +
                self.INTEL_WEIGHT * (ip_score / 100.0)
            )
            enriched = True
        else:
            hybrid   = ml_prob
            enriched = False

        return {
            "ml_score":     round(ml_prob, 4),
            "ip_risk":      ip_score,
            "hybrid_score": round(hybrid, 4),
            "enriched":     enriched,
        }

    def risk_label(self, hybrid_score: float) -> str:
        """Human-readable risk tier."""
        if hybrid_score >= 0.80:
            return "CRITICAL"
        if hybrid_score >= 0.65:
            return "HIGH"
        if hybrid_score >= 0.45:
            return "MEDIUM"
        return "LOW"

    # ── internals ───────────────────────────────────────────────────────────

    def _fetch(self, ip: str) -> Optional[int]:
        headers = {"Key": self.api_key, "Accept": "application/json"}
        params  = {"ipAddress": ip, "maxAgeInDays": 90, "verbose": ""}
        try:
            r = requests.get(
                _ABUSEIPDB_URL,
                headers=headers,
                params=params,
                timeout=_TIMEOUT_SEC,
            )
            if r.status_code == 200:
                score = int(r.json()["data"]["abuseConfidenceScore"])
                log.info("IP reputation fetched", ip=ip, score=score)
                return score
            elif r.status_code == 429:
                log.warning("AbuseIPDB rate limit reached")
            else:
                log.warning("AbuseIPDB error", status=r.status_code)
        except requests.RequestException as exc:
            log.error("AbuseIPDB request error", error=str(exc))
        return None
