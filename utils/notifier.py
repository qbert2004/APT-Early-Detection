"""
Telegram Alert Notifier
=======================
Sends an alert message to a Telegram chat when a high-confidence
ATTACK flow is detected.

Setup (one-time):
  1. Create a bot via @BotFather → copy the token
  2. Send any message to your bot, then:
     https://api.telegram.org/bot<TOKEN>/getUpdates  → copy chat_id
  3. Add to .env:
       TELEGRAM_BOT_TOKEN=123456:ABC...
       TELEGRAM_CHAT_ID=-100123456789

The notifier silently disables itself if env vars are missing.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Optional

import requests

from utils.logger import get_logger

log = get_logger(__name__)

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"

# severity → emoji
_EMOJI = {
    "attack": "🚨",
    "vpn":    "🔒",
    "normal": "✅",
}


class TelegramNotifier:
    """
    Fire-and-forget Telegram notifier.
    All sends happen in a background thread so the detection loop
    is never blocked by network latency.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
        min_interval_sec: float = 5.0,
        threshold: float = 0.75,
    ):
        self.token    = token    or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id  = chat_id  or os.getenv("TELEGRAM_CHAT_ID",   "")
        self.threshold       = threshold
        self.min_interval    = min_interval_sec
        self._last_sent: float = 0.0
        self._enabled = bool(self.token and self.chat_id)

        if self._enabled:
            log.info("Telegram notifier enabled", chat_id=self.chat_id)
        else:
            log.warning(
                "Telegram notifier disabled — set TELEGRAM_BOT_TOKEN and "
                "TELEGRAM_CHAT_ID in .env to enable"
            )

    # ── public API ──────────────────────────────────────────────────────────

    def notify_attack(
        self,
        src: str,
        dst: str,
        prob: float,
        ip_risk: Optional[int] = None,
        features: Optional[dict] = None,
    ) -> None:
        """Send an alert if enabled and confidence >= threshold."""
        if not self._enabled or prob < self.threshold:
            return

        # Rate-limit: at most one message every min_interval seconds
        now = time.time()
        if now - self._last_sent < self.min_interval:
            return
        self._last_sent = now

        msg = self._build_message(src, dst, prob, ip_risk, features)
        t = threading.Thread(
            target=self._send, args=(msg,), daemon=True, name="tg-alert"
        )
        t.start()

    def send_startup(self, model_names: list[str]) -> None:
        """Optional: notify that the detector has started."""
        if not self._enabled:
            return
        msg = (
            "🛡 *APT Early-Detection started*\n"
            f"Models: `{', '.join(model_names)}`\n"
            f"Alert threshold: `{self.threshold:.0%}`"
        )
        threading.Thread(
            target=self._send, args=(msg,), daemon=True, name="tg-startup"
        ).start()

    # ── internals ───────────────────────────────────────────────────────────

    def _build_message(
        self,
        src: str,
        dst: str,
        prob: float,
        ip_risk: Optional[int],
        features: Optional[dict],
    ) -> str:
        lines = [
            "🚨 *APT ATTACK DETECTED*",
            f"• Source:      `{src}`",
            f"• Destination: `{dst}`",
            f"• ML Confidence: `{prob:.1%}`",
        ]

        if ip_risk is not None:
            risk_label = (
                "🔴 HIGH"   if ip_risk >= 75 else
                "🟡 MEDIUM" if ip_risk >= 25 else
                "🟢 LOW"
            )
            lines.append(f"• IP Reputation: `{ip_risk}/100` {risk_label}")

        if features:
            pkts  = int(features.get("packet_count", 0))
            byt   = int(features.get("total_bytes", 0))
            bps   = int(features.get("bytes_per_second", 0))
            lines.append(f"• Packets: `{pkts}`  Bytes: `{byt}`  BPS: `{bps}`")

        lines.append(f"\n_Time: {time.strftime('%Y-%m-%d %H:%M:%S')}_")
        return "\n".join(lines)

    def _send(self, text: str) -> None:
        url = _TELEGRAM_API.format(token=self.token)
        payload = {
            "chat_id":    self.chat_id,
            "text":       text,
            "parse_mode": "Markdown",
        }
        try:
            r = requests.post(url, json=payload, timeout=8)
            if r.status_code == 200:
                log.info("Telegram alert sent")
            else:
                log.warning("Telegram send failed", status=r.status_code, body=r.text[:200])
        except requests.RequestException as exc:
            log.error("Telegram request error", error=str(exc))
