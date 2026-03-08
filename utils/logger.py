"""
Structured JSON logger for APT Early Detection.

Usage:
    from utils.logger import get_logger
    log = get_logger(__name__)
    log.info("model loaded", model="rf_early", features=14)
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        extra = {k: v for k, v in record.__dict__.items()
                 if k not in logging.LogRecord.__dict__ and not k.startswith("_")}
        payload: dict[str, Any] = {
            "ts":      datetime.now(timezone.utc).isoformat(),
            "level":   record.levelname,
            "logger":  record.name,
            "message": record.getMessage(),
            **extra,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class _KwLogger(logging.LoggerAdapter):
    """Adapter that accepts arbitrary keyword args as structured fields."""

    def process(self, msg: str, kwargs: dict) -> tuple[str, dict]:
        extra = kwargs.pop("extra", {})
        extra.update({k: v for k, v in list(kwargs.items())
                      if k not in ("exc_info", "stack_info", "stacklevel")})
        # remove those keys from kwargs so logging doesn't complain
        for k in list(extra.keys()):
            kwargs.pop(k, None)
        kwargs["extra"] = extra
        return msg, kwargs


_handlers_configured = False


def _configure_root() -> None:
    global _handlers_configured
    if _handlers_configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    _handlers_configured = True


def get_logger(name: str) -> _KwLogger:
    _configure_root()
    return _KwLogger(logging.getLogger(name), {})
