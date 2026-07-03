"""
Unified MVS logging — single format across backend/oss/ (Phase 1 gate P1-W1-005).

Usage:
    from backend.oss.log_config import get_logger
    log = get_logger(__name__)
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional

MVS_LOG_FORMAT = os.environ.get(
    "MVS_LOG_FORMAT",
    "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
MVS_LOG_DATEFMT = os.environ.get("MVS_LOG_DATEFMT", "%Y-%m-%d %H:%M:%S")
_CONFIGURED = False


def configure_mvs_logging(level: Optional[str] = None) -> None:
    """Idempotent root handler setup for OSS/MVS modules."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    lvl_name = (level or os.environ.get("MVS_LOG_LEVEL", "INFO")).upper()
    lvl = getattr(logging, lvl_name, logging.INFO)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(MVS_LOG_FORMAT, datefmt=MVS_LOG_DATEFMT))

    root = logging.getLogger("backend.oss")
    root.setLevel(lvl)
    if not root.handlers:
        root.addHandler(handler)
    root.propagate = False

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under backend.oss with unified formatting."""
    configure_mvs_logging()
    if name.startswith("backend.oss."):
        return logging.getLogger(name)
    # Allow __name__ from submodules like backend.oss.mvs
    if "backend.oss" in name:
        return logging.getLogger(name)
    return logging.getLogger(f"backend.oss.{name}")