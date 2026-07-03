"""Tailscale integration for GH05T3.

Uses the Tailscale API to list devices, check connectivity, and expose
the laptop's Tailscale IP so Android can always reach GH05T3 remotely.
"""
from __future__ import annotations
import logging
import os
import subprocess
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=True)

LOG = logging.getLogger("ghost.tailscale")

API_KEY  = os.environ.get("TAILSCALE_API_KEY", "")
TAILNET  = os.environ.get("TAILSCALE_TAILNET", "-")   # "-" = default tailnet
_BASE    = "https://api.tailscale.com/api/v2"


def _headers() -> dict:
    return {"Authorization": f"Bearer {API_KEY}"}


async def get_devices() -> list[dict]:
    """Return all devices on the tailnet."""
    if not API_KEY:
        return []
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(f"{_BASE}/tailnet/{TAILNET}/devices", headers=_headers())
        r.raise_for_status()
        devices = r.json().get("devices", [])
        return [
            {
                "name":       d.get("hostname", d.get("name", "?")),
                "ip":         d.get("addresses", ["?"])[0],
                "os":         d.get("os", "?"),
                "online":     d.get("online", False),
                "last_seen":  d.get("lastSeen", ""),
                "authorized": d.get("authorized", False),
            }
            for d in devices
        ]


async def my_ip() -> str:
    """Return this machine's Tailscale IP via CLI (fastest path)."""
    # Try CLI first
    try:
        result = subprocess.run(
            ["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=3
        )
        ip = result.stdout.strip()
        if ip:
            return ip
    except Exception:
        pass
    # Fall back to env
    return os.environ.get("TAILSCALE_IP", "")


async def status() -> dict:
    """Full Tailscale status for the dashboard."""
    ip = await my_ip()
    try:
        devices = await get_devices()
    except Exception as e:
        devices = []
        LOG.warning("tailscale API error: %s", e)

    online = [d for d in devices if d["online"]]
    android = [d for d in devices if "android" in d.get("os", "").lower()]

    return {
        "laptop_ip":      ip,
        "dashboard_url":  f"http://{ip}:3210" if ip else "",
        "api_url":        f"http://{ip}:8001" if ip else "",
        "devices":        devices,
        "online_count":   len(online),
        "android_devices": android,
        "configured":     bool(API_KEY),
    }


async def authorize_device(device_id: str) -> dict:
    """Authorize a device to join the tailnet."""
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.post(f"{_BASE}/device/{device_id}/authorized",
                         headers=_headers(), json={"authorized": True})
        r.raise_for_status()
        return {"ok": True, "device_id": device_id}
