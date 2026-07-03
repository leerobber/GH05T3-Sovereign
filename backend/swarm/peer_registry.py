"""GH05T3 — Tailscale peer auto-discovery registry.

Polls the Tailscale API to enumerate all devices on the tailnet,
probes each one's gateway port to confirm GH05T3 is running,
then maintains a live peer list for MCP cross-instance delegation.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

import httpx

log = logging.getLogger("gh0st3.peer_registry")


class PeerRegistry:
    """Discovers and tracks other GH05T3 instances on the Tailscale mesh."""

    def __init__(self) -> None:
        self._peers: list[dict] = []
        self._task: Optional[asyncio.Task] = None
        self.api_key = os.environ.get("TAILSCALE_API_KEY", "")
        self.tailnet = os.environ.get("TAILSCALE_TAILNET", "-")
        self.own_ip = os.environ.get("TAILSCALE_OWN_IP", "")
        self.probe_port = int(os.environ.get("GATEWAY_PORT", "8002"))
        self.refresh_interval = int(os.environ.get("PEER_REFRESH_INTERVAL", "300"))

    @property
    def peers(self) -> list[dict]:
        return list(self._peers)

    async def start(self) -> None:
        await self.refresh()
        self._task = asyncio.create_task(self._refresh_loop(), name="peer-registry")
        log.info("PeerRegistry started (interval=%ds)", self.refresh_interval)

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _refresh_loop(self) -> None:
        while True:
            await asyncio.sleep(self.refresh_interval)
            try:
                await self.refresh()
            except Exception as exc:
                log.warning("PeerRegistry refresh error: %s", exc)

    async def refresh(self) -> list[dict]:
        """Discover Tailscale peers and probe each for a live GH05T3 gateway."""
        if not self.api_key:
            log.debug("TAILSCALE_API_KEY not set — peer discovery disabled")
            return self._peers

        devices = await self._get_tailscale_devices()
        if not devices:
            return self._peers

        alive: list[dict] = []
        api_token = os.environ.get("GH05T3_API_TOKEN", "")
        headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}

        async with httpx.AsyncClient(timeout=3.0) as client:
            tasks = [self._probe(client, dev, headers) for dev in devices]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, dict):
                    alive.append(r)

        self._peers = alive
        log.info("PeerRegistry: %d/%d peers alive", len(alive), len(devices))
        return alive

    async def _probe(self, client: httpx.AsyncClient, dev: dict, headers: dict) -> Optional[dict]:
        """Return peer info dict if device is running a GH05T3 gateway, else None."""
        url = f"http://{dev['ip']}:{self.probe_port}"
        try:
            r = await client.get(f"{url}/health", headers=headers)
            if r.status_code == 200:
                data = r.json()
                return {
                    "label":   dev["hostname"],
                    "ip":      dev["ip"],
                    "url":     url,
                    "status":  data.get("status", "unknown"),
                    "agents":  data.get("swarm_agents", 0),
                    "version": data.get("version", "?"),
                }
        except Exception:
            pass
        return None

    async def _get_tailscale_devices(self) -> list[dict]:
        """Call Tailscale API and return a list of {hostname, ip} for peer devices."""
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                r = await client.get(
                    f"https://api.tailscale.com/api/v2/tailnet/{self.tailnet}/devices",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                if r.status_code != 200:
                    log.warning("Tailscale API %d: %s", r.status_code, r.text[:200])
                    return []

                devices = r.json().get("devices", [])
                result = []
                for dev in devices:
                    addrs = dev.get("addresses", [])
                    ts_ip = next((a for a in addrs if a.startswith("100.")), None)
                    if ts_ip and ts_ip != self.own_ip:
                        result.append({
                            "hostname": dev.get("hostname", "unknown"),
                            "ip": ts_ip,
                        })
                return result
        except Exception as exc:
            log.warning("Tailscale API error: %s", exc)
            return []
