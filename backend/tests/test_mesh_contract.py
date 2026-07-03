from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gateway_v3  # noqa: E402


class DummyPeerRegistry:
    def __init__(self, peers: list[dict]):
        self._peers = peers
        self.refresh_calls = 0

    @property
    def peers(self) -> list[dict]:
        return list(self._peers)

    async def refresh(self) -> list[dict]:
        self.refresh_calls += 1
        return list(self._peers)


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch):
    monkeypatch.setenv("TAILSCALE_OWN_LABEL", "mesh-node")
    monkeypatch.setenv("TAILSCALE_OWN_IP", "100.64.0.10")
    monkeypatch.setenv("INSTANCE_ROLE", "primary")
    monkeypatch.setenv("GATEWAY_PORT", "8002")


def test_peers_contract_includes_self_and_mesh(monkeypatch):
    registry = DummyPeerRegistry(
        [
            {
                "label": "node-b",
                "ip": "100.64.0.11",
                "url": "http://100.64.0.11:8002",
                "status": "ok",
                "agents": 5,
                "version": "3.0.0",
            }
        ]
    )
    monkeypatch.setattr(gateway_v3, "peer_registry", registry)

    data = asyncio.run(gateway_v3.get_peers())

    assert data["self"]["label"] == "mesh-node"
    assert data["self"]["role"] == "primary"
    assert data["self"]["url"] == f"http://100.64.0.10:{gateway_v3.GATEWAY_PORT}"
    assert data["count"] == 1
    assert data["peers"][0]["label"] == "node-b"
    assert data["mesh"]["discovery"]["refresh"] == "/peers/refresh"
    assert data["mesh"]["github_relay"]["sync"] == "/github/mesh/sync"


def test_peers_refresh_returns_refreshed_contract(monkeypatch):
    registry = DummyPeerRegistry(
        [
            {
                "label": "node-c",
                "ip": "100.64.0.12",
                "url": "http://100.64.0.12:8002",
                "status": "ok",
                "agents": 2,
                "version": "3.0.0",
            }
        ]
    )
    monkeypatch.setattr(gateway_v3, "peer_registry", registry)

    data = asyncio.run(gateway_v3.refresh_peers())

    assert registry.refresh_calls == 1
    assert data["count"] == 1
    assert data["peers"][0]["label"] == "node-c"
    assert data["self"]["label"] == "mesh-node"


def test_ping_alias_uses_refresh(monkeypatch):
    registry = DummyPeerRegistry([])
    monkeypatch.setattr(gateway_v3, "peer_registry", registry)

    data = asyncio.run(gateway_v3.ping_peers())

    assert registry.refresh_calls == 1
    assert data["count"] == 0
    assert data["peers"] == []


def test_mesh_sync_alias_delegates_to_github_sync(monkeypatch):
    async def _fake_sync():
        return {"ok": True, "push": {"ok": True}, "pull": {"ok": True}}

    monkeypatch.setattr(gateway_v3, "github_mesh_sync", _fake_sync)

    data = asyncio.run(gateway_v3.push_mesh_sync())

    assert data["ok"] is True
    assert data["push"]["ok"] is True
    assert data["pull"]["ok"] is True
