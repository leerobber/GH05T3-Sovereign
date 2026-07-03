"""Provider-state endpoint for Pact verification."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Body

router = APIRouter(tags=["oss-pact"])


@router.post("/_pact/provider_states")
def provider_states(payload: Dict[str, Any] | None = Body(default=None)) -> Dict[str, Any]:
    """Accept Pact provider-state setup requests and return a stable ack."""
    payload = payload or {}
    state = payload.get("state") or payload.get("name") or payload.get("providerState")
    return {
        "ok": True,
        "state": state,
        "received": payload,
    }


@router.get("/_pact/provider_states")
def provider_states_health() -> Dict[str, Any]:
    return {"ok": True, "endpoint": "/_pact/provider_states"}
