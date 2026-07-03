"""v1 agent invocation — /v1/agents/{id}/invoke."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/v1", tags=["v1"])


class InvokeRequest(BaseModel):
    message: str = ""
    context: dict[str, Any] = Field(default_factory=dict)


@router.post("/agents/{agent_id}/invoke")
async def invoke_agent(agent_id: str, body: InvokeRequest) -> dict[str, Any]:
    try:
        from agent_forge import invoke_agent as forge_invoke
        return await forge_invoke(agent_id, body.message, body.context)
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="agent_forge not installed — v1 invoke unavailable",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))