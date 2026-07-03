"""Binary transformer training dashboard endpoint.

Reads gh05t3_binary/train/checkpoints/training_state.json -- the sidecar
gh05t3_binary/train/train_binary.py writes after every epoch. This is a
real file read, not an in-memory GH05T3State reference: training runs as
its own script/process invocation with no shared memory with whatever
process serves this endpoint, so there's no live Python object to query
across that boundary.

Standalone-runnable for direct testing (`python -m backend.api.binary_training`);
wiring this router into the main gateway (gateway_v3.py/server.py) is a
separate step.
"""
from __future__ import annotations

import json
import os

from fastapi import APIRouter, FastAPI

router = APIRouter()

_TRAINING_STATE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "gh05t3_binary", "train", "checkpoints", "training_state.json"
)


@router.get("/binary/training")
def get_binary_training_state() -> dict:
    if not os.path.isfile(_TRAINING_STATE_PATH):
        return {
            "trained": False,
            "epochs": 0,
            "loss_curve": [],
            "last_checkpoint": None,
            "detail": "no training run has completed yet",
        }

    with open(_TRAINING_STATE_PATH, "r") as f:
        state = json.load(f)

    return {
        "trained": True,
        "epochs": state.get("epochs", 0),
        "last_loss": state.get("last_loss"),
        "loss_curve": state.get("loss_curve", []),
        "last_checkpoint": state.get("checkpoint"),
    }


def _standalone_app() -> FastAPI:
    app = FastAPI(title="GH05T3 binary training dashboard")
    app.include_router(router)
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(_standalone_app(), host="0.0.0.0", port=8020)
