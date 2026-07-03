"""
GH05T3 Weights & Biases logger.

Tracks KAIROS evolutionary fitness curves + training loss across runs.
Free for solo use — dashboard at wandb.ai.

Config (.env):
  WANDB_API_KEY  = your W&B API key (wandb.ai → Settings → API Keys)
  WANDB_PROJECT  = gh05t3  (default)
  WANDB_ENTITY   = your W&B username (optional)

Set WANDB_DISABLED=1 to silence without removing the calls.
"""
from __future__ import annotations

import logging
import os

LOG = logging.getLogger("ghost.wandb")

_run = None   # active wandb run, or None
_enabled = False


def _enabled_check() -> bool:
    if os.environ.get("WANDB_DISABLED") == "1":
        return False
    return bool(os.environ.get("WANDB_API_KEY"))


def init_run(run_name: str = "gh05t3", job_type: str = "training"):
    """Initialize (or re-use) the W&B run. No-op if not configured."""
    global _run, _enabled
    if not _enabled_check():
        return
    try:
        import wandb
        if _run is None or _run.id is None:
            _run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "gh05t3"),
                entity=os.environ.get("WANDB_ENTITY") or None,
                name=run_name,
                job_type=job_type,
                reinit=True,
            )
            _enabled = True
            LOG.info("W&B run initialized: %s", _run.url)
    except ImportError:
        LOG.debug("wandb not installed — skipping W&B logging")
    except Exception as e:
        LOG.debug("wandb init failed: %s", e)


def log_kairos_cycle(cycle_id: int, score: float, is_elite: bool,
                     total_cycles: int, elite_cycles: int):
    if not _enabled or _run is None:
        return
    try:
        import wandb
        _run.log({
            "kairos/score":        score,
            "kairos/is_elite":     int(is_elite),
            "kairos/total_cycles": total_cycles,
            "kairos/elite_cycles": elite_cycles,
            "kairos/elite_rate":   elite_cycles / max(total_cycles, 1),
        }, step=cycle_id)
    except Exception as e:
        LOG.debug("wandb log_kairos failed: %s", e)


def log_training_step(step: int, loss: float, dataset_size: int = 0):
    if not _enabled or _run is None:
        return
    try:
        payload = {"finetune/loss": loss}
        if dataset_size:
            payload["finetune/dataset_size"] = dataset_size
        _run.log(payload, step=step)
    except Exception as e:
        LOG.debug("wandb log_training_step failed: %s", e)


def log_training_complete(version: int, final_loss: float, steps: int,
                           dataset_size: int, model: str):
    if not _enabled or _run is None:
        return
    try:
        import wandb
        _run.summary.update({
            "finetune/version":      version,
            "finetune/final_loss":   final_loss,
            "finetune/total_steps":  steps,
            "finetune/dataset_size": dataset_size,
            "finetune/base_model":   model,
        })
        _run.finish()
    except Exception as e:
        LOG.debug("wandb log_training_complete failed: %s", e)


def log_pipeline_cost(spent: float, paid_calls: int, free_calls: int,
                      total_examples: int):
    if not _enabled or _run is None:
        return
    try:
        _run.log({
            "pipeline/cost_usd":     spent,
            "pipeline/paid_calls":   paid_calls,
            "pipeline/free_calls":   free_calls,
            "pipeline/total_examples": total_examples,
        })
    except Exception as e:
        LOG.debug("wandb log_pipeline_cost failed: %s", e)


def wandb_status() -> dict:
    enabled = _enabled_check()
    return {
        "configured": enabled,
        "active_run":  _run.url if (_enabled and _run) else None,
        "project": os.environ.get("WANDB_PROJECT", "gh05t3"),
    }
