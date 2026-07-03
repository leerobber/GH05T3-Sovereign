"""
Model Versioning System for GH05T3-Omni

Stores for every trained version:
- version string
- training data hash (for reproducibility)
- full training config
- evaluation metrics
- path to the checkpoint

Used by AgentHandle for safe reintegration of improved models.
"""

from typing import Dict, Any, List
import json
import os
from pathlib import Path

REGISTRY_PATH = Path("models/model_registry.json")


def load_registry() -> Dict[str, Any]:
    if not REGISTRY_PATH.exists():
        return {"models": []}
    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"models": []}


def save_registry(registry: Dict[str, Any]):
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def register_model(version_info: Dict[str, Any], model_path: str):
    """Register a newly trained model version."""
    registry = load_registry()
    entry = {
        "version": version_info.get("version"),
        "data_hash": version_info.get("data_hash"),
        "metrics": version_info.get("metrics", {}),
        "config": version_info.get("config", {}),
        "path": model_path,
        "timestamp": version_info.get("timestamp"),
    }
    # Avoid duplicates
    existing = [m for m in registry["models"] if m.get("version") == entry["version"]]
    if not existing:
        registry["models"].append(entry)
        save_registry(registry)
        print(f"Registered model version {entry['version']}")
    return entry


def get_latest_model() -> Dict[str, Any]:
    """Return the most recent successfully trained model entry (or empty dict)."""
    registry = load_registry()
    models = registry.get("models", [])
    if not models:
        return {}
    # Prefer omni models if present, otherwise latest
    omni_models = [m for m in models if "omni" in str(m.get("version", "")).lower() or "gh05t3_omni" in str(m.get("path", "")).lower()]
    if omni_models:
        return sorted(omni_models, key=lambda m: m.get("timestamp", ""))[-1]
    return sorted(models, key=lambda m: m.get("timestamp", ""))[-1]

def activate_latest_omni_adapter():
    """Helper to get the path of the latest omni adapter (for env or direct use)."""
    latest = get_latest_model()
    if not latest:
        return None
    path = latest.get("path")
    if path:
        adapter = os.path.join(path, "adapter")
        if os.path.exists(os.path.join(adapter, "adapter_config.json")):
            return adapter
        if os.path.exists(os.path.join(path, "adapter_config.json")):
            return path
    return None


def get_model_by_version(version: str) -> Dict[str, Any]:
    registry = load_registry()
    for m in registry.get("models", []):
        if m.get("version") == version:
            return m
    return {}


def list_models() -> List[Dict[str, Any]]:
    return load_registry().get("models", [])
