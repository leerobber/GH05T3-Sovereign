"""
Pytest configuration for backend/tests/.

Live-server tests (test_gh05t3*, test_swarm) require REACT_APP_BACKEND_URL
to be set. When it isn't (e.g. CI), they are automatically skipped.
"""
import os
import pytest

# ── Markers ───────────────────────────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live: test requires a running backend server (skipped in CI if "
        "REACT_APP_BACKEND_URL is not set)",
    )
    config.addinivalue_line(
        "markers",
        "integration: heavy integration test; opt-in only",
    )


# ── Auto-skip live tests when no backend URL is available ────────────────────

_LIVE_MODULES = {
    "test_gh05t3",
    "test_gh05t3_phase2",
    "test_gh05t3_phase3",
    "test_gh05t3_phase4",
    "test_swarm",
}

def pytest_collection_modifyitems(config, items):
    if os.environ.get("REACT_APP_BACKEND_URL"):
        return  # URL is set — let live tests run

    skip = pytest.mark.skip(reason="live server not available (REACT_APP_BACKEND_URL not set)")
    for item in items:
        module_name = getattr(item, "module", None)
        if module_name is None:
            continue
        if getattr(module_name, "__name__", "").split(".")[-1] in _LIVE_MODULES:
            item.add_marker(skip)
