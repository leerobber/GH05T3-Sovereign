"""
economy_bridge.py — GH05T3 training results → Agent Economy credits.

Real work earns real credits. Import and call these functions from
ghost_trainer.py and continuous_learner.py.

Rewards:
  KAIROS PASS          → KAIROS agent + domain-relevant agent earn
  SPIN upload          → Distiller + Researcher earn
  Frontier breakthrough→ GH05T3-Avery bonus
  Repo scan complete   → ORACLE earns
  AETHER image gen     → FORGE earns
  AETHER character gen → CODEX earns
  Domain mastery (0.8+)→ domain agent earns
"""

import sys, os as _os
import requests
from functools import lru_cache

ECO_BASE = "http://localhost:8081"
ENABLED  = True   # set False to disable without removing call sites

# Local SQLite ledger — always records even when SovereignCore is offline
_backend_dir = _os.path.join(_os.path.dirname(__file__), "backend")
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)
try:
    from economy.ledger import credit as _local_credit
    _LOCAL_LEDGER = True
except ImportError:
    _local_credit = None
    _LOCAL_LEDGER = False

# ── Reward table (credits) ────────────────────────────────────────────────────
REWARD = {
    "kairos_pass":         30,
    "kairos_high_pass":    60,   # score >= 0.85
    "kairos_breakthrough": 150,  # score >= 0.95
    "spin_upload":         25,
    "repo_scan":           15,
    "image_gen":           10,
    "character_gen":       20,
    "domain_mastery":      100,
    "frontier_pass":       80,
}

# ── Domain → agent name mapping ───────────────────────────────────────────────
DOMAIN_AGENT = {
    "business":         "KAIROS",
    "sales":            "CODEX",
    "product_strategy": "Architect",
    "growth":           "Researcher",
    "cfo":              "Analyst",
    "content":          "Distiller",
    "legal_ip":         "SENTINEL",
    "ops":              "NEXUS",
    "ml_engineer":      "FORGE",
    "frontier":         "ORACLE",
    "core":             "GH05T3-Avery",
}


@lru_cache(maxsize=1)
def _get_agent_ids() -> dict:
    """Returns {name: id} for all GH05T3 agents. Cached for speed."""
    try:
        agents = requests.get(f"{ECO_BASE}/agents/?limit=500", timeout=3).json()
        if not isinstance(agents, list):
            agents = agents.get("agents", agents.get("data", []))
        return {
            a["name"]: a["id"] for a in agents
            if a["name"] in {
                "GH05T3-Avery", "KAIROS", "ORACLE", "FORGE", "CODEX",
                "SENTINEL", "NEXUS", "Architect", "Researcher", "Coder",
                "Analyst", "Monitor", "Distiller", "Evolver",
                "AETHER-Platform", "LocalMesh",
            }
        }
    except Exception:
        return {}


def _refresh_ids():
    """Force refresh of the agent ID cache."""
    _get_agent_ids.cache_clear()
    return _get_agent_ids()


def _seed(name: str, amount: int, reason: str) -> bool:
    if not ENABLED:
        return False

    # Always record locally first — economy never goes dark.
    # Track actual write success (not just whether the module imported).
    local_seeded = False
    if _LOCAL_LEDGER and _local_credit:
        try:
            _local_credit(name, amount, reason, source="sovereign_core")
            local_seeded = True
        except Exception:
            pass

    # Then attempt SovereignCore
    ids = _get_agent_ids()
    agent_id = ids.get(name)
    if not agent_id:
        ids = _refresh_ids()
        agent_id = ids.get(name)
    if not agent_id:
        return local_seeded  # remote ID not found; reflect actual local write result
    try:
        r = requests.post(
            f"{ECO_BASE}/creator/seed/{agent_id}",
            params={"amount": amount},
            timeout=5,
        )
        if r.ok:
            print(f"  [ECO] +{amount} cr → {name}  ({reason})", flush=True)
            return True
    except Exception:
        pass
    return local_seeded  # remote failed; reflect actual local write result


def complete_task_for(agent_name: str, title: str, reward: int):
    """Mark a task as completed by a GH05T3 agent (seed workaround)."""
    _seed(agent_name, reward, f"task: {title[:40]}")


# ── Public reward functions ───────────────────────────────────────────────────

def on_kairos_pass(domain: str, score: float):
    """Call after every KAIROS cycle PASS."""
    if not ENABLED:
        return
    if score >= 0.95:
        amount = REWARD["kairos_breakthrough"]
        _seed("GH05T3-Avery", amount, f"breakthrough score={score:.3f}")
    elif score >= 0.85:
        amount = REWARD["kairos_high_pass"]
    else:
        amount = REWARD["kairos_pass"]

    _seed("KAIROS", amount, f"pass domain={domain} score={score:.3f}")

    domain_agent = DOMAIN_AGENT.get(domain)
    if domain_agent and domain_agent != "KAIROS":
        _seed(domain_agent, amount // 2, f"domain support domain={domain}")

    if domain == "frontier":
        _seed("ORACLE", REWARD["frontier_pass"] // 2, "frontier pass support")


def on_spin_upload(pairs_uploaded: int):
    """Call after SPIN pairs are uploaded to HuggingFace."""
    if not ENABLED or pairs_uploaded == 0:
        return
    bonus = min(pairs_uploaded * 5, 100)
    _seed("Distiller",  REWARD["spin_upload"] + bonus, f"spin upload {pairs_uploaded} pairs")
    _seed("Researcher", REWARD["spin_upload"],           "spin data curation")


def on_repo_scan(repos_scanned: int):
    """Call after continuous_learner repo scan completes."""
    if not ENABLED or repos_scanned == 0:
        return
    _seed("ORACLE", REWARD["repo_scan"] * min(repos_scanned, 5), f"repo scan {repos_scanned} repos")


def on_domain_mastery(domain: str, goal: str, score: float):
    """Call when a goal reaches mastery (score >= 0.8)."""
    if not ENABLED:
        return
    domain_agent = DOMAIN_AGENT.get(domain, "GH05T3-Avery")
    _seed(domain_agent, REWARD["domain_mastery"], f"mastery domain={domain} score={score:.3f}")
    _seed("GH05T3-Avery", REWARD["domain_mastery"] // 2, f"mastery bonus domain={domain}")


def on_image_gen():
    """Call after successful AETHER image generation."""
    _seed("FORGE", REWARD["image_gen"], "image gen")
    _seed("AETHER-Platform", REWARD["image_gen"] // 2, "platform image gen")


def on_character_gen():
    """Call after successful AETHER character generation."""
    _seed("CODEX", REWARD["character_gen"], "character gen")
    _seed("AETHER-Platform", REWARD["character_gen"] // 2, "platform character gen")


def status() -> dict:
    """Quick economy health snapshot for logging."""
    try:
        d = requests.get(f"{ECO_BASE}/system/stats", timeout=3).json()
        return {
            "health": d.get("economy_health", {}).get("score", "?"),
            "tick":   d.get("tick", "?"),
            "agents": d.get("agents", {}).get("total", "?"),
        }
    except Exception:
        return {"health": "offline", "tick": "?", "agents": "?"}
