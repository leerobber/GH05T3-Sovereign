"""
slack_notify.py — SovereignNation Slack notification hub for GH05T3.

Reads SLACK_BOT_TOKEN from environment or GH05T3/.env
Gracefully no-ops if token is missing or has wrong scopes.
"""
import os, json, time
from pathlib import Path

# ── Channel map ───────────────────────────────────────────────────────────────
CHANNELS = {
    "training":   "avery-training",
    "learner":    "continuous-learner",
    "alerts":     "alerts",
    "research":   "research",
    "business":   "ventures",
    "finance":    "finance",
    "crypto":     "crypto",
    "general":    "general",
    "releases":   "releases",
    "kairos":     "kairos",
}

_token: str | None = None
_session = None   # requests.Session, lazy-loaded

def _get_token() -> str | None:
    global _token
    if _token:
        return _token
    # 1. env var
    t = os.environ.get("SLACK_BOT_TOKEN", "")
    if t and t.startswith("xoxb-"):
        _token = t
        return _token
    # 2. GH05T3/.env
    env = Path(__file__).parent / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("SLACK_BOT_TOKEN="):
                t = line.split("=", 1)[1].strip().strip('"\'')
                if t.startswith("xoxb-"):
                    _token = t
                    return _token
    return None


def _session_get():
    global _session
    if _session is None:
        import requests
        _session = requests.Session()
    return _session


def post(channel_key: str, text: str, blocks: list | None = None) -> bool:
    """Post a message. channel_key maps via CHANNELS dict. Returns True on success."""
    token = _get_token()
    if not token:
        return False
    channel = CHANNELS.get(channel_key, channel_key)
    payload = {"channel": channel, "text": text}
    if blocks:
        payload["blocks"] = blocks
    try:
        r = _session_get().post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload, timeout=10
        )
        data = r.json()
        if not data.get("ok"):
            # Silently ignore missing_scope — token not yet reinstalled
            if data.get("error") not in ("missing_scope", "not_in_channel", "channel_not_found"):
                print(f"  [Slack] {data.get('error')}")
        return data.get("ok", False)
    except Exception:
        return False


def create_channel(name: str) -> str | None:
    """Create a public channel. Returns channel ID or None."""
    token = _get_token()
    if not token:
        return None
    try:
        r = _session_get().post(
            "https://slack.com/api/conversations.create",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"name": name, "is_private": False}, timeout=10
        )
        data = r.json()
        if data.get("ok"):
            return data["channel"]["id"]
        if data.get("error") == "name_taken":
            return _get_channel_id(name)
        return None
    except Exception:
        return None


def _get_channel_id(name: str) -> str | None:
    token = _get_token()
    if not token:
        return None
    try:
        cursor = ""
        while True:
            params = {"limit": 200, "types": "public_channel,private_channel"}
            if cursor:
                params["cursor"] = cursor
            r = _session_get().get(
                "https://slack.com/api/conversations.list",
                headers={"Authorization": f"Bearer {token}"},
                params=params, timeout=10
            )
            data = r.json()
            for ch in data.get("channels", []):
                if ch["name"] == name:
                    return ch["id"]
            cursor = data.get("response_metadata", {}).get("next_cursor", "")
            if not cursor:
                break
    except Exception:
        pass
    return None


def join_channel(channel_id: str) -> bool:
    token = _get_token()
    if not token:
        return False
    try:
        r = _session_get().post(
            "https://slack.com/api/conversations.join",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"channel": channel_id}, timeout=10
        )
        return r.json().get("ok", False)
    except Exception:
        return False


def setup_workspace() -> dict:
    """Create all SovereignNation channels and join them. Returns {name: id}."""
    results = {}
    channels_to_create = [
        "avery-training", "continuous-learner", "kairos", "alerts",
        "engineering", "code-reviews", "testing", "releases",
        "research", "documentation", "design",
        "ventures", "sales", "content",
        "finance", "crypto",
        "employees", "inbox",
        "gh05t3-cmd",
    ]
    print("[Slack] Setting up SovereignNation workspace channels...")
    for name in channels_to_create:
        cid = create_channel(name)
        if cid:
            join_channel(cid)
            results[name] = cid
            print(f"  [Slack] #{name} ready ({cid})")
        else:
            print(f"  [Slack] #{name} skipped (no scope yet)")
    return results


# ── Notification helpers ───────────────────────────────────────────────────────

def notify_cycle(cycle: int, goal: str, verdict: str, score: float,
                 axes: dict, domain: str, proposal_excerpt: str, elapsed: float):
    """Post a training cycle result to #avery-training."""
    if verdict == "PASS":
        emoji = ":white_check_mark:"
        color_bar = ":large_green_circle:" if score >= 0.80 else ":large_yellow_circle:"
    else:
        emoji = ":x:"
        color_bar = ":red_circle:"

    text = (
        f"{emoji} *Cycle {cycle}* | `{domain}` | {verdict} `{score:.3f}`\n"
        f"{color_bar} spec={axes.get('spec','?')} exec={axes.get('exec','?')} "
        f"innov={axes.get('innov','?')} rev={axes.get('rev','?')} | {elapsed:.0f}s\n"
        f"> *Goal:* {goal[:80]}\n"
        f"> {proposal_excerpt[:140]}"
    )
    post("training", text)


def notify_breakthrough(cycle: int, goal: str, score: float, domain: str, excerpt: str):
    """Post a breakthrough (score >= 0.90) alert."""
    text = (
        f":fire: *BREAKTHROUGH* | Cycle {cycle} | `{domain}` | score `{score:.3f}`\n"
        f"> *Goal:* {goal}\n"
        f"> {excerpt[:200]}"
    )
    post("training", text)
    post("alerts", text)


def notify_domain_rotation(from_domain: str, to_domain: str, cycle: int, spin_count: int):
    post("learner",
         f":arrows_counterclockwise: *Domain rotation* | `{from_domain}` -> `{to_domain}` "
         f"at cycle {cycle} | {spin_count} SPIN pairs total")


def notify_scan(active_repos: int, cycle: int):
    post("learner",
         f":mag: *Repo scan complete* | {active_repos} repos active | cycle {cycle}")


def notify_spin_upload(new_pairs: int, total: int, upload_num: int):
    post("learner",
         f":outbox_tray: *SPIN upload #{upload_num}* | "
         f"{new_pairs} new pairs pushed to HuggingFace | {total} total")


def notify_session_done(cycles: int, total_ever: int, spin_total: int, uploads: int):
    post("learner",
         f":checkered_flag: *Session complete* | "
         f"{cycles} cycles this run | {total_ever} total ever | "
         f"{spin_total} SPIN pairs | {uploads} HF uploads")


def notify_ollama_error(model: str, attempt: int, error: str):
    if attempt >= 3:  # only alert on repeated failures
        post("alerts",
             f":warning: *Ollama ERR* | model=`{model}` attempt={attempt} | {error[:100]}")
