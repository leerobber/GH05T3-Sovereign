"""
tunnel_watcher.py
=================
Monitors cloudflared log files for trycloudflare.com URLs.
Saves the latest URLs to data/tunnel_urls.json so other services
and the landing page can always find the current tunnel addresses.

Runs standalone or imported. Usage:
    python tunnel_watcher.py
"""

from __future__ import annotations
import json
import re
import sys
import time
import logging
from pathlib import Path
from datetime import datetime, timezone

ROOT      = Path(__file__).parent.resolve()
DATA_DIR  = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

URLS_FILE  = DATA_DIR / "tunnel_urls.json"
CHAT_LOG   = DATA_DIR / "tunnel_chat.log"
LAND_LOG   = DATA_DIR / "tunnel_landing.log"

URL_RE     = re.compile(r'https://[a-z0-9-]+\.trycloudflare\.com')
CHECK_INTERVAL = 3   # seconds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[
        logging.FileHandler(DATA_DIR / "tunnel_watcher.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("tunnel_watcher")


def _extract_url(log_path: Path) -> str | None:
    """Return the most recently seen trycloudflare URL in a log file."""
    if not log_path.exists():
        return None
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        matches = URL_RE.findall(text)
        return matches[-1] if matches else None
    except Exception:
        return None


def _load_saved() -> dict:
    try:
        return json.loads(URLS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save(urls: dict):
    URLS_FILE.write_text(
        json.dumps(urls, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def watch():
    log.info("Tunnel watcher started — monitoring %s and %s", CHAT_LOG, LAND_LOG)
    saved = _load_saved()

    while True:
        changed = False

        chat_url   = _extract_url(CHAT_LOG)
        landing_url = _extract_url(LAND_LOG)

        now = datetime.now(timezone.utc).isoformat()

        if chat_url and chat_url != saved.get("chat"):
            log.info("CHAT tunnel URL → %s", chat_url)
            saved["chat"]            = chat_url
            saved["chat_updated"]    = now
            changed = True

        if landing_url and landing_url != saved.get("landing"):
            log.info("LANDING tunnel URL → %s", landing_url)
            saved["landing"]         = landing_url
            saved["landing_updated"] = now
            changed = True

        if changed:
            saved["last_check"] = now
            _save(saved)
            log.info("Saved tunnel_urls.json  chat=%s  landing=%s",
                     saved.get("chat", "?"), saved.get("landing", "?"))

            # Print clearly so it's impossible to miss in the supervisor log
            print("\n" + "="*70)
            print("  TUNNEL URLs UPDATED")
            print(f"  Chat    : {saved.get('chat', 'unknown')}")
            print(f"  Landing : {saved.get('landing', 'unknown')}")
            print("="*70 + "\n", flush=True)

            # Slack notification so demo links can be updated
            try:
                sys.path.insert(0, str(ROOT))
                import slack_notify as _slack
                chat_url    = saved.get("chat", "?")
                landing_url = saved.get("landing", "?")
                _slack.post("alerts",
                    f"*TUNNEL URLS UPDATED*\n"
                    f"Demo (chat): {chat_url}\n"
                    f"Landing:     {landing_url}\n"
                    f"Update demo links if these changed."
                )
            except Exception as e:
                log.debug("Slack notify failed: %s", e)

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    watch()
