"""
GH05T3 Notifier — Telegram + Slack mobile alerts.

Wired to:
  • SENTINEL threat detection (MsgType.ERROR)
  • KAIROS elite cycle achievement
  • Critical backend errors

Config (.env):
  TELEGRAM_BOT_TOKEN  = bot token from @BotFather
  TELEGRAM_CHAT_ID    = your personal chat ID (@userinfobot to get it)
  SLACK_WEBHOOK_URL   = incoming webhook URL from Slack app

Both are optional — only configured channels fire.
All calls are best-effort (never raise).
"""
from __future__ import annotations

import logging
import os

LOG = logging.getLogger("ghost.notifier")


def _telegram_token() -> str:
    return os.environ.get("TELEGRAM_BOT_TOKEN", "")


def _telegram_chat() -> str:
    return os.environ.get("TELEGRAM_CHAT_ID", "")


def _slack_url() -> str:
    return os.environ.get("SLACK_WEBHOOK_URL", "")


async def _send_telegram(text: str):
    token = _telegram_token()
    chat  = _telegram_chat()
    if not token or not chat:
        return
    try:
        import httpx
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        async with httpx.AsyncClient(timeout=10) as c:
            await c.post(url, json={"chat_id": chat, "text": text,
                                    "parse_mode": "HTML"})
    except Exception as e:
        LOG.debug("telegram send failed: %s", e)


async def _send_slack(text: str):
    url = _slack_url()
    if not url:
        return
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10) as c:
            await c.post(url, json={"text": text})
    except Exception as e:
        LOG.debug("slack send failed: %s", e)


async def notify(text: str):
    """Send to all configured channels. Never raises."""
    try:
        await _send_telegram(text)
        await _send_slack(text)
    except Exception as e:
        LOG.debug("notify error: %s", e)


async def notify_threat(threat: str, source: str):
    msg = (
        f"<b>⚠ GH05T3 SENTINEL ALERT</b>\n"
        f"Threat: <code>{threat}</code>\n"
        f"Source: {source}"
    )
    await notify(msg)


async def notify_elite_cycle(cycle_id: int, score: float, proposal: str):
    preview = proposal[:120] + "..." if len(proposal) > 120 else proposal
    msg = (
        f"<b>⭐ GH05T3 Elite KAIROS Cycle #{cycle_id}</b>\n"
        f"Score: {score:.3f}\n"
        f"Proposal: {preview}"
    )
    await notify(msg)


async def notify_finetune_complete(version: int, steps: int, output_dir: str):
    msg = (
        f"<b>🧠 GH05T3 Fine-Tune Complete</b>\n"
        f"Model: gh05t3-lora-v{version}\n"
        f"Steps: {steps}\n"
        f"Saved: {output_dir}"
    )
    await notify(msg)


async def notify_error(context: str, error: str):
    msg = (
        f"<b>🔴 GH05T3 Error</b>\n"
        f"Context: {context}\n"
        f"Error: {error[:200]}"
    )
    await notify(msg)


def notifier_status() -> dict:
    return {
        "telegram": bool(_telegram_token() and _telegram_chat()),
        "slack":    bool(_slack_url()),
    }
