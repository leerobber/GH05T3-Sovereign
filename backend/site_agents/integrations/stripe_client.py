"""Stripe REST API client — revenue, subscriptions, charges, customers.

Uses raw requests to Stripe REST API — no stripe Python library required.
Set STRIPE_SECRET_KEY in backend/.env.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

LOG = logging.getLogger("site_agents.stripe_client")
STRIPE_BASE = "https://api.stripe.com/v1"


def _key() -> str | None:
    return os.environ.get("STRIPE_SECRET_KEY", "").strip() or None


def available() -> bool:
    return bool(_key())


def _get(endpoint: str, params: dict | None = None) -> dict:
    import requests
    resp = requests.get(
        f"{STRIPE_BASE}/{endpoint}",
        params=params or {},
        auth=(_key(), ""),
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def _unavailable(reason: str = "STRIPE_SECRET_KEY not set") -> dict:
    return {"available": False, "reason": reason, "mrr": 0, "arr": 0}


def get_revenue_summary() -> dict:
    """MRR, ARR, active subscriptions, avg value, churn estimate."""
    if not available():
        return _unavailable()
    try:
        subs = _get("subscriptions", {"status": "active", "limit": 100})
        items = subs.get("data", [])
        total_mrr_cents = sum(
            s.get("plan", {}).get("amount", 0) or
            sum(i.get("price", {}).get("unit_amount", 0) for i in s.get("items", {}).get("data", []))
            for s in items
        )
        mrr = round(total_mrr_cents / 100, 2)
        arr = round(mrr * 12, 2)
        avg_value = round(mrr / max(len(items), 1), 2)

        canceled = _get("subscriptions", {"status": "canceled", "limit": 100})
        canceled_count = len(canceled.get("data", []))
        total = len(items) + canceled_count
        churn_rate = round((canceled_count / max(total, 1)) * 100, 1)

        return {
            "available": True,
            "mrr": mrr,
            "arr": arr,
            "active_subscriptions": len(items),
            "avg_subscription_value": avg_value,
            "canceled_subscriptions": canceled_count,
            "churn_rate_pct": churn_rate,
        }
    except Exception as e:
        LOG.warning("[stripe] revenue summary failed: %s", e)
        return _unavailable(str(e))


def get_recent_charges(limit: int = 20) -> list[dict]:
    """Recent successful charges."""
    if not available():
        return []
    try:
        data = _get("charges", {"limit": limit, "status": "succeeded"})
        return [
            {
                "id": c["id"],
                "amount": round(c["amount"] / 100, 2),
                "currency": c.get("currency", "usd").upper(),
                "description": c.get("description", ""),
                "created": c.get("created"),
                "customer": c.get("customer"),
            }
            for c in data.get("data", [])
        ]
    except Exception as e:
        LOG.warning("[stripe] charges failed: %s", e)
        return []


def get_customer_count() -> dict:
    """Total customer count."""
    if not available():
        return {"available": False, "total": 0}
    try:
        data = _get("customers", {"limit": 1})
        total = data.get("total_count") or len(data.get("data", []))
        return {"available": True, "total": total}
    except Exception as e:
        LOG.warning("[stripe] customer count failed: %s", e)
        return {"available": False, "total": 0, "error": str(e)}


def get_pricing_tiers() -> list[dict]:
    """All active prices/products."""
    if not available():
        return []
    try:
        products = _get("products", {"active": "true", "limit": 20})
        prices = _get("prices", {"active": "true", "limit": 20})
        price_map: dict[str, list] = {}
        for p in prices.get("data", []):
            pid = p.get("product")
            price_map.setdefault(pid, []).append({
                "id": p["id"],
                "amount": round((p.get("unit_amount") or 0) / 100, 2),
                "currency": p.get("currency", "usd").upper(),
                "interval": p.get("recurring", {}).get("interval", "one_time"),
            })
        return [
            {
                "product_id": prod["id"],
                "name": prod.get("name", ""),
                "description": prod.get("description", ""),
                "prices": price_map.get(prod["id"], []),
            }
            for prod in products.get("data", [])
            if prod and isinstance(prod, dict)
        ]
    except Exception as e:
        LOG.warning("[stripe] pricing tiers failed: %s", e)
        return []
