"""Google Analytics 4 Data API client.

Set GA4_PROPERTY_ID and GA4_SERVICE_ACCOUNT_JSON (path to service account JSON) in backend/.env.
Returns {"available": False, "reason": "..."} gracefully if credentials are missing.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

LOG = logging.getLogger("site_agents.ga4")
GA4_BASE = "https://analyticsdata.googleapis.com/v1beta"


def _property_id() -> str | None:
    return os.environ.get("GA4_PROPERTY_ID", "").strip() or None


def _sa_path() -> str | None:
    return os.environ.get("GA4_SERVICE_ACCOUNT_JSON", "").strip() or None


def available() -> bool:
    return bool(_property_id() and _sa_path() and os.path.exists(_sa_path() or ""))


def _unavailable(reason: str = "GA4 credentials not configured") -> dict:
    return {"available": False, "reason": reason}


def _get_access_token() -> str | None:
    sa_path = _sa_path()
    if not sa_path or not os.path.exists(sa_path):
        return None
    try:
        from cryptography.hazmat.primitives import serialization, hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        from cryptography.hazmat.backends import default_backend
        import base64, json as _json
        import requests

        with open(sa_path) as f:
            sa = _json.load(f)

        now = int(time.time())
        claim_set = {
            "iss": sa["client_email"],
            "scope": "https://www.googleapis.com/auth/analytics.readonly",
            "aud": "https://oauth2.googleapis.com/token",
            "iat": now,
            "exp": now + 3600,
        }

        header = base64.urlsafe_b64encode(json.dumps({"alg": "RS256", "typ": "JWT"}).encode()).rstrip(b"=")
        payload_b = base64.urlsafe_b64encode(json.dumps(claim_set).encode()).rstrip(b"=")
        signing_input = header + b"." + payload_b

        private_key = serialization.load_pem_private_key(
            sa["private_key"].encode(), password=None, backend=default_backend()
        )
        signature = private_key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
        sig_b64 = base64.urlsafe_b64encode(signature).rstrip(b"=")
        jwt_token = (signing_input + b"." + sig_b64).decode()

        token_resp = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                "assertion": jwt_token,
            },
            timeout=15,
        )
        token_resp.raise_for_status()
        return token_resp.json().get("access_token")
    except ImportError:
        LOG.warning("[ga4] cryptography package not installed — run: pip install cryptography")
        return None
    except Exception as e:
        LOG.warning("[ga4] token error: %s", e)
        return None


def _run_report(body: dict) -> dict | None:
    pid = _property_id()
    token = _get_access_token()
    if not pid or not token:
        return None
    try:
        import requests
        resp = requests.post(
            f"{GA4_BASE}/properties/{pid}:runReport",
            json=body,
            headers={"Authorization": f"Bearer {token}"},
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        LOG.warning("[ga4] report failed: %s", e)
        return None


def _parse_rows(resp: dict, dim_keys: list[str], metric_keys: list[str]) -> list[dict]:
    rows = resp.get("rows", [])
    result = []
    for row in rows:
        dims = [d.get("value", "") for d in row.get("dimensionValues", [])]
        metrics = [m.get("value", "0") for m in row.get("metricValues", [])]
        record = {dim_keys[i]: dims[i] for i in range(min(len(dim_keys), len(dims)))}
        record.update({metric_keys[i]: metrics[i] for i in range(min(len(metric_keys), len(metrics)))})
        result.append(record)
    return result


def get_traffic_summary(days: int = 30) -> dict:
    """Sessions, users, pageviews, bounce rate, avg session duration, top pages, top sources."""
    if not available():
        return _unavailable()

    date_range = [{"startDate": f"{days}daysAgo", "endDate": "today"}]

    # Overview metrics
    overview = _run_report({
        "dateRanges": date_range,
        "metrics": [
            {"name": "sessions"}, {"name": "totalUsers"},
            {"name": "screenPageViews"}, {"name": "bounceRate"},
            {"name": "averageSessionDuration"},
        ],
    })
    if overview is None:
        return _unavailable("GA4 API call failed")

    totals = overview.get("totals", [{}])[0] if overview.get("totals") else {}
    mv = [m.get("value", "0") for m in totals.get("metricValues", [])]

    # Top pages
    pages_resp = _run_report({
        "dateRanges": date_range,
        "dimensions": [{"name": "pagePath"}],
        "metrics": [{"name": "screenPageViews"}, {"name": "sessions"}],
        "orderBys": [{"metric": {"metricName": "screenPageViews"}, "desc": True}],
        "limit": 10,
    })
    top_pages = _parse_rows(pages_resp or {}, ["page"], ["views", "sessions"]) if pages_resp else []

    # Top sources
    sources_resp = _run_report({
        "dateRanges": date_range,
        "dimensions": [{"name": "sessionSource"}],
        "metrics": [{"name": "sessions"}, {"name": "totalUsers"}],
        "orderBys": [{"metric": {"metricName": "sessions"}, "desc": True}],
        "limit": 10,
    })
    top_sources = _parse_rows(sources_resp or {}, ["source"], ["sessions", "users"]) if sources_resp else []

    return {
        "available": True,
        "period_days": days,
        "sessions": mv[0] if len(mv) > 0 else "0",
        "users": mv[1] if len(mv) > 1 else "0",
        "pageviews": mv[2] if len(mv) > 2 else "0",
        "bounce_rate": f"{float(mv[3] if len(mv) > 3 else 0) * 100:.1f}%" if len(mv) > 3 else "?",
        "avg_session_duration_s": round(float(mv[4] if len(mv) > 4 else 0), 1),
        "top_pages": top_pages,
        "top_sources": top_sources,
    }


def get_top_pages(limit: int = 10) -> list[dict]:
    """Most visited pages in last 30 days."""
    if not available():
        return []
    resp = _run_report({
        "dateRanges": [{"startDate": "30daysAgo", "endDate": "today"}],
        "dimensions": [{"name": "pagePath"}, {"name": "pageTitle"}],
        "metrics": [{"name": "screenPageViews"}, {"name": "sessions"}, {"name": "bounceRate"}],
        "orderBys": [{"metric": {"metricName": "screenPageViews"}, "desc": True}],
        "limit": limit,
    })
    return _parse_rows(resp or {}, ["path", "title"], ["views", "sessions", "bounce_rate"]) if resp else []
