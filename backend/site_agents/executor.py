"""
Executor — the hands of every site agent.

Gives agents real-world execution channels:
  github_push()      → push any file to aethyro-landing GitHub Pages (live in ~60s)
  github_read()      → read a file from GitHub (get SHA for updates)
  github_delete()    → remove a file from GitHub
  send_email()       → Resend API transactional email
  slack_notify()     → Slack webhook notification
  update_blog_index()→ maintain /blog/posts.json manifest
  publish_blog_post()→ full pipeline: HTML → GitHub → index update → Slack
  push_site_fix()    → push an HTML/CSS/JS fix to any landing page

All functions are synchronous — call from async contexts with asyncio.to_thread().
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

LOG = logging.getLogger("site_agents.executor")

# ─── Config ──────────────────────────────────────────────────────────────────

LANDING_REPO  = "leerobber/aethyro-landing"
LANDING_BRANCH = "main"
SITE_URL       = "https://aethyro.com"

def _gh_token() -> str:
    return os.environ.get("GITHUB_PAT", "").strip()

def _resend_key() -> str:
    return os.environ.get("RESEND_API_KEY", "").strip()

def _slack_url() -> str:
    return os.environ.get("SLACK_WEBHOOK_URL", "").strip()


# ─── GitHub ───────────────────────────────────────────────────────────────────

def github_read(file_path: str, repo: str = LANDING_REPO) -> dict:
    """Read a file from GitHub. Returns {content, sha, exists}."""
    token = _gh_token()
    if not token:
        return {"exists": False, "sha": None, "content": None}
    url = f"https://api.github.com/repos/{repo}/contents/{file_path}"
    resp = requests.get(url, headers={"Authorization": f"token {token}"}, timeout=10)
    if resp.status_code == 200:
        data = resp.json()
        raw = base64.b64decode(data["content"]).decode("utf-8")
        return {"exists": True, "sha": data["sha"], "content": raw}
    return {"exists": False, "sha": None, "content": None}


def github_push(
    file_path: str,
    content: str,
    commit_message: str,
    repo: str = LANDING_REPO,
    branch: str = LANDING_BRANCH,
    encoding: str = "utf-8",
) -> dict:
    """Push (create or update) a file to GitHub. Returns {ok, url, sha}."""
    token = _gh_token()
    if not token:
        return {"ok": False, "error": "GITHUB_PAT not set"}

    existing = github_read(file_path, repo)
    url = f"https://api.github.com/repos/{repo}/contents/{file_path}"
    payload: dict = {
        "message": commit_message,
        "content": base64.b64encode(content.encode(encoding)).decode(),
        "branch": branch,
    }
    if existing["sha"]:
        payload["sha"] = existing["sha"]

    resp = requests.put(
        url,
        json=payload,
        headers={"Authorization": f"token {token}", "Content-Type": "application/json"},
        timeout=20,
    )
    if resp.status_code in (200, 201):
        data = resp.json()
        live_url = f"{SITE_URL}/{file_path}"
        LOG.info("[executor] pushed %s → %s", file_path, live_url)
        return {"ok": True, "url": live_url, "sha": data.get("content", {}).get("sha")}
    LOG.error("[executor] github_push failed %s: %s", file_path, resp.text[:200])
    return {"ok": False, "error": resp.text[:200], "status": resp.status_code}


def github_delete(file_path: str, commit_message: str, repo: str = LANDING_REPO) -> dict:
    token = _gh_token()
    if not token:
        return {"ok": False, "error": "GITHUB_PAT not set"}
    existing = github_read(file_path, repo)
    if not existing["exists"]:
        return {"ok": True, "note": "file did not exist"}
    url = f"https://api.github.com/repos/{repo}/contents/{file_path}"
    resp = requests.delete(
        url,
        json={"message": commit_message, "sha": existing["sha"], "branch": LANDING_BRANCH},
        headers={"Authorization": f"token {token}"},
        timeout=15,
    )
    return {"ok": resp.status_code in (200, 204)}


# ─── Email (Resend) ───────────────────────────────────────────────────────────

def send_email(
    to: str | list[str],
    subject: str,
    html: str,
    from_email: str = "Robert at Aethyro <hello@aethyro.com>",
    reply_to: str = "leer4030@gmail.com",
) -> dict:
    """Send transactional email via Resend. Set RESEND_API_KEY in .env."""
    key = _resend_key()
    if not key:
        LOG.warning("[executor] RESEND_API_KEY not set — email not sent")
        return {"ok": False, "error": "RESEND_API_KEY not set"}
    recipients = [to] if isinstance(to, str) else to
    payload = {
        "from": from_email,
        "to": recipients,
        "subject": subject,
        "html": html,
        "reply_to": reply_to,
    }
    resp = requests.post(
        "https://api.resend.com/emails",
        json=payload,
        headers={"Authorization": f"Bearer {key}"},
        timeout=15,
    )
    if resp.status_code in (200, 201):
        return {"ok": True, "id": resp.json().get("id")}
    LOG.error("[executor] email failed: %s", resp.text[:200])
    return {"ok": False, "error": resp.text[:200]}


# ─── Slack ────────────────────────────────────────────────────────────────────

def slack_notify(message: str, channel: str = "#broadcast") -> dict:
    url = _slack_url()
    if not url:
        LOG.debug("[executor] SLACK_WEBHOOK_URL not set — notification skipped")
        return {"ok": False, "error": "SLACK_WEBHOOK_URL not set"}
    payload = {"text": message, "channel": channel}
    resp = requests.post(url, json=payload, timeout=8)
    return {"ok": resp.status_code == 200}


# ─── Blog Index ───────────────────────────────────────────────────────────────

BLOG_INDEX_PATH = "blog/posts.json"

def _load_blog_index() -> list[dict]:
    existing = github_read(BLOG_INDEX_PATH)
    if existing["exists"] and existing["content"]:
        try:
            return json.loads(existing["content"])
        except Exception:
            pass
    return []


def update_blog_index(post_meta: dict) -> dict:
    """Add or update a post in posts.json. post_meta must have: slug, title, date, description, path."""
    posts = _load_blog_index()
    slugs = [p["slug"] for p in posts]
    if post_meta["slug"] in slugs:
        posts = [post_meta if p["slug"] == post_meta["slug"] else p for p in posts]
    else:
        posts.insert(0, post_meta)
    content = json.dumps(posts, indent=2)
    return github_push(BLOG_INDEX_PATH, content, f"blog: update index — {post_meta['slug']}")


# ─── Blog Publisher ───────────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return re.sub(r"-+", "-", text)[:60]


def _build_blog_html(
    title: str,
    meta_title: str,
    meta_description: str,
    body_html: str,
    slug: str,
    date_str: str,
    reading_time: int = 5,
) -> str:
    canonical = f"{SITE_URL}/blog/{slug}.html"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{meta_title}</title>
  <meta name="description" content="{meta_description}"/>
  <link rel="canonical" href="{canonical}"/>
  <meta name="robots" content="index, follow"/>
  <meta property="og:type" content="article"/>
  <meta property="og:title" content="{meta_title}"/>
  <meta property="og:description" content="{meta_description}"/>
  <meta property="og:url" content="{canonical}"/>
  <meta property="og:image" content="{SITE_URL}/og-image.png"/>
  <meta name="twitter:card" content="summary_large_image"/>
  <meta name="twitter:title" content="{meta_title}"/>
  <meta name="twitter:description" content="{meta_description}"/>
  <script type="application/ld+json">{{
    "@context":"https://schema.org",
    "@type":"Article",
    "headline":"{title}",
    "datePublished":"{date_str}",
    "dateModified":"{date_str}",
    "author":{{"@type":"Organization","name":"Aethyro"}},
    "publisher":{{"@type":"Organization","name":"Aethyro","url":"{SITE_URL}"}},
    "url":"{canonical}",
    "description":"{meta_description}"
  }}</script>
  <link rel="icon" href="/favicon.svg" type="image/svg+xml"/>
  <script async src="https://www.googletagmanager.com/gtag/js?id=G-BE319Z71PD"></script>
  <script>window.dataLayer=window.dataLayer||[];function gtag(){{dataLayer.push(arguments);}}gtag('js',new Date());gtag('config','G-BE319Z71PD');</script>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet"/>
  <style>
    *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
    :root{{--bg:#030303;--s1:#0a0a0a;--s2:#111214;--b1:#1c1c1e;--b2:#2a2a2d;--orange:#ff4d00;--orange2:#ff7a3d;--cyan:#00d4ff;--green:#22c55e;--text:#f0f0f0;--t2:#aaaaaa;--t3:#555;--font:'Space Grotesk',system-ui,sans-serif;--mono:'JetBrains Mono',monospace}}
    body{{font-family:var(--font);background:var(--bg);color:var(--text);line-height:1.7}}
    nav{{position:sticky;top:0;z-index:200;background:rgba(3,3,3,.92);backdrop-filter:blur(20px);border-bottom:1px solid var(--b1)}}
    .nav-inner{{max-width:900px;margin:0 auto;display:flex;align-items:center;justify-content:space-between;padding:.7rem 1.5rem;gap:1rem}}
    .nav-logo{{font-size:1.15rem;font-weight:800;text-decoration:none;color:#fff;display:flex;align-items:center;gap:.35rem}}
    .nav-dot{{width:8px;height:8px;border-radius:50%;background:var(--orange);box-shadow:0 0 8px var(--orange)}}
    .nav-back{{font-size:.8rem;color:var(--t2);text-decoration:none}}
    .nav-back:hover{{color:var(--text)}}
    .article-wrap{{max-width:740px;margin:0 auto;padding:3.5rem 1.5rem 6rem}}
    .eyebrow{{font-family:var(--mono);font-size:.65rem;color:var(--orange);text-transform:uppercase;letter-spacing:.12em;margin-bottom:1rem}}
    h1{{font-size:clamp(1.8rem,4vw,2.8rem);font-weight:800;line-height:1.1;letter-spacing:-1px;margin-bottom:1rem}}
    .meta{{font-family:var(--mono);font-size:.72rem;color:var(--t3);margin-bottom:2.5rem;padding-bottom:1.5rem;border-bottom:1px solid var(--b1);display:flex;gap:1.5rem;flex-wrap:wrap}}
    .article-body h2{{font-size:1.35rem;font-weight:700;margin:2.5rem 0 .75rem;color:#fff}}
    .article-body h3{{font-size:1.05rem;font-weight:700;margin:1.75rem 0 .5rem;color:var(--text)}}
    .article-body p{{color:var(--t2);margin-bottom:1.1rem;font-size:.97rem;line-height:1.8}}
    .article-body ul,.article-body ol{{color:var(--t2);padding-left:1.5rem;margin-bottom:1.1rem}}
    .article-body li{{margin-bottom:.4rem;font-size:.97rem;line-height:1.7}}
    .article-body strong{{color:var(--text)}}
    .article-body blockquote{{border-left:3px solid var(--orange);padding:.75rem 1.25rem;margin:1.5rem 0;background:rgba(255,77,0,.04);border-radius:0 8px 8px 0}}
    .article-body blockquote p{{color:var(--text);font-size:1rem;font-style:italic;margin:0}}
    .article-body a{{color:var(--orange);text-decoration:none}}
    .article-body a:hover{{text-decoration:underline}}
    .article-body code{{font-family:var(--mono);font-size:.82rem;background:var(--s2);border:1px solid var(--b1);padding:.1rem .35rem;border-radius:4px;color:var(--cyan)}}
    .cta-box{{margin:3rem 0;padding:2rem;background:linear-gradient(var(--s2),var(--s2)) padding-box,linear-gradient(135deg,rgba(255,77,0,.5),rgba(0,212,255,.35)) border-box;border:1px solid transparent;border-radius:16px;text-align:center}}
    .cta-box h3{{font-size:1.1rem;font-weight:800;margin-bottom:.5rem}}
    .cta-box p{{color:var(--t2);font-size:.88rem;margin-bottom:1.25rem}}
    .cta-btn{{display:inline-flex;align-items:center;gap:.4rem;padding:.75rem 1.75rem;background:var(--orange);color:#fff;font-weight:700;border-radius:10px;text-decoration:none;font-size:.9rem;transition:background .2s}}
    .cta-btn:hover{{background:var(--orange2)}}
    footer{{border-top:1px solid var(--b1);padding:2rem 1.5rem;text-align:center;font-size:.72rem;color:var(--t3);font-family:var(--mono)}}
    footer a{{color:var(--orange);text-decoration:none}}
    @media(max-width:600px){{.article-wrap{{padding:2rem 1rem 4rem}}}}
  </style>
</head>
<body>
<nav>
  <div class="nav-inner">
    <a href="/" class="nav-logo"><span class="nav-dot"></span> Aethyro</a>
    <a href="/blog/" class="nav-back">← All articles</a>
  </div>
</nav>
<article class="article-wrap">
  <div class="eyebrow">Aethyro Blog</div>
  <h1>{title}</h1>
  <div class="meta">
    <span>{date_str}</span>
    <span>{reading_time} min read</span>
    <span>by Aethyro</span>
  </div>
  <div class="article-body">
    {body_html}
  </div>
  <div class="cta-box">
    <h3>Run AI on your own hardware — starting at $29/month</h3>
    <p>Private local AI for families and small businesses. No cloud, no data leaks, no surprise bills. 14-day free trial.</p>
    <a href="https://buy.stripe.com/4gM8wQ60HfnUbZQedx7bW00" target="_blank" rel="noopener" class="cta-btn">Start Free Trial →</a>
  </div>
</article>
<footer>
  © 2026 Aethyro &nbsp;·&nbsp; <a href="/">Home</a> &nbsp;·&nbsp; <a href="/blog/">Blog</a> &nbsp;·&nbsp; <a href="/terms.html">Terms</a> &nbsp;·&nbsp; <a href="/privacy.html">Privacy</a>
</footer>
</body>
</html>"""


def _build_blog_index_html(posts: list[dict]) -> str:
    cards = ""
    for p in posts:
        cards += f"""
    <a href="/blog/{p['slug']}.html" class="post-card">
      <div class="post-date">{p.get('date','')}</div>
      <h2>{p.get('title','')}</h2>
      <p>{p.get('description','')}</p>
      <span class="read-more">Read article →</span>
    </a>"""
    count = len(posts)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Aethyro Blog — AI Accessibility, Local AI, and the Digital Divide</title>
  <meta name="description" content="Articles on private local AI, AI accessibility for families and small businesses, and how to run AI on your own hardware without cloud costs."/>
  <link rel="canonical" href="https://aethyro.com/blog/"/>
  <meta name="robots" content="index, follow"/>
  <meta property="og:title" content="Aethyro Blog"/>
  <meta property="og:description" content="Private AI, AI accessibility, and the digital divide — from the team building local AI for everyone."/>
  <meta property="og:url" content="https://aethyro.com/blog/"/>
  <meta property="og:type" content="website"/>
  <script type="application/ld+json">{{
    "@context":"https://schema.org",
    "@type":"Blog",
    "name":"Aethyro Blog",
    "url":"https://aethyro.com/blog/",
    "description":"Private local AI for families and businesses.",
    "publisher":{{"@type":"Organization","name":"Aethyro","url":"https://aethyro.com"}}
  }}</script>
  <link rel="icon" href="/favicon.svg" type="image/svg+xml"/>
  <script async src="https://www.googletagmanager.com/gtag/js?id=G-BE319Z71PD"></script>
  <script>window.dataLayer=window.dataLayer||[];function gtag(){{dataLayer.push(arguments);}}gtag('js',new Date());gtag('config','G-BE319Z71PD');</script>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet"/>
  <style>
    *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
    :root{{--bg:#030303;--s1:#0a0a0a;--s2:#111214;--b1:#1c1c1e;--b2:#2a2a2d;--orange:#ff4d00;--orange2:#ff7a3d;--text:#f0f0f0;--t2:#aaaaaa;--t3:#555;--font:'Space Grotesk',system-ui,sans-serif;--mono:'JetBrains Mono',monospace}}
    body{{font-family:var(--font);background:var(--bg);color:var(--text);line-height:1.6}}
    nav{{position:sticky;top:0;z-index:200;background:rgba(3,3,3,.92);backdrop-filter:blur(20px);border-bottom:1px solid var(--b1)}}
    .nav-inner{{max-width:900px;margin:0 auto;display:flex;align-items:center;justify-content:space-between;padding:.7rem 1.5rem}}
    .nav-logo{{font-size:1.15rem;font-weight:800;text-decoration:none;color:#fff;display:flex;align-items:center;gap:.35rem}}
    .nav-dot{{width:8px;height:8px;border-radius:50%;background:var(--orange);box-shadow:0 0 8px var(--orange)}}
    .nav-link{{font-size:.8rem;color:var(--t2);text-decoration:none;padding:.4rem .85rem;border-radius:6px;transition:color .15s}}
    .nav-link:hover{{color:var(--text)}}
    .blog-wrap{{max-width:900px;margin:0 auto;padding:4rem 1.5rem 6rem}}
    .blog-header{{margin-bottom:3rem}}
    .eyebrow{{font-family:var(--mono);font-size:.65rem;color:var(--orange);text-transform:uppercase;letter-spacing:.12em;margin-bottom:.75rem}}
    .blog-header h1{{font-size:clamp(2rem,4vw,3rem);font-weight:800;letter-spacing:-1.5px;margin-bottom:.6rem}}
    .blog-header p{{color:var(--t2);font-size:.95rem;max-width:520px}}
    .post-count{{font-family:var(--mono);font-size:.68rem;color:var(--t3);margin-top:1.5rem}}
    .posts-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:1rem}}
    .post-card{{display:flex;flex-direction:column;gap:.6rem;background:var(--s2);border:1px solid var(--b1);border-radius:14px;padding:1.6rem;text-decoration:none;color:inherit;transition:border-color .18s,transform .15s}}
    .post-card:hover{{border-color:var(--orange);transform:translateY(-2px)}}
    .post-date{{font-family:var(--mono);font-size:.64rem;color:var(--t3)}}
    .post-card h2{{font-size:1.05rem;font-weight:700;color:#fff;line-height:1.35}}
    .post-card p{{font-size:.83rem;color:var(--t2);line-height:1.65;flex:1}}
    .read-more{{font-size:.78rem;color:var(--orange);font-weight:600;margin-top:.25rem}}
    footer{{border-top:1px solid var(--b1);padding:2rem 1.5rem;text-align:center;font-size:.72rem;color:var(--t3);font-family:var(--mono)}}
    footer a{{color:var(--orange);text-decoration:none}}
    @media(max-width:600px){{.posts-grid{{grid-template-columns:1fr}}.blog-wrap{{padding:2rem 1rem 4rem}}}}
  </style>
</head>
<body>
<nav>
  <div class="nav-inner">
    <a href="/" class="nav-logo"><span class="nav-dot"></span> Aethyro</a>
    <a href="/" class="nav-link">← Home</a>
  </div>
</nav>
<div class="blog-wrap">
  <div class="blog-header">
    <div class="eyebrow">Aethyro Blog</div>
    <h1>AI for everyone.<br/>Not just the cloud class.</h1>
    <p>Stories, guides, and research on private local AI — for families, small businesses, and communities who deserve the same tools as Fortune 500 companies.</p>
    <div class="post-count">// {count} article{"s" if count != 1 else ""} published</div>
  </div>
  <div class="posts-grid">
    {cards}
  </div>
</div>
<footer>
  © 2026 Aethyro &nbsp;·&nbsp; <a href="/">Home</a> &nbsp;·&nbsp; <a href="/terms.html">Terms</a> &nbsp;·&nbsp; <a href="/privacy.html">Privacy</a> &nbsp;·&nbsp; <a href="mailto:leer4030@gmail.com">Contact</a>
</footer>
</body>
</html>"""


def publish_blog_post(
    title: str,
    body_html: str,
    meta_title: str = "",
    meta_description: str = "",
    slug: str = "",
    reading_time: int = 5,
) -> dict:
    """Full pipeline: build HTML → push post → update index → rebuild index page → Slack."""
    date_str = datetime.now(timezone.utc).strftime("%B %d, %Y")
    slug = slug or _slugify(title)
    meta_title = meta_title or f"{title} — Aethyro Blog"
    meta_description = meta_description or title[:155]

    post_html = _build_blog_html(
        title=title,
        meta_title=meta_title[:60] if len(meta_title) > 60 else meta_title,
        meta_description=meta_description[:155],
        body_html=body_html,
        slug=slug,
        date_str=date_str,
        reading_time=reading_time,
    )

    push_result = github_push(
        f"blog/{slug}.html",
        post_html,
        f"blog: publish '{title}'",
    )
    if not push_result["ok"]:
        return {"ok": False, "stage": "post_push", "error": push_result.get("error")}

    post_meta = {
        "slug": slug,
        "title": title,
        "description": meta_description,
        "date": date_str,
        "path": f"/blog/{slug}.html",
        "url": f"{SITE_URL}/blog/{slug}.html",
    }
    update_blog_index(post_meta)

    posts = _load_blog_index()
    index_html = _build_blog_index_html(posts)
    github_push("blog/index.html", index_html, f"blog: rebuild index ({len(posts)} posts)")

    live_url = f"{SITE_URL}/blog/{slug}.html"
    slack_notify(f"📝 New blog post published: *{title}*\n{live_url}", "#broadcast")
    LOG.info("[executor] blog post live: %s", live_url)

    return {"ok": True, "url": live_url, "slug": slug, "post_meta": post_meta}


def push_site_fix(
    file_path: str,
    new_content: str,
    description: str,
) -> dict:
    """Push a fix to any landing page file. Used by SEO/design agents."""
    result = github_push(
        file_path,
        new_content,
        f"fix({file_path}): {description}",
    )
    if result["ok"]:
        slack_notify(f"🔧 Site fix deployed: `{file_path}` — {description}", "#broadcast")
    return result
