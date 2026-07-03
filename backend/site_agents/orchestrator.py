"""Site agent orchestrator — coordinates multi-agent site improvement workflows."""
from __future__ import annotations

import asyncio
import logging
import time

from .crawler import crawl_site, fetch_page
from .rag_store import upsert_page, store_knowledge, stats as rag_stats
from .memory_layer import store, recent_tasks
from .agents import get_agent, ALL_AGENTS

LOG = logging.getLogger("site_agents.orchestrator")

SITE_URL = "https://aethyro.com"


async def ingest_site(max_pages: int = 20) -> dict:
    """Crawl aethyro.com (Firecrawl if available, else httpx/BS4) and store in RAG."""
    LOG.info("[orchestrator] starting site ingest, max_pages=%d", max_pages)

    # Try Firecrawl first — richer content, JS-rendered pages
    from site_agents.integrations import firecrawl_client as fc
    pages_raw = []
    if fc.available():
        LOG.info("[orchestrator] using Firecrawl for ingest")
        fc_pages = fc.crawl(SITE_URL, max_pages=max_pages)
        if fc_pages:
            stored = 0
            for fp in fc_pages:
                pdata = fc.markdown_to_page_data(fp, SITE_URL)
                import hashlib
                page_id = hashlib.md5(pdata["url"].encode()).hexdigest()[:12]
                text = f"URL: {pdata['url']}\nTitle: {pdata['title']}\nDescription: {pdata['description']}\nContent: {pdata['markdown'][:3000]}"
                ok = await upsert_page(page_id, text, {"url": pdata["url"], "title": pdata["title"], "source": "firecrawl"})
                if ok:
                    stored += 1
            summary = {"pages_crawled": len(fc_pages), "pages_stored": stored, "errors": 0,
                       "source": "firecrawl", "rag_stats": rag_stats()}
            store("orchestrator", "ingest", "last_run", summary)
            return summary

    # Fallback to httpx/BS4 crawler
    LOG.info("[orchestrator] using httpx/BS4 crawler (no FIRECRAWL_API_KEY)")
    pages = await crawl_site(SITE_URL, max_pages=max_pages)
    stored = 0
    errors = 0
    for page in pages:
        if page.error:
            errors += 1
            continue
        ok = await upsert_page(
            page.page_id,
            page.to_rag_text(),
            {"url": page.url, "title": page.title,
             "word_count": str(page.word_count), "status_code": str(page.status_code)},
        )
        if ok:
            stored += 1

    summary = {
        "pages_crawled": len(pages),
        "pages_stored": stored,
        "errors": errors,
        "source": "httpx_bs4",
        "rag_stats": rag_stats(),
        "pages": [{"url": p.url, "title": p.title, "word_count": p.word_count,
                   "seo_score": p.seo_summary().get("score", 0), "error": p.error or None}
                  for p in pages],
    }
    store("orchestrator", "ingest", "last_run", summary)
    return summary


async def full_site_audit() -> dict:
    """Run SEO/design/content agents against all crawled pages."""
    LOG.info("[orchestrator] starting full site audit")
    pages = await crawl_site(SITE_URL, max_pages=15)
    live_pages = [p for p in pages if not p.error]

    results = {}
    seo  = get_agent("seo")
    des  = get_agent("design")
    cont = get_agent("content")

    for page in live_pages[:10]:
        pdata = {
            "url": page.url, "title": page.title, "description": page.description,
            "h1": page.h1, "h2": page.h2, "h3": page.h3, "body_text": page.body_text,
            "images": page.images, "word_count": page.word_count,
            "og_image": page.og_image, "seo_summary": page.seo_summary(),
        }
        seo_r, des_r, cont_r = await asyncio.gather(
            seo.analyze_page(pdata), des.analyze_design(pdata), cont.audit_content(pdata),
            return_exceptions=True,
        )
        results[page.url] = {
            "seo":     seo_r if not isinstance(seo_r, Exception) else {"error": str(seo_r)},
            "design":  des_r if not isinstance(des_r, Exception) else {"error": str(des_r)},
            "content": cont_r if not isinstance(cont_r, Exception) else {"error": str(cont_r)},
            "seo_score": page.seo_summary().get("score", 0),
        }

    audit_text = f"Full site audit {time.strftime('%Y-%m-%d')}: {len(results)} pages analyzed.\n"
    for url, r in results.items():
        audit_text += f"\nURL: {url}\n  SEO score: {r.get('seo_score')}\n"
    await store_knowledge("audit_full", audit_text, "site_audit", ["full_audit", "aethyro"])

    return {"pages_audited": len(results), "results": results, "timestamp": time.time()}


async def full_business_audit() -> dict:
    """Run ALL 13 agents in parallel — comprehensive site + business intelligence report."""
    LOG.info("[orchestrator] starting full business audit — %d agents", len(ALL_AGENTS))
    start = time.time()

    agents = {name: get_agent(name) for name in ALL_AGENTS}

    (
        site_result,
        revenue_result,
        traffic_result,
        brand_result,
        health_result,
        legal_result,
        email_result,
        hr_result,
        marketing_result,
        seo_kw_result,
    ) = await asyncio.gather(
        full_site_audit(),
        agents["stripe"].revenue_report(),
        agents["analytics"].traffic_report(),
        agents["brand"].brand_audit(),
        agents["ops"].service_health_check(),
        agents["legal"].generate_ai_disclaimer(),
        agents["email"].list_building_strategy(),
        agents["hr"].hiring_strategy(),
        agents["marketing"].competitive_analysis(),
        agents["seo"].keyword_research("affordable local AI for families"),
        return_exceptions=True,
    )

    def _safe(r):
        return {"error": str(r)} if isinstance(r, Exception) else r

    elapsed = round(time.time() - start, 1)
    report = {
        "audit_type": "full_business_audit",
        "agents_run": len(ALL_AGENTS),
        "elapsed_seconds": elapsed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "site_audit":       _safe(site_result),
        "revenue":          _safe(revenue_result),
        "traffic":          _safe(traffic_result),
        "brand":            _safe(brand_result),
        "ops_health":       _safe(health_result),
        "legal_disclaimer": _safe(legal_result),
        "email_strategy":   _safe(email_result),
        "hr_strategy":      _safe(hr_result),
        "competitive":      _safe(marketing_result),
        "seo_keywords":     _safe(seo_kw_result),
    }

    store("orchestrator", "business_audit", "last_run", {
        "timestamp": report["timestamp"],
        "elapsed_seconds": elapsed,
        "agents_run": report["agents_run"],
    })
    return report


async def run_task(agent_name: str, task_type: str, payload: dict) -> dict:
    """Route a task to the appropriate specialist agent."""
    agent = get_agent(agent_name)

    if agent_name == "seo":
        if task_type == "audit_page":
            url = payload.get("url", SITE_URL)
            page = await fetch_page(url)
            pdata = {"url": page.url, "title": page.title, "description": page.description,
                     "h1": page.h1, "h2": page.h2, "body_text": page.body_text,
                     "images": page.images, "word_count": page.word_count, "seo_summary": page.seo_summary()}
            return await agent.analyze_page(pdata)
        elif task_type == "keywords":
            return await agent.keyword_research(payload.get("topic", "affordable AI for families"))
        elif task_type == "fix_plan":
            return await agent.fix_recommendations(payload.get("audit_results", []))

    elif agent_name == "design":
        if task_type == "audit_page":
            url = payload.get("url", SITE_URL)
            page = await fetch_page(url)
            pdata = {"url": page.url, "title": page.title, "h1": page.h1, "h2": page.h2,
                     "h3": page.h3, "images": page.images, "body_text": page.body_text,
                     "word_count": page.word_count, "og_image": page.og_image}
            return await agent.analyze_design(pdata)
        elif task_type == "design_spec":
            return await agent.generate_design_spec(payload.get("component", "hero section"))

    elif agent_name == "content":
        if task_type == "audit_page":
            url = payload.get("url", SITE_URL)
            page = await fetch_page(url)
            pdata = {"url": page.url, "title": page.title, "description": page.description,
                     "h1": page.h1, "h2": page.h2, "body_text": page.body_text, "word_count": page.word_count}
            return await agent.audit_content(pdata)
        elif task_type == "create":
            return await agent.create_page_content(payload.get("page_type", "blog post"),
                payload.get("topic", "affordable AI"), payload.get("keywords"))
        elif task_type == "homepage":
            page = await fetch_page(SITE_URL)
            pdata = {"url": page.url, "title": page.title, "description": page.description,
                     "h1": page.h1, "body_text": page.body_text}
            return await agent.improve_homepage(pdata)

    elif agent_name == "marketing":
        if task_type == "analytics":
            pages = await crawl_site(SITE_URL, max_pages=8)
            pdata = [{"url": p.url, "word_count": p.word_count, "links": p.links} for p in pages if not p.error]
            return await agent.analytics_strategy(pdata)
        elif task_type == "campaign":
            return await agent.growth_campaign(payload.get("channel", "organic"), payload.get("budget", "$0"))
        elif task_type == "email":
            return await agent.email_strategy()
        elif task_type == "competitive":
            return await agent.competitive_analysis(payload.get("competitors"))

    elif agent_name == "journalist":
        if task_type == "blog":
            return await agent.write_blog_post(payload.get("topic", "AI equity"),
                payload.get("angle", ""), payload.get("word_count", 800))
        elif task_type == "press_release":
            return await agent.press_release(payload.get("announcement", "Aethyro launches"))
        elif task_type == "trend":
            return await agent.trend_analysis(payload.get("trend", "AI in education"))
        elif task_type == "newsletter":
            return await agent.write_newsletter(payload.get("week_of", time.strftime("%B %d, %Y")),
                payload.get("highlights"))

    elif agent_name == "marketplace":
        if task_type == "list":
            return await agent.list_on_platform(payload.get("product", "local_ai_setup"),
                payload.get("platform", "fiverr"))
        elif task_type == "proposal":
            return await agent.write_upwork_proposal(payload.get("job_description", ""))
        elif task_type == "strategy":
            return await agent.platform_strategy()
        elif task_type == "optimize":
            return await agent.optimize_existing_listing(payload.get("platform", "fiverr"),
                payload.get("current_listing", ""))
        elif task_type == "research":
            return await agent.competitor_research(payload.get("platform", "fiverr"),
                payload.get("search_term", "AI assistant setup"))
        elif task_type == "products":
            from .agents.marketplace_agent import AETHYRO_PRODUCTS, PLATFORMS
            return {"products": AETHYRO_PRODUCTS, "platforms": list(PLATFORMS.keys())}

    elif agent_name == "stripe":
        if task_type == "report":
            return await agent.revenue_report()
        elif task_type == "pricing":
            return await agent.pricing_optimization()
        elif task_type == "churn":
            return await agent.churn_analysis()
        elif task_type == "forecast":
            return await agent.growth_forecast(payload.get("months", 6))

    elif agent_name == "email":
        if task_type == "campaign":
            return await agent.write_campaign(payload.get("goal", "acquire subscribers"),
                payload.get("audience", "lower-income families"), payload.get("tone", "warm_direct"))
        elif task_type == "sequence":
            return await agent.write_sequence(payload.get("trigger", "new subscriber"), payload.get("steps", 5))
        elif task_type == "newsletter":
            return await agent.write_newsletter(payload.get("highlights", []))
        elif task_type == "list_building":
            return await agent.list_building_strategy()
        elif task_type == "deliverability":
            return await agent.deliverability_audit()

    elif agent_name == "legal":
        doc_type = task_type or payload.get("doc_type", "terms")
        dispatch = {
            "terms": agent.generate_terms_of_service,
            "privacy": agent.generate_privacy_policy,
            "disclaimer": agent.generate_ai_disclaimer,
            "refund": agent.generate_refund_policy,
            "coppa": agent.coppa_compliance_check,
        }
        if doc_type in dispatch:
            return await dispatch[doc_type]()

    elif agent_name == "hr":
        if task_type == "role":
            return await agent.design_role(payload.get("title", "Engineer"), payload.get("responsibilities", ""))
        elif task_type == "org_chart":
            return await agent.build_org_chart(payload.get("stage", "pre-revenue"))
        elif task_type == "onboarding":
            return await agent.onboarding_playbook(payload.get("role", "Engineer"))
        elif task_type == "hiring_strategy":
            return await agent.hiring_strategy()
        elif task_type == "contractor":
            return await agent.contractor_agreement_template(payload.get("role", "Developer"))

    elif agent_name == "analytics":
        if task_type == "traffic":
            return await agent.traffic_report()
        elif task_type == "funnel":
            return await agent.funnel_analysis()
        elif task_type == "cro":
            return await agent.conversion_optimization(payload.get("page_url", SITE_URL))
        elif task_type == "dashboard":
            return await agent.growth_dashboard()
        elif task_type == "kpis":
            return await agent.kpi_setup()

    elif agent_name == "brand":
        if task_type == "audit":
            return await agent.brand_audit()
        elif task_type == "messaging":
            return await agent.messaging_framework()
        elif task_type == "voice":
            return await agent.brand_voice_guide()
        elif task_type == "visual":
            return await agent.visual_identity_spec()
        elif task_type == "positioning":
            return await agent.positioning_statement()

    elif agent_name == "ops":
        if task_type == "health":
            return await agent.service_health_check()
        elif task_type == "cost":
            return await agent.cost_analysis()
        elif task_type == "security":
            return await agent.security_audit()
        elif task_type == "runbook":
            return await agent.deployment_runbook()
        elif task_type == "incident":
            return await agent.incident_response(payload.get("service", "gateway"))

    # Generic fallback
    return await agent.run_task(task_type, payload.get("prompt", task_type))
