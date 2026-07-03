"""FastAPI router for /site/* endpoints — Aethyro.com elite superagent system."""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .orchestrator import ingest_site, full_site_audit, full_business_audit, run_task
from .rag_store import query as rag_query, stats as rag_stats
from .memory_layer import recall, recent_tasks
from .agents import ALL_AGENTS

LOG = logging.getLogger("site_agents.router")

router = APIRouter(prefix="/site", tags=["site-agents"])


# ── Request models ─────────────────────────────────────────────────────────────

class TaskRequest(BaseModel):
    agent: str
    task_type: str
    payload: dict = {}


class QueryRequest(BaseModel):
    query: str
    n: int = 5
    category: Optional[str] = None


class CampaignRequest(BaseModel):
    goal: str
    audience: str = "lower-income families seeking AI tools"
    tone: str = "warm_direct"


class LegalGenerateRequest(BaseModel):
    doc_type: str  # terms | privacy | disclaimer | refund | coppa


class RoleRequest(BaseModel):
    title: str
    responsibilities: str = ""


class IncidentRequest(BaseModel):
    service: str


class BrandAuditRequest(BaseModel):
    include_competitive: bool = True

class BlogPublishRequest(BaseModel):
    topic: str
    angle: str = ""

class WelcomeEmailRequest(BaseModel):
    to_email: str
    name: str = ""

# ── Contractor agent pack models ───────────────────────────────────────────────

class InvoiceDraftRequest(BaseModel):
    description: str
    user_id: str = ""
    session_id: str = ""
    client_name: str = ""
    client_email: str = ""

class InvoiceSetRateRequest(BaseModel):
    user_id: str
    service: str
    rate: float
    unit: str = "hr"

class InvoiceSetBusinessRequest(BaseModel):
    user_id: str
    name: str
    phone: str = ""
    email: str = ""

class QuoteDraftRequest(BaseModel):
    job_description: str
    user_id: str = ""
    session_id: str = ""
    client_name: str = ""
    client_email: str = ""

class QuoteEstimateRequest(BaseModel):
    job_type: str
    user_id: str = ""
    session_id: str = ""

class ClientCommsReplyRequest(BaseModel):
    client_message: str
    user_id: str = ""
    session_id: str = ""
    client_name: str = ""
    message_type: str = "general"

class PaymentReminderRequest(BaseModel):
    client_name: str
    amount: float
    days_overdue: int
    invoice_number: str = ""
    user_id: str = ""
    session_id: str = ""

class ComplaintRequest(BaseModel):
    complaint: str
    client_name: str = ""
    user_id: str = ""
    session_id: str = ""

class InquiryReplyRequest(BaseModel):
    inquiry: str
    client_name: str = ""
    user_id: str = ""
    session_id: str = ""

class ScheduleBookRequest(BaseModel):
    request: str
    user_id: str = ""
    session_id: str = ""
    existing_events: list = []
    today: str = ""

class ScheduleListWeekRequest(BaseModel):
    user_id: str = ""
    session_id: str = ""
    week_start: str = ""

class RescheduleRequest(BaseModel):
    original_booking: str
    new_time_request: str
    user_id: str = ""
    session_id: str = ""
    client_name: str = ""

class WorkingHoursRequest(BaseModel):
    user_id: str
    hours: str

class JobDurationRequest(BaseModel):
    user_id: str
    job_type: str
    hours: float

class SignupRequest(BaseModel):
    email: str
    first_name: str = ""

class SupabaseAuthHook(BaseModel):
    type: str = ""
    record: dict = {}

class MarketplaceListRequest(BaseModel):
    platform: str = "fiverr"  # fiverr | upwork | linkedin
    product: str = "local_ai_setup"

SITE_URL = "https://aethyro.com"


# ── Core endpoints ─────────────────────────────────────────────────────────────

@router.get("/agents")
async def list_agents():
    """List all 13 superagents with roles and expertise."""
    agents = []
    for name, cls in ALL_AGENTS.items():
        inst = cls()
        agents.append({"name": inst.name, "role": inst.role, "expertise": inst.expertise})
    return {"agents": agents, "total": len(agents)}


@router.get("/status")
async def site_status():
    """RAG store stats, recent tasks, Firecrawl availability."""
    from site_agents.integrations import firecrawl_client as fc
    return {
        "rag": rag_stats(),
        "recent_tasks": recent_tasks(limit=10),
        "domain": "aethyro.com",
        "firecrawl": fc.available(),
        "agents": len(ALL_AGENTS),
    }


@router.post("/ingest")
async def ingest(max_pages: int = 20):
    """Crawl aethyro.com and store pages in RAG. Uses Firecrawl if FIRECRAWL_API_KEY is set."""
    try:
        return await ingest_site(max_pages=max_pages)
    except Exception as e:
        LOG.error("[site/ingest] %s", e)
        raise HTTPException(500, str(e))


@router.post("/audit")
async def audit_site():
    """SEO + design + content audit across all pages."""
    try:
        return await full_site_audit()
    except Exception as e:
        LOG.error("[site/audit] %s", e)
        raise HTTPException(500, str(e))


@router.post("/task")
async def agent_task(req: TaskRequest):
    """Run any task with any agent.

    Examples:
      {"agent": "seo", "task_type": "audit_page", "payload": {"url": "https://aethyro.com"}}
      {"agent": "stripe", "task_type": "report", "payload": {}}
      {"agent": "legal", "task_type": "terms", "payload": {}}
      {"agent": "brand", "task_type": "messaging", "payload": {}}
      {"agent": "ops", "task_type": "health", "payload": {}}
    """
    if req.agent not in ALL_AGENTS:
        raise HTTPException(400, f"Unknown agent '{req.agent}'. Valid: {sorted(ALL_AGENTS)}")
    try:
        return await run_task(req.agent, req.task_type, req.payload)
    except Exception as e:
        LOG.error("[site/task] agent=%s task=%s err=%s", req.agent, req.task_type, e)
        raise HTTPException(500, str(e))


@router.post("/knowledge/query")
async def knowledge_query(req: QueryRequest):
    """Semantic search across all stored site + business knowledge in RAG."""
    results = await rag_query(req.query, n=req.n)
    return {"query": req.query, "results": results, "count": len(results)}


@router.get("/knowledge/recall")
async def knowledge_recall(agent: Optional[str] = None, category: Optional[str] = None, limit: int = 20):
    """Retrieve stored memory by agent/category."""
    results = recall(agent=agent, category=category, limit=limit)
    return {"results": results, "count": len(results)}


@router.get("/knowledge/tasks")
async def task_history(agent: Optional[str] = None, limit: int = 20):
    """Task execution history across all agents."""
    results = recent_tasks(agent=agent, limit=limit)
    return {"tasks": results, "count": len(results)}


# ── Business superagent endpoints ──────────────────────────────────────────────

@router.post("/business/audit")
async def business_audit():
    """Run ALL 13 agents in parallel. Full business + site intelligence report."""
    try:
        return await full_business_audit()
    except Exception as e:
        LOG.error("[site/business/audit] %s", e)
        raise HTTPException(500, str(e))


@router.get("/stripe/report")
async def stripe_report():
    """Revenue dashboard: MRR, ARR, churn, subscriptions, strategic insights."""
    try:
        return await get_agent_result("stripe", "revenue_report")
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/stripe/pricing")
async def stripe_pricing():
    """Pricing optimization analysis and tier recommendations."""
    try:
        return await get_agent_result("stripe", "pricing_optimization")
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/analytics/report")
async def analytics_report():
    """GA4 traffic dashboard: sessions, users, top pages, top sources, insights."""
    try:
        return await get_agent_result("analytics", "traffic_report")
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/analytics/kpis")
async def analytics_kpis():
    """Define and set up Aethyro's 7 critical KPIs."""
    try:
        return await get_agent_result("analytics", "kpi_setup")
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/analytics/cro")
async def analytics_cro(page_url: str = "https://aethyro.com"):
    """Conversion rate optimization audit for a specific page."""
    try:
        from .agents import get_agent
        agent = get_agent("analytics")
        return await agent.conversion_optimization(page_url)
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/brand/audit")
async def brand_audit(req: BrandAuditRequest):
    """Brand audit + optional competitive positioning analysis."""
    try:
        from .agents import get_agent
        agent = get_agent("brand")
        result = await agent.brand_audit()
        if req.include_competitive:
            positioning = await agent.positioning_statement()
            result["positioning"] = positioning
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/brand/messaging")
async def brand_messaging():
    """Complete messaging framework: taglines, elevator pitches, objection handlers."""
    try:
        return await get_agent_result("brand", "messaging_framework")
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/brand/voice")
async def brand_voice():
    """Brand voice guide: tone, vocabulary, writing rules, copy examples."""
    try:
        return await get_agent_result("brand", "brand_voice_guide")
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/legal/generate")
async def legal_generate(req: LegalGenerateRequest):
    """Generate a legal document. doc_type: terms | privacy | disclaimer | refund | coppa"""
    valid = {"terms", "privacy", "disclaimer", "refund", "coppa"}
    if req.doc_type not in valid:
        raise HTTPException(400, f"Invalid doc_type. Valid: {sorted(valid)}")
    try:
        from .agents import get_agent
        agent = get_agent("legal")
        dispatch = {
            "terms":      agent.generate_terms_of_service,
            "privacy":    agent.generate_privacy_policy,
            "disclaimer": agent.generate_ai_disclaimer,
            "refund":     agent.generate_refund_policy,
            "coppa":      agent.coppa_compliance_check,
        }
        return await dispatch[req.doc_type]()
    except Exception as e:
        LOG.error("[site/legal/generate] %s", e)
        raise HTTPException(500, str(e))


@router.get("/ops/health")
async def ops_health():
    """Live health check of all Aethyro services with latency measurements."""
    try:
        return await get_agent_result("ops", "service_health_check")
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/ops/security")
async def ops_security():
    """Security audit of the full Aethyro stack."""
    try:
        return await get_agent_result("ops", "security_audit")
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/ops/incident")
async def ops_incident(req: IncidentRequest):
    """Incident response playbook for a specific service going down."""
    try:
        from .agents import get_agent
        agent = get_agent("ops")
        return await agent.incident_response(req.service)
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/email/campaign")
async def email_campaign(req: CampaignRequest):
    """Write a complete email campaign with subject, body, CTA, A/B variant."""
    try:
        from .agents import get_agent
        agent = get_agent("email")
        return await agent.write_campaign(req.goal, req.audience, req.tone)
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/email/list-building")
async def email_list_building():
    """Email list growth strategy with lead magnets and acquisition channels."""
    try:
        return await get_agent_result("email", "list_building_strategy")
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/hr/role")
async def hr_role(req: RoleRequest):
    """Design a complete job description for a new Aethyro role."""
    try:
        from .agents import get_agent
        agent = get_agent("hr")
        return await agent.design_role(req.title, req.responsibilities)
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/hr/hiring-strategy")
async def hr_hiring_strategy():
    """Aethyro hiring roadmap: priority roles, compensation philosophy, sourcing."""
    try:
        return await get_agent_result("hr", "hiring_strategy")
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Firecrawl utility endpoints ────────────────────────────────────────────────

@router.get("/firecrawl/status")
async def firecrawl_status():
    """Check Firecrawl availability and site map."""
    from site_agents.integrations import firecrawl_client as fc
    if not fc.available():
        return {"available": False, "reason": "Set FIRECRAWL_API_KEY in backend/.env"}
    urls = fc.map_site(SITE_URL)
    return {"available": True, "site_url": SITE_URL, "urls_found": len(urls), "urls": urls[:20]}


@router.post("/firecrawl/scrape")
async def firecrawl_scrape(url: str = "https://aethyro.com"):
    """Scrape a URL with Firecrawl for clean markdown extraction."""
    from site_agents.integrations import firecrawl_client as fc
    result = fc.scrape(url)
    if result.get("fallback"):
        raise HTTPException(503, f"Firecrawl unavailable: {result.get('error', 'no key')}")
    return result


# ── Blog publishing endpoints ──────────────────────────────────────────────────

@router.post("/blog/publish")
async def blog_publish(req: BlogPublishRequest):
    """Write AND publish a blog post live to aethyro.com/blog/ via GitHub Pages."""
    try:
        from .agents import get_agent
        agent = get_agent("journalist")
        return await agent.write_and_publish(req.topic, req.angle)
    except Exception as e:
        LOG.error("[site/blog/publish] %s", e)
        raise HTTPException(500, str(e))


@router.get("/blog/posts")
async def blog_posts():
    """List all published blog posts from the GitHub Pages manifest."""
    from site_agents.executor import github_read
    import asyncio, json
    result = await asyncio.to_thread(github_read, "blog/posts.json")
    if result["exists"]:
        posts = json.loads(result["content"])
        return {"posts": posts, "count": len(posts)}
    return {"posts": [], "count": 0}


@router.post("/blog/publish-batch")
async def blog_publish_batch(topics: list[str]):
    """Publish multiple blog posts in sequence. Max 10 at once."""
    if len(topics) > 10:
        raise HTTPException(400, "Max 10 topics per batch")
    from .agents import get_agent
    agent = get_agent("journalist")
    results = []
    for topic in topics:
        try:
            r = await agent.write_and_publish(topic)
            results.append({"topic": topic, "ok": r.get("ok"), "url": r.get("url")})
        except Exception as e:
            results.append({"topic": topic, "ok": False, "error": str(e)})
    published = sum(1 for r in results if r.get("ok"))
    return {"published": published, "total": len(topics), "results": results}


# ── Email execution endpoints ──────────────────────────────────────────────────

@router.post("/email/send-welcome")
async def email_send_welcome(req: WelcomeEmailRequest):
    """Send a real welcome email via Resend. Requires RESEND_API_KEY in .env."""
    try:
        from .agents import get_agent
        agent = get_agent("email")
        return await agent.send_welcome(req.to_email, req.name)
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/email/signup")
async def email_signup(req: SignupRequest):
    """Send Email 1 immediately and queue Emails 2-5 for the onboarding scheduler.
    Call this whenever a new user signs up."""
    try:
        from .agents import get_agent
        agent = get_agent("email")
        return await agent.schedule_onboarding(req.email, req.first_name)
    except Exception as e:
        LOG.error("[site/email/signup] %s", e)
        raise HTTPException(500, str(e))


@router.post("/email/auth-hook")
async def supabase_auth_hook(req: SupabaseAuthHook):
    """Supabase Auth webhook — fires on INSERT to auth.users.
    Set this as the Auth webhook URL in your Supabase dashboard:
      https://your-gateway-url/site/email/auth-hook
    """
    if req.type not in ("INSERT", ""):
        return {"ok": True, "skipped": True, "reason": f"event type {req.type!r} ignored"}
    record = req.record
    email = record.get("email", "")
    meta = record.get("raw_user_meta_data") or {}
    first_name = meta.get("first_name") or meta.get("name", "").split()[0] if meta.get("name") else ""
    if not email:
        return {"ok": False, "error": "no email in record"}
    try:
        from .agents import get_agent
        agent = get_agent("email")
        result = await agent.schedule_onboarding(email, first_name)
        LOG.info("[auth-hook] onboarding started for %s", email)
        return {**result, "email": email}
    except Exception as e:
        LOG.error("[site/email/auth-hook] %s", e)
        raise HTTPException(500, str(e))


@router.get("/email/schedule")
async def email_schedule_status(limit: int = 50):
    """View the onboarding email schedule — pending and sent."""
    import sqlite3
    from site_agents.memory_layer import DB_PATH
    with sqlite3.connect(str(DB_PATH)) as c:
        c.row_factory = sqlite3.Row
        rows = c.execute(
            "SELECT * FROM email_schedule ORDER BY send_after DESC LIMIT ?", (limit,)
        ).fetchall()
    return {"schedule": [dict(r) for r in rows], "count": len(rows)}


# ── Marketplace listing generator ─────────────────────────────────────────────

@router.post("/marketplace/generate-listing")
async def marketplace_generate_listing(req: MarketplaceListRequest):
    """Generate a ready-to-post Fiverr/Upwork/LinkedIn listing."""
    try:
        from .agents import get_agent
        agent = get_agent("marketplace")
        return await agent.create_listing(req.platform, req.product)
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/marketplace/upwork-proposal")
async def upwork_proposal(job_description: str):
    """Write a winning Upwork proposal for a specific job posting."""
    try:
        from .agents import get_agent
        agent = get_agent("marketplace")
        return await agent.write_upwork_proposal(job_description)
    except Exception as e:
        raise HTTPException(500, str(e))


# ── SEO auto-fix endpoint ──────────────────────────────────────────────────────

@router.post("/seo/auto-fix")
async def seo_auto_fix(file_path: str, issues: list[str]):
    """Read a landing page, fix SEO issues, push live via GitHub. file_path = e.g. 'index.html'"""
    try:
        from .agents import get_agent
        from site_agents.executor import github_read
        import asyncio
        existing = await asyncio.to_thread(github_read, file_path)
        if not existing["exists"]:
            raise HTTPException(404, f"File not found in repo: {file_path}")
        agent = get_agent("seo")
        return await agent.auto_fix_page(file_path, existing["content"], issues)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Contractor Agent Pack ──────────────────────────────────────────────────────

@router.post("/contractor/invoice/draft")
async def contractor_invoice_draft(req: InvoiceDraftRequest):
    """Draft a professional invoice from a plain-language description."""
    try:
        from .agents import get_agent
        return await get_agent("invoice").draft(
            req.description, req.user_id, req.session_id, req.client_name, req.client_email)
    except Exception as e:
        LOG.error("[contractor/invoice/draft] %s", e)
        raise HTTPException(500, str(e))


@router.post("/contractor/invoice/set-rate")
async def contractor_invoice_set_rate(req: InvoiceSetRateRequest):
    """Store a service rate to the user's memory."""
    try:
        from .agents import get_agent
        return await get_agent("invoice").set_rate(req.user_id, req.service, req.rate, req.unit)
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/contractor/invoice/set-business")
async def contractor_invoice_set_business(req: InvoiceSetBusinessRequest):
    """Store business identity (name, phone, email) to user memory."""
    try:
        from .agents import get_agent
        return await get_agent("invoice").set_business(req.user_id, req.name, req.phone, req.email)
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/contractor/quote/draft")
async def contractor_quote_draft(req: QuoteDraftRequest):
    """Generate a professional quote from a job description."""
    try:
        from .agents import get_agent
        return await get_agent("quote").draft(
            req.job_description, req.user_id, req.session_id, req.client_name, req.client_email)
    except Exception as e:
        LOG.error("[contractor/quote/draft] %s", e)
        raise HTTPException(500, str(e))


@router.post("/contractor/quote/estimate")
async def contractor_quote_estimate(req: QuoteEstimateRequest):
    """Quick ballpark estimate for a job type based on stored rates."""
    try:
        from .agents import get_agent
        return await get_agent("quote").estimate_range(req.job_type, req.user_id, req.session_id)
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/contractor/comms/reply")
async def contractor_comms_reply(req: ClientCommsReplyRequest):
    """Draft a professional reply to a client message."""
    try:
        from .agents import get_agent
        return await get_agent("client_comms").draft_reply(
            req.client_message, req.user_id, req.session_id, req.client_name, req.message_type)
    except Exception as e:
        LOG.error("[contractor/comms/reply] %s", e)
        raise HTTPException(500, str(e))


@router.post("/contractor/comms/payment-reminder")
async def contractor_payment_reminder(req: PaymentReminderRequest):
    """Draft a payment reminder — tone scales with days overdue."""
    try:
        from .agents import get_agent
        return await get_agent("client_comms").payment_reminder(
            req.client_name, req.amount, req.days_overdue,
            req.invoice_number, req.user_id, req.session_id)
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/contractor/comms/complaint")
async def contractor_complaint(req: ComplaintRequest):
    """Draft a measured response to a client complaint."""
    try:
        from .agents import get_agent
        return await get_agent("client_comms").complaint_response(
            req.complaint, req.client_name, req.user_id, req.session_id)
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/contractor/comms/inquiry")
async def contractor_inquiry(req: InquiryReplyRequest):
    """Reply to a new client inquiry — warm, move toward booking."""
    try:
        from .agents import get_agent
        return await get_agent("client_comms").new_inquiry_reply(
            req.inquiry, req.client_name, req.user_id, req.session_id)
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/contractor/schedule/book")
async def contractor_schedule_book(req: ScheduleBookRequest):
    """Book a job from a natural language request."""
    try:
        from .agents import get_agent
        return await get_agent("schedule").book(
            req.request, req.user_id, req.session_id,
            req.existing_events or None, req.today or None)
    except Exception as e:
        LOG.error("[contractor/schedule/book] %s", e)
        raise HTTPException(500, str(e))


@router.post("/contractor/schedule/week")
async def contractor_schedule_week(req: ScheduleListWeekRequest):
    """Summarize the user's booked week from memory."""
    try:
        from .agents import get_agent
        return await get_agent("schedule").list_week(
            req.user_id, req.session_id, req.week_start or None)
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/contractor/schedule/reschedule")
async def contractor_reschedule(req: RescheduleRequest):
    """Reschedule an existing booking."""
    try:
        from .agents import get_agent
        return await get_agent("schedule").reschedule(
            req.original_booking, req.new_time_request,
            req.user_id, req.session_id, req.client_name)
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/contractor/schedule/set-hours")
async def contractor_set_hours(req: WorkingHoursRequest):
    """Store working hours to user memory."""
    try:
        from .agents import get_agent
        return await get_agent("schedule").set_working_hours(req.user_id, req.hours)
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/contractor/schedule/set-duration")
async def contractor_set_duration(req: JobDurationRequest):
    """Store default job duration to user memory."""
    try:
        from .agents import get_agent
        return await get_agent("schedule").set_job_duration(req.user_id, req.job_type, req.hours)
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/contractor/agents")
async def contractor_agents():
    """List the 4 contractor agents with roles and endpoints."""
    return {
        "agents": [
            {"name": "invoice", "role": "Invoice & Billing Specialist",
             "endpoints": ["/contractor/invoice/draft", "/contractor/invoice/set-rate", "/contractor/invoice/set-business"]},
            {"name": "quote", "role": "Quote & Estimation Specialist",
             "endpoints": ["/contractor/quote/draft", "/contractor/quote/estimate"]},
            {"name": "client_comms", "role": "Client Communications Specialist",
             "endpoints": ["/contractor/comms/reply", "/contractor/comms/payment-reminder",
                           "/contractor/comms/complaint", "/contractor/comms/inquiry"]},
            {"name": "schedule", "role": "Scheduling & Calendar Specialist",
             "endpoints": ["/contractor/schedule/book", "/contractor/schedule/week",
                           "/contractor/schedule/reschedule", "/contractor/schedule/set-hours",
                           "/contractor/schedule/set-duration"]},
        ],
        "total": 4,
        "memory_enabled": True,
        "note": "Pass user_id in all requests to enable persistent memory across sessions.",
    }


# ── Helper ─────────────────────────────────────────────────────────────────────

async def get_agent_result(agent_name: str, method: str) -> dict:
    from .agents import get_agent
    agent = get_agent(agent_name)
    return await getattr(agent, method)()
