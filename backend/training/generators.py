"""
Training data generators — hybrid: Haiku (budget-capped) + Groq/Ollama (free).

Cost model (claude-haiku-4-5-20251001):
    $0.80 / MTok input  |  $4.00 / MTok output
    ~$0.0016 per training example (300 in + 350 out tokens)

Budget defaults:
    TRAIN_BUDGET_TARGET = $5.00  (soft — logs warning, keeps going)
    TRAIN_BUDGET_HARD   = $7.50  (hard stop — switches to free-only)

Set TRAIN_USE_ANTHROPIC=1 to enable paid path.
Leave 0 for fully free (Groq → Google → Ollama).
"""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

LOG = logging.getLogger("ghost.training.generators")

OUT_DIR = Path(__file__).parent / "datasets"
OUT_DIR.mkdir(exist_ok=True)

# ── Haiku pricing (as of 2025) ────────────────────────────────
_HAIKU_MODEL   = "claude-haiku-4-5-20251001"
_IN_PER_TOK    = 0.80  / 1_000_000   # $ per input token
_OUT_PER_TOK   = 4.00  / 1_000_000   # $ per output token

BUDGET_TARGET  = float(os.environ.get("TRAIN_BUDGET_TARGET", "5.00"))
BUDGET_HARD    = float(os.environ.get("TRAIN_BUDGET_HARD",   "7.50"))


# ─────────────────────────────────────────────────────────────
# Cost tracker — shared across all datasets in a pipeline run
# ─────────────────────────────────────────────────────────────
class CostTracker:
    def __init__(self, hard_limit: float = BUDGET_HARD,
                 target: float = BUDGET_TARGET):
        self.spent         = 0.0
        self.hard_limit    = hard_limit
        self.target        = target
        self.input_tokens  = 0
        self.output_tokens = 0
        self.calls         = 0
        self.free_calls    = 0

    def record_paid(self, in_tok: int, out_tok: int) -> float:
        cost = (in_tok * _IN_PER_TOK) + (out_tok * _OUT_PER_TOK)
        self.spent         += cost
        self.input_tokens  += in_tok
        self.output_tokens += out_tok
        self.calls         += 1
        if self.spent >= self.target:
            LOG.info("cost tracker: $%.4f spent (target $%.2f reached)",
                     self.spent, self.target)
        return cost

    def record_free(self):
        self.free_calls += 1

    def over_hard_limit(self) -> bool:
        return self.spent >= self.hard_limit

    def remaining(self) -> float:
        return max(0.0, self.hard_limit - self.spent)

    def use_paid(self) -> bool:
        return (os.environ.get("TRAIN_USE_ANTHROPIC") == "1"
                and bool(os.environ.get("ANTHROPIC_API_KEY"))
                and not self.over_hard_limit())

    def to_dict(self) -> dict:
        return {
            "spent":         round(self.spent, 4),
            "target":        self.target,
            "hard_limit":    self.hard_limit,
            "remaining":     round(self.remaining(), 4),
            "paid_calls":    self.calls,
            "free_calls":    self.free_calls,
            "input_tokens":  self.input_tokens,
            "output_tokens": self.output_tokens,
            "over_target":   self.spent >= self.target,
            "over_limit":    self.over_hard_limit(),
        }


# Global tracker — reset at pipeline start
_tracker: CostTracker = CostTracker()


def reset_tracker(hard_limit: float = BUDGET_HARD,
                  target: float = BUDGET_TARGET) -> CostTracker:
    global _tracker
    _tracker = CostTracker(hard_limit=hard_limit, target=target)
    return _tracker


def get_tracker() -> CostTracker:
    return _tracker


# ─────────────────────────────────────────────────────────────
# LLM routing — Haiku (paid, budget-gated) → free fallback
# ─────────────────────────────────────────────────────────────
async def _call_haiku(system: str, user: str) -> tuple[str, int, int]:
    """Returns (text, input_tokens, output_tokens)."""
    import anthropic
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    client = anthropic.AsyncAnthropic(api_key=key)
    resp = await client.messages.create(
        model=_HAIKU_MODEL,
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return (resp.content[0].text,
            resp.usage.input_tokens,
            resp.usage.output_tokens)


async def _llm(system: str, user: str) -> str:
    """
    Budget-aware hybrid routing.
    Paid (Haiku) while under hard limit, then free forever after.
    """
    if _tracker.use_paid():
        try:
            text, in_tok, out_tok = await _call_haiku(system, user)
            cost = _tracker.record_paid(in_tok, out_tok)
            LOG.debug("haiku call: $%.5f | total $%.4f / $%.2f",
                      cost, _tracker.spent, _tracker.hard_limit)
            return text.strip()
        except Exception as e:
            LOG.warning("haiku failed, switching to free: %s", e)

    _tracker.record_free()
    from ghost_llm import nightly_chat
    text, _ = await nightly_chat("training", system, user)
    return text.strip()


# ─────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────
def _write(path: Path, record: dict):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def _count(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def _extract_json(raw: str) -> dict | None:
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(raw[start:end])
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# Dataset 1 — Adversarial Defense  (target: 5,000)
# ─────────────────────────────────────────────────────────────
DEFENSE_SYS = """You are a defensive security expert generating training data.
Given a threat vector, produce a JSON object with exactly these fields:
{
  "threat_vector": "...",
  "exploitation_method": "brief technical description of how attacker exploits this",
  "detection_pattern": "specific log entries, anomalies, or indicators to detect this",
  "mitigation_strategy": "concrete defensive countermeasure to prevent or stop this"
}
Respond with valid JSON only. No explanation. Focus on detection and defense, never on weaponization."""

_THREAT_SEEDS = [
    "SQL injection via login form","XSS reflected attack on search parameter",
    "SSRF via URL parameter in image fetcher","Path traversal in file download endpoint",
    "Broken object level authorization in REST API","JWT algorithm confusion attack",
    "LDAP injection in authentication","XML external entity injection",
    "Command injection via filename parameter","Insecure deserialization in session cookie",
    "Open redirect via callback URL","CSRF on state-changing API endpoint",
    "Subdomain takeover via dangling CNAME","HTTP request smuggling",
    "GraphQL introspection abuse","Rate limiting bypass via IP rotation",
    "Mass assignment vulnerability in ORM","Insecure direct object reference in profile endpoint",
    "SSTI in Jinja2 template engine","DNS rebinding attack on localhost services",
    "Prototype pollution in JavaScript","ReDoS via malicious regex input",
    "Account enumeration via timing difference","Password reset token predictability",
    "Clickjacking via iframe embedding","Cache poisoning via unkeyed header",
    "Business logic flaw in coupon redemption","Information disclosure via verbose error",
    "Weak cryptographic key generation","Session fixation after login",
    "CORS misconfiguration allowing arbitrary origin","WebSocket hijacking",
    "NoSQL injection in MongoDB query","Template injection in email system",
    "Blind SSRF via out-of-band DNS","Race condition in concurrent requests",
    "HTTP parameter pollution","Insecure file permissions on config files",
    "Default credentials on admin interface","Unvalidated redirect after login",
]


async def generate_adversarial_defense(
    target: int = 5000,
    nvd_records: list[dict] | None = None,
    mitre_records: list[dict] | None = None,
) -> int:
    out = OUT_DIR / "adversarial_defense.jsonl"
    existing = _count(out)
    if existing >= target:
        LOG.info("adversarial_defense already at %d/%d", existing, target)
        return existing

    seeds = list(_THREAT_SEEDS)
    for rec in (nvd_records or [])[:500]:
        desc = rec.get("description", "")
        if desc and len(desc) > 40:
            seeds.append(desc[:200])
    for rec in (mitre_records or [])[:200]:
        name = rec.get("name", "")
        if name:
            seeds.append(f"ATT&CK: {name}")

    generated = existing
    random.shuffle(seeds)
    seed_cycle = (seeds * ((target // len(seeds)) + 2))

    for seed in seed_cycle[existing:target]:
        try:
            raw = await _llm(DEFENSE_SYS, f"Threat vector: {seed}")
            obj = _extract_json(raw)
            if not obj:
                continue
            if not {"threat_vector","exploitation_method",
                    "detection_pattern","mitigation_strategy"}.issubset(obj):
                continue
            _write(out, obj)
            generated += 1
            if generated % 200 == 0:
                LOG.info("adversarial_defense: %d/%d | cost: %s",
                         generated, target, _tracker.to_dict())
        except Exception as e:
            LOG.debug("defense gen error: %s", e)

    return generated


# ─────────────────────────────────────────────────────────────
# Dataset 2 — Multi-turn Reasoning Chains  (target: 3,000)
# ─────────────────────────────────────────────────────────────
REASONING_SYS = """You are generating reasoning chain training data for an AI agent.
Given a question, produce a JSON object with exactly these fields:
{
  "question": "...",
  "reasoning_steps": ["step 1", "step 2", "step 3", "step 4", "step 5"],
  "data_sources": ["what information or evidence was considered"],
  "synthesis": "how the steps combine to reach the answer",
  "final_answer": "concise conclusion"
}
Respond with valid JSON only. Make reasoning explicit, auditable, and step-by-step."""

_REASONING_SEEDS = [
    "Why should an AI agent choose rate limiting over CAPTCHA for brute force prevention?",
    "How do you determine if a CVE is critical enough to patch immediately?",
    "What factors should influence the choice of encryption algorithm for stored passwords?",
    "Why is defense-in-depth more effective than perimeter-only security?",
    "How should a security agent prioritize incidents during a multi-vector attack?",
    "What makes a bug bounty report valid vs invalid?",
    "Why is input validation at server side necessary even with client-side checks?",
    "What signals indicate a system is being used as a pivot point in an attack?",
    "Why should secrets be rotated on a schedule rather than only after compromise?",
    "How do you evaluate whether a third-party library introduces supply chain risk?",
    "What evidence distinguishes a security researcher from a malicious actor?",
    "How does threat modeling change risk prioritization decisions?",
    "Why is least privilege important and how is it violated in practice?",
    "How should a system respond to authenticated user performing unusual bulk exports?",
    "What reasoning process identifies a zero-day from behavioral anomalies alone?",
    "Why is memory-safe code preferable to manual memory management for security?",
    "How do you decide between logging more vs logging less for security monitoring?",
    "Why might two identical vulnerabilities have different risk scores?",
    "How should an AI agent handle conflicting security policies from two sources?",
    "What is the correct order of operations when responding to an active breach?",
]


async def generate_reasoning_chains(
    target: int = 3000,
    hf_records: list[dict] | None = None,
) -> int:
    out = OUT_DIR / "reasoning_chains.jsonl"
    existing = _count(out)
    if existing >= target:
        return existing

    seeds = list(_REASONING_SEEDS)
    for rec in (hf_records or [])[:300]:
        q = rec.get("question", "")
        if q and len(q) > 20:
            seeds.append(q[:300])

    generated = existing
    seed_cycle = (seeds * ((target // len(seeds)) + 2))

    for seed in seed_cycle[existing:target]:
        try:
            raw = await _llm(REASONING_SYS, f"Question: {seed}")
            obj = _extract_json(raw)
            if not obj:
                continue
            if not {"question","reasoning_steps","data_sources",
                    "synthesis","final_answer"}.issubset(obj):
                continue
            if not isinstance(obj["reasoning_steps"], list):
                continue
            _write(out, obj)
            generated += 1
            if generated % 200 == 0:
                LOG.info("reasoning_chains: %d/%d | cost: $%.4f",
                         generated, target, _tracker.spent)
        except Exception as e:
            LOG.debug("reasoning gen error: %s", e)

    return generated


# ─────────────────────────────────────────────────────────────
# Dataset 3 — CVE Pattern Analysis  (target: 3,000)
# ─────────────────────────────────────────────────────────────
CVE_SYS = """You are a security analyst generating training data about vulnerability patterns.
Given a CVE description, produce a JSON object with exactly these fields:
{
  "vulnerability_pattern": "abstract pattern this CVE represents",
  "discovery_indicators": ["observable signs that this class of vulnerability exists"],
  "exploitation_timeline": "typical time from discovery to weaponization for this pattern",
  "defensive_lessons": "what defenders should implement to prevent this class of issue"
}
Respond with valid JSON only. Focus on pattern recognition and defense."""


async def generate_cve_patterns(
    target: int = 3000,
    nvd_records: list[dict] | None = None,
) -> int:
    out = OUT_DIR / "cve_patterns.jsonl"
    existing = _count(out)
    if existing >= target:
        return existing

    records = [r for r in (nvd_records or [])
               if r.get("description") and len(r["description"]) > 60]
    if not records:
        LOG.warning("no NVD records — run collection first")
        return existing

    random.shuffle(records)
    generated = existing

    for rec in records[existing:existing + (target - existing)]:
        try:
            desc   = rec["description"][:500]
            cve_id = rec.get("cve_id", "unknown")
            raw    = await _llm(CVE_SYS, f"CVE ID: {cve_id}\nDescription: {desc}")
            obj    = _extract_json(raw)
            if not obj:
                continue
            if not {"vulnerability_pattern","discovery_indicators",
                    "exploitation_timeline","defensive_lessons"}.issubset(obj):
                continue
            obj["source_cve"]  = cve_id
            obj["cvss_score"]  = rec.get("cvss_score", 0)
            _write(out, obj)
            generated += 1
            if generated % 200 == 0:
                LOG.info("cve_patterns: %d/%d | cost: $%.4f",
                         generated, target, _tracker.spent)
        except Exception as e:
            LOG.debug("cve gen error: %s", e)

    return generated


# ─────────────────────────────────────────────────────────────
# Dataset 4 — Bug Bounty Methodologies  (target: 5,000)
# ─────────────────────────────────────────────────────────────
BOUNTY_SYS = """You are a security researcher generating ethical bug bounty training data.
Given a target system type and vulnerability class, produce a JSON object:
{
  "target_system": "type of system",
  "recon_method": "passive or non-intrusive discovery technique",
  "vulnerability_found": "specific vulnerability class and location",
  "non_weaponized_poc": "proof of concept that ONLY demonstrates existence",
  "impact_assessment": "business impact if exploited",
  "remediation": "specific code-level or configuration fix"
}
All examples must be from a defensive/researcher perspective. Valid JSON only."""

_BOUNTY_SEEDS = [
    ("REST API", "IDOR on user profile endpoint"),
    ("web application", "stored XSS in comment field"),
    ("mobile app", "insecure data storage in local database"),
    ("GraphQL API", "introspection revealing sensitive schema"),
    ("OAuth flow", "state parameter missing CSRF protection"),
    ("file upload endpoint", "unrestricted file type allowing SSRF"),
    ("admin panel", "broken access control on user management"),
    ("password reset flow", "token reuse after password change"),
    ("API gateway", "rate limiting absent on authentication endpoint"),
    ("webhook handler", "SSRF via controllable callback URL"),
    ("email verification", "link still valid after email change"),
    ("payment form", "amount parameter tampering in checkout"),
    ("search functionality", "SQL injection via sort parameter"),
    ("image processing", "SSRF via image URL fetch"),
    ("export feature", "path traversal in filename parameter"),
    ("user settings", "account takeover via email change without verification"),
    ("API versioning", "old API version bypasses new auth controls"),
    ("CDN configuration", "cache poisoning via unkeyed Host header"),
    ("session management", "session not invalidated after logout"),
    ("debug endpoint", "stack trace leaking internal paths in production"),
    ("mobile API", "certificate pinning bypass in debug build"),
    ("third-party integration", "OAuth token leakage via referrer header"),
    ("multi-tenant SaaS", "tenant data isolation failure"),
    ("CI/CD pipeline", "secrets exposed in build logs"),
    ("kubernetes dashboard", "unauthenticated access via exposed NodePort"),
]


async def generate_bug_bounty(
    target: int = 5000,
    mitre_records: list[dict] | None = None,
) -> int:
    out = OUT_DIR / "bug_bounty.jsonl"
    existing = _count(out)
    if existing >= target:
        return existing

    seeds = list(_BOUNTY_SEEDS)
    for rec in (mitre_records or [])[:100]:
        name = rec.get("name", "")
        if name:
            seeds.append(("web application", name))

    generated = existing
    seed_cycle = (seeds * ((target // len(seeds)) + 2))

    for target_sys, vuln_class in seed_cycle[existing:target]:
        try:
            raw = await _llm(BOUNTY_SYS,
                             f"Target system: {target_sys}\nVulnerability: {vuln_class}")
            obj = _extract_json(raw)
            if not obj:
                continue
            if not {"target_system","recon_method","vulnerability_found",
                    "non_weaponized_poc","impact_assessment","remediation"}.issubset(obj):
                continue
            _write(out, obj)
            generated += 1
            if generated % 200 == 0:
                LOG.info("bug_bounty: %d/%d | cost: $%.4f",
                         generated, target, _tracker.spent)
        except Exception as e:
            LOG.debug("bounty gen error: %s", e)

    return generated


# ─────────────────────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────────────────────
def dataset_stats() -> dict:
    stats = {}
    for p in OUT_DIR.glob("*.jsonl"):
        stats[p.stem] = _count(p)
    return stats
