"""
kairos_dataset_gen.py — Generate KAIROS-structured SFT pairs for Avery fine-tuning.

Produces instruction/response pairs where every response walks through all 6 KAIROS
phases: Kickoff, Alignment, Implementation, Refinement, Optimization, Scaling.

Providers (auto-detected from .env, or pass --provider):
  groq      — FREE tier, llama-3.3-70b-versatile. Get key: console.groq.com
  anthropic — Claude Opus (credits required)
  ollama    — Local Ollama model (no key, needs Ollama running)

Run:
  python kairos_dataset_gen.py                          # auto-detect provider, 500 pairs
  python kairos_dataset_gen.py --provider groq          # force Groq
  python kairos_dataset_gen.py --provider ollama        # use local Ollama
  python kairos_dataset_gen.py --pairs 200              # custom count
  python kairos_dataset_gen.py --dry-run                # preview prompts, no API calls
  python kairos_dataset_gen.py --append                 # append to existing file
"""
import argparse, json, os, sys, time, random, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

ROOT   = Path(__file__).parent
DATA   = ROOT / "data"
OUTPUT = DATA / "kairos_dataset.jsonl"

GROQ_BASE_URL    = "https://api.groq.com/openai/v1"
GROQ_MODEL       = "llama-3.3-70b-versatile"
GROQ_MODEL_FAST  = "llama-3.1-8b-instant"   # ~4x cheaper tokens, use when TPD quota is tight
OLLAMA_BASE_URL  = "http://localhost:11434/v1"
OLLAMA_MODEL     = "avery"
MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
MISTRAL_MODEL    = "open-mistral-7b"
ANTHROPIC_MODEL  = "claude-opus-4-7"


def _load_env():
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))

_load_env()

# ── KAIROS system prompt ──────────────────────────────────────────────────────

SYSTEM = (
    "You are Avery, the sovereign business strategist for SovereignNation — "
    "a fixed-cost AI platform built for lower and middle class families, "
    "children's education, and affordable connectivity. "
    "You always reason through strategy using the KAIROS framework, labeling each phase clearly:\n"
    "  K — Kickoff: Define the mission, stakeholders, and success criteria.\n"
    "  A — Alignment: Assess resources, gaps, constraints, and opportunities.\n"
    "  I — Implementation: Concrete action steps with owners and timelines.\n"
    "  R — Refinement: What to test, validate, and iterate on.\n"
    "  O — Optimization: KPIs, friction reduction, efficiency gains.\n"
    "  S — Scaling: How to expand once the model is proven.\n"
    "Be direct, structured, and actionable. Always include all 6 phases."
)

# ── Prompt bank ───────────────────────────────────────────────────────────────

PROMPT_BANK = [
    # Market strategy
    "Build a go-to-market strategy for SovereignNation's $29/month family tier targeting rural communities.",
    "Design a competitive positioning strategy against OpenAI and Google for underserved markets.",
    "Create a B2B strategy selling SovereignNation to employers as an employee benefit.",
    "Develop a partnership strategy with rural ISPs to bundle SovereignNation at no added cost.",
    "Build a market entry plan for SovereignNation in a new geographic region.",
    "Design a strategy for SovereignNation to win government education contracts.",
    "Create a viral referral program that incentivizes existing subscribers to bring in neighbors.",
    "Develop a strategy for SovereignNation to partner with food banks and social services.",
    "Build a corporate social responsibility partnership strategy with Fortune 500 companies.",
    "Design a franchise model for SovereignNation community hubs in underserved areas.",
    "Create a market segmentation strategy for SovereignNation's three tiers: individual, family, education.",
    "Build a strategy to enter the healthcare-adjacent market with AI wellness coaching.",
    "Develop a white-label strategy for SovereignNation to license its platform to nonprofits.",
    "Design a subscription bundle strategy with streaming services and digital tools.",
    "Create a back-to-school campaign targeting families with school-aged children.",
    # Pricing and revenue
    "Design a revenue model that keeps SovereignNation fixed-cost at scale for 100,000 families.",
    "Build a pricing strategy that maintains affordability while covering infrastructure costs.",
    "Create a tiered pricing architecture from free community to premium family to enterprise.",
    "Develop a freemium conversion funnel for SovereignNation's free community tier.",
    "Design a credit-based usage system for SovereignNation's enterprise tier.",
    "Build a revenue sharing model with SovereignNation content partners.",
    "Create a pricing strategy for SovereignNation's children's education module.",
    "Develop a grants and subsidies strategy to make SovereignNation free for qualifying families.",
    "Design a usage-based billing system that stays predictable for low-income households.",
    "Build a loyalty rewards program that increases lifetime value without raising costs.",
    # Product and roadmap
    "Design the 90-day launch roadmap for SovereignNation's Phase 1 market entry.",
    "Develop the product roadmap for SovereignNation's children's education module Q3-Q4.",
    "Create a feature prioritization framework for SovereignNation's next 6 months.",
    "Build a mobile-first strategy for SovereignNation serving users on low-end Android devices.",
    "Design an offline-first capability strategy for areas with unreliable internet.",
    "Create a product strategy for SovereignNation's AI tutoring feature.",
    "Develop a roadmap for multi-language support starting with Spanish.",
    "Build a strategy for integrating SovereignNation with existing school district software.",
    "Design a parental controls and family safety feature rollout plan.",
    "Create a product strategy for SovereignNation's community discussion feature.",
    "Develop a strategy for SovereignNation's AI career counseling module.",
    "Build a voice-interface strategy for users with limited literacy.",
    "Design an accessibility roadmap covering screen readers, low-vision, and motor limitations.",
    "Create a strategy for SovereignNation's job placement and skills training vertical.",
    # Operations and infrastructure
    "Create a cost optimization strategy to keep infrastructure under $5/user/month.",
    "Build an infrastructure scaling plan from 10,000 to 1,000,000 users.",
    "Design a disaster recovery strategy ensuring 99.9% uptime for SovereignNation.",
    "Develop a multi-cloud strategy reducing vendor lock-in and lowering costs.",
    "Create a content delivery strategy for SovereignNation in bandwidth-constrained regions.",
    "Build an edge computing strategy to reduce latency for rural users.",
    "Design a data sovereignty strategy ensuring family data stays private and local.",
    "Develop a GDPR and COPPA compliance roadmap for SovereignNation.",
    "Create a cybersecurity strategy for SovereignNation's consumer platform.",
    "Build a vendor negotiation strategy for SovereignNation's GPU compute costs.",
    # Growth and retention
    "Build a community-driven growth strategy leveraging existing users as advocates.",
    "Create a retention strategy for SovereignNation's fixed-cost subscribers.",
    "Develop a churn prevention strategy for families who haven't logged in for 30 days.",
    "Design a win-back campaign for cancelled SovereignNation subscribers.",
    "Build a net promoter score improvement strategy for SovereignNation.",
    "Create a content strategy that drives weekly active usage.",
    "Develop a social proof strategy using real family success stories.",
    "Design a gamification strategy for SovereignNation's learning features.",
    "Build a cross-sell strategy from individual to family tier.",
    "Create a seasonal engagement strategy tied to school calendars.",
    # Funding and sustainability
    "Develop a grant and funding strategy targeting social impact investors.",
    "Build an impact investment pitch for SovereignNation's Series A.",
    "Create a government funding strategy leveraging broadband equity programs.",
    "Develop a nonprofit endowment strategy for SovereignNation's long-term sustainability.",
    "Design a crowdfunding campaign for SovereignNation community hubs.",
    "Build a revenue diversification strategy to reduce dependence on subscriptions.",
    "Create a corporate sponsorship framework for SovereignNation's education content.",
    "Build a financial model showing break-even at 50,000 subscribers.",
    # People and culture
    "Build a talent acquisition strategy for SovereignNation that aligns with its mission.",
    "Create a remote-first culture strategy for SovereignNation's distributed team.",
    "Develop an advisory board strategy bringing in telecom and education experts.",
    "Design a community moderator program for SovereignNation's local chapters.",
    "Build a volunteer and intern program supporting SovereignNation's mission.",
    # Metrics and analytics
    "Develop a metrics framework to track SovereignNation's social impact.",
    "Build a KPI dashboard strategy for SovereignNation's quarterly reviews.",
    "Create an experiment framework for SovereignNation's feature testing.",
    "Develop a cohort analysis strategy for understanding subscriber lifetime value.",
    "Build a customer satisfaction measurement strategy beyond NPS.",
    # Crisis and risk
    "Create a crisis management plan for SovereignNation's service continuity.",
    "Build a competitive response plan if a major tech company launches a free alternative.",
    "Design a regulatory change response framework for SovereignNation.",
    "Develop a data breach response plan for SovereignNation.",
    "Build a public relations crisis playbook for SovereignNation.",
    # AI and technology strategy
    "Design SovereignNation's AI model update and deployment strategy.",
    "Build a responsible AI governance framework for SovereignNation.",
    "Create a strategy for SovereignNation to fine-tune models on community feedback.",
    "Develop a bias detection and mitigation strategy for SovereignNation's AI outputs.",
    "Design a strategy for SovereignNation to offer local-only AI with no cloud dependency.",
    "Build a model evaluation framework for SovereignNation's quarterly model reviews.",
    "Develop a data flywheel strategy where usage improves model quality over time.",
    # Education vertical
    "Build a K-12 curriculum alignment strategy for SovereignNation's AI tutor.",
    "Create a strategy for SovereignNation to partner with homeschool communities.",
    "Develop a teacher professional development program for SovereignNation in schools.",
    "Build a strategy for SovereignNation's adult literacy and GED preparation module.",
    "Create a college access strategy helping first-generation students through SovereignNation.",
    "Develop a STEM enrichment strategy using SovereignNation for underrepresented youth.",
    # Community and social impact
    "Design a strategy for SovereignNation to build digital equity in tribal communities.",
    "Build a refugee and immigrant onboarding strategy for SovereignNation.",
    "Create a senior digital inclusion strategy for SovereignNation.",
    "Develop a reentry program strategy for SovereignNation serving formerly incarcerated individuals.",
    "Design a veterans' support integration strategy for SovereignNation.",
    "Build a rural small business support strategy within SovereignNation.",
    # Agent reasoning and AI system intelligence
    "ORACLE receives a request to retrieve SovereignNation's full pricing history. Walk through the retrieval, synthesis, and delivery strategy.",
    "FORGE is asked to build a FastAPI endpoint for family account management with payment integration. Walk through the complete code generation and architecture strategy.",
    "CODEX reviews a Python script that processes family payment data and personal records. Walk through the full security audit and quality review process.",
    "SENTINEL detects an anomalous spike in API calls from a single IP. Walk through the threat assessment, escalation, and response strategy.",
    "NEXUS must orchestrate a complex task across ORACLE, FORGE, CODEX, and SENTINEL simultaneously. Design the full coordination and dependency workflow.",
    "Design a self-improvement loop where SovereignNation's agents learn from user feedback and usage patterns each week.",
    "Build a strategy for SovereignNation's AI agents to handle ambiguous, conflicting, or incomplete user requests without breaking.",
    "Develop a multi-agent collaboration protocol for completing a research-to-implementation task end-to-end without human intervention.",
    "Create a strategy for maintaining coherent context across a 30-day multi-session user journey within SovereignNation.",
    "Design a knowledge distillation pipeline that compresses expert human knowledge into SovereignNation's AI training data.",
    "Build a strategy for SovereignNation's agents to detect when they are uncertain and escalate gracefully to human review.",
    "Develop a continuous evaluation framework for SovereignNation's AI agents — measuring accuracy, helpfulness, and safety weekly.",
    "Design an agent specialization roadmap: how ORACLE, FORGE, CODEX, SENTINEL, and NEXUS each deepen their domain expertise over 12 months.",
    "Build a cross-agent memory sharing strategy so insights from ORACLE are automatically available to FORGE and CODEX.",
    "Create a red-team strategy where SENTINEL adversarially tests FORGE's outputs before they reach the user.",
    # Token economics and AI economy
    "Design a community token economy for SovereignNation that rewards contribution and engagement without creating inflation.",
    "Build an economic model for SovereignNation's cooperative ownership structure where subscribers earn equity.",
    "Create a digital micro-economy strategy within SovereignNation where members trade skills, time, and knowledge.",
    "Develop a peer-to-peer lending circle strategy powered by SovereignNation's AI for underbanked families.",
    "Design an AI-powered financial literacy curriculum that moves families from financial fragility to stability in 12 months.",
    "Build a strategy for SovereignNation to create a community jobs board with AI-powered matching for gig and full-time roles.",
    "Create a cooperative profit-sharing model where SovereignNation's most engaged users receive dividends proportional to contribution.",
    "Design a universal basic compute strategy — how SovereignNation gives every member free baseline AI access regardless of ability to pay.",
    "Build a strategy for SovereignNation to issue verifiable digital credentials for skills learned on the platform.",
    "Develop a community investment strategy where SovereignNation pools subscriber capital for local economic development.",
    # Specific scenarios
    "SovereignNation just lost its main cloud provider. Build a 30-day migration strategy.",
    "A large EdTech company wants to acquire SovereignNation. Build a response strategy.",
    "SovereignNation's growth has stalled at 15,000 subscribers. Build a growth unlock plan.",
    "A community college wants to embed SovereignNation in their remedial math program. Plan it.",
    "SovereignNation needs to cut costs by 30% without degrading service. Build the plan.",
    "Three competing community AI platforms launched last month. Build a differentiation strategy.",
    "A government agency wants SovereignNation as standard software for housing voucher recipients. Plan the implementation.",
    "SovereignNation's AI tutor is getting 2-star reviews for math. Build a remediation plan.",
    "A viral social media post claims SovereignNation sells user data. Build the response strategy.",
    "SovereignNation has 6 months of runway left. Build a survival and fundraising strategy.",
    "A major school district wants to pilot SovereignNation for 5,000 students next semester. Plan it.",
    "SovereignNation's top 3 engineers are leaving. Build a knowledge retention and hiring strategy.",
    "Usage spikes 10x during back-to-school. Build the infrastructure surge strategy.",
]


def _build_user_msg(instruction: str) -> str:
    return (
        f"Strategic challenge for SovereignNation:\n\n{instruction}\n\n"
        "Respond as Avery using the full KAIROS framework. Label each phase clearly with "
        "## K — Kickoff, ## A — Alignment, ## I — Implementation, ## R — Refinement, "
        "## O — Optimization, ## S — Scaling. "
        "Each section: 3-6 bullet points or a short structured paragraph. "
        "Be specific, concrete, and SovereignNation-context-aware."
    )


# ── Provider clients ──────────────────────────────────────────────────────────

class _GroqRotatingClient:
    """Wraps multiple Groq API keys, rotating on TPD (daily token) 429s."""

    def __init__(self, keys: list[str], base_url: str, OpenAI):
        self._clients = [OpenAI(api_key=k, base_url=base_url) for k in keys]
        self._idx = 0

    def _client(self):
        return self._clients[self._idx]

    def chat_complete(self, model: str, messages: list, max_tokens: int, temperature: float):
        import time
        for attempt in range(len(self._clients) * 2):
            try:
                return self._client().chat.completions.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=messages,
                    temperature=temperature,
                )
            except Exception as e:
                err = str(e)
                tpd   = "rate_limit_exceeded" in err and "tokens per day" in err.lower()
                inv   = "invalid_api_key" in err or '"code": 401' in err or "Error code: 401" in err
                if tpd or inv:
                    reason = "TPD exhausted" if tpd else "invalid key"
                    # Find next untried key index
                    tried = getattr(self, "_tried", set())
                    tried.add(self._idx)
                    self._tried = tried
                    remaining = [i for i in range(len(self._clients)) if i not in tried]
                    if not remaining:
                        raise RuntimeError("All Groq keys exhausted or invalid — use --provider anthropic")
                    self._idx = remaining[0]
                    print(f"  [key {list(tried)[-1]+1} {reason} → rotating to key {self._idx+1}]")
                elif "rate_limit_exceeded" in err:
                    # RPM/TPM limit — brief wait, same key
                    wait = 12
                    print(f"  [rate limit → waiting {wait}s]")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("All Groq keys exhausted their daily token quota")


def _make_client(provider: str, model_override: str | None = None):
    if provider == "groq":
        try:
            from openai import OpenAI
        except ImportError:
            print("ERROR: openai package not installed. Run: pip install openai")
            sys.exit(1)
        # Support multiple keys: GROQ_API_KEY, GROQ_API_KEY_2, GROQ_API_KEY_3
        keys = [v for k, v in sorted(os.environ.items())
                if k.startswith("GROQ_API_KEY") and v]
        if not keys:
            print("ERROR: GROQ_API_KEY not set. Get free key at: https://console.groq.com")
            sys.exit(1)
        # Return a wrapper that holds all keys and can rotate on 429
        model = model_override or GROQ_MODEL
        print(f"  Groq keys found : {len(keys)}")
        return _GroqRotatingClient(keys, GROQ_BASE_URL, OpenAI), model

    elif provider == "mistral":
        try:
            from openai import OpenAI
        except ImportError:
            print("ERROR: openai package not installed. Run: pip install openai")
            sys.exit(1)
        key = os.environ.get("MISTRAL_API_KEY", "")
        if not key:
            print("ERROR: MISTRAL_API_KEY not set"); sys.exit(1)
        model = model_override or MISTRAL_MODEL
        return OpenAI(api_key=key, base_url=MISTRAL_BASE_URL), model

    elif provider == "ollama":
        try:
            from openai import OpenAI
        except ImportError:
            print("ERROR: openai package not installed. Run: pip install openai")
            sys.exit(1)
        return OpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL), OLLAMA_MODEL

    elif provider == "anthropic":
        try:
            import anthropic
        except ImportError:
            print("ERROR: anthropic package not installed. Run: pip install anthropic")
            sys.exit(1)
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            print("ERROR: ANTHROPIC_API_KEY not set in .env")
            sys.exit(1)
        return anthropic.Anthropic(api_key=key), ANTHROPIC_MODEL

    else:
        print(f"ERROR: Unknown provider '{provider}'")
        sys.exit(1)


def _auto_detect_provider() -> str:
    """Pick the best available provider based on set API keys."""
    if os.environ.get("MISTRAL_API_KEY"):    return "mistral"
    if os.environ.get("GROQ_API_KEY"):       return "groq"
    if os.environ.get("ANTHROPIC_API_KEY"):  return "anthropic"
    return "ollama"


def _call_openai_compat(client, model: str, instruction: str) -> str | None:
    """Call OpenAI-compatible API (Groq rotating client, Ollama, etc.)."""
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": _build_user_msg(instruction)},
    ]
    if isinstance(client, _GroqRotatingClient):
        resp = client.chat_complete(model=model, messages=msgs, max_tokens=1800, temperature=0.8)
    else:
        resp = client.chat.completions.create(
            model=model, max_tokens=1800, messages=msgs, temperature=0.8)
    return resp.choices[0].message.content.strip()


def _call_anthropic(client, model: str, instruction: str) -> str | None:
    msg = client.messages.create(
        model=model,
        max_tokens=1800,
        system=SYSTEM,
        messages=[{"role": "user", "content": _build_user_msg(instruction)}],
    )
    return msg.content[0].text.strip()


def _generate_pair(client, model: str, provider: str, instruction: str, dry_run: bool = False) -> dict | None:
    if dry_run:
        print(f"  [DRY RUN] {instruction[:80]}...")
        return {"instruction": instruction, "response": "[DRY RUN]", "source": f"kairos_{provider}", "framework": "KAIROS"}
    try:
        if provider == "anthropic":
            text = _call_anthropic(client, model, instruction)
        else:
            text = _call_openai_compat(client, model, instruction)

        if not text or len(text) < 200:
            print(f"  SKIP — response too short ({len(text) if text else 0} chars)")
            return None
        phases_found = sum(1 for p in ["Kickoff", "Alignment", "Implementation", "Refinement", "Optimization", "Scaling"]
                          if p.lower() in text.lower())
        if phases_found < 4:
            print(f"  SKIP — only {phases_found}/6 KAIROS phases found")
            return None
        return {"instruction": instruction, "response": text, "source": f"kairos_{provider}", "framework": "KAIROS"}
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def _variations(base: str) -> list[str]:
    return [
        base,
        f"Using KAIROS, {base[0].lower()}{base[1:]}",
        f"Walk me through the KAIROS framework applied to: {base}",
        f"As SovereignNation's strategist, {base[0].lower()}{base[1:]}",
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs",    type=int, default=500)
    ap.add_argument("--provider", choices=["groq", "mistral", "anthropic", "ollama"], default=None,
                    help="LLM provider. Default: auto-detect from .env")
    ap.add_argument("--model",    choices=["fast", "full"], default="full",
                    help="fast=llama-3.1-8b-instant (~4x cheaper tokens), full=llama-3.3-70b-versatile")
    ap.add_argument("--workers",  type=int, default=1,
                    help="Parallel generation workers (default 1; use 3-5 for Anthropic, 1 for Groq)")
    ap.add_argument("--dry-run",  action="store_true")
    ap.add_argument("--append",   action="store_true")
    args = ap.parse_args()

    provider = args.provider or (_auto_detect_provider() if not args.dry_run else "groq")
    model_override = GROQ_MODEL_FAST if args.model == "fast" else None

    print(f"\n{'='*58}")
    print("  KAIROS DATASET GENERATOR")
    print(f"{'='*58}")
    print(f"  Provider : {provider}")
    print(f"  Target   : {args.pairs} pairs")
    print(f"  Output   : {OUTPUT}")
    print(f"  Mode     : {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"  File     : {'append' if args.append else 'overwrite'}")
    print()

    if not args.dry_run:
        client, model = _make_client(provider, model_override)
        print(f"  Model    : {model}")
    else:
        client, model = None, "dry-run"

    print()

    # Build instruction list
    all_instructions: list[str] = []
    for prompt in PROMPT_BANK:
        all_instructions.extend(_variations(prompt))
    seen: set[str] = set()
    unique = [x for x in all_instructions if not (x in seen or seen.add(x))]
    random.shuffle(unique)
    while len(unique) < args.pairs:
        extra = list(unique); random.shuffle(extra); unique.extend(extra)
    instructions = unique[:args.pairs]

    DATA.mkdir(exist_ok=True)
    file_mode = "a" if args.append else "w"
    generated = 0
    failed = 0
    workers = max(1, min(args.workers, 10))

    # Groq free tier: force single-worker to avoid 429s
    if provider == "groq" and workers > 1:
        print(f"  [Note] Groq rate limits → forcing --workers 1")
        workers = 1

    print(f"  Workers  : {workers}")
    print()

    _lock = threading.Lock()

    def _run(idx_instr):
        idx, instruction = idx_instr
        print(f"[{idx+1}/{args.pairs}] {instruction[:72]}...")
        return idx, instruction, _generate_pair(client, model, provider, instruction, dry_run=args.dry_run)

    with open(OUTPUT, file_mode, encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run, (i, instr)): i for i, instr in enumerate(instructions)}
            for future in as_completed(futures):
                idx, instruction, pair = future.result()
                if pair:
                    with _lock:
                        f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                        f.flush()
                        generated += 1
                    print(f"  [{idx+1}] OK ({len(pair['response'])} chars) — {generated} total")
                else:
                    with _lock:
                        failed += 1

    print(f"\n{'='*58}")
    print(f"  Done: {generated} generated, {failed} failed")
    print(f"  Output: {OUTPUT}")
    print(f"{'='*58}\n")


if __name__ == "__main__":
    main()
