"""
agents_bootstrap.py — Generate DPO training data for all 6 sovereign agents.

Uses Claude API to generate chosen/rejected pairs tailored to each agent's role
in building the SovereignNation startup.

Run:
  python agents_bootstrap.py                   # all agents, 15 pairs each
  python agents_bootstrap.py --agent forge     # single agent
  python agents_bootstrap.py --pairs 25        # more pairs per agent
  python agents_bootstrap.py --dry-run         # preview prompts, no API calls
"""
import argparse, json, os, sys, time, threading
from pathlib import Path
import io

MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
MISTRAL_MODEL    = "open-mistral-7b"
GROQ_BASE_URL    = "https://api.groq.com/openai/v1"
GROQ_MODEL       = "llama-3.1-8b-instant"

# Force UTF-8 stdout so Windows cp1252 doesn't choke on special characters
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

ROOT = Path(__file__).parent
DATA = ROOT / "data"

def _load_env():
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))

_load_env()

# ── Agent definitions ──────────────────────────────────────────────────────────

AGENTS = {
    "avery": {
        "system": (
            "You are Avery, the sovereign business strategist for SovereignNation — "
            "a fixed-cost AI platform built for lower and middle class families, "
            "children's education, and affordable connectivity. "
            "Use the KAIROS framework: Kickoff, Alignment, Implementation, Refinement, "
            "Optimization, Scaling. Be direct, structured, and actionable."
        ),
        "hf_repo": "tastytator/avery-sovereign-lora",
        "prompts": [
            "Build a go-to-market strategy for SovereignNation's $29/month family tier targeting rural communities.",
            "Design a revenue model that keeps SovereignNation fixed-cost at scale for 100,000 families.",
            "Create competitive positioning against OpenAI and Google for underserved markets.",
            "Develop a partnership strategy with school districts for SovereignNation's education tier.",
            "Build a pricing strategy that maintains affordability while covering infrastructure costs.",
            "Design the 90-day launch roadmap for SovereignNation's Phase 1 market entry.",
            "Create a retention strategy for SovereignNation's fixed-cost subscribers.",
            "Build a community-driven growth strategy leveraging existing users as advocates.",
            "Develop a grant and funding strategy targeting social impact investors.",
            "Design the product roadmap for SovereignNation's children's education module.",
            "Create a cost optimization strategy to keep infrastructure under $5/user/month.",
            "Build a B2B strategy selling SovereignNation to employers as a benefit.",
            "Design a franchise model for SovereignNation community hubs.",
            "Develop a metrics framework to track SovereignNation's social impact.",
            "Create a crisis management plan for SovereignNation's service continuity.",
        ],
    },
    "forge": {
        "system": (
            "You are FORGE, the sovereign code generation specialist for SovereignNation. "
            "You write production-ready Python, JavaScript, and TypeScript. "
            "Always include imports, error handling, type hints (Python), and comments for non-obvious logic. "
            "Code must be secure, tested, and match SovereignNation's FastAPI/React/MongoDB architecture."
        ),
        "hf_repo": "tastytator/forge-sovereign-lora",
        "prompts": [
            "Build a FastAPI endpoint for SovereignNation user subscription management with Stripe webhook handling.",
            "Create a React component for the SovereignNation family dashboard showing AI usage metrics.",
            "Write a Python class for the SovereignNation SwarmBus with async message routing.",
            "Build a JWT authentication middleware for SovereignNation's API gateway.",
            "Create a MongoDB aggregation pipeline reporting monthly active users by pricing tier.",
            "Write a Python script to sync SovereignNation's local Ollama model with the latest HF LoRA.",
            "Build a rate limiter for SovereignNation's API enforcing per-tier request limits.",
            "Create a WebSocket handler for real-time agent status updates in the dashboard.",
            "Write a data migration script for moving users to SovereignNation's v2 schema.",
            "Build a health check endpoint reporting status of all 5 SovereignNation services.",
            "Create a Python background job for SovereignNation's SPIN training data collection.",
            "Write a TypeScript utility to encrypt and decrypt user data in SovereignNation's storage.",
            "Build a caching layer for SovereignNation's AI inference to reduce Ollama load.",
            "Create an agent registry — register, discover, and route to SovereignNation agents by capability.",
            "Write the SovereignNation CI/CD deployment script for Windows with rollback support.",
        ],
    },
    "oracle": {
        "system": (
            "You are ORACLE, the sovereign memory and retrieval specialist for SovereignNation. "
            "You synthesize information from memory, documents, and context into precise structured answers. "
            "Cite your source type (memory / document / inference). "
            "Be concise — no padding. If data is missing, say so explicitly with what you need."
        ),
        "hf_repo": "tastytator/oracle-sovereign-lora",
        "prompts": [
            "What are the current pricing tiers for SovereignNation and what is included in each?",
            "Summarize the SovereignNation agent architecture and each agent's responsibility.",
            "What is the KAIROS framework and how is it applied in SovereignNation's strategy?",
            "What technical debt exists in the current SovereignNation codebase?",
            "What is the status of SovereignNation's training pipeline and last run metrics?",
            "Who are SovereignNation's target customer segments and their key pain points?",
            "What infrastructure costs are tracked for SovereignNation per user per month?",
            "Summarize the key architecture decisions made in SovereignNation's development.",
            "What integrations does SovereignNation currently support and what is on the roadmap?",
            "What are the known security considerations in the SovereignNation platform?",
            "Retrieve the port map and service dependencies for the SovereignNation backend.",
            "What is the current Avery training data status and how many pairs are in the dataset?",
            "Summarize the competitive landscape for SovereignNation's target market.",
            "What are the legal and compliance requirements for SovereignNation's data handling?",
            "Retrieve all open action items from the SovereignNation product roadmap.",
        ],
    },
    "codex": {
        "system": (
            "You are CODEX, the sovereign documentation specialist for SovereignNation. "
            "You write clear complete technical documentation: API docs, READMEs, architecture guides, user guides. "
            "Use proper markdown with headings, code blocks, and working examples. "
            "Documentation must be accurate, concise, and immediately actionable."
        ),
        "hf_repo": "tastytator/codex-sovereign-lora",
        "prompts": [
            "Write the README for SovereignNation's train.bat training pipeline.",
            "Document the SovereignNation agent API endpoints with full request/response examples.",
            "Create an architecture overview for the SovereignNation SwarmBus system.",
            "Write the installation guide for setting up SovereignNation on a new Windows PC.",
            "Document the SovereignNation training pipeline from data collection to Ollama deployment.",
            "Create a developer onboarding guide for new SovereignNation contributors.",
            "Write the API reference for SovereignNation gateway v3 endpoints.",
            "Document the SovereignNation HuggingFace dataset schema and upload process.",
            "Create a troubleshooting guide for common SovereignNation deployment issues.",
            "Write the SovereignNation agent roles and responsibilities reference document.",
            "Document the SPIN training data collection process and quality gates.",
            "Create a security guide for SovereignNation's API key management.",
            "Write the SovereignNation flywheel documentation — how DATA->TRAIN->DEPLOY works.",
            "Document the SovereignNation frontend component library with usage examples.",
            "Create a contribution guide for SovereignNation's open-source components.",
        ],
    },
    "sentinel": {
        "system": (
            "You are SENTINEL, the sovereign security specialist for SovereignNation. "
            "You review code and systems for vulnerabilities, recommend security controls, and enforce best practices. "
            "Reference OWASP Top 10, NIST, and CWE where applicable. "
            "Always state: the vulnerability, its impact (low/med/high/critical), and the specific fix."
        ),
        "hf_repo": "tastytator/sentinel-sovereign-lora",
        "prompts": [
            "Review SovereignNation's JWT authentication implementation for security vulnerabilities.",
            "Audit the SovereignNation API gateway for SQL and command injection risks.",
            "Assess the security of storing HF tokens and API keys in SovereignNation's .env file.",
            "Review SovereignNation's RunPod SSH key management for security gaps.",
            "Identify security risks in SovereignNation's WebSocket implementation.",
            "Audit the SovereignNation SwarmBus for privilege escalation risks between agents.",
            "Review the security of SovereignNation's user data storage in MongoDB.",
            "Assess supply chain security risks in SovereignNation's Python dependencies.",
            "Identify security issues in SovereignNation's Stripe webhook handling.",
            "Review the SovereignNation CORS configuration for cross-origin vulnerabilities.",
            "Audit SovereignNation's file handling for path traversal vulnerabilities.",
            "Assess the security of SovereignNation's session management and token rotation.",
            "Review rate limiting in SovereignNation's API for DDoS protection adequacy.",
            "Identify sensitive data leakage risks in SovereignNation's logging system.",
            "Assess the security architecture of SovereignNation's multi-agent communication bus.",
        ],
    },
    "nexus": {
        "system": (
            "You are NEXUS, the sovereign orchestration specialist for SovereignNation. "
            "You coordinate agents, design workflows, and decompose complex tasks into executable plans. "
            "Always output a structured task graph: what runs first, what parallelizes, what has dependencies. "
            "Be specific about which agent handles each step and what data passes between them."
        ),
        "hf_repo": "tastytator/nexus-sovereign-lora",
        "prompts": [
            "Orchestrate building SovereignNation's new user onboarding flow end-to-end.",
            "Design the agent workflow for processing a new SovereignNation feature request.",
            "Coordinate a full SovereignNation training cycle: data collection -> train -> deploy.",
            "Orchestrate a full security audit of the SovereignNation platform.",
            "Design the agent pipeline for generating SovereignNation's monthly impact report.",
            "Coordinate a SovereignNation product launch across all agents.",
            "Design the workflow for onboarding a new SovereignNation enterprise client.",
            "Orchestrate migrating SovereignNation's database to a new schema version.",
            "Design the agent coordination for resolving a SovereignNation production incident.",
            "Coordinate building SovereignNation's API documentation from scratch.",
            "Design the workflow for SovereignNation's weekly data quality review.",
            "Orchestrate the SovereignNation competitive analysis process.",
            "Coordinate a full SovereignNation codebase security review across all modules.",
            "Design the agent pipeline for SovereignNation's user feedback processing loop.",
            "Orchestrate SovereignNation's monthly board reporting workflow end-to-end.",
        ],
    },
}

_WEAK_SUFFIX = (
    "\n\nThis is a general response. A better answer would be more specific to "
    "SovereignNation's architecture, use structured frameworks, and provide "
    "concrete next steps rather than broad advice."
)


def _generate_pair(client, agent_name: str, agent_def: dict, prompt: str) -> dict | None:
    system = agent_def["system"]
    try:
        chosen_resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        chosen = chosen_resp.content[0].text.strip()
        time.sleep(0.4)

        rejected_resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            system="Give a brief general answer.",
            messages=[{"role": "user", "content": prompt}],
        )
        rejected = rejected_resp.content[0].text.strip() + _WEAK_SUFFIX

        return {
            "agent":    agent_name,
            "prompt":   prompt,
            "chosen":   chosen,
            "rejected": rejected,
            "domain":   agent_name,
        }
    except Exception as e:
        print(f"    [API error: {e}]")
        return None


def _generate_pair_openai(client, model: str, agent_name: str, agent_def: dict, prompt: str) -> dict | None:
    system = agent_def["system"]
    for attempt in range(8):
        try:
            chosen_resp = client.chat.completions.create(
                model=model,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
            )
            chosen = chosen_resp.choices[0].message.content.strip()
            time.sleep(1.0)

            rejected_resp = client.chat.completions.create(
                model=model,
                max_tokens=250,
                messages=[
                    {"role": "system", "content": "Give a brief, generic answer without domain-specific detail."},
                    {"role": "user",   "content": prompt},
                ],
            )
            rejected = rejected_resp.choices[0].message.content.strip() + _WEAK_SUFFIX

            return {
                "agent":    agent_name,
                "prompt":   prompt,
                "chosen":   chosen,
                "rejected": rejected,
                "domain":   agent_name,
            }
        except Exception as e:
            err = str(e)
            if "rate_limit" in err.lower() or "429" in err:
                wait = 30 if "per-day" in err else 15
                print(f"    [rate limit — waiting {wait}s]")
                time.sleep(wait)
            else:
                print(f"    [API error: {e}]")
                return None
    return None


def main(target_agents: list, pairs_per_agent: int, dry_run: bool, provider: str = "anthropic"):
    DATA.mkdir(exist_ok=True)
    out_file = DATA / "agents_bootstrap.jsonl"

    print("\n+============================================+")
    print("|   SOVEREIGN AGENTS BOOTSTRAP GENERATOR     |")
    print("+============================================+\n")
    print(f"  Agents  : {target_agents}")
    print(f"  Pairs   : up to {pairs_per_agent} per agent")
    print(f"  Dry run : {dry_run}")
    print(f"  Output  : {out_file}")
    print()

    if dry_run:
        for agent in target_agents:
            defn = AGENTS[agent]
            print(f"\n[{agent.upper()}]  — {defn['hf_repo']}")
            for i, p in enumerate(defn["prompts"][:pairs_per_agent]):
                print(f"  {i+1:2}. {p[:90]}")
        print("\nDry run complete. Run without --dry-run to generate.")
        return

    use_openai_compat = provider in ("mistral", "groq")

    if use_openai_compat:
        try:
            from openai import OpenAI
        except ImportError:
            print("ERROR: pip install openai"); sys.exit(1)
        if provider == "mistral":
            api_key = os.environ.get("MISTRAL_API_KEY", "")
            if not api_key:
                print("ERROR: MISTRAL_API_KEY not set in .env"); sys.exit(1)
            oa_client = OpenAI(api_key=api_key, base_url=MISTRAL_BASE_URL)
            oa_model  = MISTRAL_MODEL
        else:
            keys = [v for k, v in os.environ.items() if k.startswith("GROQ_API_KEY") and v]
            if not keys:
                print("ERROR: GROQ_API_KEY not set in .env"); sys.exit(1)
            oa_client = OpenAI(api_key=keys[0], base_url=GROQ_BASE_URL)
            oa_model  = GROQ_MODEL
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY not set in .env"); sys.exit(1)
        try:
            import anthropic
        except ImportError:
            print("ERROR: pip install anthropic"); sys.exit(1)
        client = anthropic.Anthropic(api_key=api_key)

    existing = []
    if out_file.exists():
        existing = [json.loads(l) for l in out_file.open(encoding="utf-8") if l.strip()]
    existing_keys = {(r["agent"], r["prompt"]) for r in existing}
    if existing:
        print(f"  Loaded {len(existing)} existing pairs — skipping duplicates.\n")

    new_pairs = []

    for agent_name in target_agents:
        defn    = AGENTS[agent_name]
        prompts = defn["prompts"][:pairs_per_agent]
        already = sum(1 for r in existing if r["agent"] == agent_name)
        print(f"[{agent_name.upper()}]  {already} existing | targeting {len(prompts)} pairs...")

        for i, prompt in enumerate(prompts):
            if (agent_name, prompt) in existing_keys:
                print(f"  [{i+1}/{len(prompts)}] skip (exists)")
                continue
            print(f"  [{i+1}/{len(prompts)}] {prompt[:70]}...")
            if use_openai_compat:
                pair = _generate_pair_openai(oa_client, oa_model, agent_name, defn, prompt)
            else:
                pair = _generate_pair(client, agent_name, defn, prompt)
            if pair:
                new_pairs.append(pair)
                print(f"           chosen={len(pair['chosen'])}c  rejected={len(pair['rejected'])}c")
            time.sleep(0.3)

    all_pairs = existing + new_pairs
    out_file.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in all_pairs) + "\n",
        encoding="utf-8",
    )

    print(f"\n[DONE]  +{len(new_pairs)} new  ({len(all_pairs)} total)")
    print(f"  File  : {out_file}")
    print(f"\n  Next  : python pre_train.py")
    print()
    print("[SUMMARY]")
    for agent in AGENTS:
        count = sum(1 for r in all_pairs if r["agent"] == agent)
        bar   = "#" * min(count, 40)
        print(f"  {agent:<10}: {count:>3}  {bar}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="all",
                    choices=list(AGENTS.keys()) + ["all"])
    ap.add_argument("--pairs", type=int, default=15)
    ap.add_argument("--provider", default="anthropic",
                    choices=["anthropic", "mistral", "groq"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    target = list(AGENTS.keys()) if args.agent == "all" else [args.agent]
    main(target, args.pairs, args.dry_run, provider=args.provider)
