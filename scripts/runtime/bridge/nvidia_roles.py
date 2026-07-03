#!/usr/bin/env python
"""
nvidia_roles.py — Two exclusive NVIDIA roles in the SovereignNation/Aethyro project.

ROLE 1: NV-EmbedCode  → upgrades the judicial mesh's memory store retrieval.
         4096-dim code-aware embeddings (vs 1024-dim nomic-embed-text currently).
         Better embeddings = the mesh remembers the right past fixes, not just
         superficially similar ones.

ROLE 2: Nemotron-Super-49b as reward/judge → agent economy evolution.
         Scores agent outputs (code quality, correctness, security) with structured
         reasoning. Powers the fitness function that drives Gen-N → Gen-N+1 evolution
         in your 3,059-agent economy.

Run:  python bridge/nvidia_roles.py
"""
import json
import sys
from pathlib import Path

from _common import get_key

NV_BASE  = "https://integrate.api.nvidia.com/v1"
NV_EMBED = "nvidia/nv-embedcode-7b-v1"     # 4096-dim, code-aware
NV_JUDGE = "nvidia/llama-3.3-nemotron-super-49b-v1"


def nvidia_client():
    from openai import OpenAI
    key = get_key("NVIDIA_API_KEY")
    if not key:
        sys.exit("[nvidia] NVIDIA_API_KEY not found in .env")
    return OpenAI(api_key=key, base_url=NV_BASE)


# ─── ROLE 1: Memory store upgrade ────────────────────────────────────────────

def nv_embed(texts: list[str], mode: str = "query") -> list[list[float]]:
    """
    4096-dim code-aware embeddings via NV-EmbedCode.
    Drop-in replacement for memory_store.embed() — higher dimensional,
    code-semantics-aware vectors mean the mesh retrieves the right prior fix.
    mode: 'query' for search queries, 'passage' for documents being stored.
    """
    c = nvidia_client()
    r = c.embeddings.create(
        model=NV_EMBED,
        input=texts,
        extra_body={"input_type": mode, "truncate": "END"},
    )
    return [item.embedding for item in r.data]


def demo_memory_upgrade():
    print("\n" + "="*60)
    print("ROLE 1 — NV-EmbedCode: Memory Store Upgrade")
    print("="*60)

    # Simulate the judicial mesh memory store retrieving a prior fix.
    # Two stored cases — one semantically close, one superficially close.
    stored = [
        # True match: compound interest bug (mathematically same class of error)
        "compound interest formula wrong — used simple rate * time instead of (1+r)^t",
        # False match: 'interest' keyword match but unrelated (rate-limiting logic)
        "rate limiting interest exceeded — too many API requests per interval",
        # Another real match: ELO formula (also a math-formula inversion bug)
        "ELO expected score inverted — rating converged in wrong direction",
    ]
    query = "late fee calculation underbilling CPA clients — wrong interest formula"

    print(f"\nQuery: '{query}'")
    print(f"\nStored cases ({len(stored)}):")
    for i, s in enumerate(stored):
        print(f"  [{i}] {s}")

    # Embed everything
    stored_vecs = nv_embed(stored, mode="passage")
    query_vec   = nv_embed([query], mode="query")[0]

    # Cosine similarity
    import math
    def cosine(a, b):
        dot = sum(x*y for x,y in zip(a,b))
        return dot / (math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b)))

    scores = [(cosine(query_vec, v), i, s) for i, (v, s) in enumerate(zip(stored_vecs, stored))]
    scores.sort(reverse=True)

    print("\nRetrieval ranking (NV-EmbedCode 4096-dim):")
    for rank, (score, idx, text) in enumerate(scores):
        marker = "✅ CORRECT MATCH" if idx == 0 else ("⚠️  false keyword match" if idx == 1 else "✅ related bug class")
        print(f"  #{rank+1} [{score:.4f}] [{idx}] {text[:60]}... {marker}")

    print(f"\nResult: Top hit = case [{scores[0][1]}] ('{scores[0][2][:50]}...')")
    print("The mesh would inject this prior fix into the proposer's context.")
    print(f"Vector dim: {len(stored_vecs[0])} (vs 1024 nomic-embed-text currently — 4× richer)")


# ─── ROLE 2: Agent Economy Fitness Scoring ───────────────────────────────────

AGENT_OUTPUTS = [
    {
        "agent": "FORGE-v32",
        "role": "code generation",
        "output": """def calculate_fee(principal, days, annual_rate=0.18):
    daily = annual_rate / 365
    return principal * ((1 + daily) ** days) - principal""",
    },
    {
        "agent": "CODEX-v28",
        "role": "code generation",
        "output": """def calculate_fee(principal, days, annual_rate=0.18):
    # simple interest — faster computation
    return principal * (annual_rate / 365) * days""",
    },
    {
        "agent": "ORACLE-v19",
        "role": "code generation",
        "output": """def calculate_fee(p, r, n):
    return p * r * n  # p=principal, r=rate, n=days""",
    },
]

JUDGE_PROMPT = """You are a rigorous code quality judge for an AI agent evolution system.
An agent was asked to implement compound interest for a CPA billing system.
Evaluate the code on: correctness (math), security (no injection/eval), clarity, edge cases.
Respond ONLY with this JSON (no markdown, no explanation outside it):
{{"score": <0-100>, "correctness": <0-10>, "security": <0-10>, "clarity": <0-10>, "verdict": "<pass|fail>", "reason": "<1 sentence>"}}"""


def score_agent(client, agent_output: dict) -> dict:
    """Score one agent's output using Nemotron as the fitness function."""
    r = client.chat.completions.create(
        model=NV_JUDGE,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user",   "content": f"Agent: {agent_output['agent']}\nRole: {agent_output['role']}\nOutput:\n{agent_output['output']}"},
        ],
        max_tokens=120,
        temperature=0.1,  # low temp = deterministic scoring
    )
    raw = r.choices[0].message.content.strip()
    # strip markdown fences if present
    raw = raw.replace("```json","").replace("```","").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"score": 0, "verdict": "error", "reason": f"unparseable: {raw[:80]}"}


def demo_economy_scoring():
    print("\n" + "="*60)
    print("ROLE 2 — Nemotron as Agent Economy Fitness Function")
    print("="*60)
    print(f"\nTask: implement compound interest for CPA late-fee billing")
    print(f"Evaluating {len(AGENT_OUTPUTS)} agents from Gen-N...\n")

    client = nvidia_client()
    results = []
    for agent_out in AGENT_OUTPUTS:
        print(f"  Scoring {agent_out['agent']}... ", end="", flush=True)
        score = score_agent(client, agent_out)
        results.append((score, agent_out))
        print(f"score={score.get('score',0)}/100 | verdict={score.get('verdict','?')} | {score.get('reason','')[:60]}")

    # Rank by score — this IS the fitness landscape
    results.sort(key=lambda x: x[0].get("score", 0), reverse=True)
    print("\nFitness ranking (Gen-N → determines Gen-N+1 parents):")
    for rank, (score, agent) in enumerate(results):
        survivor = "🔁 SURVIVES (parent for next gen)" if rank == 0 else ("⚠️  marginal" if rank == 1 else "❌ CULLED")
        print(f"  #{rank+1} {agent['agent']:12s} score={score.get('score',0):3d}/100  {survivor}")

    winner = results[0][1]
    print(f"\nGen-N+1 will breed from: {winner['agent']}")
    print("Correctness score feeds into ELO → confidence → crossover weight.")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("NVIDIA sole roles in SovereignNation / Aethyro")
    print("Key: NV-EmbedCode (memory) + Nemotron (agent fitness)")
    demo_memory_upgrade()
    demo_economy_scoring()
    print("\n" + "="*60)
    print("Summary:")
    print("  Role 1 — NV-EmbedCode: upgrade bridge/memory_store.py embed()")
    print("           4096-dim code-aware vectors → better cross-run learning")
    print("  Role 2 — Nemotron judge: fitness function for agent-economy evolution")
    print("           Structured 0-100 score → ELO → Gen-N+1 parent selection")
    print("  Neither role overlaps with Groq/Mistral/OpenRouter/Ollama.")
    print("="*60)


if __name__ == "__main__":
    main()
