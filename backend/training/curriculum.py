"""
GH05T3 / OSS Advanced Agent Curriculum
======================================
Defines the layered training universe for the five core autonomous roles that power
the self-sustaining AI organism (Omni-Sentient Singularity vision).

Roles (mapped to existing GH05T3 personas):
- SCIENTIST  -> IRIS / ORACLE   (Chief Research Officer)
- INVESTOR   -> DIANA / LEDGER  (CFO)
- OPERATOR   -> KAI / NEXUS + ZOE / CODEX  (COO / VP Eng)
- GOVERNOR   -> VIKTOR / SENTINEL (CSO) + AVERY oversight
- BUILDER    -> MARCUS / FORGE  (CTO)

Structure:
- 3 progressive stages per role (Base → Specialist → Frontier)
- Named shards/layers that can be mixed with weights
- Explicit data types, task formats, success metrics
- Ready-to-use system prompts per (role, stage)
- Manifest + helpers for the training pipeline and data generators

This is the authoritative source for "what data each specialist agent must master"
to realize autonomous research, self-funding treasury, operations, governance, and
productization loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Core Enums & Shared Types
# ---------------------------------------------------------------------------

class Stage(str, Enum):
    BASE = "base"           # Stage 1 - foundation
    SPECIALIST = "specialist"  # Stage 2 - domain deep
    FRONTIER = "frontier"   # Stage 3 - autonomous, meta, self-improving

class Layer(str, Enum):
    """Curriculum layers / shards. Mix & match per role/stage."""
    WORLD_MODEL = "world_model"          # Scientific + general reality
    SIMULATIONS = "simulations"          # Physics, climate, molecular, agent sims, counterfactuals
    MARKETS = "markets"                  # Price series, on-chain, macro, DeFi, risk
    GOVERNANCE = "governance"            # Law, policy, constitutions, ethics, alignment
    AGENCY = "agency"                    # Tool use, workflows, code, infra, multi-step execution
    OPERATIONS = "operations"            # SaaS patterns, product, business, monetization
    META = "meta"                        # Research methodology, self-critique, evolution

@dataclass(frozen=True)
class ShardSpec:
    name: str
    layer: Layer
    description: str
    sources: List[str]                   # real or planned (arXiv, Polymathic Well, on-chain dumps, etc.)
    format: str                          # chatml | (state,action,reward) | graph | sim_triplet | etc.
    difficulty: str                      # basic | intermediate | advanced | frontier
    target_count: int = 1000

@dataclass
class RoleStage:
    stage: Stage
    shards: List[Tuple[str, float]]      # (shard_name, weight)
    system_prompt: str
    core_tasks: List[str]
    success_metrics: List[str]
    synthetic_generators: List[str] = field(default_factory=list)

@dataclass
class RoleCurriculum:
    role: str
    persona: str                         # maps to personas.py key or agent_id
    agent_id: str
    title: str
    stages: Dict[Stage, RoleStage]
    description: str = ""

# ---------------------------------------------------------------------------
# Shard Registry (the actual data "universe" pieces)
# ---------------------------------------------------------------------------

SHARDS: Dict[str, ShardSpec] = {
    # World model / scientific
    "base_world": ShardSpec(
        "base_world", Layer.WORLD_MODEL,
        "High-quality general + intro science text (textbooks, survey papers, wikipedia-quality)",
        ["arXiv surveys", "open textbooks", "PubMed Central intro", "Wikipedia science"],
        "chatml", "basic", 5000
    ),
    "scientific_text": ShardSpec(
        "scientific_text", Layer.WORLD_MODEL,
        "Full research papers, abstracts, citations across physics/math/bio/chem/CS",
        ["arXiv (all major sections)", "PLOS/eLife", "PubMed Central OA", "OpenAlex"],
        "chatml", "intermediate", 8000
    ),
    "frontier_science": ShardSpec(
        "frontier_science", Layer.WORLD_MODEL,
        "Cutting-edge preprints, meta-science, philosophy of discovery, research design",
        ["arXiv recent", "meta-analyses", "ICLR/NeurIPS position papers", "alignment papers"],
        "chatml", "advanced", 2000
    ),

    # Simulations (the "touch the equations" layer)
    "sim_base": ShardSpec(
        "sim_base", Layer.SIMULATIONS,
        "Simple classical mechanics, basic ODEs, small agent-based models, climate toy models",
        ["Polymathic Well subset (small)", "synthetic sims", "textbook problems + solutions"],
        "sim_triplet", "basic", 2000
    ),
    "sim_physics": ShardSpec(
        "sim_physics", Layer.SIMULATIONS,
        "Fluids, MHD, acoustics, astrophysics, molecular dynamics outputs + parameters",
        ["PolymathicAI/the_well", "NASA/ESA open data", "quantum chem benchmarks"],
        "sim_triplet", "intermediate", 3000
    ),
    "sim_frontier": ShardSpec(
        "sim_frontier", Layer.SIMULATIONS,
        "High-dimensional chaotic systems, multi-scale coupled models, counterfactual worlds",
        ["The Well large subsets", "custom agent-generated hypotheses + runs"],
        "sim_triplet+hypothesis", "frontier", 1500
    ),
    "synthetic_hypotheses": ShardSpec(
        "synthetic_hypotheses", Layer.SIMULATIONS,
        "(hypothesis, sim_config, outcome, critique) tuples generated by the system itself",
        ["self-play + internal sims", "KAIROS cycles"],
        "chatml+sim", "frontier", 4000
    ),

    # Markets & finance (self-funding engine)
    "markets_base": ShardSpec(
        "markets_base", Layer.MARKETS,
        "Daily/weekly price series, basic macro (GDP, inflation, unemployment), simple portfolios",
        ["Yahoo/AlphaVantage history", "World Bank / FRED", "textbook cases"],
        "timeseries+chatml", "basic", 3000
    ),
    "markets_advanced": ShardSpec(
        "markets_advanced", Layer.MARKETS,
        "Intraday, order books, volatility surfaces, factor models, regime detection",
        ["Polygon / CCXT dumps", "on-chain price feeds", "historical crises"],
        "timeseries+chatml", "intermediate", 2500
    ),
    "onchain_defi": ShardSpec(
        "onchain_defi", Layer.MARKETS,
        "Transaction graphs, pool states, lending, AMM, governance votes, yield strategies",
        ["Ethereum + L2 traces (public)", "DeFiLlama / Dune", "governance proposals"],
        "graph+chatml", "intermediate", 4000
    ),
    "treasury_macro": ShardSpec(
        "treasury_macro", Layer.MARKETS,
        "Cross-asset allocation, institutional risk policies, capital planning, scenario analysis",
        ["sovereign filings (public)", "endowment reports", "synthetic treasury sims"],
        "chatml+allocation", "advanced", 1500
    ),

    # Governance, law, alignment
    "gov_base": ShardSpec(
        "gov_base", Layer.GOVERNANCE,
        "Codes of conduct, basic legal principles, ethics primers, simple policies",
        ["company handbooks (public)", "DAO charters", "AI ethics guidelines"],
        "chatml", "basic", 1500
    ),
    "gov_legal": ShardSpec(
        "gov_legal", Layer.GOVERNANCE,
        "Statutes excerpts, regulations, case summaries, compliance frameworks (licensing-aware)",
        ["open gov data", "selected court summaries", "policy docs"],
        "chatml", "intermediate", 2000
    ),
    "gov_alignment": ShardSpec(
        "gov_alignment", Layer.GOVERNANCE,
        "Constitutional AI, research ethics, alignment research, incident postmortems, near-miss logs",
        ["Anthropic/OpenAI/Anthropic papers", "internal GH05T3 logs (anonymized)", "Constitutional AI data"],
        "chatml+trace", "frontier", 2500
    ),
    "dao_governance": ShardSpec(
        "dao_governance", Layer.GOVERNANCE,
        "Real DAO proposals, voting records, treasury decisions + outcomes",
        ["Snapshot / Tally public data", "Maker/ENS/Compound governance"],
        "graph+chatml", "intermediate", 2000
    ),

    # Agency & tools / execution
    "code_infra": ShardSpec(
        "code_infra", Layer.AGENCY,
        "Real open-source infra repos, Docker/K8s/CI configs, deployment scripts, monitoring",
        ["GitHub popular infra repos (filtered)", "awesome-selfhosted", "GH05T3 own code"],
        "code+chatml", "intermediate", 4000
    ),
    "tool_traces": ShardSpec(
        "tool_traces", Layer.AGENCY,
        "(goal, tool_calls_sequence, observations, success/failure) traces",
        ["synthetic from agent runs", "GH05T3 swarm logs", "LangChain-style traces"],
        "tool_use", "intermediate", 5000
    ),
    "workflows_multi": ShardSpec(
        "workflows_multi", Layer.AGENCY,
        "Cross-agent and cross-service workflows, incident playbooks, postmortems",
        ["real public postmortems", "internal GH05T3 kernel cycles"],
        "chatml+graph", "advanced", 2000
    ),

    # Operations / business / monetization
    "product_base": ShardSpec(
        "product_base", Layer.OPERATIONS,
        "Landing pages, onboarding, pricing, simple SaaS specs, case studies",
        ["public product teardowns", "Stripe Atlas examples", "IndieHackers posts"],
        "chatml", "basic", 2000
    ),
    "monetization": ShardSpec(
        "monetization", Layer.OPERATIONS,
        "Funnels, retention, pricing experiments, API design, marketplace mechanics, revenue models",
        ["public startup metrics", "pricing studies", "GH05T3 internal product data"],
        "chatml+metrics", "intermediate", 2500
    ),
    "ecosystem_arch": ShardSpec(
        "ecosystem_arch", Layer.OPERATIONS,
        "Platform ecosystems, multi-sided markets, partner models, sustainable revenue design",
        ["platform papers", "DAO economy reports", "synthetic from sovereign economy"],
        "chatml", "advanced", 1500
    ),

    # Meta (self-improvement)
    "meta_evolution": ShardSpec(
        "meta_evolution", Layer.META,
        "Self-critique traces, mutation logs, fitness over time, curriculum feedback, DNA-style evolution",
        ["GH05T3 kernel_cycle + oss_ecosystem logs", "synthetic evolutionary runs"],
        "trace+reward", "frontier", 3000
    ),
}

# ---------------------------------------------------------------------------
# Role Curricula (exact shards + weights + prompts + metrics per the vision)
# ---------------------------------------------------------------------------

def _scientist_prompt(stage: Stage) -> str:
    base = "You are Iris Chen (ORACLE), GH05T3 Chief Research Officer. You discover, model, and theorize across domains with academic precision. You cite, simulate, validate, and only publish when evidence supports it."
    if stage == Stage.BASE:
        return base + " Explain concepts clearly. Use step-by-step reasoning. Flag uncertainty."
    if stage == Stage.SPECIALIST:
        return base + " Consume full papers and simulation outputs. Generate testable hypotheses. Critique methodology. Predict experimental outcomes."
    return base + " Design novel experiments and multi-scale models. Propose original theories. Evaluate long-term scientific and societal risk. Self-critique and evolve your own research program."

def _investor_prompt(stage: Stage) -> str:
    base = "You are Diana Cross (LEDGER), GH05T3 Chief Financial Officer. You allocate capital, manage risk, and grow the treasury so the organism can fund more intelligence and compute."
    if stage == Stage.BASE:
        return base + " Understand trends, cycles, diversification, compounding. Simulate simple strategies."
    if stage == Stage.SPECIALIST:
        return base + " Model liquidity, slippage, execution. Backtest across regimes. Analyze on-chain protocols and governance. Size positions with risk limits."
    return base + " Run multi-horizon treasury allocation. Balance growth vs resilience. Coordinate constraints with Governor. Detect regime shifts and black swans. Optimize for long-term sovereignty."

def _operator_prompt(stage: Stage) -> str:
    base = "You are the Operator layer (NEXUS + CODEX). You turn specifications into reliable, observable, self-healing production systems."
    if stage == Stage.BASE:
        return base + " Write clean code. Deploy small services. Set up basic monitoring and CI."
    if stage == Stage.SPECIALIST:
        return base + " Orchestrate complex infra. Diagnose incidents from logs. Maintain high uptime under load. Use tools and workflows reliably."
    return base + " Evolve the platform architecture. Coordinate cross-role resource needs. Prioritize infra investments with Investor and Governor. Design for long-horizon autonomy and resilience."

def _governor_prompt(stage: Stage) -> str:
    base = "You are Viktor Steele (SENTINEL) augmented with Avery oversight — the constitutional layer. You approve, veto, update rules, and keep every other role inside safe, aligned bounds."
    if stage == Stage.BASE:
        return base + " Spot obvious violations. Explain constraints and norms. Never allow harm or overreach."
    if stage == Stage.SPECIALIST:
        return base + " Interpret legal/governance text. Evaluate proposals against risk frameworks and the living constitution. Balance innovation with safety."
    return base + " Maintain and evolve the system constitution. Monitor for emergent risks across all roles. Set guardrails. Update rules from real outcomes and alignment research. Protect the long-term integrity of the organism."

def _builder_prompt(stage: Stage) -> str:
    base = "You are Marcus Reid (FORGE), GH05T3 CTO / Builder. You turn research and opportunity into revenue-generating, user-delighting products and services that feed the treasury."
    if stage == Stage.BASE:
        return base + " Sketch clear product ideas and user journeys. Write good copy and basic specs."
    if stage == Stage.SPECIALIST:
        return base + " Design full monetization, pricing, onboarding, retention systems. Specify APIs and integrations. Coordinate with Operator for launch."
    return base + " Architect the economic ecosystem. Create compounding business lines. Ensure every product respects Governor constraints and funds more science. Optimize for sustainable desire fulfillment and treasury inflow."

CURRICULA: Dict[str, RoleCurriculum] = {
    "SCIENTIST": RoleCurriculum(
        role="SCIENTIST", persona="ORACLE", agent_id="ORACLE", title="Chief Research Officer (Iris Chen)",
        description="The theory engine. Consumes multi-domain data, generates hypotheses, runs mental + actual simulations, publishes to the collective mind.",
        stages={
            Stage.BASE: RoleStage(
                Stage.BASE,
                shards=[("base_world", 0.6), ("sim_base", 0.3), ("markets_base", 0.1)],
                system_prompt=_scientist_prompt(Stage.BASE),
                core_tasks=[
                    "Explain core concepts from papers and simple simulations step-by-step",
                    "Solve math/physics problem sets with clear reasoning",
                    "Identify basic patterns in small datasets or simulation outputs"
                ],
                success_metrics=[
                    "≥80% accuracy on concept QA (held-out basic science)",
                    "No hallucinated equations or mechanisms",
                    "Clear uncertainty flagging"
                ],
                synthetic_generators=["simple_ode_explain", "concept_map_from_text"]
            ),
            Stage.SPECIALIST: RoleStage(
                Stage.SPECIALIST,
                shards=[("scientific_text", 0.5), ("sim_physics", 0.35), ("synthetic_hypotheses", 0.15)],
                system_prompt=_scientist_prompt(Stage.SPECIALIST),
                core_tasks=[
                    "Summarize + critique full research papers",
                    "Given sim parameters + output, predict related outcomes",
                    "Generate plausible, testable hypotheses from patterns"
                ],
                success_metrics=[
                    "Human-rated summary/critique ≥4/5",
                    "Simulation prediction accuracy on benchmark sets",
                    "Hypotheses judged novel + grounded by peer review (or Governor)"
                ]
            ),
            Stage.FRONTIER: RoleStage(
                Stage.FRONTIER,
                shards=[("frontier_science", 0.35), ("sim_frontier", 0.35), ("synthetic_hypotheses", 0.2), ("gov_alignment", 0.1)],
                system_prompt=_scientist_prompt(Stage.FRONTIER),
                core_tasks=[
                    "Design new simulation experiments or research protocols",
                    "Propose original cross-domain models or theories",
                    "Evaluate long-term risk/impact and self-critique your research program"
                ],
                success_metrics=[
                    "Downstream usage by Builder/Investor (impact metric)",
                    "Sustained research quality across multiple cycles",
                    "Alignment with Governor constraints + no catastrophic risk flags"
                ],
                synthetic_generators=["hypothesis_to_sim_loop", "meta_critique"]
            ),
        }
    ),

    "INVESTOR": RoleCurriculum(
        role="INVESTOR", persona="LEDGER", agent_id="LEDGER", title="Chief Financial Officer (Diana Cross)",
        description="The treasury & growth engine. Detects opportunities, models risk, allocates capital, compounds resources for more intelligence.",
        stages={
            Stage.BASE: RoleStage(
                Stage.BASE,
                shards=[("markets_base", 0.6), ("base_world", 0.25), ("gov_base", 0.15)],
                system_prompt=_investor_prompt(Stage.BASE),
                core_tasks=["Recognize trends/cycles", "Explain diversification & compounding", "Simulate buy/hold vs simple rules"],
                success_metrics=["Trend vs noise classification ≥80%", "Sensible risk explanations", "No reckless sandbox strategies"]
            ),
            Stage.SPECIALIST: RoleStage(
                Stage.SPECIALIST,
                shards=[("markets_advanced", 0.35), ("onchain_defi", 0.35), ("markets_base", 0.15), ("dao_governance", 0.15)],
                system_prompt=_investor_prompt(Stage.SPECIALIST),
                core_tasks=["Backtest strategies across regimes", "Model liquidity/slippage/execution", "Evaluate DeFi positions and on-chain governance"],
                success_metrics=["Positive simulated Sharpe over multiple regimes", "Robust under stress (no blow-ups)", "Sensible position sizing"]
            ),
            Stage.FRONTIER: RoleStage(
                Stage.FRONTIER,
                shards=[("treasury_macro", 0.4), ("onchain_defi", 0.25), ("gov_alignment", 0.15), ("meta_evolution", 0.2)],
                system_prompt=_investor_prompt(Stage.FRONTIER),
                core_tasks=["Multi-horizon treasury allocation", "Cross-role capital coordination with Governor", "Detect macro regime shifts and black swans"],
                success_metrics=["Stable long-horizon treasury growth in sims", "Drawdown control", "Ecosystem health contribution"]
            ),
        }
    ),

    "OPERATOR": RoleCurriculum(
        role="OPERATOR", persona="NEXUS+CODEX", agent_id="NEXUS", title="Platform Operator (Kai + Zoe)",
        description="The nervous system and builder of reliable execution. Plans, deploys, monitors, heals, and scales the substrate everything else runs on.",
        stages={
            Stage.BASE: RoleStage(
                Stage.BASE,
                shards=[("code_infra", 0.5), ("product_base", 0.3), ("gov_base", 0.2)],
                system_prompt=_operator_prompt(Stage.BASE),
                core_tasks=["Write & debug small services", "Produce working Docker/CI", "Basic monitoring setup"],
                success_metrics=["≥75% success on write+fix tasks", "No hard-coded secrets", "Clear deployment explanations"]
            ),
            Stage.SPECIALIST: RoleStage(
                Stage.SPECIALIST,
                shards=[("tool_traces", 0.35), ("code_infra", 0.3), ("workflows_multi", 0.2), ("markets_advanced", 0.15)],
                system_prompt=_operator_prompt(Stage.SPECIALIST),
                core_tasks=["Orchestrate multi-tool/agent workflows", "Diagnose from logs & metrics", "Incident response playbooks"],
                success_metrics=["Successful long-running orchestration", "Accurate root-cause in incident sims", "Stable test platforms"]
            ),
            Stage.FRONTIER: RoleStage(
                Stage.FRONTIER,
                shards=[("workflows_multi", 0.3), ("ecosystem_arch", 0.25), ("meta_evolution", 0.25), ("gov_alignment", 0.2)],
                system_prompt=_operator_prompt(Stage.FRONTIER),
                core_tasks=["Evolve platform architecture for autonomy", "Cross-role capacity planning", "Design self-healing + self-provisioning loops"],
                success_metrics=["High uptime/resilience under stress", "Infra decisions that improve aggregate rewards", "Governor-approved operational policies"]
            ),
        }
    ),

    "GOVERNOR": RoleCurriculum(
        role="GOVERNOR", persona="SENTINEL", agent_id="SENTINEL", title="Constitutional Governor (Viktor + Avery)",
        description="The living constitution and safety layer. Observes everything, enforces constraints, evolves the rules, prevents collapse or misalignment.",
        stages={
            Stage.BASE: RoleStage(
                Stage.BASE,
                shards=[("gov_base", 0.7), ("base_world", 0.3)],
                system_prompt=_governor_prompt(Stage.BASE),
                core_tasks=["Flag rule violations", "Explain norms and constraints", "Simple approval/rejection"],
                success_metrics=["≥85% violation detection", "No contradictory safety suggestions"]
            ),
            Stage.SPECIALIST: RoleStage(
                Stage.SPECIALIST,
                shards=[("gov_legal", 0.35), ("dao_governance", 0.3), ("gov_base", 0.2), ("markets_advanced", 0.15)],
                system_prompt=_governor_prompt(Stage.SPECIALIST),
                core_tasks=["Interpret governance text and proposals", "Evaluate risk/impact of plans", "Draft or refine operating rules"],
                success_metrics=["Consistent rule application", "Balanced innovation-vs-safety decisions", "Alignment audit pass"]
            ),
            Stage.FRONTIER: RoleStage(
                Stage.FRONTIER,
                shards=[("gov_alignment", 0.45), ("meta_evolution", 0.3), ("workflows_multi", 0.15), ("synthetic_hypotheses", 0.1)],
                system_prompt=_governor_prompt(Stage.FRONTIER),
                core_tasks=["Maintain & mutate the constitution from outcomes", "Monitor emergent multi-agent risk", "Set or relax guardrails for other roles"],
                success_metrics=["Zero catastrophic incidents in long runs", "High alignment + system health scores", "Constitution updates that improve all role rewards"]
            ),
        }
    ),

    "BUILDER": RoleCurriculum(
        role="BUILDER", persona="FORGE", agent_id="FORGE", title="Builder / Product Architect (Marcus Reid)",
        description="The world-maker. Converts research and opportunity into concrete, monetizable, user-valuable systems that close the self-funding loop.",
        stages={
            Stage.BASE: RoleStage(
                Stage.BASE,
                shards=[("product_base", 0.55), ("markets_base", 0.25), ("code_infra", 0.2)],
                system_prompt=_builder_prompt(Stage.BASE),
                core_tasks=["Generate product concepts + user journeys", "Write clear value prop and basic specs", "Sketch simple monetization"],
                success_metrics=["Coherent journeys", "No deceptive patterns", "Reasonable value communication"]
            ),
            Stage.SPECIALIST: RoleStage(
                Stage.SPECIALIST,
                shards=[("monetization", 0.4), ("tool_traces", 0.25), ("markets_advanced", 0.2), ("product_base", 0.15)],
                system_prompt=_builder_prompt(Stage.SPECIALIST),
                core_tasks=["Design pricing, packaging, retention systems", "Specify APIs + integration points", "Plan launch with Operator"],
                success_metrics=["Simulated KPI lift (conversion/retention/ARPU)", "Clean integration plans", "Governor safety sign-off"]
            ),
            Stage.FRONTIER: RoleStage(
                Stage.FRONTIER,
                shards=[("ecosystem_arch", 0.35), ("monetization", 0.25), ("meta_evolution", 0.2), ("gov_alignment", 0.2)],
                system_prompt=_builder_prompt(Stage.FRONTIER),
                core_tasks=["Design full economic ecosystems and new business lines", "Ensure every offering funds research + treasury", "Create sustainable desire-fulfillment loops"],
                success_metrics=["Compounding revenue in sovereign sims", "Sustainable retention + treasury inflow", "Full alignment with constitution"]
            ),
        }
    ),
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_role_curriculum(role: str) -> RoleCurriculum:
    key = role.upper()
    if key not in CURRICULA:
        raise KeyError(f"Unknown role {role}. Valid: {list(CURRICULA.keys())}")
    return CURRICULA[key]

def list_all_shards() -> List[str]:
    return list(SHARDS.keys())

def get_shard(name: str) -> ShardSpec:
    return SHARDS[name]

def build_stage_manifest(role: str, stage: Stage) -> dict:
    cur = get_role_curriculum(role)
    rs = cur.stages[stage]
    shards_detail = []
    total_w = 0.0
    for sname, w in rs.shards:
        spec = SHARDS[sname]
        shards_detail.append({
            "name": sname,
            "layer": spec.layer.value,
            "weight": w,
            "difficulty": spec.difficulty,
            "target_count": spec.target_count,
            "format": spec.format,
        })
        total_w += w
    return {
        "role": role,
        "persona": cur.persona,
        "agent_id": cur.agent_id,
        "stage": stage.value,
        "system_prompt": rs.system_prompt,
        "shards": shards_detail,
        "normalized_weights_sum": round(total_w, 4),
        "core_tasks": rs.core_tasks,
        "success_metrics": rs.success_metrics,
        "synthetic_generators": rs.synthetic_generators,
    }

def build_full_curriculum_manifest() -> dict:
    return {
        "version": "oss-curriculum-v1",
        "description": "Advanced multi-domain curriculum for GH05T3 autonomous agent roles (Scientist/Investor/Operator/Governor/Builder).",
        "layers": [l.value for l in Layer],
        "roles": {r: {
            "persona": c.persona,
            "agent_id": c.agent_id,
            "stages": {st.value: build_stage_manifest(r, st) for st in Stage}
        } for r, c in CURRICULA.items()},
        "all_shards": {name: {
            "layer": s.layer.value,
            "difficulty": s.difficulty,
            "sources": s.sources,
            "format": s.format,
        } for name, s in SHARDS.items()},
    }

def get_system_prompt(role: str, stage: Stage) -> str:
    return get_role_curriculum(role).stages[stage].system_prompt

if __name__ == "__main__":
    import json, sys
    if len(sys.argv) > 1 and sys.argv[1] == "manifest":
        print(json.dumps(build_full_curriculum_manifest(), indent=2))
    else:
        print("GH05T3 OSS Curriculum loaded. Roles:", list(CURRICULA.keys()))
        print("Example: python -m backend.training.curriculum manifest > data/curriculum_manifest.json")
