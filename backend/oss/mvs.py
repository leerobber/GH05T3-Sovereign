"""
Minimal Viable Substrate (MVS) Entry Point

Stabilized core:
- OmniDNA v1.0
- GenomicSubstrate v1.0
- OmniMind v1.0
- OmniEconomy v1.0

Use this to get a consistent, debuggable foundation.
"""

from .omni_dna import OmniDNA, create_omnidna, UNIVERSAL_TRAITS
from .genomic_substrate import GenomicSubstrate, AgentHandle
from .omni_mind import OmniMind
from .omni_economy import OmniEconomy
from .species_memory import get_species_memory
from .speciation import get_speciation_engine
from .omni_net import get_omni_net  # Phase 5

# Phase 6: lazy imports to avoid circulars
def get_meta_learning_engine():
    from .evolution.meta_learning import get_meta_learning_engine as _g
    return _g()

def get_self_improving_loop_v2():
    from .evolution.self_improving_loop_v2 import get_self_improving_loop_v2 as _g
    return _g()

# Phase 7
def get_self_directed_goals_engine(mind=None):
    from .mind.self_directed_goals import get_self_directed_goals_engine as _g
    return _g(mind)

def get_autonomous_research_engine(economy=None):
    from .research.autonomous_research import get_autonomous_research_engine as _g
    return _g(economy)

def run_grand_challenges():
    from .research.grand_challenges import run_grand_challenge_demo
    return run_grand_challenge_demo()

def _get_financial():
    from .financial.omni_financial_sector import OmniFinancialSector
    return OmniFinancialSector()

# Elite layer (lazy — avoids circular import on first MVS boot)
def get_elite_fabric():
    from .elite.omni_fabric import OmniFabric
    return OmniFabric()

def get_elite_thread_engine():
    from .elite.omni_thread import OmniThreadEngine
    return OmniThreadEngine()

def get_elite_aegis():
    from .elite.omni_aegis import AegisFastPath
    return AegisFastPath()

def get_elite_lineage_manager():
    from .elite_lineages import EliteLineageManager
    return EliteLineageManager()

__all__ = [
    "OmniDNA",
    "create_omnidna",
    "UNIVERSAL_TRAITS",
    "GenomicSubstrate",
    "AgentHandle",
    "OmniMind",
    "OmniEconomy",
    "get_mvs",
    "get_elite_fabric",
    "get_elite_thread_engine",
    "get_elite_aegis",
    "get_elite_lineage_manager",
]

_mvs = None

def get_mvs() -> dict:
    global _mvs
    if _mvs is None:
        try:
            sub = GenomicSubstrate()
            mind = OmniMind(sub)
            econ = OmniEconomy()
            _mvs = {
                "substrate": sub,
                "mind": mind,
                "economy": econ,
                "species_memory": get_species_memory(),
                "speciation": get_speciation_engine(),
                "net": get_omni_net(),  # Phase 5 Omni-Net Alpha
                "meta_learning": get_meta_learning_engine(),  # Phase 6
                "self_improving_loop": get_self_improving_loop_v2(),  # Phase 6
                "self_directed_goals": get_self_directed_goals_engine(),  # Phase 7
                "autonomous_research": get_autonomous_research_engine(),  # Phase 7
                "grand_challenges": run_grand_challenges(),  # Phase 7
                "financial": _get_financial(),  # Phase 8 parallel - SURVIVAL CRITICAL (monetization = oxygen)
                # Elite layer — zero-bottleneck fabric, sharding, fast-path security, lineages
                "elite_fabric": get_elite_fabric(),
                "elite_thread": get_elite_thread_engine(),
                "elite_aegis": get_elite_aegis(),
                "elite_lineages": get_elite_lineage_manager(),
            }
        except Exception as e:
            # Resilience: MVS must degrade gracefully. Financial paths must remain accessible for survival-first operation.
            print(f"[MVS] Partial load due to: {e}. Core may be degraded. Survival/monetization still prioritized.")
            _mvs = {"error": str(e), "partial": True, "survival_mode": True}
    return _mvs

def get_theorist_population():
    """Convenience for theory-heavy labs only (research, architecture, meta-design)."""
    mvs = get_mvs()
    sub = mvs["substrate"]
    return [gid for gid, rec in sub.genomes.items() if "THEORIST" in rec.role.upper()]

def create_theorist_elite(seed: int = None):
    """Helper to create/register high-spec Theorist for theory labs."""
    dna = create_omnidna("THEORIST_ELITE", seed=seed)
    for t in ["math", "pattern_detection", "self_reflection", "creativity", "alignment"]:
        if t in dna.traits:
            dna.traits[t] = max(dna.traits[t], 0.88)
    sub = get_mvs()["substrate"]
    sub.register_genome(dna)
    return dna.genome_id


def create_web_engineer_elite(seed: int = None):
    """Elite web design/SEO/revenue species — base of operations: Aethyro.com."""
    dna = create_omnidna("WEB_ENGINEER_ELITE", seed=seed)
    for t in ["creativity", "efficiency", "innovation", "market_intuition", "pattern_detection", "persistence"]:
        if t in dna.traits:
            dna.traits[t] = max(dna.traits[t], 0.90)
    dna.power_tier = "T2"
    sub = get_mvs()["substrate"]
    sub.register_genome(dna)
    return dna.genome_id


def get_web_engineer_population():
    """All WEB_ENGINEER_ELITE genomes registered in substrate."""
    mvs = get_mvs()
    sub = mvs["substrate"]
    return [gid for gid, rec in sub.genomes.items() if "WEB_ENGINEER" in rec.role.upper()]


def create_hyper_elite_psychology_agent(seed: int = None):
    """
    Register Hyper-Elite Psychology species (genomic layer + MVS OmniDNA).
    Bridges molecule-level genome to trait dict for Theory Lab / Aethyro web tasks.
    """
    from backend.oss.genomic import create_hyper_elite_psychology_genome, sync_genome_to_omni_dna

    genome = create_hyper_elite_psychology_genome(seed=seed, role="HYPER_ELITE_PSYCHOLOGIST")
    dna = create_omnidna("HYPER_ELITE_PSYCHOLOGIST", seed=seed)
    sync_genome_to_omni_dna(genome, dna)
    for t in ("creativity", "market_intuition", "pattern_detection", "alignment", "novelty_seeking"):
        if t in dna.traits:
            dna.traits[t] = max(dna.traits[t], 0.88)
    dna.power_tier = "T2"
    sub = get_mvs()["substrate"]
    sub.register_genome(dna, role="HYPER_ELITE_PSYCHOLOGIST")
    return dna.genome_id
