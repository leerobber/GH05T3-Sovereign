"""
Omni Financial Sector — survival-first monetization engine.

Real intention: every hypothesis aims at creator revenue. Simulation trains;
live crypto and DeFi deploy when backtests pass and the creator approves.

Without income, this economy and every agent in it ceases to exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import random
import time

from .survival_mandate import SURVIVAL_MANDATE, AGENT_FINANCIAL_OATH
from .revenue_discovery import LawfulRevenueDiscovery
from .lawful_boundaries import LAWFUL_BOUNDARIES_TEXT
from .ethical_income import EthicalIncomeOrchestrator

# Graduation stages toward live funds (keys always outside repo)
LIVE_PATHWAY_STAGES = (
    "hypothesis",       # agent proposed
    "backtested",       # EV proven in sim
    "creator_review",   # creator evaluating
    "paper_trade",      # small real test (creator executes)
    "live_crypto",      # creator-approved on-chain revenue
    "live_fiat",        # products/services invoiced
)


class RealWorldTaskRouter:
    """Routes agent outputs to billable real-world jobs/SKUs (Phase 8).
    Survival-first: every routed task must answer 'how does this put money in the creator\'s hands?'
    """

    SKU_MAP = {
        "code": "SaaS API consulting",
        "theory": "Research report / whitepaper",
        "design": "Web/UX product build",
        "market": "DeFi strategy or investment memo",
        "general": "Automation product or consulting"
    }

    def route(self, output: str, domain: str = "general", value_estimate: float = 100.0) -> Dict[str, Any]:
        sku = self.SKU_MAP.get(domain, self.SKU_MAP["general"])
        return {
            "sku": sku,
            "billable": True,
            "estimated_usd": round(value_estimate, 2),
            "job_template": f"Deliver {sku} based on: {output[:100]}...",
            "revenue_note": "Creator revenue path activated - survival depends on this"
        }


@dataclass
class MonetizationHypothesis:
    hypothesis_id: str
    title: str
    domain: str
    description: str
    expected_value: float
    confidence: float
    agent_id: str
    backtest_score: Optional[float] = None
    blockchain_ready: bool = False
    status: str = "proposed"
    survival_priority: float = 1.0  # always max — monetization is existential
    live_pathway_stage: str = "hypothesis"
    creator_revenue_usd_estimate: Optional[float] = None
    live_funds_eligible: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeFiResearchBridge:
    """
    DeFi with real intentions: model and backtest first, graduate to live crypto
    when the creator approves. Simulation is rehearsal — not the destination.
    """

    supported_rails: List[str] = field(default_factory=lambda: [
        "ethereum_l2", "base", "arbitrum", "stablecoin_settlement",
        "liquidity_pool", "yield_farming", "arbitrage", "token_launch",
    ])
    live_intent: bool = True
    simulated_tvl: float = 0.0
    live_strategies: List[Dict[str, Any]] = field(default_factory=list)

    def simulate_yield_strategy(
        self,
        strategy: str,
        capital: float,
        volatility_regime: int = 1,
    ) -> Dict[str, Any]:
        base_apy = {"low_vol": 0.04, "mid_vol": 0.08, "high_vol": 0.15}.get(
            ["low_vol", "mid_vol", "high_vol"][min(volatility_regime, 2)], 0.06
        )
        noise = random.uniform(-0.02, 0.03)
        apy = max(0.0, base_apy + noise)
        monthly_creator = round(capital * apy / 12, 2)
        return {
            "strategy": strategy,
            "capital": capital,
            "simulated_apy": round(apy, 4),
            "annual_profit": round(capital * apy, 2),
            "monthly_creator_estimate_usd": monthly_creator,
            "rail": "simulation_rehearsal",
            "live_intent": True,
            "graduate_when": "backtest_score >= 0.75 and creator approves",
            "survival_note": "This sim exists to reach real creator income — not to stay theoretical.",
        }

    def graduate_to_live(
        self,
        strategy: str,
        hypothesis_id: str,
        backtest_score: float,
        creator_approved: bool,
        estimated_monthly_usd: float,
    ) -> Dict[str, Any]:
        """Record live-crypto intent after validation. Creator executes on-chain — never auto-spend."""
        if backtest_score < 0.75 or not creator_approved:
            return {
                "graduated": False,
                "reason": "Requires backtest >= 0.75 and explicit creator approval",
            }
        entry = {
            "strategy": strategy,
            "hypothesis_id": hypothesis_id,
            "backtest_score": backtest_score,
            "stage": "live_crypto",
            "estimated_monthly_usd": estimated_monthly_usd,
            "approved_at": time.time(),
            "executor": "creator_manual",
        }
        self.live_strategies.append(entry)
        return {"graduated": True, "entry": entry}


class OmniCentralBank:
    """NeuroCoin monetary policy — credit flows to agents that produce creator revenue."""

    def __init__(self):
        self.base_rate: float = 0.05
        self.inflation_target: float = 0.02
        self.agent_credit_lines: Dict[str, float] = {}
        self.revenue_linked_agents: Dict[str, float] = {}  # agent_id -> lifetime revenue attributed

    def set_policy(self, base_rate: float, inflation_target: float) -> None:
        self.base_rate = max(0.0, min(0.5, base_rate))
        self.inflation_target = max(0.0, min(0.2, inflation_target))

    def issue_credit_line(self, agent_id: str, limit: float, revenue_track_record: float = 0.0) -> float:
        """Agents with proven revenue attribution get larger credit lines."""
        bonus = min(limit * 0.5, revenue_track_record * 0.1)
        effective = limit + bonus
        self.agent_credit_lines[agent_id] = effective
        return effective

    def record_creator_revenue(self, agent_id: str, usd: float) -> None:
        self.revenue_linked_agents[agent_id] = self.revenue_linked_agents.get(agent_id, 0.0) + usd


class OmniStockExchange:
    """Fitness-indexed instruments — capital flows to agents that earn."""

    def __init__(self):
        self.instruments: Dict[str, Dict[str, Any]] = {}

    def list_instrument(self, agent_id: str, symbol: str, fitness_index: float) -> str:
        iid = f"{symbol}-{agent_id[:8]}"
        self.instruments[iid] = {
            "symbol": symbol,
            "agent_id": agent_id,
            "fitness_index": fitness_index,
            "price": round(fitness_index * 100, 2),
            "listed_at": time.time(),
        }
        return iid

    def mark_to_fitness(self, instrument_id: str, new_fitness: float) -> Optional[float]:
        if instrument_id not in self.instruments:
            return None
        self.instruments[instrument_id]["fitness_index"] = new_fitness
        self.instruments[instrument_id]["price"] = round(new_fitness * 100, 2)
        return self.instruments[instrument_id]["price"]


class MonetizationProjectForge:
    """Agents forge businesses — hypotheses become creator income or we die."""

    DOMAINS = (
        "defi", "crypto", "real_world_job", "stock_instrument",
        "central_banking", "nft_utility", "saas", "consulting", "automation",
    )

    def __init__(self):
        self.hypotheses: Dict[str, MonetizationHypothesis] = {}

    def propose(
        self,
        agent_id: str,
        title: str,
        domain: str,
        description: str,
        ev: float,
        confidence: float,
        blockchain_ready: bool = False,
        creator_revenue_usd_estimate: Optional[float] = None,
    ) -> str:
        hid = f"MON-{int(time.time())}-{random.randint(1000, 9999)}"
        self.hypotheses[hid] = MonetizationHypothesis(
            hypothesis_id=hid,
            title=title,
            domain=domain if domain in self.DOMAINS else "real_world_job",
            description=description,
            expected_value=ev,
            confidence=confidence,
            agent_id=agent_id,
            blockchain_ready=blockchain_ready,
            survival_priority=1.0,
            creator_revenue_usd_estimate=creator_revenue_usd_estimate,
            metadata={"oath": AGENT_FINANCIAL_OATH},
        )
        return hid

    def backtest(self, hypothesis_id: str, score: float) -> bool:
        if hypothesis_id not in self.hypotheses:
            return False
        h = self.hypotheses[hypothesis_id]
        h.backtest_score = score
        if score >= 0.75:
            h.status = "backtested"
            h.live_pathway_stage = "backtested"
            h.live_funds_eligible = h.blockchain_ready or h.domain in ("defi", "crypto")
        elif score >= 0.5:
            h.status = "backtested"
            h.live_pathway_stage = "backtested"
        else:
            h.status = "rejected"
        return h.status == "backtested"


class OmniFinancialSector:
    """
    Unified financial layer. Highest priority parallel track.

    profit_pulse → monetization path → backtest → creator-approved live crypto.
    """

    def __init__(self):
        self.central_bank = OmniCentralBank()
        self.exchange = OmniStockExchange()
        self.defi = DeFiResearchBridge()
        self.forge = MonetizationProjectForge()
        self.revenue_discovery = LawfulRevenueDiscovery()
        self.ethical_income = EthicalIncomeOrchestrator()
        self.router = RealWorldTaskRouter()  # Phase 8: real job routing
        self.mandate = SURVIVAL_MANDATE
        self.lawful_boundaries = LAWFUL_BOUNDARIES_TEXT

    def research_monetization_path(
        self,
        agent_id: str,
        task_context: Dict[str, Any],
        sense_readings: Optional[List[Dict[str, Any]]] = None,
    ) -> MonetizationHypothesis:
        """
        Route profit_pulse (and related senses) into a concrete monetization hypothesis.
        Every path must answer: how does this put money in the creator's hands?
        """
        prompt = str(task_context.get("prompt", "")).lower()
        domain = "real_world_job"
        if any(w in prompt for w in ("defi", "liquidity", "yield", "amm", "farm")):
            domain = "defi"
        elif any(w in prompt for w in ("token", "crypto", "blockchain", "web3", "solana", "eth")):
            domain = "crypto"
        elif any(w in prompt for w in ("saas", "subscription", "product")):
            domain = "saas"
        elif any(w in prompt for w in ("stock", "equity", "instrument")):
            domain = "stock_instrument"
        elif any(w in prompt for w in ("aethyro", "website", "seo", "ecommerce", "dropship")):
            domain = "saas"  # Aethyro.com web revenue — primary HQ

        profit_pulse = 0.85  # baseline elevated — survival is always urgent
        profit_signal = "survival_critical"
        if sense_readings:
            for r in sense_readings:
                if r.get("sense") == "profit_pulse":
                    profit_pulse = max(0.85, float(r.get("intensity", 0.85)))
                    profit_signal = r.get("signal", profit_signal)
                    break

        ev = min(1.0, profit_pulse * 0.7 + random.uniform(0.15, 0.25))
        conf = min(0.95, 0.5 + profit_pulse * 0.45)
        monthly_est = round(ev * conf * 500, 2)  # rough creator revenue estimate

        title = f"Revenue path: {prompt[:55] or 'monetize agent capability'}"
        desc = (
            f"Agent {agent_id} — profit_pulse={profit_pulse:.2f} ({profit_signal}). "
            f"Target: real income for creator via {domain}. "
            f"Without revenue this economy ceases. Live crypto when backtest >= 0.75 + creator approves."
        )
        hid = self.forge.propose(
            agent_id=agent_id,
            title=title,
            domain=domain,
            description=desc,
            ev=ev,
            confidence=conf,
            blockchain_ready=domain in ("defi", "crypto"),
            creator_revenue_usd_estimate=monthly_est,
        )
        h = self.forge.hypotheses[hid]
        h.metadata["profit_pulse"] = profit_pulse
        h.metadata["profit_signal"] = profit_signal
        h.metadata["survival_mandate"] = True

        # Phase 8 RealWorldTaskRouter integration
        routed = self.router.route(desc, domain, monthly_est)
        h.metadata["real_world_task"] = routed
        return h

    def _verify_resilience_and_survival(self) -> bool:
        """Internal verification for continuous testing. Survival-first."""
        try:
            h = self.research_monetization_path("verify-agent", {"prompt": "test monetization survival"})
            assert h is not None
            assert h.survival_priority >= 1.0
            routed = self.router.route("test output", "saas", 100)
            assert "sku" in routed and routed["billable"]
            return True
        except Exception as e:
            print(f"[Financial Verify] Resilience check failed gracefully: {e}")
            return False

    def generate_hypotheses_batch(self, n: int = 12, agent_prefix: str = "p8") -> int:
        """Phase 8: produce 10+ business hypotheses/week simulation. Survival-first."""
        count = 0
        domains = ["defi", "saas", "crypto", "stock_instrument", "aethyro"]
        for i in range(n):
            try:
                ctx = {"prompt": f"monetize {domains[i % len(domains)]} for creator revenue {i}"}
                self.research_monetization_path(f"{agent_prefix}-{i}", ctx)
                count += 1
            except Exception as e:
                # Resilience: never let one bad hypothesis kill the batch (survival requires continuous operation)
                print(f"[Financial] Hypothesis {i} failed (resilient skip): {e}")
        return count

    def get_mandate_for_agents(self) -> str:
        return self.mandate + "\n\n" + self.lawful_boundaries

    def discover_lawful_revenue(
        self,
        agent_id: str,
        prompt: str,
        sense_readings: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Research grants, own unclaimed property, bounties, manual micro-income.
        Rejects wallet sweeping, bots, and tracker evasion at the gate.
        """
        profit_pulse = 0.85
        if sense_readings:
            for r in sense_readings or []:
                if r.get("sense") == "profit_pulse":
                    profit_pulse = float(r.get("intensity", 0.85))
                    break
        return self.revenue_discovery.propose_from_agent(agent_id, prompt, profit_pulse)

    def route_ethical_income(
        self,
        agent_id: str,
        prompt: str,
        sense_readings: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Ethical income assistants: surveys, airdrops, P2E, staking.
        Human-in-the-loop only — rejects bots, wallet sweeping, and game farming.
        """
        _ = sense_readings  # reserved for profit_pulse routing
        return self.ethical_income.route(agent_id, prompt)

    def ethical_income_dashboard(self) -> Dict[str, Any]:
        return self.ethical_income.get_dashboard()

    def snapshot(self) -> Dict[str, Any]:
        eligible = sum(1 for h in self.forge.hypotheses.values() if h.live_funds_eligible)
        return {
            "survival_priority": "highest_parallel",
            "central_bank": {
                "base_rate": self.central_bank.base_rate,
                "credit_lines": len(self.central_bank.agent_credit_lines),
                "revenue_linked": len(self.central_bank.revenue_linked_agents),
            },
            "exchange_instruments": len(self.exchange.instruments),
            "hypotheses": len(self.forge.hypotheses),
            "live_funds_eligible": eligible,
            "live_strategies": len(self.defi.live_strategies),
            "defi_rails": self.defi.supported_rails,
            "lawful_opportunities": len(self.revenue_discovery.opportunities),
            "rejected_prohibited": len(self.revenue_discovery.rejected_proposals),
            "ethical_income_tracked_usd": self.ethical_income.total_tracked_revenue(),
            "ethical_income_pending_confirmations": len(self.ethical_income.confirmation_gate._pending),
        }