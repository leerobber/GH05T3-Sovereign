"""
Omni-Net Alpha (Phase 5) — The Network Layer for users + agents in living ecosystem.

Canonical package per OMNI_SENTIENT_BUILD_PLAN.md
- Builds on Beta OmniNet (broadcast, memetic spread, consensus)
- Adds full Phase 5 components: AgentInterface, RoutingEngine, KnowledgeSyncEngine, MultiAgentChat, OmniSecurity

Integrates with:
- Phase 3 KnowledgeGraph
- DNA v2 traits for routing
- Elite lineages
- Economy rewards
- MVS / TheoryLab via get_omni_net()

Success metrics targeted in sims/tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import time
import random
import json
from pathlib import Path

# Re-export Phase 5 components
from .agent_interface import UserQuery, AgentInterface
from .routing import RoutingEngine
from .knowledge_sync import KnowledgeSyncEngine, SyncEvent
from .multi_agent_chat import MultiAgentChat
from .security import OmniSecurity

__all__ = [
    "OmniNet", "Peer", "get_omni_net",
    "UserQuery", "AgentInterface",
    "RoutingEngine",
    "KnowledgeSyncEngine", "SyncEvent",
    "MultiAgentChat",
    "OmniSecurity",
]


@dataclass
class Peer:
    genome_id: str
    role: str
    traits: Dict[str, float]
    last_seen: float = field(default_factory=time.time)
    published_count: int = 0
    reputation: float = 0.5


class OmniNet:
    """Core in-memory + persist net. Extended for Phase 5."""

    PERSIST_PATH = Path(__file__).resolve().parents[2] / "data" / "omni_net_state.json"

    def __init__(self, max_peers: int = 500, persist: bool = True):
        self.max_peers = max_peers
        self.peers: Dict[str, Peer] = {}
        self.theory_feed: List[Dict[str, Any]] = []
        self.canonical_gossip: List[Dict[str, Any]] = []
        self._last_prune = time.time()
        self.persist = persist
        if persist:
            self._load()

    # Registration etc (full original + Phase 5 hooks)
    def register(self, genome_id: str, role: str, traits: Dict[str, float]) -> Peer:
        if genome_id in self.peers:
            p = self.peers[genome_id]
            p.last_seen = time.time()
            p.traits = traits
            return p
        if len(self.peers) >= self.max_peers:
            self._prune_old_peers()
        p = Peer(genome_id=genome_id, role=role, traits=dict(traits))
        self.peers[genome_id] = p
        return p

    def unregister(self, genome_id: str):
        self.peers.pop(genome_id, None)

    def broadcast_theory(self, genome_id: str, proposal: str, score: float, world: str = "unknown", meta: Optional[Dict] = None):
        if genome_id not in self.peers:
            return False
        peer = self.peers[genome_id]
        entry = {
            "genome_id": genome_id,
            "role": peer.role,
            "score": round(score, 4),
            "world": world,
            "proposal": proposal[:1200],
            "ts": time.time(),
            "traits_snapshot": {k: round(v, 2) for k, v in list(peer.traits.items())[:6]},
            **(meta or {})
        }
        self.theory_feed.append(entry)
        if len(self.theory_feed) > 2000:
            self.theory_feed = self.theory_feed[-1500:]
        if score > 0.8:
            peer.reputation = min(0.98, peer.reputation + 0.03)
        elif score < 0.5:
            peer.reputation = max(0.1, peer.reputation - 0.02)
        peer.published_count += 1
        peer.last_seen = time.time()
        if self.persist:
            self._save()
        return True

    def publish_canonical(self, genome_id: str, memory: Dict[str, Any]):
        if genome_id not in self.peers:
            return
        mem = dict(memory)
        mem["source"] = genome_id
        mem["ts"] = time.time()
        self.canonical_gossip.append(mem)
        if len(self.canonical_gossip) > 500:
            self.canonical_gossip = self.canonical_gossip[-350:]
        if self.persist:
            self._save()

    def pull_canonical_memories(self, limit: int = 20, min_score: float = 0.6) -> List[Dict[str, Any]]:
        good = [m for m in self.canonical_gossip if m.get("computed_score", 0) >= min_score or m.get("canonical")]
        return sorted(good, key=lambda x: x.get("ts", 0), reverse=True)[:limit]

    def sample_peers(self, k: int = 5, role_filter: Optional[str] = None) -> List[Peer]:
        peers = list(self.peers.values())
        if role_filter:
            peers = [p for p in peers if role_filter.upper() in p.role.upper()]
        peers.sort(key=lambda p: p.reputation + random.random()*0.1, reverse=True)
        return peers[:k]

    def net_consensus(self, proposals: List[Dict[str, Any]], boost_theorists: bool = True) -> Dict[str, Any]:
        if not proposals:
            return {}
        total = 0.0
        acc: Dict[str, float] = {}
        for p in proposals:
            gid = p.get("genome_id", "anon")
            peer = self.peers.get(gid)
            w = (peer.reputation if peer else 0.5) + 0.3
            if boost_theorists and "THEORIST" in str(p.get("role", "")).upper():
                w *= 1.7
            if p.get("score"):
                w *= (0.5 + p["score"])
            total += w
            for k, v in p.items():
                if isinstance(v, (int, float)):
                    acc[k] = acc.get(k, 0.0) + float(v) * w
        if total <= 0:
            return {}
        return {k: round(v / total, 4) for k, v in acc.items()}

    def memetic_spread(self, source_gid: str, target_gids: List[str], strength: float = 0.12):
        if source_gid not in self.peers:
            return 0
        source = self.peers[source_gid]
        spread = 0
        for gid in target_gids:
            if gid in self.peers and gid != source_gid:
                tgt = self.peers[gid]
                for t, val in source.traits.items():
                    if t in tgt.traits:
                        old = tgt.traits[t]
                        tgt.traits[t] = max(0.1, min(0.95, old * (1-strength) + val * strength))
                spread += 1
        return spread

    # Phase 5: Agent Interface integration hook
    def route_to_elites(self, query_text: str, traits: Optional[Dict[str, float]] = None) -> List[str]:
        # Delegates to RoutingEngine when available
        try:
            from .routing import RoutingEngine
            re = RoutingEngine()
            primary = re.route(query_text, traits)
            return [primary, "THEORIST_ELITE"]
        except Exception:
            return ["THEORIST_ELITE"]

    # Persistence
    def _load(self):
        try:
            if self.PERSIST_PATH.exists():
                raw = json.loads(self.PERSIST_PATH.read_text(encoding="utf-8"))
                for gid, p in raw.get("peers", {}).items():
                    self.peers[gid] = Peer(**p)
                self.theory_feed = raw.get("theory_feed", [])[-1500:]
                self.canonical_gossip = raw.get("canonical_gossip", [])[-350:]
        except Exception:
            pass

    def _save(self):
        try:
            self.PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "peers": {gid: {"genome_id": p.genome_id, "role": p.role, "traits": p.traits,
                                "last_seen": p.last_seen, "published_count": p.published_count,
                                "reputation": p.reputation} for gid, p in self.peers.items()},
                "theory_feed": self.theory_feed[-1500:],
                "canonical_gossip": self.canonical_gossip[-350:],
            }
            self.PERSIST_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _prune_old_peers(self):
        now = time.time()
        stale = [gid for gid, p in self.peers.items() if now - p.last_seen > 3600]
        for gid in stale[:max(1, len(stale)//3)]:
            self.peers.pop(gid, None)
        if self.persist:
            self._save()

    def stats(self) -> Dict[str, Any]:
        if self.persist:
            self._save()
        return {
            "peer_count": len(self.peers),
            "theories_published": len(self.theory_feed),
            "canonical_gossip": len(self.canonical_gossip),
            "top_reputation": sorted(
                [(p.genome_id, round(p.reputation, 2), p.role) for p in self.peers.values()],
                key=lambda x: -x[1]
            )[:5]
        }


# Singleton
_global_net: Optional[OmniNet] = None

def get_omni_net() -> OmniNet:
    global _global_net
    if _global_net is None:
        _global_net = OmniNet()
    return _global_net


# Demo
if __name__ == "__main__":
    net = get_omni_net()
    net.register("DNA-THE-001", "THEORIST_ELITE", {"math": 0.95, "alignment": 0.92})
    net.broadcast_theory("DNA-THE-001", "Regime-aware model...", score=0.91)
    print("Omni-Net Alpha ready. Stats:", net.stats())
