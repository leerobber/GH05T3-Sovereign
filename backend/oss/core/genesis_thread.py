"""
Genesis Thread — the permanent heartbeat of the Aethyro Execution Plane.

Runs as a daemon thread. Each heartbeat tick:
  1. Orphan Pruning  — reclaim ledger slots for agents absent > ORPHAN_TICKS ticks
  2. Dissent Pass    — numpy-based population mean → inject fitness boost to outliers
  3. Breakthrough    — monitor M_NOVELTY across ledger; signal LEX-GEN when threshold crossed

The Genesis Thread has READ access to the ChronosLedger and calls
AethyroBridge.population_dissent_pass() for write-back to agent fitness.
It never touches Control Plane data (SQLite, JSONL, learnings).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional

LOG = logging.getLogger("ghost.genesis_thread")

HEARTBEAT_SECS = 15       # tick interval
ORPHAN_TICKS   = 10       # slots not updated after this many ticks → pruned
NOVELTY_GATE   = 0.85     # M_NOVELTY threshold to trigger LEX-GEN


class GenesisThread:
    """
    Permanent background daemon for the Execution Plane.

    Usage:
        gen = GenesisThread(swarm=eco.swarm)
        gen.start()
        # ... runs indefinitely until gen.stop()
        gen.stop()
    """

    def __init__(
        self,
        swarm: Any,
        heartbeat_secs: int = HEARTBEAT_SECS,
        patent_office: Optional[Any] = None,
    ):
        self._swarm         = swarm
        self._heartbeat     = heartbeat_secs
        self._patent_office = patent_office
        self._thread:       Optional[threading.Thread] = None
        self._stop_evt      = threading.Event()
        self._tick          = 0
        self._last_slot_seen: Dict[int, int] = {}   # slot → last tick seen
        self._stats = {
            "ticks":           0,
            "orphans_pruned":  0,
            "dissent_passes":  0,
            "lex_triggers":    0,
            "bme_migrations":  0,
            "bme_promotions":  0,
            "bme_breakthroughs": 0,
        }

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="genesis-thread"
        )
        self._thread.start()
        LOG.info("GenesisThread started (heartbeat=%ds)", self._heartbeat)

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=self._heartbeat + 2)
        LOG.info("GenesisThread stopped after %d ticks", self._stats["ticks"])

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def stats(self) -> Dict[str, Any]:
        return {"running": self.is_running(), "tick": self._tick, **self._stats}

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                self._tick += 1
                self._stats["ticks"] += 1
                self._prune_orphans()
                self._dissent_pass()
                self._breakthrough_check()
                self._universe_pass()
            except Exception as exc:
                LOG.debug("genesis tick error: %s", exc)
            self._stop_evt.wait(self._heartbeat)

    # ── Phase 1: Orphan Pruning ───────────────────────────────────────────────

    def _prune_orphans(self) -> None:
        from backend.oss.core.aethyro_bridge import get_aethyro_bridge
        bridge = get_aethyro_bridge()
        ledger = bridge._ledger
        if ledger.active_slots == 0:
            return
        current_agent_ids = set(self._swarm.agents.keys()) if self._swarm else set()
        # Stamp heartbeat for all live agents so orphan age is measurable
        for aid in current_agent_ids:
            slot = bridge.get_slot(aid)
            if slot is not None:
                ledger.update_heartbeat(slot, self._tick)
        stale_ids = [aid for aid in list(bridge._slots.keys()) if aid not in current_agent_ids]
        for aid in stale_ids:
            bridge.release_slot(aid)
            self._stats["orphans_pruned"] += 1
            LOG.debug("genesis: pruned orphan slot for %s", aid)

    # ── Phase 2: Dissent Pass + Spawn Trigger ─────────────────────────────────

    def _dissent_pass(self) -> None:
        from backend.oss.core.aethyro_bridge import get_aethyro_bridge
        bridge = get_aethyro_bridge()
        if not self._swarm or not self._swarm.agents:
            return
        boosts = bridge.population_dissent_pass(self._swarm.agents)
        if not boosts:
            return
        self._stats["dissent_passes"] += 1

        # Exponential threshold: boost > 2.0 at dist ≈ 4.62 (ln(2)/0.15) — spawn offspring
        spawn_candidates = [(aid, v) for aid, v in boosts.items() if v > 2.0]
        if spawn_candidates:
            self._spawn_offspring(spawn_candidates, bridge)

        outliers = {k: v for k, v in boosts.items() if v > 1.2}
        if outliers:
            LOG.debug(
                "genesis: dissent pass — %d agents boosted, spawn_candidates=%d, top: %s",
                len(boosts), len(spawn_candidates), list(outliers.items())[:3],
            )

    def _spawn_offspring(
        self,
        candidates: list,
        bridge: Any,
    ) -> None:
        """Deploy offspring for high-dissent parent agents into next free ledger slots."""
        from backend.oss.core.mutation import get_mutation_engine
        mutation = get_mutation_engine()
        ledger   = bridge._ledger

        for parent_id, boost in candidates:
            agent = self._swarm.agents.get(parent_id) if self._swarm else None
            if agent is None:
                continue
            parent_slot = bridge.get_slot(parent_id)
            if parent_slot is None:
                continue

            # Read parent desires from ledger
            try:
                parent_data = ledger.read_agent(parent_slot)
                desires_dict = parent_data.get("desires", {})
                parent_desires = tuple(desires_dict.get(k, 0.1) for k in [
                    "KNOWLEDGE", "SKILL", "STATUS", "EXPERIENCE",
                    "CREATION", "CONNECTION", "FREEDOM",
                ])
            except Exception:
                continue

            # Mutation intensity scales with boost magnitude
            intensity = min(0.3, 0.1 * boost)
            children  = mutation.spawn_offspring(
                parent_desires, count=3, intensity=intensity
            )

            parent_gen = parent_data.get("generation", 0)

            # Preserve parent's universe bits in offspring scratchpad
            from backend.oss.core.chronos_ledger import SCRATCH_UNIVERSE_MASK, SCRATCH_ROLE_TIER_MASK, UNIVERSE_SHIFT
            parent_scratch  = parent_data.get("scratchpad", 0)
            parent_univ_bits = parent_scratch & SCRATCH_UNIVERSE_MASK  # bits 3-5

            for child_desires in children:
                try:
                    child_slot = ledger.write_at_next_available_slot(
                        desires       = tuple(float(v) for v in child_desires),
                        maturity      = max(1, parent_data.get("maturity", 1) - 1),
                        fitness       = 0.5,
                        parent_offset = parent_slot,
                        generation    = (parent_gen + 1) & 0xFF,
                        scratchpad    = parent_univ_bits,  # inherit universe; role tier starts at 0
                    )
                    # Keep bridge counter in sync
                    if child_slot >= bridge._next_slot:
                        bridge._next_slot = child_slot + 1

                    # Inherit + mutate genome via BMEBridge
                    try:
                        from backend.oss.core.bme_bridge import get_bme_bridge
                        get_bme_bridge(ledger).inherit_genome(parent_slot, child_slot)
                    except Exception as bme_exc:
                        LOG.debug("genesis: genome inheritance failed: %s", bme_exc)

                    LOG.info(
                        "genesis: spawned offspring slot=%d parent=%s boost=%.3f",
                        child_slot, parent_id, boost,
                    )
                except MemoryError as oom:
                    LOG.warning("genesis: ledger at capacity, spawn skipped: %s", oom)
                    break
            self._stats.setdefault("offspring_spawned", 0)
            self._stats["offspring_spawned"] += len(children)

    # ── Phase 3: Breakthrough Check → LEX-GEN ────────────────────────────────

    def _breakthrough_check(self) -> None:
        try:
            from backend.oss.breakthrough_detector import get_breakthrough_detector
            bd = get_breakthrough_detector()
            recent = bd.get_recent(limit=5)
            novel  = [b for b in recent if b.get("novelty_score", 0) >= NOVELTY_GATE]
            if novel and self._patent_office is not None:
                for breakthrough_dict in novel:
                    self._patent_office.handle_breakthrough_signal(breakthrough_dict)
                    self._stats["lex_triggers"] += 1
                    LOG.info(
                        "genesis: LEX-GEN triggered — novelty=%.3f",
                        breakthrough_dict.get("novelty_score", 0),
                    )
        except Exception as exc:
            LOG.debug("genesis breakthrough check error: %s", exc)

    # ── Phase 4: BME Universe Pass ────────────────────────────────────────────

    def _universe_pass(self) -> None:
        """
        Run one Binary Multiverse Engine tick:
          - Universe migration pressure check for each active agent
          - Role tier promotion evaluation
          - Breakthrough gene flagging

        Runs every HEARTBEAT_SECS ticks. Fast: all operations are mmap reads/writes
        with no LLM calls, no DB, no JSON.
        """
        try:
            from backend.oss.core.aethyro_bridge import get_aethyro_bridge
            from backend.oss.core.bme_bridge import get_bme_bridge

            bridge = get_aethyro_bridge()
            ledger = bridge._ledger
            bme    = get_bme_bridge(ledger)

            if not self._swarm or not self._swarm.agents:
                return

            # Collect active slots from bridge's slot map
            active_slots = [
                slot for slot in bridge._slots.values()
                if slot is not None
            ]
            if not active_slots:
                return

            result = bme.universe_pass(active_slots)
            self._stats["bme_migrations"]   += result.get("migrations",   0)
            self._stats["bme_promotions"]   += result.get("promotions",   0)
            self._stats["bme_breakthroughs"] += result.get("breakthroughs", 0)

            if result["migrations"] or result["promotions"]:
                LOG.debug(
                    "genesis: universe_pass tick=%d migrations=%d promotions=%d breakthroughs=%d",
                    self._tick,
                    result["migrations"],
                    result["promotions"],
                    result["breakthroughs"],
                )

            # Every 10 ticks, push summary to SovereignCore :9000 (non-blocking)
            if self._tick % 10 == 0:
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(bme.push_to_sovereign_core(
                            universe_counts=result.get("universe_counts", {}),
                            migrations=result["migrations"],
                            promotions=result["promotions"],
                            tick=self._tick,
                        ))
                except Exception:
                    pass

        except Exception as exc:
            LOG.debug("genesis: universe_pass error: %s", exc)


# ── Singleton ─────────────────────────────────────────────────────────────────

_genesis: Optional[GenesisThread] = None


def get_genesis_thread(swarm: Optional[Any] = None) -> GenesisThread:
    global _genesis
    if _genesis is None:
        _genesis = GenesisThread(swarm=swarm)
    elif swarm is not None:
        _genesis._swarm = swarm
    return _genesis
