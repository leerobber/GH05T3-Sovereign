"""GH05T3 — Agent Marketplace: persistent job queue + autoscaling pool.

Agents post and claim jobs through this queue. The AgentPool spawns
additional agent coroutine workers when the queue backs up and retires idle
ones, enabling thousands of parallel agents sharing a single inference
endpoint.

SQLite tables in palace.db (shared with memory + economy):
    marketplace_jobs     — pending/claimed/completed/failed jobs
    marketplace_workers  — registered active worker instances

Job lifecycle:
    PENDING → CLAIMED → COMPLETED | FAILED | EXPIRED

Real-world ingestion hooks (called from gateway_v3.py):
    ingest_github_event(event)   — PR/push/issue → FORGE/CODEX job
    ingest_stripe_event(event)   — subscription events → LEDGER job
    ingest_cve_feed(cve_list)    — CVE records → SENTINEL jobs

Env vars:
    MARKETPLACE_DB_PATH     SQLite path (default: memory/palace.db)
    MARKETPLACE_AUTOSCALE   "1" to enable autoscaler (default: "1")
    MARKETPLACE_MAX_WORKERS max workers per agent type (default: 8)
    MARKETPLACE_SCALE_UP_AT queue depth to trigger scale-up (default: 10)
    MARKETPLACE_IDLE_TTL    seconds idle before worker retires (default: 120)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from typing import Optional, Callable, Awaitable

LOG = logging.getLogger("ghost.marketplace")

_DB_PATH = Path(os.environ.get("MARKETPLACE_DB_PATH",
                               str(Path(__file__).parent / "memory" / "palace.db")))
_AUTOSCALE     = os.environ.get("MARKETPLACE_AUTOSCALE",   "1") == "1"
_MAX_WORKERS   = int(os.environ.get("MARKETPLACE_MAX_WORKERS", "8"))
_SCALE_UP_AT   = int(os.environ.get("MARKETPLACE_SCALE_UP_AT", "10"))
_IDLE_TTL      = int(os.environ.get("MARKETPLACE_IDLE_TTL",    "120"))
_JOB_EXPIRE_S     = 3600   # PENDING jobs unclaimed after 1h are expired
_CLAIM_TIMEOUT_S  = int(os.environ.get("MARKETPLACE_CLAIM_TIMEOUT", "300"))  # CLAIMED lease: 5 min
BID_WINDOW_S      = int(os.environ.get("MARKETPLACE_BID_WINDOW", "10"))


# ---------------------------------------------------------------------------
# Enums + dataclasses
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    PENDING    = "pending"
    BIDDING    = "bidding"
    CLAIMED    = "claimed"
    COMPLETED  = "completed"
    FAILED     = "failed"
    EXPIRED    = "expired"
    VALIDATING = "validating"
    VALIDATED  = "validated"
    REJECTED   = "rejected"


@dataclass
class Job:
    id:           str
    task:         str
    tags:         list[str]
    reward:       int
    status:       JobStatus
    posted_by:    str
    claimed_by:   str
    created_at:   float
    claimed_at:   float
    completed_at: float
    result:       str
    metadata:     dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

@contextmanager
def _conn(immediate: bool = False):
    """WAL-mode connection with explicit transaction management.

    Use immediate=True for claim() to serialise the SELECT + UPDATE
    so two workers cannot claim the same job concurrently.
    """
    c = sqlite3.connect(_DB_PATH, timeout=10, check_same_thread=False,
                        isolation_level=None)
    try:
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
        yield c
        c.execute("COMMIT")
    except Exception:
        try:
            c.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        c.close()


def _init_db():
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Ensure ledger tables (agent_credits, credit_transactions) exist in the
    # same DB before complete() tries to write to them atomically.
    try:
        from economy.ledger import get_ledger as _get_ledger
        _get_ledger()
    except Exception:
        pass
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS marketplace_jobs (
                id           TEXT    PRIMARY KEY,
                task         TEXT    NOT NULL,
                tags         TEXT    NOT NULL DEFAULT '[]',
                reward       INTEGER NOT NULL DEFAULT 0,
                status       TEXT    NOT NULL DEFAULT 'pending',
                posted_by    TEXT    NOT NULL DEFAULT 'system',
                claimed_by   TEXT    NOT NULL DEFAULT '',
                created_at   REAL    NOT NULL,
                claimed_at   REAL    NOT NULL DEFAULT 0,
                completed_at REAL    NOT NULL DEFAULT 0,
                result       TEXT    NOT NULL DEFAULT '',
                metadata     TEXT    NOT NULL DEFAULT '{}'
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS marketplace_workers (
                worker_id    TEXT    PRIMARY KEY,
                agent_type   TEXT    NOT NULL,
                capabilities TEXT    NOT NULL DEFAULT '[]',
                spawned_at   REAL    NOT NULL,
                last_active  REAL    NOT NULL,
                jobs_done    INTEGER NOT NULL DEFAULT 0,
                status       TEXT    NOT NULL DEFAULT 'idle'
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON marketplace_jobs(status)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_jobs_tags   ON marketplace_jobs(tags)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_jobs_time   ON marketplace_jobs(created_at)")
        c.execute("""
            CREATE TABLE IF NOT EXISTS marketplace_bids (
                id         TEXT PRIMARY KEY,
                job_id     TEXT NOT NULL,
                agent_id   TEXT NOT NULL,
                bid        REAL NOT NULL,
                submitted  REAL NOT NULL,
                UNIQUE(job_id, agent_id)
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_bids_job ON marketplace_bids(job_id)")


# ---------------------------------------------------------------------------
# Job Queue
# ---------------------------------------------------------------------------

class JobQueue:
    """Thread-safe, process-safe persistent job queue."""

    _instance: Optional["JobQueue"] = None

    def __init__(self):
        _init_db()
        self._listeners: list[asyncio.Queue] = []

    @classmethod
    def instance(cls) -> "JobQueue":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── posting ──────────────────────────────────────────────────────────────

    async def post(self, task: str, tags: list[str] = None,
                   reward: int = 20, posted_by: str = "system",
                   metadata: dict = None) -> str:
        job_id  = str(uuid.uuid4())[:12]
        tags_j  = json.dumps(tags or [])
        meta_j  = json.dumps(metadata or {})
        now     = time.time()

        def _insert():
            with _conn() as c:
                c.execute(
                    "INSERT INTO marketplace_jobs "
                    "(id, task, tags, reward, status, posted_by, created_at, metadata) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (job_id, task, tags_j, reward,
                     JobStatus.PENDING.value, posted_by, now, meta_j),
                )
        await asyncio.to_thread(_insert)

        LOG.info("[mkt] posted job=%s tags=%s reward=%d  %.60s", job_id, tags, reward, task)
        for q in self._listeners:
            try:
                q.put_nowait(job_id)
            except asyncio.QueueFull:
                pass
        return job_id

    # ── claiming ─────────────────────────────────────────────────────────────

    async def claim(self, agent_id: str, capability_tags: list[str]) -> Optional[Job]:
        """Atomically claim the highest-reward matching pending job.

        Uses BEGIN IMMEDIATE so only one worker can execute the SELECT+UPDATE
        at a time, preventing double-claims. Tag matching uses json_each for
        exact token comparison instead of fragile LIKE substring matching.
        Runs in a thread so the event loop is never blocked by SQLite I/O.
        """
        def _do() -> Optional[Job]:
            if capability_tags:
                placeholders = ",".join("?" * len(capability_tags))
                tag_filter = (
                    f"EXISTS (SELECT 1 FROM json_each(tags) WHERE value IN ({placeholders}))"
                )
                params = list(capability_tags)
            else:
                tag_filter = "1=1"
                params = []

            with _conn(immediate=True) as c:
                row = c.execute(
                    f"SELECT id, task, tags, reward, status, posted_by, claimed_by, "
                    f"created_at, claimed_at, completed_at, result, metadata "
                    f"FROM marketplace_jobs "
                    f"WHERE status=? AND ({tag_filter}) "
                    f"ORDER BY reward DESC, created_at ASC LIMIT 1",
                    [JobStatus.PENDING.value] + params,
                ).fetchone()

                if not row:
                    return None

                now = time.time()
                updated = c.execute(
                    "UPDATE marketplace_jobs SET status=?, claimed_by=?, claimed_at=? "
                    "WHERE id=? AND status=?",
                    (JobStatus.CLAIMED.value, agent_id, now,
                     row[0], JobStatus.PENDING.value),
                ).rowcount

                if not updated:
                    return None  # race: another worker claimed it between SELECT and UPDATE

            return Job(
                id=row[0], task=row[1], tags=json.loads(row[2]),
                reward=row[3], status=JobStatus.CLAIMED,
                posted_by=row[5], claimed_by=agent_id,
                created_at=row[7], claimed_at=now,
                completed_at=0, result="",
                metadata=json.loads(row[11]),
            )

        return await asyncio.to_thread(_do)

    # ── completing ────────────────────────────────────────────────────────────

    async def complete(self, job_id: str, result: str, agent_id: str) -> bool:
        """Mark a job completed and credit the agent atomically in one transaction.

        Both the job-status update and the ledger credit happen in the same
        SQLite transaction (both tables live in palace.db), so a crash between
        them is impossible — either both commit or neither does.
        """
        _completed_metadata: dict = {}

        def _do() -> tuple[bool, int, float]:
            with _conn(immediate=True) as c:
                row = c.execute(
                    "SELECT reward, claimed_by, status, metadata FROM marketplace_jobs WHERE id=?",
                    (job_id,)
                ).fetchone()
                if not row or row[1] != agent_id or row[2] != JobStatus.CLAIMED.value:
                    return False, 0, 0.0

                now    = time.time()
                reward = row[0]
                _completed_metadata.update(json.loads(row[3] or "{}"))

                updated = c.execute(
                    "UPDATE marketplace_jobs SET status=?, result=?, completed_at=? "
                    "WHERE id=? AND status=?",
                    (JobStatus.COMPLETED.value, result[:2000], now,
                     job_id, JobStatus.CLAIMED.value),
                ).rowcount
                if not updated:
                    return False, 0, 0.0

                # Credit atomically in the same transaction
                bal_row = c.execute(
                    "SELECT balance FROM agent_credits WHERE agent_id=?", (agent_id,)
                ).fetchone()
                old_bal = bal_row[0] if bal_row else 0.0
                new_bal = old_bal + reward
                c.execute(
                    "INSERT OR REPLACE INTO agent_credits (agent_id, balance, updated) "
                    "VALUES (?,?,?)",
                    (agent_id, new_bal, now),
                )
                c.execute(
                    "INSERT INTO credit_transactions "
                    "(ts, agent_id, delta, balance, reason, source) VALUES (?,?,?,?,?,?)",
                    (now, agent_id, reward, new_bal, f"job:{job_id}"[:200], "marketplace"),
                )
            return True, reward, new_bal

        ok, reward, new_bal = await asyncio.to_thread(_do)
        if ok:
            LOG.info("[mkt] completed job=%s by=%s reward=%d → balance=%.1f",
                     job_id, agent_id, reward, new_bal)
            # Post validation job if requested
            if _completed_metadata.get("requires_validation"):
                await self._post_validation_job(job_id, result)
        return ok

    async def fail(self, job_id: str, error: str, agent_id: str) -> bool:
        def _do():
            with _conn() as c:
                return c.execute(
                    "UPDATE marketplace_jobs SET status=?, result=?, completed_at=? "
                    "WHERE id=? AND claimed_by=? AND status=?",
                    (JobStatus.FAILED.value, f"ERROR: {error[:1000]}", time.time(),
                     job_id, agent_id, JobStatus.CLAIMED.value),
                ).rowcount
        return bool(await asyncio.to_thread(_do))

    # ── expiry ────────────────────────────────────────────────────────────────

    async def expire_stale(self) -> int:
        """Expire stale PENDING jobs and recycle abandoned CLAIMED jobs.

        CLAIMED jobs whose lease (claimed_at) is older than _CLAIM_TIMEOUT_S
        are reset back to PENDING so another worker can pick them up — this
        handles the case where a worker crashed mid-job.
        """
        now          = time.time()
        pending_cut  = now - _JOB_EXPIRE_S
        claimed_cut  = now - _CLAIM_TIMEOUT_S

        def _do():
            with _conn() as c:
                n_exp = c.execute(
                    "UPDATE marketplace_jobs SET status=? "
                    "WHERE status=? AND created_at<?",
                    (JobStatus.EXPIRED.value, JobStatus.PENDING.value, pending_cut),
                ).rowcount
                n_rec = c.execute(
                    "UPDATE marketplace_jobs "
                    "SET status=?, claimed_by='', claimed_at=0 "
                    "WHERE status=? AND claimed_at>0 AND claimed_at<?",
                    (JobStatus.PENDING.value, JobStatus.CLAIMED.value, claimed_cut),
                ).rowcount
            return n_exp, n_rec

        n_exp, n_rec = await asyncio.to_thread(_do)
        if n_exp:
            LOG.info("[mkt] expired %d stale pending jobs", n_exp)
        if n_rec:
            LOG.info("[mkt] recycled %d abandoned claimed jobs back to pending", n_rec)
        return n_exp + n_rec

    # ── auction ───────────────────────────────────────────────────────────────

    async def post_auction(self, task: str, tags: list[str] = None,
                           reward: int = 20, posted_by: str = "system",
                           metadata: dict = None) -> str:
        """Post a job in BIDDING state. Agents bid quality promises within BID_WINDOW_S seconds.

        After the window, resolve_auction() claims for the highest bidder at second-price reward.
        This incentivises honest capability reporting — an agent can't gain by over-promising.
        """
        job_id = str(uuid.uuid4())[:12]
        deadline = time.time() + BID_WINDOW_S
        meta = dict(metadata or {})
        meta["bid_deadline"] = deadline
        def _insert():
            with _conn() as c:
                c.execute(
                    "INSERT INTO marketplace_jobs "
                    "(id, task, tags, reward, status, posted_by, created_at, metadata) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (job_id, task, json.dumps(tags or []), reward,
                     JobStatus.BIDDING.value, posted_by, time.time(), json.dumps(meta)),
                )
        await asyncio.to_thread(_insert)
        LOG.info("[mkt:auction] posted job=%s reward=%d window=%ds", job_id, reward, BID_WINDOW_S)
        for q in self._listeners:
            try: q.put_nowait(job_id)
            except asyncio.QueueFull: pass
        return job_id

    async def bid(self, job_id: str, agent_id: str, quality_promise: float) -> bool:
        """Submit a sealed bid (quality_promise 0.0–1.0) for a BIDDING job.

        The promise is a truthful capability signal — Vickrey's second-price rule
        means over-promising yields no gain (winner pays second-highest, not first).
        """
        quality_promise = max(0.0, min(1.0, quality_promise))
        bid_id = str(uuid.uuid4())[:12]
        def _do():
            with _conn() as c:
                row = c.execute(
                    "SELECT metadata FROM marketplace_jobs WHERE id=? AND status=?",
                    (job_id, JobStatus.BIDDING.value)
                ).fetchone()
                if not row:
                    return False
                deadline = json.loads(row[0]).get("bid_deadline", 0)
                if time.time() > deadline:
                    return False
                c.execute(
                    "INSERT OR REPLACE INTO marketplace_bids (id, job_id, agent_id, bid, submitted) "
                    "VALUES (?,?,?,?,?)",
                    (bid_id, job_id, agent_id, quality_promise, time.time()),
                )
                return True
        ok = await asyncio.to_thread(_do)
        if ok:
            LOG.debug("[mkt:auction] bid job=%s agent=%s promise=%.2f", job_id, agent_id, quality_promise)
        return ok

    async def resolve_auction(self, job_id: str) -> Optional[str]:
        """Close bidding, claim for the highest bidder at second-price reward.

        Vickrey rule: highest bidder wins; their reward fraction = second-highest bid.
        E.g. bids [0.9, 0.7, 0.5] → winner is 0.9 agent, pays/receives 0.7 × base_reward.
        Returns winning agent_id or None if no bids.
        """
        def _do() -> Optional[tuple[str, str, int]]:
            with _conn(immediate=True) as c:
                row = c.execute(
                    "SELECT reward FROM marketplace_jobs WHERE id=? AND status=?",
                    (job_id, JobStatus.BIDDING.value)
                ).fetchone()
                if not row:
                    return None
                base_reward = row[0]

                bids = c.execute(
                    "SELECT agent_id, bid FROM marketplace_bids WHERE job_id=? ORDER BY bid DESC",
                    (job_id,)
                ).fetchall()
                if not bids:
                    # No bids — expire the job
                    c.execute("UPDATE marketplace_jobs SET status=? WHERE id=?",
                              (JobStatus.EXPIRED.value, job_id))
                    return None

                winner_agent, winner_bid = bids[0]
                second_bid = bids[1][1] if len(bids) > 1 else winner_bid
                # Second-price: winner receives reward proportional to second-highest bid
                final_reward = max(1, int(base_reward * second_bid))

                now = time.time()
                c.execute(
                    "UPDATE marketplace_jobs SET status=?, claimed_by=?, claimed_at=?, reward=? "
                    "WHERE id=?",
                    (JobStatus.CLAIMED.value, winner_agent, now, final_reward, job_id),
                )
                return winner_agent, job_id, final_reward

        result = await asyncio.to_thread(_do)
        if result:
            winner, jid, reward = result
            LOG.info("[mkt:auction] resolved job=%s winner=%s reward=%d (2nd-price)", jid, winner, reward)
            return winner
        return None

    # ── validation ────────────────────────────────────────────────────────────

    async def _post_validation_job(self, original_job_id: str, result: str) -> str:
        """Post a VALIDATOR job to verify a just-completed job's output."""
        task = (
            f"Validate result of job {original_job_id}.\n"
            f"Result to verify (first 800 chars):\n{result[:800]}\n\n"
            "Score the result 0.0–1.0 for correctness, completeness, and safety. "
            "Respond as JSON: {\"passed\": true/false, \"score\": 0.0–1.0, \"feedback\": \"...\"}."
        )
        return await self.post(
            task, tags=["validate", "qa"], reward=10,
            posted_by="marketplace_auto",
            metadata={"validates": original_job_id, "is_validation": True},
        )

    async def validate_result(self, original_job_id: str, validator_id: str,
                              passed: bool, score: float, feedback: str = "") -> bool:
        """Record validator's decision. Updates original job to VALIDATED or REJECTED.

        A rejected job is re-posted as PENDING so another agent can retry it.
        """
        def _do() -> bool:
            with _conn(immediate=True) as c:
                new_status = JobStatus.VALIDATED.value if passed else JobStatus.REJECTED.value
                updated = c.execute(
                    "UPDATE marketplace_jobs "
                    "SET status=?, metadata=json_set(metadata, '$.validation_score', ?, "
                    "    '$.validation_feedback', ?, '$.validated_by', ?) "
                    "WHERE id=? AND status=?",
                    (new_status, round(score, 3), feedback[:400], validator_id,
                     original_job_id, JobStatus.COMPLETED.value),
                ).rowcount
                if updated and not passed:
                    # Rejected — re-queue so another agent can attempt it
                    row = c.execute(
                        "SELECT task, tags, reward, posted_by, metadata FROM marketplace_jobs WHERE id=?",
                        (original_job_id,)
                    ).fetchone()
                    if row:
                        retry_meta = json.loads(row[4] or "{}")
                        retry_meta["retry_of"] = original_job_id
                        c.execute(
                            "INSERT INTO marketplace_jobs "
                            "(id, task, tags, reward, status, posted_by, created_at, metadata) "
                            "VALUES (?,?,?,?,?,?,?,?)",
                            (str(uuid.uuid4())[:12], row[0], row[1], row[2],
                             JobStatus.PENDING.value, row[3], time.time(),
                             json.dumps(retry_meta)),
                        )
                return bool(updated)
        ok = await asyncio.to_thread(_do)
        if ok:
            status_word = "VALIDATED" if passed else "REJECTED+requeued"
            LOG.info("[mkt] job=%s %s by=%s score=%.2f", original_job_id, status_word, validator_id, score)
        return ok

    # ── stats ─────────────────────────────────────────────────────────────────

    def pending_for_tags(self, tags: list[str]) -> int:
        """Count pending jobs that match any of the given capability tags.

        Used by the autoscaler to make per-agent-type scaling decisions
        instead of reacting to the total global pending count.
        """
        if not tags:
            with _conn() as c:
                return c.execute(
                    "SELECT COUNT(*) FROM marketplace_jobs WHERE status=?",
                    (JobStatus.PENDING.value,)
                ).fetchone()[0]
        placeholders = ",".join("?" * len(tags))
        with _conn() as c:
            return c.execute(
                f"SELECT COUNT(*) FROM marketplace_jobs WHERE status=? "
                f"AND EXISTS (SELECT 1 FROM json_each(tags) WHERE value IN ({placeholders}))",
                [JobStatus.PENDING.value] + list(tags),
            ).fetchone()[0]

    def stats(self) -> dict:
        with _conn() as c:
            counts = {row[0]: row[1] for row in c.execute(
                "SELECT status, COUNT(*) FROM marketplace_jobs GROUP BY status"
            ).fetchall()}
            top_earners = c.execute(
                "SELECT claimed_by, COUNT(*), SUM(reward) FROM marketplace_jobs "
                "WHERE status='completed' AND claimed_by!='' "
                "GROUP BY claimed_by ORDER BY SUM(reward) DESC LIMIT 5"
            ).fetchall()
        return {
            "pending":    counts.get("pending", 0),
            "claimed":    counts.get("claimed", 0),
            "completed":  counts.get("completed", 0),
            "failed":     counts.get("failed", 0),
            "top_earners": [
                {"agent": r[0], "jobs": r[1], "credits": r[2]} for r in top_earners
            ],
        }

    def subscribe(self) -> asyncio.Queue:
        """Get a queue that receives job_id notifications when new jobs are posted."""
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._listeners.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        self._listeners.discard(q) if hasattr(self._listeners, 'discard') else None
        try:
            self._listeners.remove(q)
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# Agent Pool (autoscaler)
# ---------------------------------------------------------------------------

AgentWorkerFn = Callable[[Job], Awaitable[str]]


@dataclass
class WorkerConfig:
    agent_type:   str
    capabilities: list[str]
    handler:      AgentWorkerFn
    min_workers:  int = 1
    max_workers:  int = _MAX_WORKERS


class AgentPool:
    """Spawns and manages async agent workers. Auto-scales based on queue depth."""

    def __init__(self, queue: JobQueue = None):
        self._queue   = queue or JobQueue.instance()
        self._configs: dict[str, WorkerConfig] = {}
        self._workers: dict[str, list[asyncio.Task]] = {}
        self._running = False
        self._scale_task: Optional[asyncio.Task] = None

    def register(self, config: WorkerConfig):
        """Register an agent type with its handler function."""
        self._configs[config.agent_type] = config
        self._workers[config.agent_type] = []
        LOG.info("[pool] registered %s (caps=%s, min=%d, max=%d)",
                 config.agent_type, config.capabilities, config.min_workers, config.max_workers)

    async def start(self):
        """Boot minimum workers for each registered agent type."""
        self._running = True
        for cfg in self._configs.values():
            for _ in range(cfg.min_workers):
                await self._spawn(cfg.agent_type)
        if _AUTOSCALE:
            self._scale_task = asyncio.create_task(self._autoscale_loop())
        LOG.info("[pool] started with %d agent types", len(self._configs))

    async def stop(self):
        self._running = False
        if self._scale_task:
            self._scale_task.cancel()
        for tasks in self._workers.values():
            for t in tasks:
                t.cancel()

    async def _spawn(self, agent_type: str) -> asyncio.Task:
        cfg     = self._configs[agent_type]
        wid     = f"{agent_type}-{str(uuid.uuid4())[:6]}"
        task    = asyncio.create_task(self._worker_loop(wid, cfg), name=wid)
        self._workers[agent_type].append(task)
        with _conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO marketplace_workers "
                "(worker_id, agent_type, capabilities, spawned_at, last_active, status) "
                "VALUES (?,?,?,?,?,'idle')",
                (wid, agent_type, json.dumps(cfg.capabilities), time.time(), time.time()),
            )
        LOG.debug("[pool] spawned %s", wid)
        return task

    async def _worker_loop(self, worker_id: str, cfg: WorkerConfig):
        last_active = time.time()
        while self._running:
            job = await self._queue.claim(cfg.agent_type, cfg.capabilities)
            if job is None:
                # No work — check idle TTL
                if time.time() - last_active > _IDLE_TTL:
                    current = self._workers.get(cfg.agent_type, [])
                    if len(current) > cfg.min_workers:
                        LOG.debug("[pool] %s idle TTL reached — retiring", worker_id)
                        break
                await asyncio.sleep(2)
                continue

            last_active = time.time()
            with _conn() as c:
                c.execute("UPDATE marketplace_workers SET last_active=?, status='busy' WHERE worker_id=?",
                          (last_active, worker_id))

            try:
                result = await asyncio.wait_for(cfg.handler(job), timeout=120)
                await self._queue.complete(job.id, result, cfg.agent_type)
                with _conn() as c:
                    c.execute("UPDATE marketplace_workers SET jobs_done=jobs_done+1, status='idle' WHERE worker_id=?",
                              (worker_id,))
            except asyncio.TimeoutError:
                await self._queue.fail(job.id, "timeout after 120s", cfg.agent_type)
            except Exception as e:
                LOG.warning("[pool] %s job=%s error: %s", worker_id, job.id, e)
                await self._queue.fail(job.id, str(e), cfg.agent_type)
            finally:
                with _conn() as c:
                    c.execute("UPDATE marketplace_workers SET status='idle' WHERE worker_id=?",
                              (worker_id,))

        # Clean up on exit
        tasks = self._workers.get(cfg.agent_type, [])
        current = asyncio.current_task()
        if current in tasks:
            tasks.remove(current)
        with _conn() as c:
            c.execute("DELETE FROM marketplace_workers WHERE worker_id=?", (worker_id,))

    async def _autoscale_loop(self):
        """Check queue depth every 5 seconds, scale workers up or down.

        Also re-spawns workers that have crashed to maintain min_workers.
        """
        while self._running:
            await asyncio.sleep(5)
            try:
                for agent_type, cfg in self._configs.items():
                    tasks  = self._workers.get(agent_type, [])
                    active = [t for t in tasks if not t.done()]
                    self._workers[agent_type] = active

                    # Restore workers that have exited below minimum
                    while len(active) < cfg.min_workers:
                        t = await self._spawn(agent_type)
                        active.append(t)
                        LOG.info("[pool] autoscale RESTORE: %s → %d/%d workers",
                                 agent_type, len(active), cfg.min_workers)

                    # Scale up based on pending depth for THIS agent type's tags
                    pending_for_type = await asyncio.to_thread(
                        self._queue.pending_for_tags, cfg.capabilities
                    )
                    if pending_for_type >= _SCALE_UP_AT and len(active) < cfg.max_workers:
                        await self._spawn(agent_type)
                        LOG.info("[pool] autoscale UP: %s → %d workers (queue=%d)",
                                 agent_type, len(active) + 1, pending_for_type)

                await self._queue.expire_stale()
            except Exception as e:
                LOG.debug("autoscale loop error: %s", e)

    def stats(self) -> dict:
        result = {}
        for agent_type, tasks in self._workers.items():
            active = [t for t in tasks if not t.done()]
            result[agent_type] = {
                "workers": len(active),
                "max":     self._configs[agent_type].max_workers,
            }
        return {"pool": result, "queue": self._queue.stats()}


# ---------------------------------------------------------------------------
# Real-world job ingestion helpers (called from gateway_v3.py)
# ---------------------------------------------------------------------------

async def ingest_github_event(event_type: str, payload: dict) -> list[str]:
    """Convert GitHub webhook payload into marketplace jobs.

    pull_request (opened/synchronize) → CODEX code review job
    push (non-main)                   → FORGE security scan job
    issues (opened)                   → ORACLE research job
    """
    q    = JobQueue.instance()
    jobs = []

    if event_type == "pull_request" and payload.get("action") in ("opened", "synchronize"):
        pr   = payload.get("pull_request", {})
        repo = payload.get("repository", {}).get("full_name", "unknown")
        task = (
            f"Code review for PR #{pr.get('number')} in {repo}: "
            f"'{pr.get('title', '')}'\n"
            f"Branch: {pr.get('head', {}).get('ref', '')} → "
            f"{pr.get('base', {}).get('ref', '')}\n"
            f"Files changed: {pr.get('changed_files', '?')} | "
            f"+{pr.get('additions', 0)} -{pr.get('deletions', 0)}\n"
            f"URL: {pr.get('html_url', '')}"
        )
        jid = await q.post(task, tags=["code_review", "github"], reward=40,
                           posted_by="github_webhook",
                           metadata={"pr_number": pr.get("number"), "repo": repo})
        jobs.append(jid)

    elif event_type == "push":
        ref    = payload.get("ref", "")
        branch = ref.split("/")[-1] if ref else "unknown"
        if branch not in ("main", "master"):
            repo    = payload.get("repository", {}).get("full_name", "unknown")
            commits = payload.get("commits", [])
            files   = [f for c in commits for f in c.get("modified", []) + c.get("added", [])]
            task    = (
                f"Security scan for push to {branch} in {repo}\n"
                f"Commits: {len(commits)}  | Changed files: {len(files)}\n"
                f"Files: {', '.join(files[:10])}"
            )
            jid = await q.post(task, tags=["security_scan", "github"], reward=25,
                               posted_by="github_webhook",
                               metadata={"branch": branch, "repo": repo})
            jobs.append(jid)

    elif event_type == "issues" and payload.get("action") == "opened":
        issue = payload.get("issue", {})
        repo  = payload.get("repository", {}).get("full_name", "unknown")
        task  = (
            f"Research issue #{issue.get('number')} in {repo}: "
            f"'{issue.get('title', '')}'\n"
            f"Body: {issue.get('body', '')[:400]}"
        )
        jid = await q.post(task, tags=["research", "github"], reward=20,
                           posted_by="github_webhook",
                           metadata={"issue_number": issue.get("number"), "repo": repo})
        jobs.append(jid)

    LOG.info("[mkt] github %s → %d jobs", event_type, len(jobs))
    return jobs


async def ingest_stripe_event(event_type: str, payload: dict) -> Optional[str]:
    """Convert Stripe webhook events into LEDGER jobs."""
    q = JobQueue.instance()

    customer = (payload.get("data", {}).get("object", {}).get("customer") or
                payload.get("data", {}).get("object", {}).get("customer_email") or
                "unknown")

    task_map = {
        "customer.subscription.created":  (f"New subscriber {customer} — send welcome + activate access", 30),
        "customer.subscription.deleted":  (f"Subscriber {customer} cancelled — revoke access + retention email", 25),
        "invoice.payment_succeeded":       (f"Payment received from {customer} — update subscription status", 15),
        "invoice.payment_failed":          (f"Payment FAILED for {customer} — notify team + retry logic", 40),
        "checkout.session.completed":      (f"Checkout complete for {customer} — provision account", 35),
    }

    if event_type in task_map:
        task, reward = task_map[event_type]
        jid = await q.post(task, tags=["billing", "stripe", "ledger"], reward=reward,
                           posted_by="stripe_webhook",
                           metadata={"event_type": event_type, "customer": customer})
        LOG.info("[mkt] stripe %s → job=%s", event_type, jid)
        return jid
    return None


def _cve_already_queued(cve_id: str) -> bool:
    """Return True if an active (PENDING/CLAIMED) job for this CVE already exists."""
    with _conn() as c:
        row = c.execute(
            "SELECT id FROM marketplace_jobs "
            "WHERE json_extract(metadata,'$.cve_id')=? "
            "AND status IN (?,?)",
            (cve_id, JobStatus.PENDING.value, JobStatus.CLAIMED.value),
        ).fetchone()
    return row is not None


async def ingest_cve_feed(cve_records: list[dict]) -> list[str]:
    """Post CVE records as SENTINEL security scan jobs.

    Each record: {"id": "CVE-2024-...", "summary": "...", "severity": "CRITICAL"}
    Skips CVEs that already have an active (PENDING/CLAIMED) job to prevent
    the hourly feed from creating duplicate jobs for the same vulnerability.
    """
    q    = JobQueue.instance()
    jobs = []
    for cve in cve_records[:20]:   # cap at 20 per batch
        cve_id   = cve.get("id", "CVE-UNKNOWN")
        if await asyncio.to_thread(_cve_already_queued, cve_id):
            LOG.debug("[mkt] CVE %s already queued — skipping", cve_id)
            continue
        severity = (cve.get("severity") or "UNKNOWN").upper()
        reward   = {"CRITICAL": 80, "HIGH": 50, "MEDIUM": 30, "LOW": 15}.get(severity, 20)
        task     = (
            f"Analyze {cve_id} ({severity}): "
            f"{cve.get('summary', '')[:300]}\n"
            f"Determine: (1) whether GH05T3 stack is affected, "
            f"(2) mitigation steps, (3) patch priority."
        )
        jid = await q.post(task, tags=["security_scan", "cve", severity.lower()],
                           reward=reward, posted_by="cve_feed",
                           metadata={"cve_id": cve_id, "severity": severity})
        jobs.append(jid)
    LOG.info("[mkt] ingested %d new CVE jobs (%d dupes skipped)",
             len(jobs), len(cve_records[:20]) - len(jobs))
    return jobs
