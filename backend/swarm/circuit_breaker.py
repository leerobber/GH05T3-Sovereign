"""Per-agent circuit breaker for fault isolation in the SwarmBus.

State machine: CLOSED → OPEN → HALF_OPEN → CLOSED
  CLOSED    — normal operation; failures are counted
  OPEN      — agent is suspended; calls fail fast for `reset_timeout` seconds
  HALF_OPEN — one probe request allowed; success → CLOSED, failure → OPEN

Usage:
    cb = CircuitBreaker("ORACLE", failure_threshold=3, reset_timeout=30)

    async with cb:
        result = await oracle.on_message(msg)

    # or without context manager:
    if cb.allow():
        try:
            result = await oracle.on_message(msg)
            cb.success()
        except Exception as e:
            cb.failure(e)
            raise
"""
from __future__ import annotations

import logging
import time
from typing import Callable

LOG = logging.getLogger("ghost.swarm.cb")


class CircuitBreakerOpen(RuntimeError):
    """Raised when a call is attempted while the circuit is OPEN."""


class CircuitBreaker:
    """Thread-safe (asyncio-safe) circuit breaker for a single agent."""

    _CLOSED    = "closed"
    _OPEN      = "open"
    _HALF_OPEN = "half_open"

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        reset_timeout: float   = 30.0,
        success_threshold: int = 1,
        on_state_change: Callable[[str, str, str], None] | None = None,
    ):
        self.name              = name
        self.failure_threshold = failure_threshold
        self.reset_timeout     = reset_timeout
        self.success_threshold = success_threshold
        self._on_state_change  = on_state_change

        self._state            = self._CLOSED
        self._failures         = 0
        self._successes        = 0
        self._opened_at        = 0.0
        self._last_failure     = ""
        self._probe_in_flight  = False  # enforces single probe in HALF_OPEN
        self._probe_started_at = 0.0   # wall-clock when probe was issued

    # ── state checks ──────────────────────────────────────────────────────────

    @property
    def state(self) -> str:
        if self._state == self._OPEN:
            if time.monotonic() >= self._opened_at + self.reset_timeout:
                self._transition(self._HALF_OPEN)
        return self._state

    def allow(self) -> bool:
        """Return True if a call should proceed.

        In HALF_OPEN exactly one probe is allowed at a time — the flag is
        cleared by success(), failure(), or _transition() so the next
        probe can proceed after the current one resolves.  If success() /
        failure() are never called (caller bypassed the context manager and
        crashed), the probe token auto-releases after reset_timeout seconds.
        """
        s = self.state
        if s == self._CLOSED:
            return True
        if s == self._HALF_OPEN:
            if self._probe_in_flight:
                # Stale probe: auto-release after reset_timeout so a new probe
                # can be issued even if the previous caller never reported back.
                if time.monotonic() - self._probe_started_at > self.reset_timeout:
                    self._probe_in_flight = False
                else:
                    return False  # another probe already in flight
            self._probe_in_flight  = True
            self._probe_started_at = time.monotonic()
            return True
        return False      # OPEN — fast-fail

    # ── feedback ──────────────────────────────────────────────────────────────

    def success(self) -> None:
        self._probe_in_flight = False
        s = self.state
        if s == self._HALF_OPEN:
            self._successes += 1
            if self._successes >= self.success_threshold:
                self._failures  = 0
                self._successes = 0
                self._transition(self._CLOSED)
        elif s == self._CLOSED:
            self._failures = max(0, self._failures - 1)

    def failure(self, exc: Exception | None = None) -> None:
        self._probe_in_flight = False
        self._last_failure = str(exc) if exc else "unknown"
        s = self.state
        if s in (self._CLOSED, self._HALF_OPEN):
            self._failures  += 1
            self._successes  = 0
            if self._failures >= self.failure_threshold:
                self._opened_at = time.monotonic()
                self._transition(self._OPEN)

    # ── context manager ───────────────────────────────────────────────────────

    async def __aenter__(self):
        if not self.allow():
            remaining = (self._opened_at + self.reset_timeout) - time.monotonic()
            raise CircuitBreakerOpen(
                f"Agent {self.name!r} circuit OPEN "
                f"(last error: {self._last_failure!r}, "
                f"reopens in {remaining:.0f}s)"
            )
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.success()
        else:
            self.failure(exc)
        return False   # don't suppress exceptions

    # ── helpers ───────────────────────────────────────────────────────────────

    def _transition(self, new_state: str) -> None:
        if new_state == self._state:
            return
        self._probe_in_flight = False  # reset probe token on every state change
        old = self._state
        self._state = new_state
        LOG.info("[cb] %s: %s → %s", self.name, old, new_state)
        if self._on_state_change:
            try:
                self._on_state_change(self.name, old, new_state)
            except Exception as e:
                LOG.warning("[cb] %s state-change hook failed: %s", self.name, e)

    def stats(self) -> dict:
        return {
            "agent":             self.name,
            "state":             self.state,
            "failures":          self._failures,
            "failure_threshold": self.failure_threshold,
            "last_failure":      self._last_failure,
            "reset_timeout":     self.reset_timeout,
        }


# ---------------------------------------------------------------------------
# Registry — one breaker per agent, shared across the process
# ---------------------------------------------------------------------------
_registry: dict[str, CircuitBreaker] = {}


def get_breaker(name: str, **kwargs) -> CircuitBreaker:
    """Return (or create) the circuit breaker for a given agent name."""
    if name not in _registry:
        _registry[name] = CircuitBreaker(name, **kwargs)
    return _registry[name]


def all_stats() -> list[dict]:
    return [cb.stats() for cb in _registry.values()]
