"""
FREE FALLBACK LLM ROUTING
========================
When Anthropic API is unavailable or credits exhausted,
automatically route to free local models in degradation order:
  1. vLLM (RTX 5050) — fastest, most powerful
  2. llama.cpp verifier (Radeon 780M) — mid-tier
  3. llama.cpp CPU (any hardware) — always available

Transparent to callers — same interface as ClaudeClient.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple
import httpx

log = logging.getLogger("gh0st3.fallback")

# Port map for local inference
VLLM_URL = "http://localhost:8010/v1/chat/completions"
LLAMA_VERIFIER_URL = "http://localhost:8011/v1/chat/completions"
LLAMA_CPU_URL = "http://localhost:8012/v1/chat/completions"

# Model names for each endpoint
VLLM_MODEL = "meta-llama/Llama-3.3-70B-Instruct"  # or whatever is loaded
LLAMA_MODEL = "llama2"  # Generic for llama.cpp


@dataclass
class FallbackUsage:
    """Compatible with ClaudeUsage for unified logging."""
    timestamp: float = None
    role: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    task: str = ""
    fallback_tier: str = ""  # "vllm", "llama_verifier", "llama_cpu"

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class FallbackLLMClient:
    """
    Unified free LLM client. Tries endpoints in degradation order.
    Detects credit exhaustion (429, 401) from Anthropic and auto-routes to local.
    """

    def __init__(self):
        self._client = httpx.AsyncClient(timeout=120.0)
        self._bus = None
        self._tier_health = {
            "vllm": True,
            "llama_verifier": True,
            "llama_cpu": True,
        }

    def set_bus(self, bus):
        """Wire in SwarmBus for logging. Sync — pure assignment, no event loop needed."""
        self._bus = bus

    async def call(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
        role_label: str = "fallback",
        task_label: str = "",
    ) -> Tuple[str, FallbackUsage]:
        """
        Try local LLM endpoints in degradation order.
        Returns (content, usage).
        """
        endpoints = [
            ("vllm", VLLM_URL, VLLM_MODEL),
            ("llama_verifier", LLAMA_VERIFIER_URL, LLAMA_MODEL),
            ("llama_cpu", LLAMA_CPU_URL, LLAMA_MODEL),
        ]

        for tier_name, url, model_name in endpoints:
            if not self._tier_health.get(tier_name, True):
                log.debug(f"[Fallback] Skipping {tier_name} (marked unhealthy)")
                continue

            try:
                content, usage = await self._call_endpoint(
                    url=url,
                    model=model_name,
                    system=system,
                    user=user,
                    max_tokens=max_tokens,
                    tier=tier_name,
                    role=role_label,
                    task=task_label,
                )
                usage.fallback_tier = tier_name
                self._tier_health[tier_name] = True

                if self._bus:
                    try:
                        from swarm.bus import MsgType
                        await self._bus.emit(
                            src=f"FALLBACK-{tier_name.upper()}",
                            content=f"[{usage.total_tokens} tokens, {usage.latency_ms:.0f}ms] {content[:150]}",
                            channel="#fallback",
                            msg_type=MsgType.TASK,
                            fallback_tier=tier_name,
                            usage={"in": usage.input_tokens, "out": usage.output_tokens},
                        )
                    except Exception as e:
                        log.debug(f"[Fallback] Bus emit failed: {e}")

                return content, usage

            except Exception as e:
                log.warning(f"[Fallback] {tier_name} failed: {e}")
                self._tier_health[tier_name] = False
                continue

        return (
            "[All fallback endpoints exhausted — no local models available]",
            FallbackUsage(role=role_label, task=task_label, fallback_tier="none"),
        )

    async def _call_endpoint(
        self,
        url: str,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
        tier: str,
        role: str,
        task: str,
    ) -> Tuple[str, FallbackUsage]:
        """Call a single local LLM endpoint (vLLM or llama.cpp)."""
        t0 = time.perf_counter()

        # Prepare messages in OpenAI-compatible format
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }

        try:
            resp = await self._client.post(url, json=payload)
            resp.raise_for_status()

            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            usage = data.get("usage", {})

            u = FallbackUsage(
                role=role,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                latency_ms=(time.perf_counter() - t0) * 1000,
                task=task,
            )

            log.info(f"[Fallback] {tier}: {u.total_tokens} tokens, {u.latency_ms:.0f}ms")
            return content, u

        except asyncio.TimeoutError:
            raise RuntimeError(f"{tier} endpoint timeout (120s)")
        except httpx.ConnectError:
            raise RuntimeError(f"{tier} endpoint unreachable")
        except Exception as e:
            raise RuntimeError(f"{tier} request failed: {e}")

    async def close(self):
        await self._client.aclose()


async def detect_credit_exhaustion(error: Exception) -> bool:
    """
    Check if an exception is due to API credit exhaustion.
    Returns True if we should fallback to free models.
    """
    error_str = str(error).lower()

    # Anthropic-specific error codes
    if any(code in error_str for code in ["429", "401", "429", "quota", "rate limit", "overloaded"]):
        return True

    # Generic credit/payment errors
    if any(phrase in error_str for phrase in [
        "credit",
        "payment",
        "subscription",
        "billing",
        "insufficient",
        "exhausted",
    ]):
        return True

    return False
