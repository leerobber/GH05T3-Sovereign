"""ContentWorkflow — multi-step plan → critique → emit pipeline."""
from __future__ import annotations

from src.semantics.semantic_word import SemanticWord, IntentType, WordType, ChannelType
from backend.core.moe_router import MOERouter


class ContentWorkflow:
    """Runs a plan → critique → emit pipeline via the MOE router."""

    def __init__(self, router: MOERouter) -> None:
        self._router = router

    def run(self, input_word_int: int) -> list[int]:
        """
        1. Route input word through PLAN expert.
        2. Pass plan output to CRITIQUE expert.
        3. Return all emitted words from the EMIT step.
        """
        # Step 1: plan
        plan_words = self._router.route(input_word_int)

        # Step 2: critique each plan word
        critique_words: list[int] = []
        for pw in plan_words:
            critique_word = SemanticWord.make(
                type=WordType.CONTROL,
                intent=IntentType.CRITIQUE,
                channel=ChannelType.INTERNAL,
                priority=180,
                confidence=1.0,
            ).encode()
            # deliver plan result to critic's inbox via adapter
            self._router._adapter.send(0, self._router._expert_ids["critic"], pw)
            critique_words.extend(self._router.route(critique_word))

        # Step 3: emit
        emit_words: list[int] = []
        for cw in critique_words:
            emit_word = SemanticWord.make(
                type=WordType.CONTROL,
                intent=IntentType.EMIT,
                channel=ChannelType.INTERNAL,
                priority=200,
                confidence=1.0,
            ).encode()
            self._router._adapter.send(0, self._router._expert_ids["builder"], cw)
            emit_words.extend(self._router.route(emit_word))

        return emit_words
