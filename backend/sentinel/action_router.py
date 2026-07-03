import asyncio
from .threat_classifier import SentinelVerdict
from .sentinel_event import SentinelEvent
from .theorist_elite_client import TheoristEliteClient
import logging

logger = logging.getLogger(__name__)

class ActionRouter:
    """Routes events to the appropriate action based on verdict."""

    @staticmethod
    async def route(verdict: SentinelVerdict) -> bool:
        """Execute the action for a verdict."""
        event = verdict.event
        action = verdict.action

        logger.info(f"Routing event {event.event_id} with action: {action}")

        if action == "block":
            return ActionRouter._block_event(event)
        elif action == "notify":
            return ActionRouter._notify_omnimind(event)
        elif action == "escalate":
            return ActionRouter._escalate_to_theorist(event, verdict.theorist_elite_notes)
        elif action == "log":
            return ActionRouter._log_event(event)
        else:
            logger.warning(f"Unknown action: {action}")
            return False

    @staticmethod
    def _block_event(event: SentinelEvent) -> bool:
        """Block the event (e.g., kill agent, revert mutation)."""
        logger.warning(f"BLOCKED event {event.event_id} (type: {event.event_type})")
        # TODO: Integrate with Aethyro to kill rogue agents
        return True

    @staticmethod
    def _notify_omnimind(event: SentinelEvent) -> bool:
        """Notify OmniMind of a suspicious event."""
        logger.info(f"Notified OmniMind about event {event.event_id}")
        # TODO: Integrate with OmniMind API
        return True

    @staticmethod
    def _escalate_to_theorist(event: SentinelEvent, notes: str) -> bool:
        """Escalate to THEORIST_ELITE for review."""
        return asyncio.run(TheoristEliteClient.escalate_event(event.event_id, notes))

    @staticmethod
    def _log_event(event: SentinelEvent) -> bool:
        """Log the event for auditing."""
        logger.info(f"Logged event {event.event_id}")
        # TODO: Integrate with long-term memory
        return True