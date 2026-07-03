from .ingestion_router import IngestionRouter
from .threat_classifier import ThreatClassifier
from .action_router import ActionRouter
from .sentinel_event import SentinelEvent
import logging
import asyncio

logger = logging.getLogger(__name__)

class SentinelCore:
    """Orchestrates the Sentinel pipeline."""

    @staticmethod
    async def process_event(raw_event: dict) -> bool:
        """Process a raw event through the Sentinel pipeline."""
        try:
            # 1. Ingest
            event = await IngestionRouter.ingest_raw_event(raw_event)

            # 2. Classify
            verdict = ThreatClassifier.classify_event(event)

            # 3. Route
            success = await ActionRouter.route(verdict)

            logger.info(f"Processed event {event.event_id}: {verdict}")
            return success

        except Exception as e:
            logger.error(f"Failed to process event: {e}")
            return False

    @staticmethod
    async def process_events_batch(raw_events: list[dict]) -> list[bool]:
        """Process a batch of events concurrently."""
        tasks = [SentinelCore.process_event(event) for event in raw_events]
        return await asyncio.gather(*tasks)