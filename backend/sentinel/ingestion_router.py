from typing import Dict, Any
from .sentinel_event import SentinelEvent, EventType
from .dna_verifier import extract_dna_signature
import logging
from ulid import ulid

logger = logging.getLogger(__name__)

class IngestionRouter:
    """Routes raw events into the Sentinel pipeline."""

    @staticmethod
    async def ingest_raw_event(raw_event: Dict[str, Any]) -> SentinelEvent:
        """Convert a raw event into a validated SentinelEvent."""
        try:
            # Generate ULID if not provided
            event_id = raw_event.get("event_id", str(ulid.new()))

            # Extract DNA signature
            dna_signature = extract_dna_signature(raw_event)

            # Build lineage chain
            lineage = raw_event.get("lineage", [])
            if not lineage and "source" in raw_event:
                lineage = [raw_event["source"]]

            # Normalize event
            event = SentinelEvent(
                event_id=event_id,
                event_type=EventType(raw_event.get("event_type", "unknown")),
                source=raw_event["source"],
                payload=raw_event.get("payload", {}),
                dna_signature=dna_signature,
                lineage=lineage,
                risk_factors=raw_event.get("risk_factors", []),
                metadata=raw_event.get("metadata", {})
            )

            logger.info(f"Ingested event: {event}")
            return event

        except Exception as e:
            logger.error(f"Failed to ingest event: {e}")
            raise ValueError(f"Invalid event: {e}")
