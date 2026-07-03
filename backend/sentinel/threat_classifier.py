from typing import Dict, Any, Optional
from .sentinel_event import SentinelEvent, EventType, RiskFactor
from .dna_verifier import verify_dna
import logging

logger = logging.getLogger(__name__)

class SentinelVerdict:
    """The result of a Sentinel classification."""
    def __init__(
        self,
        event: SentinelEvent,
        threat_score: float,
        is_threat: bool,
        anomalies: Dict[str, Any],
        action: str,
        theorist_elite_notes: Optional[str] = None
    ):
        self.event = event
        self.threat_score = threat_score
        self.is_threat = is_threat
        self.anomalies = anomalies
        self.action = action
        self.theorist_elite_notes = theorist_elite_notes

    def __str__(self) -> str:
        return f"Verdict(threat={self.is_threat}, score={self.threat_score}, action={self.action})"

class ThreatClassifier:
    """Core Sentinel logic for threat detection."""

    @staticmethod
    def classify_event(event: SentinelEvent) -> SentinelVerdict:
        """Classify an event and produce a verdict."""
        threat_score = 0.0
        anomalies = {}
        action = "log"
        theorist_notes = None

        # 1. Check DNA signature
        dna_ok, dna_anomalies = verify_dna(event)
        if not dna_ok:
            threat_score += 0.7
            anomalies["dna_violation"] = dna_anomalies

        # 2. Check lineage
        lineage_ok, lineage_anomalies = ThreatClassifier._check_lineage(event)
        if not lineage_ok:
            threat_score += 0.5
            anomalies["lineage_violation"] = lineage_anomalies

        # 3. Check event-specific risks
        event_risk = ThreatClassifier._check_event_type(event)
        threat_score += event_risk
        if event_risk > 0.3:
            anomalies["event_risk"] = f"High risk for {event.event_type}"

        # 4. Apply THEORIST_ELITE evaluation (placeholder)
        if threat_score > 0.8:
            theorist_notes = ThreatClassifier._theorist_elite_eval(event)
            action = "escalate"
        elif threat_score > 0.5:
            action = "block"
        elif threat_score > 0.2:
            action = "notify"

        return SentinelVerdict(
            event=event,
            threat_score=threat_score,
            is_threat=threat_score > 0.5,
            anomalies=anomalies,
            action=action,
            theorist_elite_notes=theorist_notes
        )

    @staticmethod
    def _check_lineage(event: SentinelEvent) -> tuple[bool, Dict[str, Any]]:
        """Check if the event's lineage is valid."""
        # Already verified in DNA verifier
        return True, {}

    @staticmethod
    def _check_event_type(event: SentinelEvent) -> float:
        """Score risk based on event type."""
        risk_map = {
            EventType.AGENT_SPAWN: 0.2,
            EventType.AGENT_MUTATION: 0.4,
            EventType.MEMORY_ACCESS: 0.3,
            EventType.SWARM_COMMAND: 0.5,
            EventType.EXTERNAL_API_CALL: 0.6,
            EventType.EVOLUTION_CYCLE: 0.1,
            EventType.THREAT_DETECTED: 0.9,
            EventType.LINEAGE_VIOLATION: 0.8
        }
        return risk_map.get(event.event_type, 0.0)

    @staticmethod
    def _theorist_elite_eval(event: SentinelEvent) -> str:
        """Placeholder for THEORIST_ELITE evaluation."""
        return f"THEORIST_ELITE: Event {event.event_id} requires review due to high threat score."