from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
from ulid import ulid

class EventType(str, Enum):
    """Canonical event types for Omni Sentinel."""
    AGENT_SPAWN = "agent_spawn"
    AGENT_MUTATION = "agent_mutation"
    MEMORY_ACCESS = "memory_access"
    SWARM_COMMAND = "swarm_command"
    EXTERNAL_API_CALL = "external_api_call"
    EVOLUTION_CYCLE = "evolution_cycle"
    THREAT_DETECTED = "threat_detected"
    LINEAGE_VIOLATION = "lineage_violation"

class RiskFactor(BaseModel):
    """Risk assessment for an event."""
    factor: str
    severity: float = Field(..., ge=0, le=1)
    description: str

class SentinelEvent(BaseModel):
    """Canonical event format for Omni Sentinel."""
    event_id: str = Field(default_factory=lambda: str(ulid.new()), description="Unique ULID for the event")
    event_type: EventType = Field(..., description="Type of event")
    source: str = Field(..., description="Source agent/process/module")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Event data")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="UTC timestamp")
    dna_signature: str = Field(..., description="OmniDNA signature of the event")
    lineage: List[str] = Field(default_factory=list, description="Agent lineage chain (agent IDs)")
    risk_factors: List[RiskFactor] = Field(default_factory=list, description="Risk assessment")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context")

    def __str__(self) -> str:
        return f"SentinelEvent(id={self.event_id}, type={self.event_type}, source={self.source})"
