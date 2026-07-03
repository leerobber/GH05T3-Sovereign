from typing import Dict, Any, Tuple, List
import mmap
import struct
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Aethyro binary ledger constants
LEDGER_PATH = Path("C:/Users/leer4/GH05T3/aethyro_swarm.bin")
SLOT_SIZE = 32  # 32-byte slots
AGENT_ID_OFFSET = 0  # First 8 bytes = agent ID
DNA_OFFSET = 8  # Next 8 bytes = DNA signature
PARENT_OFFSET = 16  # Next 8 bytes = parent agent ID
FITNESS_OFFSET = 24  # Last 8 bytes = fitness vector

def read_ledger_slot(slot_index: int) -> Dict[str, Any]:
    """Read a single slot from the Aethyro binary ledger."""
    if not LEDGER_PATH.exists():
        raise FileNotFoundError(f"Aethyro ledger not found at {LEDGER_PATH}")

    with open(LEDGER_PATH, "r+b") as f:
        with mmap.mmap(f.fileno(), 0) as mm:
            if slot_index * SLOT_SIZE >= mm.size():
                raise IndexError(f"Slot {slot_index} out of bounds")

            mm.seek(slot_index * SLOT_SIZE)
            slot_data = mm.read(SLOT_SIZE)

            # Unpack 32-byte slot
            agent_id = struct.unpack("<Q", slot_data[AGENT_ID_OFFSET:AGENT_ID_OFFSET+8])[0]
            dna = struct.unpack("<Q", slot_data[DNA_OFFSET:DNA_OFFSET+8])[0]
            parent_id = struct.unpack("<Q", slot_data[PARENT_OFFSET:PARENT_OFFSET+8])[0]
            fitness = struct.unpack("<d", slot_data[FITNESS_OFFSET:FITNESS_OFFSET+8])[0]

            return {
                "agent_id": f"agent_{agent_id}",
                "dna": f"dna_{dna}",
                "parent_id": f"agent_{parent_id}" if parent_id else None,
                "fitness": fitness
            }

def extract_dna_signature(raw_event: Dict[str, Any]) -> str:
    """Extract or generate a DNA signature for an event."""
    # Use source + payload hash for DNA
    source = raw_event.get("source", "unknown")
    payload_hash = hash(frozenset(raw_event.get("payload", {}).items()))
    return f"dna_{hash((source, payload_hash))}"

def verify_dna(event: "SentinelEvent") -> Tuple[bool, Dict[str, Any]]:
    """Verify the DNA signature and lineage against Aethyro ledger."""
    anomalies = {}

    # 1. Check if source agent exists in ledger
    source_agent = None
    for slot_idx in range(1024):  # Scan first 1024 slots (adjust as needed)
        try:
            slot = read_ledger_slot(slot_idx)
            if slot["agent_id"] == event.source:
                source_agent = slot
                break
        except IndexError:
            break

    if not source_agent:
        anomalies["source_agent_missing"] = f"Source agent {event.source} not found in ledger"
        return False, anomalies

    # 2. Verify DNA signature
    if source_agent["dna"] != event.dna_signature:
        anomalies["dna_mismatch"] = f"DNA mismatch: expected {source_agent['dna']}, got {event.dna_signature}"
        return False, anomalies

    # 3. Verify lineage chain
    current_agent = source_agent
    for ancestor_id in event.lineage:
        if not current_agent["parent_id"]:
            anomalies["lineage_break"] = f"Lineage break at {current_agent['agent_id']}"
            return False, anomalies
        if current_agent["parent_id"] != ancestor_id:
            anomalies["lineage_mismatch"] = f"Lineage mismatch: expected {current_agent['parent_id']}, got {ancestor_id}"
            return False, anomalies

        # Find parent in ledger
        parent_found = False
        for slot_idx in range(1024):
            try:
                slot = read_ledger_slot(slot_idx)
                if slot["agent_id"] == current_agent["parent_id"]:
                    current_agent = slot
                    parent_found = True
                    break
            except IndexError:
                break

        if not parent_found:
            anomalies["parent_missing"] = f"Parent agent {current_agent['parent_id']} not found in ledger"
            return False, anomalies

    return True, anomalies