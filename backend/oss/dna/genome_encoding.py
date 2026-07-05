"""Real (de)serialization for a Genome's full identity -- id, traits, and
metadata -- as opposed to the traits-only encode() paired with a stubbed
decode() (raise NotImplementedError) from the original rebuild spec. A
one-way encoder that can't decode back into a genome isn't a real codec,
it's a write-only log; this makes both directions actually work so
GenomePlane.encode/decode is a true round trip, not a stub.

JSON is deliberately the wire format here (not a binary pack) -- genomes
are small (a handful of scalar/string traits), and human-readable genome
snapshots are worth far more for debugging an evolution run than the few
bytes JSON costs over a packed binary format.
"""
from __future__ import annotations

import json
from typing import Any


def encode_genome(genome_id: str, traits: dict[str, Any], metadata: dict[str, Any]) -> bytes:
    """Encodes a genome's full identity into bytes. Raises TypeError
    immediately (via json.dumps) if traits/metadata contain anything
    non-JSON-serializable, rather than silently corrupting data that
    would only surface as a mysterious decode failure later."""
    payload = {"id": genome_id, "traits": traits, "metadata": metadata}
    return json.dumps(payload, sort_keys=True).encode("utf-8")


def decode_genome(data: bytes) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Inverse of encode_genome. Returns (genome_id, traits, metadata).
    Raises ValueError (not a bare KeyError/JSONDecodeError) on malformed
    input, naming what was actually wrong."""
    try:
        payload = json.loads(data.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"genome data is not valid encoded JSON: {e}") from e

    if not isinstance(payload, dict) or "id" not in payload or "traits" not in payload:
        raise ValueError(f"decoded genome payload missing required fields: {payload!r}")

    return payload["id"], payload["traits"], payload.get("metadata", {})
