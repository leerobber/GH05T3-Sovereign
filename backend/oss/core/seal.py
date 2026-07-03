"""
LexGenSeal — SHA256-signed IP vault for Aethyro breakthrough events.

Called by PatentOffice when a valid disclosure is generated. Produces a
tamper-evident JSON artifact that proves:
  - WHAT was discovered (data_snapshot)
  - WHEN it was discovered (timestamp from args, not runtime — for resumability)
  - WHO discovered it (agent lineage: parent_offset → child)
  - THAT the record hasn't been altered (SHA256 signature)

Output directory: data/ip_vault/
Each file: IP-{agent_id}-{seal_id}.json

The vault is append-only — seals are never modified after creation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

LOG = logging.getLogger("ghost.seal")

_VAULT_DIR = Path("data/ip_vault")


def _make_seal_id() -> str:
    return f"SEAL_{uuid.uuid4().hex[:12].upper()}"


class LexGenSeal:
    """
    Write-once IP vault with SHA256 content signatures.

    seal_breakthrough() is idempotent on the same (agent_id, disclosure_id):
    if a seal file already exists for that pair it returns the existing record
    without overwriting (append-only invariant preserved).
    """

    def __init__(self, vault_dir: Optional[Path] = None):
        self._vault = Path(vault_dir) if vault_dir else _VAULT_DIR
        self._vault.mkdir(parents=True, exist_ok=True)

    def seal_breakthrough(
        self,
        agent_id:      str,
        parent_id:     str,
        data_snapshot: Dict[str, Any],
        timestamp:     str,                # pass from caller — no runtime Date.now()
        disclosure_id: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Create a signed vault record.

        Returns:
            (seal_id, sha256_hex_signature)
        """
        seal_id = disclosure_id or _make_seal_id()

        record = {
            "seal_id":      seal_id,
            "agent_id":     agent_id,
            "parent_id":    parent_id,
            "timestamp":    timestamp,
            "lineage":      {"parent": parent_id, "child": agent_id},
            "data_snapshot": data_snapshot,
        }

        # Build signature BEFORE adding it to the record
        canonical      = json.dumps(record, sort_keys=True, separators=(",", ":"))
        signature      = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        record["sig"]  = signature

        # Write to vault (append-only — skip if file already exists)
        outpath = self._vault / f"IP-{agent_id}-{seal_id}.json"
        if not outpath.exists():
            try:
                with open(outpath, "w", encoding="utf-8") as fh:
                    json.dump(record, fh, indent=2)
                LOG.info("LexGenSeal: sealed %s → %s", seal_id, outpath.name)
            except OSError as exc:
                LOG.warning("LexGenSeal: could not write vault file: %s", exc)

        return seal_id, signature

    def verify(self, seal_id: str, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Load and verify a vault record. Returns the record if signature is valid,
        None if the file doesn't exist or the signature doesn't match.
        """
        path = self._vault / f"IP-{agent_id}-{seal_id}.json"
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                record = json.load(fh)
            stored_sig = record.pop("sig", None)
            canonical  = json.dumps(record, sort_keys=True, separators=(",", ":"))
            expected   = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            if stored_sig != expected:
                LOG.warning("LexGenSeal: signature mismatch for %s", seal_id)
                return None
            record["sig"] = stored_sig   # restore
            return record
        except Exception as exc:
            LOG.debug("LexGenSeal verify error: %s", exc)
            return None

    def list_seals(self, limit: int = 50) -> list:
        """Return filenames of the most recent vault entries."""
        try:
            files = sorted(
                self._vault.glob("IP-*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            return [f.name for f in files[:limit]]
        except Exception:
            return []


# ── Singleton ─────────────────────────────────────────────────────────────────

_seal: Optional[LexGenSeal] = None


def get_lex_gen_seal() -> LexGenSeal:
    global _seal
    if _seal is None:
        _seal = LexGenSeal()
    return _seal
