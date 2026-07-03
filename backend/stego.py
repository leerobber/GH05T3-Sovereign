"""Real lexicographic steganography.
Encodes bits into a cover text by choosing between synonym pairs at selected
positions. Decodes by reading back which synonym was used.
Capacity: ~12 bytes per ~1000 word cover (matches GhostVeil spec).
"""
from __future__ import annotations
import re
from typing import Tuple

# Synonym pairs — bit 0 uses left, bit 1 uses right. Case-insensitive.
# Pairs chosen to be low-salience substitutions.
SYNONYM_PAIRS = [
    ("fast", "quick"),
    ("big", "large"),
    ("small", "tiny"),
    ("begin", "start"),
    ("end", "finish"),
    ("use", "employ"),
    ("help", "assist"),
    ("show", "display"),
    ("make", "create"),
    ("find", "locate"),
    ("check", "verify"),
    ("build", "construct"),
    ("send", "dispatch"),
    ("get", "obtain"),
    ("keep", "retain"),
    ("need", "require"),
    ("want", "desire"),
    ("give", "provide"),
    ("buy", "purchase"),
    ("fix", "repair"),
    ("think", "believe"),
    ("happy", "glad"),
    ("smart", "clever"),
    ("brave", "bold"),
    ("calm", "serene"),
    ("right", "correct"),
    ("wrong", "incorrect"),
    ("easy", "simple"),
    ("hard", "difficult"),
    ("clear", "obvious"),
    ("true", "factual"),
    ("also", "additionally"),
    ("soon", "shortly"),
    ("often", "frequently"),
    ("never", "rarely"),
]

# Build lookup: word -> (pair_idx, bit)
LOOKUP = {}
for idx, (a, b) in enumerate(SYNONYM_PAIRS):
    LOOKUP[a.lower()] = (idx, 0)
    LOOKUP[b.lower()] = (idx, 1)


DEFAULT_COVER = """Ghost Protocol status update. Today our mission is to build a quick and
reliable pipeline, help the team begin the next sprint, find the correct
defaults, and verify the outputs so nothing slips through. If we need to
fix something mid-flight we will send a note, keep the archive clean, and
make sure the small regressions never reach production. It is often easy
to overlook simple assumptions, so additionally we will display the right
metrics clearly and also think carefully before we dispatch anything.
Robert, stay calm — we will start smart, employ the right tools, provide
clear logs, obtain useful feedback, and finish with a bold but correct
summary. You can buy back time by asking early. Tiny friction compounds,
so let us simply locate the next bottleneck and repair it shortly."""


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+|[^A-Za-z]+", text)


def encode(secret: str, cover: str | None = None) -> Tuple[str, int]:
    """Return (covertext, bits_encoded). secret is UTF-8 str, truncated to fit."""
    cover = cover or DEFAULT_COVER
    bits = []
    for ch in secret.encode("utf-8"):
        bits.extend([(ch >> i) & 1 for i in range(7, -1, -1)])

    toks = _tokens(cover)
    out: list[str] = []
    bit_idx = 0
    for tok in toks:
        low = tok.lower()
        if low in LOOKUP and bit_idx < len(bits):
            pair_idx, _ = LOOKUP[low]
            target = SYNONYM_PAIRS[pair_idx][bits[bit_idx]]
            # preserve original capitalization pattern
            if tok[0].isupper():
                target = target.capitalize()
            out.append(target)
            bit_idx += 1
        else:
            out.append(tok)
    return "".join(out), bit_idx


def decode(covertext: str, byte_count: int | None = None) -> str:
    """Recover secret bytes; if byte_count provided, stops there."""
    bits = []
    for tok in _tokens(covertext):
        low = tok.lower()
        if low in LOOKUP:
            bits.append(LOOKUP[low][1])
    # group into bytes
    bytes_out = bytearray()
    limit_bits = byte_count * 8 if byte_count else (len(bits) // 8) * 8
    for i in range(0, min(len(bits), limit_bits), 8):
        byte = 0
        for b in bits[i:i + 8]:
            byte = (byte << 1) | b
        bytes_out.append(byte)
    try:
        # strip trailing zeros that come from unused bits
        return bytes_out.rstrip(b"\x00").decode("utf-8", errors="replace")
    except Exception:
        return bytes_out.hex()


def max_bytes(cover: str | None = None) -> int:
    cover = cover or DEFAULT_COVER
    slots = sum(1 for t in _tokens(cover) if t.lower() in LOOKUP)
    return slots // 8
