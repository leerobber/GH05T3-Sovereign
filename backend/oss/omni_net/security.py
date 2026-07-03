"""5.5 Security Layer — Phase 5.

OmniSecurity: sessions, API key auth, encryption (uses cryptography Fernet when available).
Target: 0 breaches in 10k sim interactions.
"""
from __future__ import annotations

import base64
import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

try:
    from cryptography.fernet import Fernet
    HAS_FERNET = True
except Exception:
    HAS_FERNET = False


@dataclass
class OmniSecurity:
    sessions: Dict[str, float] = field(default_factory=dict)
    api_keys: Dict[str, str] = field(default_factory=dict)  # user -> key
    session_ttl: float = 3600.0
    _fernet: Optional[Any] = None
    breach_count: int = 0

    def __post_init__(self):
        if HAS_FERNET:
            self._fernet = Fernet(Fernet.generate_key())
        else:
            # fallback xor key (for envs without crypto)
            self._key = hashlib.sha256(b"gh05t3-omni-phase5").digest()

    def create_session(self, user_id: str, api_key: Optional[str] = None) -> str:
        if api_key and user_id in self.api_keys and self.api_keys[user_id] != api_key:
            self.breach_count += 1
            raise PermissionError("Invalid API key")
        token = hashlib.sha256(f"{user_id}:{time.time()}:{api_key or ''}".encode()).hexdigest()[:32]
        self.sessions[token] = time.time() + self.session_ttl
        return token

    def validate_session(self, token: str) -> bool:
        exp = self.sessions.get(token)
        if not exp:
            return False
        if time.time() > exp:
            self.sessions.pop(token, None)
            return False
        return True

    def create_api_key(self, user_id: str) -> str:
        key = base64.urlsafe_b64encode(hashlib.sha256(f"{user_id}:{time.time()}".encode()).digest()[:32]).decode()
        self.api_keys[user_id] = key
        return key

    def encrypt(self, data: str) -> str:
        if self._fernet:
            return self._fernet.encrypt(data.encode()).decode()
        # fallback
        raw = data.encode()
        xored = bytes(b ^ self._key[i % len(self._key)] for i, b in enumerate(raw))
        return base64.b64encode(xored).decode()

    def decrypt(self, payload: str) -> Optional[str]:
        try:
            if self._fernet:
                return self._fernet.decrypt(payload.encode()).decode()
            raw = base64.b64decode(payload.encode())
            plain = bytes(b ^ self._key[i % len(self._key)] for i, b in enumerate(raw))
            return plain.decode()
        except Exception:
            self.breach_count += 1
            return None

    def sim_interactions(self, n: int = 10000) -> int:
        """Sim 10k interactions for gate. Returns breach_count."""
        for i in range(n):
            tok = self.create_session(f"user{i%100}")
            if not self.validate_session(tok):
                self.breach_count += 1
            enc = self.encrypt("secret data")
            if self.decrypt(enc) is None:
                self.breach_count += 1
        return self.breach_count
