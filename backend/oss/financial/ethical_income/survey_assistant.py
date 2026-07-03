"""
Survey & microtask assistant — discovery and session prep only.

DOES NOT: auto-login, auto-fill answers, multi-account, or bypass CAPTCHA.
Creator completes every task manually on one honest account per platform.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
import re
import time

from .core import (
    UserConsentRecord,
    RateLimiter,
    HumanConfirmationGate,
    EthicalActionResult,
    ASSISTANCE_MODES,
)


@dataclass
class SurveyOpportunity:
    platform: str
    title: str
    reward_usd: float
    time_minutes: float
    url: str
    task_type: str
    hourly_rate_usd: float = 0.0
    requirements: List[str] = field(default_factory=list)
    tos_notes: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.time_minutes > 0 and self.hourly_rate_usd == 0.0:
            self.hourly_rate_usd = round((self.reward_usd / self.time_minutes) * 60, 2)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Curated lawful platforms — agents rank and route; creator executes.
SURVEY_PLATFORMS: Dict[str, Dict[str, Any]] = {
    "prolific": {
        "name": "Prolific",
        "url": "https://www.prolific.com/studies",
        "type": "academic_surveys",
        "hourly_range": (5.0, 20.0),
        "tos": "One account, honest responses, manual completion, no bots",
        "capabilities": ["survey_navigation", "reward_ranking", "time_estimation"],
    },
    "mturk": {
        "name": "Amazon Mechanical Turk",
        "url": "https://worker.mturk.com/",
        "type": "microtasks",
        "hourly_range": (3.0, 10.0),
        "tos": "Requester rules apply; no automated HIT acceptance at scale",
        "capabilities": ["task_routing", "quality_checklist"],
    },
    "clickworker": {
        "name": "Clickworker",
        "url": "https://www.clickworker.com/",
        "type": "data_entry_moderation",
        "hourly_range": (2.0, 8.0),
        "tos": "Manual tasks, one account, follow project guidelines",
        "capabilities": ["task_routing", "qc_prep"],
    },
    "microworkers": {
        "name": "Microworkers",
        "url": "https://microworkers.com/",
        "type": "microtasks",
        "hourly_range": (1.0, 5.0),
        "tos": "Small tasks — low pay; manual only",
        "capabilities": ["multi_task_queue", "efficiency_tips"],
    },
    "appen": {
        "name": "Appen",
        "url": "https://appen.com/",
        "type": "ai_training_data",
        "hourly_range": (5.0, 15.0),
        "tos": "Annotation/transcription per project rules",
        "capabilities": ["consistency_checklist", "pattern_guidance"],
    },
}


class SurveyAssistant:
    """
    Ethical survey assistant: discover → rank → prepare session → user completes.

    Example flow:
      1. creator grants consent for platform
      2. assistant lists/ranks opportunities (static catalog + future API hooks)
      3. assistant opens checklist + URL — creator logs in and completes manually
    """

    def __init__(
        self,
        rate_limiter: Optional[RateLimiter] = None,
        confirmation_gate: Optional[HumanConfirmationGate] = None,
    ):
        self.rate_limiter = rate_limiter or RateLimiter()
        self.confirmation_gate = confirmation_gate or HumanConfirmationGate()
        self.consents: Dict[str, UserConsentRecord] = {}
        self.session_log: List[Dict[str, Any]] = []

    def register_consent(self, platform: str, scopes: Optional[List[str]] = None) -> UserConsentRecord:
        scopes = scopes or ["discover", "prepare_session", "track_earnings"]
        record = UserConsentRecord(
            platform=platform.lower(),
            consented_at=time.time(),
            scopes=scopes,
        )
        self.consents[platform.lower()] = record
        return record

    def revoke_consent(self, platform: str) -> bool:
        key = platform.lower()
        if key not in self.consents:
            return False
        self.consents[key].revoked = True
        return True

    def _require_consent(self, platform: str, scope: str) -> Optional[EthicalActionResult]:
        rec = self.consents.get(platform.lower())
        if not rec or not rec.is_active(scope):
            return EthicalActionResult(
                accepted=False,
                mode="discover_only",
                platform=platform,
                message=f"Creator must opt in to '{scope}' for {platform} before assistance.",
                next_steps=[
                    f"Call register_consent('{platform}', scopes=['discover','prepare_session'])",
                    "Read platform ToS — one account, manual completion",
                ],
            )
        return None

    def discover_opportunities(
        self,
        platform: Optional[str] = None,
        min_hourly_usd: float = 5.0,
    ) -> EthicalActionResult:
        """List ranked survey/microtask opportunities (catalog + estimates)."""
        platforms = [platform.lower()] if platform else list(SURVEY_PLATFORMS.keys())
        opportunities: List[SurveyOpportunity] = []

        for key in platforms:
            if key not in SURVEY_PLATFORMS:
                continue
            if not self.rate_limiter.allow(key):
                continue
            meta = SURVEY_PLATFORMS[key]
            low, high = meta["hourly_range"]
            mid_reward = round((low + high) / 2 * 0.25, 2)  # ~15 min task estimate
            opp = SurveyOpportunity(
                platform=meta["name"],
                title=f"Available {meta['type']} on {meta['name']}",
                reward_usd=mid_reward,
                time_minutes=15.0,
                url=meta["url"],
                task_type=meta["type"],
                requirements=["One account", "Manual completion", "Honest responses"],
                tos_notes=[meta["tos"], "NO bots, NO multi-accounting"],
            )
            if opp.hourly_rate_usd >= min_hourly_usd:
                opportunities.append(opp)

        opportunities.sort(key=lambda o: o.hourly_rate_usd, reverse=True)
        return EthicalActionResult(
            accepted=True,
            mode=ASSISTANCE_MODES[0],
            platform=platform or "all",
            message=f"Found {len(opportunities)} lawful opportunities (creator completes manually).",
            opportunities=[o.to_dict() for o in opportunities],
            next_steps=[
                "Log in to platform with YOUR credentials (never store in repo)",
                "Complete tasks manually — assistant does not submit answers",
                "Log earnings via track_session()",
            ],
            requires_human_confirmation=False,
            metadata={"automation": "FORBIDDEN", "human_in_loop": True},
        )

    def prepare_session(
        self,
        platform: str,
        opportunity_title: str,
    ) -> EthicalActionResult:
        """Prepare a session checklist — user must log in and complete manually."""
        denied = self._require_consent(platform, "prepare_session")
        if denied:
            return denied

        key = platform.lower()
        if key not in SURVEY_PLATFORMS:
            return EthicalActionResult(
                accepted=False,
                mode="prepare_session",
                platform=platform,
                message=f"Unknown platform: {platform}",
            )

        meta = SURVEY_PLATFORMS[key]
        action_id = f"SURVEY-{int(time.time())}-{key}"
        details = {
            "platform": meta["name"],
            "url": meta["url"],
            "opportunity": opportunity_title,
            "automation": "FORBIDDEN",
        }
        result = self.confirmation_gate.request(
            action_id,
            "open_survey_session",
            details,
        )
        result.next_steps = [
            f"1. Open {meta['url']} in your browser",
            f"2. Log in with your own account",
            f"3. Find: {opportunity_title}",
            "4. Complete honestly — do not use bots or scripted answers",
            f"5. Confirm action_id={action_id} after completion for earnings tracking",
        ]
        result.metadata["tos"] = meta["tos"]
        return result

    def track_session(
        self,
        platform: str,
        reward_usd: float,
        minutes_spent: float,
        notes: str = "",
    ) -> Dict[str, Any]:
        """Creator-reported earnings — assistant never invents income."""
        entry = {
            "platform": platform,
            "reward_usd": round(reward_usd, 2),
            "minutes_spent": minutes_spent,
            "hourly_rate": round((reward_usd / minutes_spent) * 60, 2) if minutes_spent else 0,
            "notes": notes,
            "timestamp": time.time(),
            "source": "creator_reported",
        }
        self.session_log.append(entry)
        return entry

    def earnings_summary(self) -> Dict[str, Any]:
        total = sum(e["reward_usd"] for e in self.session_log)
        hours = sum(e["minutes_spent"] for e in self.session_log) / 60.0
        return {
            "sessions": len(self.session_log),
            "total_usd": round(total, 2),
            "hours": round(hours, 2),
            "effective_hourly": round(total / hours, 2) if hours else 0.0,
        }

    @staticmethod
    def parse_hourly_from_text(reward_text: str, time_text: str) -> float:
        """Helper to parse '$5.00' and '15 min' style strings for ranking."""
        reward = 0.0
        minutes = 15.0
        m = re.search(r"[\d.]+", reward_text.replace(",", ""))
        if m:
            reward = float(m.group())
        t = re.search(r"(\d+)", time_text)
        if t:
            minutes = float(t.group())
        return round((reward / minutes) * 60, 2) if minutes else 0.0