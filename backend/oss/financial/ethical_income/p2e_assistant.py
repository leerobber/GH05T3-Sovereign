"""
Play-to-earn gaming assistant — strategy suggestions only.

DOES NOT: automated gameplay, bot farming, or submitting moves without user confirmation.
Creator owns all NFTs/assets and confirms every action.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
import time

from .core import HumanConfirmationGate, EthicalActionResult, UserConsentRecord


@dataclass
class P2EGameProfile:
    platform: str
    game_type: str
    daily_earn_range_usd: tuple
    url: str
    tos_notes: str
    assistance_allowed: List[str]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["daily_earn_range_usd"] = list(self.daily_earn_range_usd)
        return d


@dataclass
class BattleSuggestion:
    platform: str
    battle_id: str
    suggested_team: List[str]
    rationale: str
    confidence: float
    requires_confirmation: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


P2E_GAMES: Dict[str, P2EGameProfile] = {
    "splinterlands": P2EGameProfile(
        platform="Splinterlands",
        game_type="nft_card_battling",
        daily_earn_range_usd=(1.0, 20.0),
        url="https://splinterlands.com/",
        tos_notes="No bots; user confirms team submission",
        assistance_allowed=["team_suggestion", "rental_research", "daily_quest_checklist"],
    ),
    "axie_infinity": P2EGameProfile(
        platform="Axie Infinity",
        game_type="nft_battling",
        daily_earn_range_usd=(5.0, 50.0),
        url="https://axieinfinity.com/",
        tos_notes="User-owned Axies only; no automated battling",
        assistance_allowed=["battle_strategy", "energy_tracking", "market_research"],
    ),
    "gods_unchained": P2EGameProfile(
        platform="Gods Unchained",
        game_type="tcg",
        daily_earn_range_usd=(10.0, 100.0),
        url="https://godsunchained.com/",
        tos_notes="Deck-building advice only",
        assistance_allowed=["deck_suggestion", "card_trade_research"],
    ),
    "stepn": P2EGameProfile(
        platform="STEPN",
        game_type="move_to_earn",
        daily_earn_range_usd=(1.0, 10.0),
        url="https://stepn.com/",
        tos_notes="Real movement required — no GPS spoofing",
        assistance_allowed=["sneaker_management", "energy_optimization"],
    ),
    "defi_kingdoms": P2EGameProfile(
        platform="DeFi Kingdoms",
        game_type="nft_rpg",
        daily_earn_range_usd=(5.0, 50.0),
        url="https://defikingdoms.com/",
        tos_notes="Quest planning only — user executes",
        assistance_allowed=["quest_routing", "resource_planning"],
    ),
}


class P2EAssistant:
    """Suggest teams, decks, and daily routines — never auto-play."""

    def __init__(self, confirmation_gate: Optional[HumanConfirmationGate] = None):
        self.confirmation_gate = confirmation_gate or HumanConfirmationGate()
        self.consents: Dict[str, UserConsentRecord] = {}
        self.session_log: List[Dict[str, Any]] = []
        self.team_prefs: Dict[str, Dict[str, List[str]]] = {}

    def register_consent(self, platform: str) -> UserConsentRecord:
        record = UserConsentRecord(
            platform=platform.lower(),
            consented_at=time.time(),
            scopes=["discover", "suggest_strategy", "track_earnings"],
        )
        self.consents[platform.lower()] = record
        return record

    def set_team_preferences(self, platform: str, prefs: Dict[str, List[str]]) -> None:
        self.team_prefs[platform.lower()] = prefs

    def list_games(self) -> EthicalActionResult:
        games = [g.to_dict() for g in P2E_GAMES.values()]
        return EthicalActionResult(
            accepted=True,
            mode="discover_only",
            platform="p2e",
            message=f"{len(games)} P2E platforms — assistance only, no bots.",
            opportunities=games,
            next_steps=[
                "Register consent per game",
                "Set team_preferences for card games",
                "User confirms every submission",
            ],
            metadata={"game_bot_farming": "PROHIBITED"},
        )

    def suggest_team(
        self,
        platform: str,
        battle_id: str = "daily",
    ) -> EthicalActionResult:
        key = platform.lower().replace(" ", "_")
        if key not in P2E_GAMES:
            return EthicalActionResult(
                accepted=False,
                mode="suggest_strategy",
                platform=platform,
                message=f"Unknown P2E platform: {platform}",
            )

        game = P2E_GAMES[key]
        prefs = self.team_prefs.get(key, {})
        team: List[str] = []
        for card_type in ("monster", "summoner", "spell"):
            team.extend(prefs.get(card_type, [])[:2])
        if not team:
            team = ["user_card_1", "user_card_2", "user_card_3", "user_card_4", "user_card_5", "user_card_6"]

        suggestion = BattleSuggestion(
            platform=game.platform,
            battle_id=battle_id,
            suggested_team=team[:6],
            rationale=(
                f"Team based on your preferences for {game.platform}. "
                "Review before submitting — assistant does not auto-play."
            ),
            confidence=0.65,
        )

        action_id = f"P2E-{int(time.time())}-{key}"
        self.confirmation_gate.request(
            action_id,
            "submit_p2e_team",
            {"platform": game.platform, "battle_id": battle_id, "team": suggestion.suggested_team},
        )

        return EthicalActionResult(
            accepted=True,
            mode="suggest_strategy",
            platform=game.platform,
            message="Team suggested — creator must review and confirm in-game.",
            opportunities=[suggestion.to_dict()],
            next_steps=[
                f"Open {game.url}",
                f"Review suggested team for battle {battle_id}",
                f"Confirm action_id={action_id} only after YOU submit in-game",
                game.tos_notes,
            ],
            requires_human_confirmation=True,
            metadata={"action_id": action_id, "auto_play": False},
        )

    def rank_daily_opportunities(self) -> List[Dict[str, Any]]:
        ranked = []
        for key, game in P2E_GAMES.items():
            low, high = game.daily_earn_range_usd
            ranked.append({
                "platform": game.platform,
                "url": game.url,
                "daily_usd_mid": round((low + high) / 2, 2),
                "assistance": game.assistance_allowed,
            })
        ranked.sort(key=lambda x: x["daily_usd_mid"], reverse=True)
        return ranked

    def track_session(self, platform: str, earned_usd: float, minutes: float) -> Dict[str, Any]:
        entry = {
            "platform": platform,
            "earned_usd": round(earned_usd, 2),
            "minutes": minutes,
            "timestamp": time.time(),
            "source": "creator_reported",
        }
        self.session_log.append(entry)
        return entry