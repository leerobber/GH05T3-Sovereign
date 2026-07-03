"""Ethical income assistants — human-in-the-loop, consent-gated, ToS-compliant."""

from .core import (
    UserConsentRecord,
    RateLimiter,
    HumanConfirmationGate,
    ComplianceGate,
    EthicalActionResult,
    ASSISTANCE_MODES,
)
from .survey_assistant import SurveyAssistant, SurveyOpportunity
from .airdrop_tracker import AirdropTracker, AirdropListing
from .p2e_assistant import P2EAssistant, P2EGameProfile, BattleSuggestion
from .staking_optimizer import StakingOptimizer, YieldMarket, StakingStrategy
from .orchestrator import EthicalIncomeOrchestrator

__all__ = [
    "UserConsentRecord",
    "RateLimiter",
    "HumanConfirmationGate",
    "ComplianceGate",
    "EthicalActionResult",
    "ASSISTANCE_MODES",
    "SurveyAssistant",
    "SurveyOpportunity",
    "AirdropTracker",
    "AirdropListing",
    "P2EAssistant",
    "P2EGameProfile",
    "BattleSuggestion",
    "StakingOptimizer",
    "YieldMarket",
    "StakingStrategy",
    "EthicalIncomeOrchestrator",
]