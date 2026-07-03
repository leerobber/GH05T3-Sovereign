from .seo_agent import SEOAgent
from .design_agent import DesignAgent
from .content_agent import ContentAgent
from .marketing_agent import MarketingAgent
from .journalist_agent import JournalistAgent
from .marketplace_agent import MarketplaceAgent
from .stripe_agent import StripeAgent
from .email_agent import EmailAgent
from .legal_agent import LegalAgent
from .hr_agent import HRAgent
from .analytics_agent import AnalyticsAgent
from .brand_agent import BrandAgent
from .ops_agent import OpsAgent
# Contractor agent pack
from .invoice_agent import InvoiceAgent
from .quote_agent import QuoteAgent
from .client_comms_agent import ClientCommsAgent
from .schedule_agent import ScheduleAgent

ALL_AGENTS = {
    # Site agents
    "seo":          SEOAgent,
    "design":       DesignAgent,
    "content":      ContentAgent,
    "marketing":    MarketingAgent,
    "journalist":   JournalistAgent,
    "marketplace":  MarketplaceAgent,
    # Business superagents
    "stripe":       StripeAgent,
    "email":        EmailAgent,
    "legal":        LegalAgent,
    "hr":           HRAgent,
    "analytics":    AnalyticsAgent,
    "brand":        BrandAgent,
    "ops":          OpsAgent,
    # Contractor agent pack
    "invoice":      InvoiceAgent,
    "quote":        QuoteAgent,
    "client_comms": ClientCommsAgent,
    "schedule":     ScheduleAgent,
}


def get_agent(name: str):
    cls = ALL_AGENTS.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown agent '{name}'. Available: {sorted(ALL_AGENTS)}")
    return cls()
