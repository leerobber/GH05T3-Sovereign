"""
Market trust framework for Aethyro.com — prove credibility to visitors and search engines.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List


@dataclass
class TrustSignal:
    signal_id: str
    category: str  # security, social_proof, transparency, compliance, performance
    title: str
    description: str
    implementation: str
    priority: str  # critical, high, medium
    seo_impact: float  # 0-1 E-E-A-T contribution
    implemented: bool = False


@dataclass
class TrustScorecard:
    overall_score: float
    signals: List[TrustSignal]
    critical_gaps: List[str]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "signal_count": len(self.signals),
            "implemented": sum(1 for s in self.signals if s.implemented),
            "critical_gaps": self.critical_gaps,
            "recommendations": self.recommendations[:10],
        }


class TrustSignalEngine:
    """
    Defines and scores trust elements for enterprise-grade market credibility.
    WEB_ENGINEER_ELITE agents implement these on Aethyro.com.
    """

    CANONICAL_SIGNALS: List[Dict[str, Any]] = [
        {
            "signal_id": "ssl_https",
            "category": "security",
            "title": "HTTPS everywhere",
            "description": "Valid TLS certificate, HSTS, no mixed content",
            "implementation": "Cloudflare or host TLS; redirect HTTP→HTTPS",
            "priority": "critical",
            "seo_impact": 0.9,
        },
        {
            "signal_id": "privacy_policy",
            "category": "compliance",
            "title": "Published privacy policy",
            "description": "GDPR/CCPA-aligned data handling for AI services",
            "implementation": "/legal/privacy page linked in footer",
            "priority": "critical",
            "seo_impact": 0.7,
        },
        {
            "signal_id": "terms_of_service",
            "category": "compliance",
            "title": "Terms of service",
            "description": "Clear service terms for products and AI offerings",
            "implementation": "/legal/terms page",
            "priority": "critical",
            "seo_impact": 0.6,
        },
        {
            "signal_id": "stripe_payments",
            "category": "transparency",
            "title": "Stripe-verified checkout",
            "description": "Secure payment processing with recognizable trust badge",
            "implementation": "Stripe Checkout or Payment Element; display Stripe trust",
            "priority": "critical",
            "seo_impact": 0.5,
        },
        {
            "signal_id": "testimonials",
            "category": "social_proof",
            "title": "Customer testimonials",
            "description": "Real names, photos, or video where possible",
            "implementation": "Homepage + product pages; schema Review markup",
            "priority": "high",
            "seo_impact": 0.85,
        },
        {
            "signal_id": "case_studies",
            "category": "social_proof",
            "title": "Case studies / success stories",
            "description": "Documented outcomes for families and small businesses",
            "implementation": "/stories or blog case study posts",
            "priority": "high",
            "seo_impact": 0.9,
        },
        {
            "signal_id": "about_founder",
            "category": "transparency",
            "title": "Founder story (E-E-A-T)",
            "description": "Who built Aethyro and why — authentic creator narrative",
            "implementation": "/about with Person schema markup",
            "priority": "high",
            "seo_impact": 0.95,
        },
        {
            "signal_id": "open_source_proof",
            "category": "transparency",
            "title": "Open development proof",
            "description": "Link to public repos, changelogs, or technical blog",
            "implementation": "Footer link to GitHub/docs; builds technical trust",
            "priority": "medium",
            "seo_impact": 0.75,
        },
        {
            "signal_id": "contact_real",
            "category": "transparency",
            "title": "Real contact channels",
            "description": "Email, form, or chat with response SLA stated",
            "implementation": "/contact + business email on domain",
            "priority": "high",
            "seo_impact": 0.7,
        },
        {
            "signal_id": "core_web_vitals",
            "category": "performance",
            "title": "Core Web Vitals pass",
            "description": "LCP <2.5s, INP <200ms, CLS <0.1",
            "implementation": "Optimize images, CDN, lazy load; verify in Search Console",
            "priority": "high",
            "seo_impact": 0.8,
        },
        {
            "signal_id": "schema_organization",
            "category": "seo",
            "title": "Organization + Product schema",
            "description": "JSON-LD for Organization, Product, FAQPage",
            "implementation": "Inject schema in layout; validate in Rich Results Test",
            "priority": "high",
            "seo_impact": 0.85,
        },
        {
            "signal_id": "money_back_guarantee",
            "category": "social_proof",
            "title": "Clear refund / satisfaction policy",
            "description": "Reduces purchase anxiety for lower-income audience",
            "implementation": "Visible on checkout and product pages",
            "priority": "medium",
            "seo_impact": 0.4,
        },
        {
            "signal_id": "security_page",
            "category": "security",
            "title": "Security & data practices page",
            "description": "How local AI keeps data private — key differentiator",
            "implementation": "/security explaining local-first architecture",
            "priority": "high",
            "seo_impact": 0.8,
        },
        {
            "signal_id": "live_demo",
            "category": "social_proof",
            "title": "Interactive product demo",
            "description": "Let visitors experience value before buying",
            "implementation": "Embedded demo, sandbox, or video walkthrough on homepage",
            "priority": "critical",
            "seo_impact": 0.9,
        },
    ]

    def build_scorecard(self, implemented_ids: List[str] | None = None) -> TrustScorecard:
        implemented_ids = set(implemented_ids or [])
        signals = []
        for raw in self.CANONICAL_SIGNALS:
            sig = TrustSignal(**raw, implemented=raw["signal_id"] in implemented_ids)
            signals.append(sig)

        total_seo = sum(s.seo_impact for s in signals)
        earned = sum(s.seo_impact for s in signals if s.implemented)
        overall = round(earned / total_seo, 3) if total_seo else 0.0

        critical_gaps = [
            s.title for s in signals
            if s.priority == "critical" and not s.implemented
        ]
        recommendations = [
            f"[{s.priority.upper()}] {s.title}: {s.implementation}"
            for s in sorted(signals, key=lambda x: -x.seo_impact)
            if not s.implemented
        ]

        return TrustScorecard(
            overall_score=overall,
            signals=signals,
            critical_gaps=critical_gaps,
            recommendations=recommendations,
        )