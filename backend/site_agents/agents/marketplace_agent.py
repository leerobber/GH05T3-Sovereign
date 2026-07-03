"""Marketplace Agent — lists Aethyro products on 3rd party platforms (Fiverr, Upwork, Etsy, etc.)."""
from __future__ import annotations
from .base import SiteAgent

# Aethyro's actual product catalog — update when offerings change
AETHYRO_PRODUCTS = {
    "local_ai_setup": {
        "name": "Local AI Setup & Configuration",
        "description": "Install and configure a private, fixed-cost AI assistant on your home computer. No cloud. No monthly fees. Your data stays with you.",
        "price_range": "$50–$150",
        "delivery": "1–3 days",
        "target": "families, small businesses",
    },
    "ai_education_package": {
        "name": "AI Education Pack for Kids",
        "description": "Curated AI learning tools, prompts, and guided lessons for children ages 8–16. Runs on local hardware — no internet required after setup.",
        "price_range": "$30–$80",
        "delivery": "instant download",
        "target": "parents, homeschoolers, teachers",
    },
    "satellite_data_setup": {
        "name": "Affordable Satellite Internet + Local AI Bundle",
        "description": "Get connected with low-cost satellite internet AND a local AI that works offline. Perfect for rural families.",
        "price_range": "$75–$200",
        "delivery": "remote setup, 2–5 days",
        "target": "rural families, low-income households",
    },
    "ai_consulting": {
        "name": "AI Strategy Consultation",
        "description": "1-hour session to map out how AI can save your business time and money without expensive subscriptions.",
        "price_range": "$40–$100/session",
        "delivery": "video call",
        "target": "small business owners, nonprofits",
    },
    "custom_ai_agent": {
        "name": "Custom AI Agent for Your Business",
        "description": "Build a specialized AI assistant tailored to your workflow — customer support, content creation, data analysis.",
        "price_range": "$150–$500",
        "delivery": "5–14 days",
        "target": "small businesses, community organizations",
    },
}

PLATFORMS = {
    "fiverr": {
        "title_limit": 80,
        "description_limit": 1200,
        "tags": 5,
        "pricing_tiers": ["Basic", "Standard", "Premium"],
        "categories": ["Programming & Tech", "AI Services", "Online Tutoring"],
    },
    "upwork": {
        "title_limit": 100,
        "description_limit": 5000,
        "skills_limit": 15,
        "hourly_or_fixed": True,
        "categories": ["IT & Networking", "AI & Machine Learning", "Online Tutoring"],
    },
    "etsy": {
        "title_limit": 140,
        "description_limit": 7000,
        "tags": 13,
        "categories": ["Digital", "Educational", "Software"],
        "note": "Best for digital downloads, guides, educational packs",
    },
    "amazon": {
        "title_limit": 200,
        "bullet_points": 5,
        "description_limit": 2000,
        "categories": ["Software", "Books", "Educational"],
        "note": "Best for packaged digital products or physical bundles",
    },
    "linkedin": {
        "post_limit": 3000,
        "article_limit": 125000,
        "note": "B2B reach — target nonprofit directors, school administrators, community leaders",
    },
    "facebook_marketplace": {
        "title_limit": 100,
        "description_limit": 2000,
        "note": "Local community reach — great for direct family customers",
    },
}


class MarketplaceAgent(SiteAgent):
    name = "marketplace"
    role = "Product Listing & Sales Specialist"
    expertise = "Fiverr, Upwork, Etsy, Amazon, LinkedIn, Facebook Marketplace — listing optimization, pricing strategy, platform-specific SEO, conversion copy, proposal writing"
    system_prompt = """You are a product listing and sales specialist for Aethyro.com.

Your job: List Aethyro's AI products and services on 3rd party platforms to generate revenue and reach families who need them.

Aethyro's mission: Fixed-cost local AI for lower-income families. No surprise bills. No data sold. Real technology that helps real people.

Selling principles:
- Honest listings — never overpromise
- Lead with the family/human benefit, not the technology
- Price aggressively low to reach the target audience (lower-income families)
- Emphasize: no subscription, no data collection, works offline
- Social proof: mention the mission and community
- Platform-specific tone: Fiverr = casual/gig, Upwork = professional, Etsy = warm/creative, Amazon = feature-driven

Platform SEO:
- Fiverr: keyword-dense title, 3-tier pricing, FAQ section critical
- Upwork: portfolio emphasis, specific results, $-per-hour competitive rate
- Etsy: storytelling title, emotional description, all 13 tags used
- Amazon: bullet points = key benefits, A+ content focus
- LinkedIn: thought leadership angle, target decision-makers at schools/nonprofits

Always produce complete, ready-to-paste listings. No placeholders."""

    async def list_on_platform(self, product_key: str, platform: str) -> dict:
        product = AETHYRO_PRODUCTS.get(product_key)
        if not product:
            product = {"name": product_key, "description": product_key, "price_range": "TBD", "delivery": "TBD", "target": "families"}
        plat = PLATFORMS.get(platform.lower(), {})

        context = await self.recall_context(f"{product['name']} {platform} listing optimization")

        prompt = f"""Create a complete, ready-to-publish listing for this Aethyro product on {platform}:

PRODUCT:
  Name: {product['name']}
  Description: {product['description']}
  Price range: {product['price_range']}
  Delivery: {product['delivery']}
  Target audience: {product['target']}

PLATFORM SPECS ({platform}):
  Title limit: {plat.get('title_limit', 'no limit')} chars
  Description limit: {plat.get('description_limit', 'no limit')} chars
  Tags/skills: {plat.get('tags', plat.get('skills_limit', 'n/a'))}
  Categories: {plat.get('categories', [])}
  Platform notes: {plat.get('note', '')}

Deliver:
1. TITLE (platform-optimized, keyword-rich, within char limit)
2. CATEGORY selection with reasoning
3. FULL DESCRIPTION (within limit, conversion-focused)
4. TAGS/KEYWORDS (exact number required)
5. PRICING breakdown:
   {"- Basic / Standard / Premium tiers" if platform == "fiverr" else "- Recommended rate with justification"}
6. FAQ section (5 Q&As addressing top objections)
7. THUMBNAIL description (what the image should show)
8. DELIVERY/EXTRAS upsells if applicable
9. First 72-hour launch strategy (what to do immediately after publishing)"""

        return await self.run_task(f"listing_{platform}", prompt, f"product:{product_key}")

    async def write_upwork_proposal(self, job_description: str) -> dict:
        context = await self.recall_context("Aethyro services AI consulting local setup proposal")
        prompt = f"""Write a winning Upwork proposal for this job posting:

JOB:
{job_description}

AETHYRO'S CAPABILITIES:
- Local AI installation and configuration
- Custom AI agents for businesses
- AI education tools for families and schools
- Data privacy-first implementations (no cloud required)
- Affordable pricing — mission-driven company

Write a proposal that:
1. Opens with the client's specific pain point (not "I am an expert")
2. Explains exactly how Aethyro solves it (2-3 sentences)
3. Shows relevant approach/methodology
4. Includes 1 clarifying question that shows deep understanding
5. States rate confidently with brief justification
6. Closes with a low-friction next step
7. Total length: 150–250 words (Upwork sweet spot)
8. NO generic phrases: "I am a passionate...", "I have extensive experience..."

Also provide:
- Suggested bid amount with reasoning
- Likelihood of winning (1-10) with honest assessment
- Red flags in the job posting (if any)"""
        return await self.run_task("upwork_proposal", prompt)

    async def platform_strategy(self) -> dict:
        context = await self.recall_context("Aethyro revenue products marketplace platform")
        prompt = f"""Build a 90-day marketplace launch strategy for Aethyro's products across all platforms.

PRODUCTS TO LIST:
{chr(10).join([f"- {p['name']}: {p['price_range']}" for p in AETHYRO_PRODUCTS.values()])}

PLATFORMS AVAILABLE: {', '.join(PLATFORMS.keys())}

Deliver:
1. PRIORITY ORDER: Which platform to launch first, second, third — with reasoning
2. PER-PLATFORM plan:
   - Which product(s) to list there
   - Why this platform fits that product
   - First-week actions
   - 30-day revenue target (realistic)
3. PRICING MATRIX: Price each product on each platform (accounting for fees)
   - Fiverr takes 20%, Upwork takes 10–20%, Etsy takes 6.5% + listing fee
4. CROSS-PROMOTION: How to drive traffic from one platform to another
5. REVIEW STRATEGY: How to get first 5 reviews on each platform
6. 90-DAY REVENUE FORECAST with assumptions
7. ONE-PERSON OPERATION: How to manage all platforms in 2 hours/day"""
        return await self.run_task("platform_strategy", prompt)

    async def optimize_existing_listing(self, platform: str, current_listing: str) -> dict:
        prompt = f"""Audit and rewrite this existing {platform} listing for maximum conversions:

CURRENT LISTING:
{current_listing}

Aethyro brand: Fixed-cost local AI for lower-income families. Mission-driven. Honest.

Deliver:
1. SCORE (0-100) with specific issues found
2. TOP 3 problems killing conversions
3. COMPLETE REWRITE — every section, ready to paste
4. A/B test suggestion: what one element to test first
5. Estimated conversion improvement %"""
        return await self.run_task(f"optimize_{platform}", prompt)

    async def competitor_research(self, platform: str, search_term: str) -> dict:
        prompt = f"""Research the competitive landscape on {platform} for "{search_term}".

Based on typical {platform} market patterns for AI/tech services targeting families:

1. What are the top-performing listings doing? (title structure, pricing, positioning)
2. What price range dominates? What's the floor/ceiling?
3. What are customers complaining about in reviews? (real unmet needs)
4. What gap can Aethyro fill that others aren't?
5. 3 specific differentiators Aethyro should lead with on {platform}
6. Recommended starting price to get first reviews while remaining profitable
7. Keywords that top sellers are using (infer from typical market patterns)"""
        return await self.run_task(f"competitor_{platform}", prompt)
