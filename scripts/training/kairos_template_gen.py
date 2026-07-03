"""
kairos_template_gen.py — Generate KAIROS SFT pairs from templates (no API required).

Uses structured templates with randomized scenario variables to produce varied
instruction/response pairs covering all 6 KAIROS phases. Run AFTER kairos_dataset_gen.py
when API credits are available, or standalone when they aren't.

Run:
  python kairos_template_gen.py               # generate 500 template pairs
  python kairos_template_gen.py --pairs 200   # custom count
  python kairos_template_gen.py --append      # append to existing file
"""
import argparse, json, os, random, sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT   = Path(__file__).parent
DATA   = ROOT / "data"
OUTPUT = DATA / "kairos_dataset.jsonl"

# ── Domain vocabulary pools ────────────────────────────────────────────────────

TIERS = ["individual ($19/mo)", "family ($29/mo)", "education ($9/student/mo)", "enterprise ($199/mo)"]
MARKETS = ["rural communities", "urban low-income families", "school districts",
           "tribal nations", "refugee communities", "senior citizens", "veterans",
           "small businesses", "community colleges", "housing voucher recipients"]
DURATIONS = ["30 days", "60 days", "90 days", "6 months", "12 months", "Q3", "Q4", "fiscal year"]
METRICS = ["monthly active users", "churn rate", "net promoter score", "cost per user",
           "task completion rate", "session duration", "referral rate", "revenue per user",
           "subscriber growth rate", "engagement depth"]
PARTNERS = ["local libraries", "community colleges", "rural ISPs", "school districts",
            "faith communities", "labor unions", "food banks", "healthcare clinics",
            "government housing agencies", "tribal councils"]
RISKS = ["funding shortfall", "competitive entrant", "regulatory change", "key personnel departure",
         "infrastructure failure", "data breach", "viral negative press", "GPU cost spike",
         "partner withdrawal", "compliance violation"]
LEVERS = ["referral incentive program", "retention email campaign", "partnership co-marketing",
          "freemium onboarding flow", "community ambassador network", "targeted discount offer",
          "content drip campaign", "usage-based nudge alerts", "social proof showcase",
          "local champion program"]
KPIS = ["CAC < $15", "LTV > $200", "churn < 5%/month", "NPS > 50", "DAU/MAU > 40%",
        "task completion > 70%", "session > 8 min/day", "referral rate > 15%"]
STAKEHOLDERS = ["pioneer families", "school administrators", "rural broadband ISPs",
                "social service agencies", "content partners", "government sponsors",
                "community moderators", "angel investors", "advisory board"]
CONSTRAINTS = ["no external credit injection", "< $5/user/month infra cost",
               "must maintain COPPA compliance", "6-month runway", "team of 8",
               "no paid advertising budget", "must be offline-capable",
               "FERPA compliance required", "no price increase allowed"]

# ── Scenario templates ────────────────────────────────────────────────────────
# Each entry: (instruction_template, response_template_key)
# Response templates are keyed by type and filled with scenario vars.

SCENARIOS = [
    # Growth & Market Entry
    {
        "instruction": "Build a {duration} go-to-market strategy for SovereignNation's {tier} targeting {market}.",
        "k": "Establish SovereignNation's presence in the {market} segment with the {tier}. Success = {kpi1} within {duration}. Key stakeholders: {stakeholder1}, {stakeholder2}.",
        "a": "Current: 0 users in {market}. Gap: no brand recognition, no local presence. Opportunity: underserved segment with high need for affordable AI. Constraint: {constraint}.",
        "i": "- Week 1-2: Partner with {partner1} to host free demo events\n- Week 3-4: Launch referral program with non-cash community rewards\n- Month 2: Activate {lever1} targeting {market} households\n- Month 3: Scale to 3 additional {market} clusters based on pilot results",
        "r": "- Test: Do in-person demos outperform digital outreach in {market}?\n- Validate: Referral rate should hit 15%+ by month 2\n- Iterate: Adjust messaging if conversion < 2% at demo events",
        "o": "- KPIs: {kpi1}, {kpi2}, cost per acquisition < $15\n- Reduce: Manual outreach → automated follow-up sequences\n- Instrument: Cohort dashboard tracking {market} users separately",
        "s": "- Phase 2: Replicate {market} playbook to 3 adjacent markets\n- Formalize: Codify {lever1} as standard launch protocol\n- Measure: Track {market} cohort LTV over 12 months before scaling"
    },
    # Retention & Churn
    {
        "instruction": "Design a retention strategy for SovereignNation's {tier} subscribers who haven't logged in for {duration}.",
        "k": "Recover {tier} subscribers dormant for {duration}. Success = reactivate 30%+ within 30 days. Stakeholders: {stakeholder1}, product team, content team.",
        "a": "Root cause likely: low perceived value, forgot about subscription, or friction in re-entry. Constraint: {constraint}. Available lever: {lever1}.",
        "i": "- Day 1: Trigger personalized re-engagement email with 'what's new' highlights\n- Day 3: Send family-specific use case: AI homework helper / career coach\n- Day 7: Offer {lever1} for account re-activation\n- Day 14: Phone or text outreach for highest-value dormant accounts\n- Day 21: Cancellation-save offer with {kpi1} guarantee",
        "r": "- A/B test: Personalized vs generic re-engagement email subject lines\n- Validate: Open rate > 25%, click rate > 8% on re-engagement flow\n- Iterate: Test SMS vs email for {market} segment specifically",
        "o": "- KPI: Reactivation rate > 30%, churn reduced by {kpi2}\n- Automate: Dormancy trigger → campaign launch without manual intervention\n- Reduce: Time-to-reactivate from 21 days to 10 days",
        "s": "- Scale: Apply re-engagement protocol to all tiers, not just {tier}\n- Predict: Build ML model flagging dormancy risk at day 7 before it hits day {duration}\n- Replicate: Share playbook with {partner1} for co-branded re-engagement"
    },
    # Infrastructure & Cost
    {
        "instruction": "Create a cost optimization strategy to keep SovereignNation's infrastructure under $5/user/month at {market} scale.",
        "k": "Reduce per-user infrastructure cost to < $5/month while maintaining quality for {market} users. Success measured by {kpi1}. Stakeholders: engineering, finance, {stakeholder1}.",
        "a": "Current cost drivers: GPU inference, storage, bandwidth. Constraint: {constraint}. Opportunity: batch inference, model quantization, edge caching.",
        "i": "- Sprint 1: Quantize Avery LoRA to Q8_0 — 40% inference cost reduction\n- Sprint 2: Implement response caching for top-200 common queries\n- Sprint 3: Edge deployment for {market} users reducing bandwidth costs\n- Sprint 4: Shared inference cluster across tiers with priority queuing",
        "r": "- Test: Does Q8_0 quantization degrade response quality by > 5%?\n- Validate: Cost/user drops to < $5 at 10,000 user load test\n- Iterate: Adjust cache hit-rate target if quality complaints increase",
        "o": "- KPIs: {kpi1}, {kpi2}, latency P95 < 3 seconds\n- Eliminate: Idle GPU time by implementing auto-scale-to-zero\n- Instrument: Real-time cost-per-user dashboard for engineering team",
        "s": "- Scale: Apply edge caching architecture to all geographic markets\n- Generalize: Cost optimization playbook for each new agent deployment\n- Target: Reach $3/user/month at 100,000 users through economies of scale"
    },
    # Partnership
    {
        "instruction": "Build a partnership strategy for SovereignNation with {partner1} to expand reach in {market}.",
        "k": "Formalize a partnership with {partner1} to reach {market} families. Success = {kpi1} via partner channel within {duration}. Stakeholders: {stakeholder1}, {stakeholder2}, {partner1} leadership.",
        "a": "Gap: SovereignNation lacks trust in {market}. {partner1} has established relationships. Opportunity: co-branding increases credibility. Constraint: {constraint}.",
        "i": "- Month 1: Sign MOU with {partner1} defining co-marketing terms and data sharing limits\n- Month 1: Co-host 2 community info sessions at {partner1} locations\n- Month 2: Launch {lever1} exclusively for {partner1} referrals\n- Month 3: Integrate {partner1} content into SovereignNation's platform\n- Month 4: Evaluate pilot; negotiate revenue share for continued partnership",
        "r": "- Test: Do {partner1}-referred users retain better than organic sign-ups?\n- Validate: Partner channel should account for > 20% of {market} acquisitions\n- Iterate: Adjust co-marketing materials based on {market} community feedback",
        "o": "- KPIs: {kpi1}, {kpi2}, partner-referred LTV vs organic LTV\n- Streamline: Single intake form for {partner1} referrals vs manual outreach\n- Reduce: Onboarding friction for {partner1}-referred users to < 5 minutes",
        "s": "- Replicate: {partner1} model to {partner2} within 6 months of successful pilot\n- Formalize: Partnership playbook for all future community partner launches\n- Scale: Aim for 10 active partnerships covering {market} by end of year"
    },
    # Crisis Response
    {
        "instruction": "SovereignNation is facing {risk}. Apply KAIROS to build a {duration} response strategy.",
        "k": "Address {risk} threatening SovereignNation's operations within {duration}. Success = minimize impact to < 10% user/revenue loss. Stakeholders: {stakeholder1}, {stakeholder2}, legal, communications.",
        "a": "Severity: {risk} could impact {kpi1} by 15-30% if unaddressed. Root cause analysis underway. Available levers: {lever1}, {lever2}. Constraint: {constraint}.",
        "i": "- Day 1: Convene crisis team; assign {stakeholder1} as incident lead\n- Day 1-3: Activate {lever1} to stabilize immediate impact\n- Day 3-7: Deploy {lever2} as secondary containment measure\n- Week 2: Communicate transparently with users via email and in-app\n- Week 3-4: Conduct root cause review and implement systemic fix",
        "r": "- Test: Is {lever1} sufficient, or does {risk} require escalation?\n- Validate: User retention should stay above 90% within 14 days\n- Iterate: Adjust communication cadence based on user sentiment signals",
        "o": "- KPIs: User retention > 90%, {kpi1} impact < 10%, recovery time < {duration}\n- Automate: Early warning system for {risk} indicators using {kpi2} monitoring\n- Reduce: Response time from detection to action to < 4 hours",
        "s": "- Formalize: {risk} playbook as standing crisis protocol\n- Train: Quarterly tabletop exercise with all stakeholders for {risk} scenario\n- Build: Automated monitoring dashboard covering top 5 risk categories"
    },
    # Product Roadmap
    {
        "instruction": "Design the {duration} product roadmap for SovereignNation's AI tutoring feature targeting {market}.",
        "k": "Ship a production-ready AI tutoring feature for {market} within {duration}. Success = {kpi1} and {kpi2}. Stakeholders: {stakeholder1}, engineering, content partners.",
        "a": "Current: prototype AI tutor exists but lacks {market}-specific content. Gap: curriculum alignment, language support, low-bandwidth optimization. Constraint: {constraint}.",
        "i": "- Sprint 1-2: Integrate curriculum standards for {market} grade levels\n- Sprint 3-4: Build offline-capable mode for {market} bandwidth constraints\n- Sprint 5: Add parent progress dashboard with {kpi1} tracking\n- Sprint 6: Beta test with 200 {market} students; collect feedback\n- Sprint 7-8: Iterate on feedback; ship production release",
        "r": "- Test: Do {market} students show measurable learning gains in 4-week pilot?\n- Validate: Session length > 15 min/day, return rate > 60%\n- Iterate: Adjust difficulty algorithm if 40%+ of sessions end in < 5 minutes",
        "o": "- KPIs: {kpi1}, {kpi2}, learning gain score > 15% vs control\n- Reduce: Onboarding from 10 steps to 3 for {market} households\n- Instrument: Live teacher dashboard showing per-student engagement",
        "s": "- Expand: AI tutor to 3 additional subject areas in next {duration}\n- Replicate: {market} curriculum model to adjacent demographic\n- Measure: 6-month academic outcome study before scaling nationally"
    },
    # Funding
    {
        "instruction": "Build a funding strategy for SovereignNation targeting {stakeholder1} with a {duration} timeline.",
        "k": "Secure funding from {stakeholder1} within {duration} to extend runway to 24 months. Success = term sheet signed. Key stakeholders: {stakeholder1}, {stakeholder2}, SovereignNation board.",
        "a": "Current runway: 6 months. Gap: need $2-4M to reach 50,000 subscribers (break-even). {stakeholder1} funding criteria: social impact, financial sustainability, {kpi1}. Constraint: {constraint}.",
        "i": "- Week 1-2: Audit all metrics and compile due diligence package\n- Week 2-3: Identify 15 {stakeholder1} targets using impact investment databases\n- Week 3-4: Warm introductions through {stakeholder2} network\n- Month 2: First-round pitch to 5 highest-probability {stakeholder1} targets\n- Month 3: Term sheet negotiation with top 2 interested parties",
        "r": "- Test: Does impact narrative + {kpi1} data resonate better than pure revenue pitch?\n- Validate: At least 3 of 15 targets should advance to second meeting\n- Iterate: Refine pitch deck after first 5 presentations based on objections",
        "o": "- KPIs: 15 qualified outreach → 5 meetings → 2 term sheets within {duration}\n- Automate: CRM tracking of all investor interactions and follow-ups\n- Reduce: Due diligence prep time from 3 weeks to 1 week via data room",
        "s": "- Beyond seed: Build {stakeholder1} relationship for Series A in 18 months\n- Diversify: Pursue 2 additional funding streams (grants, revenue-based financing)\n- Formalize: Investor relations protocol for ongoing {stakeholder1} engagement"
    },
    # Metrics & Analytics
    {
        "instruction": "Develop a metrics framework for SovereignNation to track {kpi1} and {kpi2} across the {tier}.",
        "k": "Build a unified metrics framework tracking {kpi1} and {kpi2} for the {tier}. Success = weekly automated report to leadership within {duration}. Stakeholders: engineering, product, {stakeholder1}.",
        "a": "Current: metrics scattered across 3 systems; no single source of truth. Gap: no automated alerting on {kpi1} degradation. Constraint: {constraint}. Lever: PostgreSQL + Grafana stack already in place.",
        "i": "- Week 1: Define exact SQL queries for {kpi1} and {kpi2} calculations\n- Week 2: Build Grafana dashboard with 24-hour, 7-day, 30-day views\n- Week 3: Configure alerts: {kpi1} drops > 5% → Slack notification\n- Week 4: Automate weekly PDF report sent to {stakeholder1} every Monday\n- Month 2: Add cohort analysis for {tier} users by acquisition source",
        "r": "- Test: Do automated alerts catch issues before {stakeholder1} notices manually?\n- Validate: Dashboard accuracy — cross-check vs manual CSV export for 2 weeks\n- Iterate: Refine {kpi2} formula if it produces counterintuitive results",
        "o": "- KPIs: {kpi1} visibility < 1 hour lag, {kpi2} reported weekly\n- Eliminate: Manual report preparation (currently 3 hours/week)\n- Expand: Add {market} segmentation to all key metrics by quarter end",
        "s": "- Replicate: Framework to all 6 SovereignNation agents (Avery/FORGE/ORACLE/etc.)\n- Generalize: Metrics standard applied to all future product launches\n- Benchmark: Compare {kpi1} against industry standards quarterly"
    },
]


def _fill(template: str, **ctx) -> str:
    try:
        return template.format(**ctx)
    except KeyError:
        return template


def _build_pair(scenario: dict) -> dict:
    ctx = {
        "tier":        random.choice(TIERS),
        "market":      random.choice(MARKETS),
        "duration":    random.choice(DURATIONS),
        "kpi1":        (kpi_sample := random.sample(KPIS, 2))[0],
        "kpi2":        kpi_sample[1],
        "partner1":    (partner_sample := random.sample(PARTNERS, 2))[0],
        "partner2":    partner_sample[1],
        "stakeholder1": (sh_sample := random.sample(STAKEHOLDERS, 2))[0],
        "stakeholder2": sh_sample[1],
        "lever1":      random.choice(LEVERS),
        "lever2":      random.choice(LEVERS),
        "constraint":  random.choice(CONSTRAINTS),
        "risk":        random.choice(RISKS),
    }

    instruction = _fill(scenario["instruction"], **ctx)
    response = (
        f"## K — Kickoff\n{_fill(scenario['k'], **ctx)}\n\n"
        f"## A — Alignment\n{_fill(scenario['a'], **ctx)}\n\n"
        f"## I — Implementation\n{_fill(scenario['i'], **ctx)}\n\n"
        f"## R — Refinement\n{_fill(scenario['r'], **ctx)}\n\n"
        f"## O — Optimization\n{_fill(scenario['o'], **ctx)}\n\n"
        f"## S — Scaling\n{_fill(scenario['s'], **ctx)}"
    )
    return {
        "instruction": instruction,
        "response": response,
        "source": "kairos_template",
        "framework": "KAIROS",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs",  type=int, default=500)
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    print(f"\n{'='*56}")
    print("  KAIROS TEMPLATE GENERATOR (offline, no API)")
    print(f"{'='*56}")
    print(f"  Target pairs : {args.pairs}")
    print(f"  Output       : {OUTPUT}")
    print(f"  File mode    : {'append' if args.append else 'overwrite'}")
    print()

    DATA.mkdir(exist_ok=True)
    file_mode = "a" if args.append else "w"
    generated = 0

    with open(OUTPUT, file_mode, encoding="utf-8") as f:
        for i in range(args.pairs):
            scenario = random.choice(SCENARIOS)
            pair = _build_pair(scenario)
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            generated += 1
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{args.pairs}] {generated} pairs written...")

    print(f"\n  Done: {generated} pairs written to {OUTPUT.name}")
    print(f"{'='*56}\n")


if __name__ == "__main__":
    main()
