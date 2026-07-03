"""CFO Agent for SovereignNation — full startup CFO execution layer.

Covers all 10 CFO domains:
  1. Financial Modeling   — 3-statement model, scenario analysis, valuation
  2. Burn Rate / Runway   — weekly burn, months of runway, break-even
  3. Unit Economics       — CAC, LTV, LTV:CAC, payback period, margins
  4. Cash Flow            — 13-week rolling forecast, AR/AP, treasury
  5. Budgeting            — annual op plan, dept budgets, variance tracking
  6. Cap Table            — equity rounds, SAFE/notes, dilution, option pool
  7. Fundraising          — pitch financials, investor reporting
  8. Financial Reporting  — P&L, balance sheet, board package narrative
  9. Compliance Calendar  — tax deadlines, state registrations, audit prep
 10. KPI Dashboard        — north-star metric, cohort analysis, SaaS metrics

All state persisted in MongoDB `cfo_*` collections.
LLM used for narrative generation and scenario analysis via ghost_llm.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Any

LOG = logging.getLogger("ghost.cfo")

# ---------------------------------------------------------------------------
# MongoDB collection names
# ---------------------------------------------------------------------------
COL_FINANCIALS  = "cfo_financials"
COL_BUDGETS     = "cfo_budgets"
COL_CASH_FLOW   = "cfo_cash_flow"
COL_CAP_TABLE   = "cfo_cap_table"
COL_KPIS        = "cfo_kpis"
COL_COMPLIANCE  = "cfo_compliance"
COL_BOARD       = "cfo_board_packages"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _now() -> datetime:
    return datetime.now(timezone.utc)


def _months_between(start: datetime, end: datetime) -> float:
    delta = (end - start).days
    return round(delta / 30.44, 1)


def _pct(part: float, whole: float) -> float:
    if whole == 0:
        return 0.0
    return round((part / whole) * 100, 2)


# ===========================================================================
# 1. BURN RATE & RUNWAY
# ===========================================================================

async def calculate_burn_runway(db, cash_on_hand: float,
                                 monthly_expenses: dict[str, float],
                                 monthly_revenue: float = 0.0) -> dict:
    """Compute net burn, gross burn, runway months, and break-even projection."""
    gross_burn = sum(monthly_expenses.values())
    net_burn = max(0.0, gross_burn - monthly_revenue)
    runway_months = (cash_on_hand / net_burn) if net_burn > 0 else 9999.0
    runway_date = _now() + timedelta(days=runway_months * 30.44)

    result = {
        "cash_on_hand":      round(cash_on_hand, 2),
        "gross_burn_mo":     round(gross_burn, 2),
        "net_burn_mo":       round(net_burn, 2),
        "monthly_revenue":   round(monthly_revenue, 2),
        "runway_months":     round(runway_months, 1),
        "runway_date":       runway_date.strftime("%Y-%m-%d"),
        "break_even_revenue": round(gross_burn, 2),
        "expenses_breakdown": monthly_expenses,
        "calculated_at":     _now().isoformat(),
        "alert": None,
    }
    if runway_months < 6:
        result["alert"] = f"CRITICAL: Only {runway_months:.1f} months runway. Raise or cut immediately."
    elif runway_months < 12:
        result["alert"] = f"WARNING: {runway_months:.1f} months runway. Begin fundraising now."

    await db[COL_FINANCIALS].update_one(
        {"_id": "burn_runway"},
        {"$set": result},
        upsert=True,
    )
    return result


# ===========================================================================
# 2. UNIT ECONOMICS
# ===========================================================================

async def calculate_unit_economics(db,
                                    cac: float,
                                    arpu: float,
                                    gross_margin_pct: float,
                                    avg_customer_lifetime_months: float,
                                    monthly_churn_pct: float = None) -> dict:
    """CAC, LTV, LTV:CAC ratio, payback period, gross margin."""
    if monthly_churn_pct and monthly_churn_pct > 0:
        avg_customer_lifetime_months = 1.0 / (monthly_churn_pct / 100.0)

    monthly_gross_profit = arpu * (gross_margin_pct / 100.0)
    ltv = monthly_gross_profit * avg_customer_lifetime_months
    ltv_cac_ratio = ltv / cac if cac > 0 else 0.0
    payback_months = cac / monthly_gross_profit if monthly_gross_profit > 0 else 9999.0

    health = "EXCELLENT" if ltv_cac_ratio >= 3 else "GOOD" if ltv_cac_ratio >= 2 else "POOR"

    result = {
        "cac":                         round(cac, 2),
        "arpu_monthly":                round(arpu, 2),
        "gross_margin_pct":            round(gross_margin_pct, 2),
        "avg_lifetime_months":         round(avg_customer_lifetime_months, 1),
        "ltv":                         round(ltv, 2),
        "ltv_cac_ratio":               round(ltv_cac_ratio, 2),
        "payback_months":              round(payback_months, 1),
        "monthly_gross_profit":        round(monthly_gross_profit, 2),
        "health":                      health,
        "recommendation":              _unit_econ_recommendation(ltv_cac_ratio, payback_months),
        "calculated_at":               _now().isoformat(),
    }

    await db[COL_FINANCIALS].update_one(
        {"_id": "unit_economics"},
        {"$set": result},
        upsert=True,
    )
    return result


def _unit_econ_recommendation(ltv_cac: float, payback: float) -> str:
    if ltv_cac >= 3 and payback <= 12:
        return "Strong unit economics. Scale paid acquisition."
    if ltv_cac >= 3 and payback > 12:
        return "Good LTV:CAC but long payback. Optimize onboarding to reduce churn earlier."
    if ltv_cac < 2:
        return "LTV:CAC below 2x. Do NOT scale marketing yet. Fix retention or reduce CAC first."
    return "Unit economics acceptable but room to improve. Monitor monthly churn closely."


# ===========================================================================
# 3. 13-WEEK ROLLING CASH FLOW FORECAST
# ===========================================================================

async def build_cash_flow_forecast(db,
                                    opening_cash: float,
                                    weekly_inflows: list[dict],
                                    weekly_outflows: list[dict]) -> dict:
    """13-week rolling cash flow. Each entry: {'week': 1, 'amount': X, 'category': str}."""
    weeks = []
    balance = opening_cash

    for w in range(1, 14):
        inflow  = sum(x["amount"] for x in weekly_inflows  if x.get("week") == w)
        outflow = sum(x["amount"] for x in weekly_outflows if x.get("week") == w)
        net     = inflow - outflow
        balance = balance + net
        weeks.append({
            "week":           w,
            "date":           (_now() + timedelta(weeks=w - 1)).strftime("%Y-%m-%d"),
            "inflows":        round(inflow, 2),
            "outflows":       round(outflow, 2),
            "net":            round(net, 2),
            "closing_balance": round(balance, 2),
            "alert":          "LOW CASH" if balance < 50_000 else None,
        })

    result = {
        "opening_cash":  round(opening_cash, 2),
        "forecast":      weeks,
        "minimum_cash":  round(min(w["closing_balance"] for w in weeks), 2),
        "generated_at":  _now().isoformat(),
    }

    await db[COL_CASH_FLOW].update_one(
        {"_id": "rolling_13wk"},
        {"$set": result},
        upsert=True,
    )
    return result


async def get_cash_flow_forecast(db) -> dict | None:
    return await db[COL_CASH_FLOW].find_one({"_id": "rolling_13wk"})


# ===========================================================================
# 4. BUDGET — Annual Operating Plan
# ===========================================================================

async def set_budget(db, year: int, departments: dict[str, dict[str, float]]) -> dict:
    """
    departments = {
        "engineering": {"headcount": 120000, "tooling": 24000, "infra": 36000},
        "marketing":   {"ads": 60000, "events": 10000},
        ...
    }
    """
    totals_by_dept = {dept: sum(v.values()) for dept, v in departments.items()}
    grand_total = sum(totals_by_dept.values())

    doc = {
        "_id":            f"budget_{year}",
        "year":           year,
        "departments":    departments,
        "totals_by_dept": totals_by_dept,
        "grand_total":    round(grand_total, 2),
        "monthly_burn":   round(grand_total / 12, 2),
        "created_at":     _now().isoformat(),
        "actuals":        {},
        "variance":       {},
    }
    await db[COL_BUDGETS].update_one({"_id": f"budget_{year}"}, {"$set": doc}, upsert=True)
    return doc


async def record_actual(db, year: int, month: int, department: str,
                         category: str, amount: float) -> dict:
    """Record actual spend against budget."""
    key = f"{year}-{month:02d}-{department}-{category}"
    await db[COL_BUDGETS].update_one(
        {"_id": f"budget_{year}"},
        {"$set": {f"actuals.{key}": amount}},
        upsert=True,
    )
    budget_doc = await db[COL_BUDGETS].find_one({"_id": f"budget_{year}"})
    if not budget_doc:
        return {"error": "budget not found"}

    budgeted = budget_doc.get("departments", {}).get(department, {}).get(category, 0)
    variance = amount - budgeted
    variance_pct = _pct(variance, budgeted) if budgeted else 0
    return {
        "department": department,
        "category":   category,
        "budgeted":   budgeted,
        "actual":     amount,
        "variance":   round(variance, 2),
        "variance_pct": variance_pct,
        "status":     "OVER" if variance > 0 else "UNDER" if variance < 0 else "ON_TARGET",
    }


async def get_budget_variance(db, year: int) -> dict:
    doc = await db[COL_BUDGETS].find_one({"_id": f"budget_{year}"})
    if not doc:
        return {"error": f"No budget for {year}"}

    actuals = doc.get("actuals", {})
    budgets = doc.get("departments", {})

    variance_report = {}
    for dept, cats in budgets.items():
        dept_actual = sum(
            v for k, v in actuals.items()
            if k.split("-")[2] == dept
        )
        dept_budget = sum(cats.values())
        variance_report[dept] = {
            "budgeted": dept_budget,
            "actual":   round(dept_actual, 2),
            "variance": round(dept_actual - dept_budget, 2),
            "pct":      _pct(dept_actual - dept_budget, dept_budget),
        }
    return {"year": year, "variance": variance_report}


# ===========================================================================
# 5. CAP TABLE
# ===========================================================================

async def initialize_cap_table(db, company_name: str, authorized_shares: int) -> dict:
    doc = {
        "_id":              "cap_table",
        "company":          company_name,
        "authorized_shares": authorized_shares,
        "issued_shares":    0,
        "option_pool":      0,
        "rounds":           [],
        "holders":          {},
        "updated_at":       _now().isoformat(),
    }
    await db[COL_CAP_TABLE].update_one({"_id": "cap_table"}, {"$set": doc}, upsert=True)
    return doc


async def add_equity_round(db, round_name: str, instrument: str,
                            investors: list[dict],
                            pre_money_valuation: float = 0,
                            option_pool_increase: int = 0) -> dict:
    """
    instrument: 'equity' | 'safe' | 'convertible_note'
    investors: [{"name": str, "shares": int, "investment": float}]
    """
    cap = await db[COL_CAP_TABLE].find_one({"_id": "cap_table"}) or {}
    holders = cap.get("holders", {})
    issued = cap.get("issued_shares", 0)
    option_pool = cap.get("option_pool", 0)

    round_shares = sum(i.get("shares", 0) for i in investors)
    round_dollars = sum(i.get("investment", 0) for i in investors)

    for inv in investors:
        name = inv["name"]
        shares = inv.get("shares", 0)
        holders[name] = holders.get(name, 0) + shares

    issued += round_shares
    option_pool += option_pool_increase

    total_fully_diluted = issued + option_pool
    ownership = {
        name: {"shares": sh, "pct": _pct(sh, total_fully_diluted)}
        for name, sh in holders.items()
    }

    round_doc = {
        "name":               round_name,
        "instrument":         instrument,
        "pre_money_val":      pre_money_valuation,
        "post_money_val":     pre_money_valuation + round_dollars,
        "raised":             round(round_dollars, 2),
        "shares_issued":      round_shares,
        "price_per_share":    round(pre_money_valuation / issued, 4) if issued else 0,
        "investors":          investors,
        "date":               _now().strftime("%Y-%m-%d"),
    }

    await db[COL_CAP_TABLE].update_one(
        {"_id": "cap_table"},
        {"$set": {
            "issued_shares": issued,
            "option_pool": option_pool,
            "holders": holders,
            "ownership_pct": ownership,
            "updated_at": _now().isoformat(),
        },
         "$push": {"rounds": round_doc}},
        upsert=True,
    )
    return {"round": round_doc, "ownership": ownership, "total_fully_diluted": total_fully_diluted}


async def get_cap_table(db) -> dict | None:
    return await db[COL_CAP_TABLE].find_one({"_id": "cap_table"})


async def model_dilution(db, new_shares: int, investment: float) -> dict:
    """Show current owners' dilution if a new round issues new_shares."""
    cap = await db[COL_CAP_TABLE].find_one({"_id": "cap_table"}) or {}
    issued = cap.get("issued_shares", 0)
    option_pool = cap.get("option_pool", 0)
    holders = cap.get("holders", {})

    total_pre = issued + option_pool
    total_post = total_pre + new_shares
    price = investment / new_shares if new_shares > 0 else 0

    dilution = {}
    for name, shares in holders.items():
        pre_pct  = _pct(shares, total_pre)
        post_pct = _pct(shares, total_post)
        dilution[name] = {
            "shares":       shares,
            "pre_pct":      pre_pct,
            "post_pct":     post_pct,
            "dilution_pct": round(pre_pct - post_pct, 2),
        }
    return {
        "new_shares":      new_shares,
        "investment":      investment,
        "price_per_share": round(price, 4),
        "total_pre_diluted": total_pre,
        "total_post_diluted": total_post,
        "dilution":        dilution,
    }


# ===========================================================================
# 6. KPI DASHBOARD — SaaS / AI-service metrics
# ===========================================================================

async def record_kpis(db, period: str, metrics: dict) -> dict:
    """
    period: 'YYYY-MM'
    metrics: {
        'mrr': float,       'arr': float,        'new_customers': int,
        'churned_customers': int, 'total_customers': int,
        'cac': float,       'arpu': float,       'nrr_pct': float,
        'gross_margin_pct': float, 'burn': float, 'cash': float,
        'headcount': int,   'revenue_per_employee': float,
    }
    """
    doc = {"_id": f"kpi_{period}", "period": period, **metrics, "recorded_at": _now().isoformat()}

    # Derive churn rate and NRR if not provided
    if "total_customers" in metrics and "churned_customers" in metrics:
        tc = metrics["total_customers"]
        ch = metrics["churned_customers"]
        doc["churn_rate_pct"] = _pct(ch, tc)

    if "mrr" in metrics and "arr" not in metrics:
        doc["arr"] = round(metrics["mrr"] * 12, 2)

    await db[COL_KPIS].update_one({"_id": f"kpi_{period}"}, {"$set": doc}, upsert=True)
    return doc


async def get_kpi_trend(db, periods: int = 6) -> list[dict]:
    """Return last N months of KPIs sorted oldest first."""
    cursor = db[COL_KPIS].find({}).sort("period", -1).limit(periods)
    docs = await cursor.to_list(length=periods)
    return list(reversed(docs))


async def saas_scorecard(db) -> dict:
    """Compute SaaS health scorecard from latest KPIs."""
    latest = await db[COL_KPIS].find_one({}, sort=[("period", -1)])
    if not latest:
        return {"error": "No KPI data recorded yet."}

    score = {}
    mrr = latest.get("mrr", 0)
    arr = latest.get("arr", mrr * 12)
    churn = latest.get("churn_rate_pct", 0)
    nrr = latest.get("nrr_pct", 100)
    ltv_cac = latest.get("ltv_cac_ratio", 0)
    gm = latest.get("gross_margin_pct", 0)
    burn = latest.get("burn", 0)
    cash = latest.get("cash", 0)
    runway = (cash / burn) if burn > 0 else 999

    score["period"] = latest.get("period")
    score["arr"] = arr
    score["mrr"] = mrr
    score["churn_rate_pct"] = churn
    score["nrr_pct"] = nrr
    score["ltv_cac_ratio"] = ltv_cac
    score["gross_margin_pct"] = gm
    score["runway_months"] = round(runway, 1)

    # Health flags
    flags = []
    if churn > 5:    flags.append("HIGH_CHURN")
    if nrr < 100:    flags.append("NEGATIVE_NRR")
    if ltv_cac < 3:  flags.append("POOR_UNIT_ECON")
    if gm < 60:      flags.append("LOW_MARGIN")
    if runway < 6:   flags.append("CRITICAL_RUNWAY")
    if runway < 12:  flags.append("LOW_RUNWAY")

    score["health_flags"] = flags
    score["overall_health"] = "GREEN" if not flags else "YELLOW" if len(flags) <= 2 else "RED"
    return score


# ===========================================================================
# 7. COMPLIANCE CALENDAR
# ===========================================================================

COMPLIANCE_EVENTS = [
    {"name": "Federal Q1 Estimated Tax",          "month": 4,  "day": 15, "category": "tax"},
    {"name": "Federal Q2 Estimated Tax",          "month": 6,  "day": 15, "category": "tax"},
    {"name": "Federal Q3 Estimated Tax",          "month": 9,  "day": 15, "category": "tax"},
    {"name": "Federal Q4 Estimated Tax",          "month": 1,  "day": 15, "category": "tax"},
    {"name": "Corporate Income Tax Return (C-Corp)", "month": 4, "day": 15, "category": "tax"},
    {"name": "Annual Report / State Filing",      "month": 3,  "day": 31, "category": "entity"},
    {"name": "W-2 / 1099 Distribution",           "month": 1,  "day": 31, "category": "payroll"},
    {"name": "W-2 / 1099 Filing (IRS)",           "month": 3,  "day": 31, "category": "payroll"},
    {"name": "Delaware Franchise Tax",            "month": 3,  "day": 1,  "category": "entity"},
    {"name": "Annual Audit Kickoff",              "month": 1,  "day": 15, "category": "audit"},
    {"name": "Board Package — Q1",                "month": 4,  "day": 20, "category": "board"},
    {"name": "Board Package — Q2",                "month": 7,  "day": 20, "category": "board"},
    {"name": "Board Package — Q3",                "month": 10, "day": 20, "category": "board"},
    {"name": "Board Package — Q4 / Full Year",    "month": 1,  "day": 20, "category": "board"},
    {"name": "SOC 2 Audit Window Opens",          "month": 10, "day": 1,  "category": "compliance"},
    {"name": "409A Valuation Refresh",            "month": 1,  "day": 31, "category": "equity"},
    {"name": "Option Grants — Board Approval",    "month": 3,  "day": 15, "category": "equity"},
]


async def get_compliance_calendar(db, year: int = None) -> list[dict]:
    year = year or _now().year
    today = _now().date()
    events = []
    for e in COMPLIANCE_EVENTS:
        try:
            due = datetime(year, e["month"], e["day"]).date()
        except ValueError:
            continue
        days_until = (due - today).days
        events.append({
            **e,
            "due_date":   due.strftime("%Y-%m-%d"),
            "days_until": days_until,
            "status":     "OVERDUE" if days_until < 0 else "DUE_SOON" if days_until <= 30 else "UPCOMING",
        })
    events.sort(key=lambda x: x["due_date"])
    return events


async def get_upcoming_deadlines(db, days_ahead: int = 60) -> list[dict]:
    cal = await get_compliance_calendar(db)
    return [e for e in cal if 0 <= e["days_until"] <= days_ahead]


# ===========================================================================
# 8. BOARD PACKAGE GENERATOR
# ===========================================================================

async def generate_board_package(db, quarter: str, llm_fn=None) -> dict:
    """
    quarter: 'Q1-2025'
    llm_fn: async callable(system, user) -> str  — used for narrative generation
    """
    burn = await db[COL_FINANCIALS].find_one({"_id": "burn_runway"}) or {}
    ue   = await db[COL_FINANCIALS].find_one({"_id": "unit_economics"}) or {}
    kpis = await saas_scorecard(db)
    cap  = await get_cap_table(db) or {}
    deadlines = await get_upcoming_deadlines(db, 90)

    package = {
        "quarter":   quarter,
        "generated": _now().isoformat(),
        "sections": {
            "financial_summary": {
                "cash_on_hand":    burn.get("cash_on_hand"),
                "net_burn_mo":     burn.get("net_burn_mo"),
                "runway_months":   burn.get("runway_months"),
                "runway_alert":    burn.get("alert"),
                "mrr":             kpis.get("mrr"),
                "arr":             kpis.get("arr"),
                "gross_margin_pct": kpis.get("gross_margin_pct"),
            },
            "unit_economics": {
                "ltv":             ue.get("ltv"),
                "cac":             ue.get("cac"),
                "ltv_cac_ratio":   ue.get("ltv_cac_ratio"),
                "payback_months":  ue.get("payback_months"),
                "health":          ue.get("health"),
            },
            "kpi_scorecard":    kpis,
            "cap_table_summary": {
                "issued_shares":   cap.get("issued_shares"),
                "option_pool":     cap.get("option_pool"),
                "last_round":      cap.get("rounds", [{}])[-1] if cap.get("rounds") else None,
            },
            "upcoming_compliance": deadlines,
        },
        "narrative": None,
    }

    if llm_fn:
        summary_text = json.dumps(package["sections"]["financial_summary"], indent=2)
        kpi_text = json.dumps(package["sections"]["kpi_scorecard"], indent=2)
        prompt = (
            f"You are the CFO of SovereignNation, an AI-native startup. "
            f"Write a concise, honest board package narrative for {quarter}. "
            f"Tone: direct, data-driven, no spin. Cover: highlights, concerns, asks.\n\n"
            f"Financial Summary:\n{summary_text}\n\n"
            f"KPI Scorecard:\n{kpi_text}"
        )
        try:
            narrative, _ = await llm_fn("You are a senior CFO writing a board package.", prompt)
            package["narrative"] = narrative
        except Exception as e:
            package["narrative"] = f"[Narrative generation failed: {e}]"

    await db[COL_BOARD].update_one(
        {"_id": f"board_{quarter}"},
        {"$set": package},
        upsert=True,
    )
    return package


# ===========================================================================
# 9. FINANCIAL REPORTING — P&L, Balance Sheet summary
# ===========================================================================

async def record_pnl(db, period: str, revenue: float,
                     cogs: float, opex: dict[str, float]) -> dict:
    """
    period: 'YYYY-MM' or 'YYYY-QN'
    opex: {'engineering': X, 'marketing': X, 'g&a': X, ...}
    """
    gross_profit = revenue - cogs
    gross_margin = _pct(gross_profit, revenue)
    total_opex = sum(opex.values())
    ebitda = gross_profit - total_opex
    ebitda_margin = _pct(ebitda, revenue)

    doc = {
        "_id":            f"pnl_{period}",
        "period":         period,
        "revenue":        round(revenue, 2),
        "cogs":           round(cogs, 2),
        "gross_profit":   round(gross_profit, 2),
        "gross_margin_pct": gross_margin,
        "opex":           opex,
        "total_opex":     round(total_opex, 2),
        "ebitda":         round(ebitda, 2),
        "ebitda_margin_pct": ebitda_margin,
        "net_income":     round(ebitda, 2),
        "recorded_at":    _now().isoformat(),
    }
    await db[COL_FINANCIALS].update_one({"_id": f"pnl_{period}"}, {"$set": doc}, upsert=True)
    return doc


async def get_pnl_history(db, periods: int = 12) -> list[dict]:
    cursor = db[COL_FINANCIALS].find({"_id": {"$regex": "^pnl_"}}).sort("period", -1).limit(periods)
    docs = await cursor.to_list(length=periods)
    return list(reversed(docs))


# ===========================================================================
# 10. THREE-STATEMENT MODEL (snapshot)
# ===========================================================================

async def three_statement_snapshot(db,
                                    revenue_scenarios: dict[str, float],
                                    fixed_costs: float,
                                    variable_cost_pct: float,
                                    cash_on_hand: float,
                                    existing_arr: float = 0.0) -> dict:
    """
    revenue_scenarios: {'bear': X, 'base': Y, 'bull': Z}
    Returns P&L, runway, and valuation range for each scenario.
    """
    results = {}
    for scenario, revenue_12mo in revenue_scenarios.items():
        monthly_rev = revenue_12mo / 12
        variable_costs = revenue_12mo * (variable_cost_pct / 100)
        total_costs = fixed_costs + variable_costs
        gross_profit = revenue_12mo - variable_costs
        ebitda = gross_profit - fixed_costs
        monthly_burn = max(0, (total_costs - revenue_12mo) / 12)
        runway = (cash_on_hand / monthly_burn) if monthly_burn > 0 else 999

        # Simple ARR-multiple valuation (SaaS typical 5x-10x ARR)
        arr = existing_arr + revenue_12mo
        val_low  = arr * 5
        val_high = arr * 10

        results[scenario] = {
            "revenue_12mo":       round(revenue_12mo, 2),
            "monthly_rev":        round(monthly_rev, 2),
            "gross_profit":       round(gross_profit, 2),
            "ebitda":             round(ebitda, 2),
            "monthly_burn":       round(monthly_burn, 2),
            "runway_months":      round(runway, 1),
            "arr":                round(arr, 2),
            "valuation_low":      round(val_low, 2),
            "valuation_high":     round(val_high, 2),
        }

    doc = {
        "_id":        "three_statement",
        "scenarios":  results,
        "inputs": {
            "fixed_costs":         fixed_costs,
            "variable_cost_pct":   variable_cost_pct,
            "cash_on_hand":        cash_on_hand,
            "existing_arr":        existing_arr,
        },
        "generated_at": _now().isoformat(),
    }
    await db[COL_FINANCIALS].update_one({"_id": "three_statement"}, {"$set": doc}, upsert=True)
    return doc


# ===========================================================================
# 11. FUNDRAISING — SAFE / Convertible Note Modeling
# ===========================================================================

def model_safe(investment: float, valuation_cap: float,
               discount_pct: float, next_round_price: float) -> dict:
    """Model SAFE conversion at next priced round."""
    cap_price      = valuation_cap / (valuation_cap / next_round_price) if next_round_price else 0
    discount_price = next_round_price * (1 - discount_pct / 100)
    conversion_price = min(cap_price, discount_price) if cap_price else discount_price
    shares_issued  = investment / conversion_price if conversion_price else 0

    return {
        "investment":        investment,
        "valuation_cap":     valuation_cap,
        "discount_pct":      discount_pct,
        "next_round_price":  next_round_price,
        "cap_price":         round(cap_price, 4),
        "discount_price":    round(discount_price, 4),
        "conversion_price":  round(conversion_price, 4),
        "shares_at_conversion": round(shares_issued, 2),
        "investor_note":     "Converts at lower of cap price or discounted price.",
    }


# ===========================================================================
# 12. API HELPERS — thin wrappers for server.py route handlers
# ===========================================================================

async def cfo_dashboard(db) -> dict:
    """Single call returns everything the dashboard needs."""
    burn  = await db[COL_FINANCIALS].find_one({"_id": "burn_runway"})
    ue    = await db[COL_FINANCIALS].find_one({"_id": "unit_economics"})
    score = await saas_scorecard(db)
    cap   = await get_cap_table(db)
    deadlines = await get_upcoming_deadlines(db, 60)
    cf    = await get_cash_flow_forecast(db)

    return {
        "burn_runway":      burn,
        "unit_economics":   ue,
        "kpi_scorecard":    score,
        "cap_table":        cap,
        "upcoming_deadlines": deadlines,
        "cash_flow_13wk":   cf,
        "as_of":            _now().isoformat(),
    }
