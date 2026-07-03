"""
herald.py — Sovereign Herald: Training Intelligence Briefer

Reads training state, SPIN dataset, and recent cycle performance,
then sends a structured briefing to the Sovereign Leader via Slack.

Signals reported:
  BREAKTHROUGH  : >10% improvement on any domain
  INCREMENTAL   : 1-10% improvement
  PLATEAU       : no change across last 10 cycles
  REGRESSION    : score declining
  THRESHOLD     : SPIN pairs near/at upload trigger
  UPLOAD        : HuggingFace push completed

Usage:
  python herald.py                    # send briefing now
  python herald.py --console          # print to console only, no Slack
  python herald.py --full             # include last 10 SPIN samples
"""
import argparse, json, os, sys
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE          = Path(__file__).parent
STATE_FILE    = BASE / "data" / "continuous_state.json"
SPIN_FILE     = BASE / "data" / "spin_dataset.jsonl"
AMP_STATE     = BASE / "data" / "amplifier_state.json"

DOMAIN_ROTATION = [
    "business", "sales", "product_strategy", "growth", "cfo",
    "content", "legal_ip", "ops", "ml_engineer", "frontier", "core",
]
UPLOAD_THRESHOLD = 150


# ── Data loaders ──────────────────────────────────────────────────────────────

def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _load_spin() -> list[dict]:
    if not SPIN_FILE.exists():
        return []
    rows = []
    with SPIN_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _load_amp_state() -> dict:
    if AMP_STATE.exists():
        try:
            return json.loads(AMP_STATE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"amplified_goals": [], "total_variants_written": 0}


# ── Analysis ──────────────────────────────────────────────────────────────────

def _score_signal(scores: list[float]) -> tuple[str, float]:
    """Return (signal_type, delta) from a list of recent scores."""
    if len(scores) < 2:
        return "UNKNOWN", 0.0
    recent_avg  = sum(scores[-5:])  / min(len(scores), 5)
    earlier_avg = sum(scores[:5])   / min(len(scores), 5)
    delta = recent_avg - earlier_avg
    if delta > 0.10:
        return "BREAKTHROUGH", delta
    elif delta > 0.01:
        return "INCREMENTAL", delta
    elif delta < -0.05:
        return "REGRESSION", delta
    else:
        return "PLATEAU", delta


_DOMAIN_KEYWORDS = {
    "content":          ["linkedin", "content", "blog", "newsletter", "seo", "marketing", "copywriting"],
    "sales":            ["sales", "outreach", "pipeline", "closing", "prospect", "crm", "cold"],
    "cfo":              ["financial", "revenue", "cost", "budget", "funding", "capital", "unit economics"],
    "growth":           ["growth", "acquisition", "retention", "viral", "conversion", "funnel", "referral"],
    "legal_ip":         ["legal", " ip ", "contract", "entity", "license", "trademark", "patent"],
    "product_strategy": ["product", "roadmap", "prd", "mvp", "user research", "feature", "sprint"],
    "business":         ["business", "venture", "startup", "market", "monetize", "b2b", "saas"],
    "ops":              ["ops", "hiring", "vendor", "sop", "process", "scale", "team"],
    "ml_engineer":      ["model", "training", "fine-tune", "deploy", "mlops", "inference", "dataset"],
    "frontier":         ["agentic", "frontier", "autonomous", "post-human", "agi", "emergent"],
    "core":             ["sovereign", "economy", "agent", "kairos", "spin", "loop"],
}

def _infer_domain(goal: str) -> str:
    g = goal.lower()
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if any(kw in g for kw in keywords):
            return domain
    return "business"


def _analyze_spin(rows: list[dict]) -> dict:
    """Analyze SPIN dataset composition and quality."""
    if not rows:
        return {}

    total         = len(rows)
    amplified     = sum(1 for r in rows if r.get("source") == "amplified")
    original      = total - amplified

    by_domain: dict[str, list] = {}
    for r in rows:
        # Use stored domain or infer from goal text
        d = r.get("domain") or _infer_domain(r.get("goal", ""))
        if d not in by_domain:
            by_domain[d] = []
        score = r.get("rejected_score", 0.0)
        if score:
            by_domain[d].append(score)

    avg_reject_gap = {}
    for d, scores in by_domain.items():
        if scores:
            avg_reject_gap[d] = round(1.0 - (sum(scores) / len(scores)), 3)

    recent_goals = [r.get("goal", "")[:80] for r in rows[-3:]]

    return {
        "total":          total,
        "original":       original,
        "amplified":      amplified,
        "by_domain":      {d: len(v) for d, v in by_domain.items()},
        "quality_gap":    avg_reject_gap,
        "recent_goals":   recent_goals,
    }


# ── Briefing builder ──────────────────────────────────────────────────────────

def build_briefing(full: bool = False) -> str:
    now       = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    state     = _load_state()
    rows      = _load_spin()
    amp       = _load_amp_state()
    analysis  = _analyze_spin(rows)

    total_cycles   = state.get("total_cycles", 0)
    domain_idx     = state.get("domain_index", 0)
    domain_cycles  = state.get("domain_cycles", 0)
    spin_uploads   = state.get("spin_uploads", 0)
    active_domain  = DOMAIN_ROTATION[domain_idx % len(DOMAIN_ROTATION)]
    spin_total     = analysis.get("total", 0)
    spin_amplified = analysis.get("amplified", 0)
    spin_original  = analysis.get("original", 0)

    # Training velocity signal — measure pair accumulation rate
    # Split dataset into first half vs second half: if second half grew faster = improving
    mid = max(len(rows) // 2, 1)
    first_half_domains  = len(set(_infer_domain(r.get("goal","")) for r in rows[:mid]))
    second_half_domains = len(set(_infer_domain(r.get("goal","")) for r in rows[mid:]))
    first_half_scores   = [r.get("rejected_score",0.5) for r in rows[:mid]  if r.get("rejected_score")]
    second_half_scores  = [r.get("rejected_score",0.5) for r in rows[mid:] if r.get("rejected_score")]
    all_scores = first_half_scores + second_half_scores
    # Measure: are recent scores more spread from 0.5 (wider chosen/rejected gap = better training)?
    def _gap(scores): return sum(abs(s - 0.5) for s in scores) / len(scores) if scores else 0.5
    gap_delta = _gap(second_half_scores) - _gap(first_half_scores)
    signal = ("BREAKTHROUGH" if gap_delta > 0.05
              else "INCREMENTAL" if gap_delta > 0.01
              else "REGRESSION" if gap_delta < -0.05
              else "PLATEAU")
    delta = gap_delta

    signal_emoji = {
        "BREAKTHROUGH": "BREAKTHROUGH",
        "INCREMENTAL":  "INCREMENTAL",
        "PLATEAU":      "PLATEAU",
        "REGRESSION":   "REGRESSION",
    }.get(signal, "UNKNOWN")

    # Threshold alert
    threshold_pct = min(int((spin_total / UPLOAD_THRESHOLD) * 100), 100)
    threshold_bar = ("=" * (threshold_pct // 10)) + ("." * (10 - threshold_pct // 10))
    threshold_msg = (
        f"UPLOAD TRIGGERED ({spin_total} pairs)"
        if spin_total >= UPLOAD_THRESHOLD
        else f"{spin_total}/{UPLOAD_THRESHOLD} [{threshold_bar}] {threshold_pct}%"
    )

    # Domain progress
    domain_lines = []
    by_domain = analysis.get("by_domain", {})
    quality   = analysis.get("quality_gap", {})
    for d in DOMAIN_ROTATION:
        count = by_domain.get(d, 0)
        gap   = quality.get(d, None)
        marker = "<-- ACTIVE" if d == active_domain else ""
        gap_str = f"gap={gap:.2f}" if gap is not None else "no data"
        domain_lines.append(f"  {d:<20} {count:>4} pairs  {gap_str}  {marker}")

    # Recent goals
    recent = analysis.get("recent_goals", [])

    lines = [
        f"*SOVEREIGN HERALD BRIEFING*",
        f"_{now}_",
        f"",
        f"*TRAINING STATUS*",
        f"  Total cycles   : {total_cycles}",
        f"  Active domain  : {active_domain} (cycle {domain_cycles})",
        f"  Signal         : {signal_emoji}  delta={delta:+.3f}",
        f"  HF uploads     : {spin_uploads}",
        f"",
        f"*SPIN DATASET*",
        f"  Total pairs    : {spin_total}",
        f"  Original       : {spin_original}",
        f"  Amplified      : {spin_amplified}",
        f"  Upload status  : {threshold_msg}",
        f"",
        f"*DOMAIN COVERAGE*",
    ] + domain_lines + [
        f"",
        f"*RECENT TRAINING GOALS*",
    ] + [f"  - {g}" for g in recent]

    if amp.get("total_variants_written", 0):
        lines += [
            f"",
            f"*AMPLIFIER*",
            f"  Variants written : {amp['total_variants_written']}",
            f"  Pairs amplified  : {len(amp.get('amplified_goals', []))}",
        ]

    if full and rows:
        lines += ["", "*LAST 5 SPIN SAMPLES*"]
        for r in rows[-5:]:
            lines.append(f"  [{r.get('domain','?')}] {r.get('goal','')[:80]}")

    lines += ["", "_SovereignNation Training Intelligence Agency_"]
    return "\n".join(lines)


# ── Slack send ────────────────────────────────────────────────────────────────

def _send_slack(text: str) -> bool:
    try:
        import slack_notify as _slack
        return _slack.post("training", text)
    except Exception as e:
        print(f"  [HERALD] Slack error: {e}")
    return False


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.chdir(BASE)

    p = argparse.ArgumentParser(description="Sovereign Herald — Training Briefer")
    p.add_argument("--console",     action="store_true",
                   help="Print to console only, skip Slack")
    p.add_argument("--full",        action="store_true",
                   help="Include last 5 SPIN samples in briefing")
    args = p.parse_args()

    briefing = build_briefing(full=args.full)
    print(briefing)

    if not args.console:
        sent = _send_slack(briefing)
        if sent:
            print("\n  [HERALD] Briefing sent to Slack.")
        else:
            print("\n  [HERALD] Slack send failed — check SLACK_WEBHOOK_URL in .env")
