"""
amplifier.py — Sovereign SPIN Amplifier

Expands the SPIN dataset by generating N variant goals per existing pair,
then writing new chosen/rejected pairs. Triggers HuggingFace upload when
the dataset crosses the upload threshold.

Strategy:
  - llama3.2:3b  : fast goal variant generation (5 templates)
  - gemma3:12b   : quality chosen response (same model as main Proposer)
  - rejected     : reuses original rejected (still valid — chosen > rejected)

Usage:
  python amplifier.py                  # 3 variants per pair, auto-upload
  python amplifier.py --variants 5     # 5 variants per pair
  python amplifier.py --dry-run        # show plan, write nothing
  python amplifier.py --skip-upload    # amplify but don't push to HF
  python amplifier.py --reset          # clear amplifier state, start fresh
"""
import argparse, json, os, sys, time
from pathlib import Path

import requests as _req

try:
    import slack_notify as _slack
except ImportError:
    _slack = None

# ── Config ────────────────────────────────────────────────────────────────────
SPIN_FILE       = Path("data/spin_dataset.jsonl")
STATE_FILE      = Path("data/continuous_state.json")
AMP_STATE_FILE  = Path("data/amplifier_state.json")
OLLAMA_URL      = "http://localhost:11434/api/chat"
GOAL_MODEL      = "llama3.2:3b"    # tiny + fast for paraphrase tasks
CHOSEN_MODEL    = "gh05t3:latest"  # Avery self-play: she generates her own training data
DEFAULT_VARIANTS = 3
UPLOAD_THRESHOLD = 150
HF_DATASET_NAME  = "tastytator/sovereign-economy"

VARIANT_PROMPTS = [
    ("paraphrase",
     "Rewrite this business goal with different wording but the same intent. "
     "Stay in the same product/repo context. Output ONLY the rewritten goal, "
     "no explanation, no quotes.\n\nGoal: {goal}"),
    ("escalate",
     "Create a harder, more ambitious version of this goal. Raise the stakes: "
     "bigger scope, tighter deadline, or higher target metrics. Same repo. "
     "Output ONLY the new goal.\n\nGoal: {goal}"),
    ("pivot",
     "Adapt this goal for an underserved customer: low-income families, "
     "K-12 educators, or rural small business owners. Same product/repo. "
     "Output ONLY the adapted goal.\n\nGoal: {goal}"),
    ("bootstrap",
     "Reframe this goal for a bootstrapped solo founder: zero paid marketing, "
     "community-first, manual before automated. Same product. "
     "Output ONLY the reframed goal.\n\nGoal: {goal}"),
    ("invert",
     "Flip the core approach to achieve the same outcome through the opposite "
     "method (inbound vs outbound, free vs paid, B2B vs B2C). Same product. "
     "Output ONLY the inverted goal.\n\nGoal: {goal}"),
]

CHOSEN_SYSTEM = (
    "You are a sovereign business strategist for SovereignNation — a fixed-cost "
    "AI platform built for lower and middle class users, children's education, "
    "and affordable connectivity. Design specific, actionable business strategies "
    "grounded in real open-source repositories. Use markdown headers and clear "
    "numbered steps. Address weaknesses. Be concrete."
)


# ── Ollama call ───────────────────────────────────────────────────────────────

def _call(model: str, messages: list, max_tok: int = 512, temp: float = 0.7,
          retries: int = 3) -> str:
    delay = 10
    for attempt in range(retries):
        try:
            r = _req.post(OLLAMA_URL, json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temp, "num_predict": max_tok},
            }, timeout=300)
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        except Exception as e:
            if attempt < retries - 1:
                jitter = delay * (0.8 + 0.4 * (time.time() % 1))
                print(f"    [retry {attempt+1} in {jitter:.0f}s] {str(e)[:60]}")
                time.sleep(jitter)
                delay *= 2
            else:
                raise


# ── Amplifier state (tracks which pairs already amplified) ───────────────────

def _load_amp_state() -> dict:
    if AMP_STATE_FILE.exists():
        try:
            return json.loads(AMP_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"amplified_goals": [], "total_variants_written": 0}


def _save_amp_state(state: dict):
    AMP_STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ── SPIN I/O ──────────────────────────────────────────────────────────────────

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


def _append_spin(pair: dict):
    with SPIN_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(pair, ensure_ascii=False) + "\n")


def _count_spin() -> int:
    if not SPIN_FILE.exists():
        return 0
    try:
        return sum(1 for _ in SPIN_FILE.open(encoding="utf-8"))
    except Exception:
        return 0


# ── Upload (reuses continuous_learner logic) ──────────────────────────────────

def _upload_to_hf(skip: bool = False):
    if skip:
        print("  [UPLOAD] skipped (--skip-upload)")
        return

    print("\n  [UPLOAD] Pushing dataset to HuggingFace...")
    try:
        from datasets import Dataset
        env_path = Path(__file__).parent / ".env"
        hf_token = os.environ.get("HF_TOKEN", "")
        if not hf_token and env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("HF_TOKEN="):
                    hf_token = line.split("=", 1)[1].strip().strip('"\'')

        if not hf_token:
            print("  [UPLOAD] HF_TOKEN not found — add to .env")
            return

        rows = _load_spin()
        examples = []
        for row in rows:
            goal    = row.get("goal", "")
            chosen  = row.get("chosen", "")
            rejected = row.get("rejected", "")
            if not (goal and chosen):
                continue
            examples.append({
                "instruction": f"Business strategy goal: {goal}",
                "context":     f"Rejected approach: {rejected[:300]}" if rejected else "",
                "response":    chosen,
                "source":      row.get("source", "ghost_trainer_spin"),
                "domain":      row.get("domain", "business"),
            })

        ds = Dataset.from_list(examples)
        ds.push_to_hub(HF_DATASET_NAME, split="spin_business",
                       token=hf_token, private=False)
        print(f"  [UPLOAD] Done — {len(examples)} examples pushed to {HF_DATASET_NAME}")

        # Update continuous state
        if STATE_FILE.exists():
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            state["last_upload_count"] = _count_spin()
            state["spin_uploads"] = state.get("spin_uploads", 0) + 1
            STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")

        if _slack:
            try:
                _slack.notify_spin_upload(len(examples), _count_spin(), 1)
            except Exception:
                pass

    except ImportError:
        print("  [UPLOAD] Run: pip install datasets huggingface_hub")
    except Exception as e:
        print(f"  [UPLOAD] ERROR: {e}")


# ── Core amplification ────────────────────────────────────────────────────────

def _generate_variant_goal(original_goal: str, template: str) -> str:
    prompt = template.format(goal=original_goal)
    result = _call(GOAL_MODEL, [{"role": "user", "content": prompt}],
                   max_tok=128, temp=0.8)
    result = result.strip().strip('"').strip("'")
    if len(result) < 10:
        raise ValueError(f"Goal too short: {result!r}")
    return result


def _generate_chosen(goal: str) -> str:
    return _call(CHOSEN_MODEL, [
        {"role": "system", "content": CHOSEN_SYSTEM},
        {"role": "user",   "content": f"GOAL: {goal}\n\nProvide a detailed, structured proposal:"},
    ], max_tok=600, temp=0.72)


def amplify(variants_per_pair: int = DEFAULT_VARIANTS,
            dry_run: bool = False,
            skip_upload: bool = False,
            reset: bool = False):

    Path("data").mkdir(exist_ok=True)
    amp_state = _load_amp_state()
    if reset:
        amp_state = {"amplified_goals": [], "total_variants_written": 0}
        print("  [AMP] State reset.")

    original_pairs = _load_spin()
    if not original_pairs:
        print("  [AMP] No SPIN pairs found. Run ghost_trainer first.")
        return

    # Skip already-amplified pairs
    done_goals = set(amp_state["amplified_goals"])
    to_amplify = [p for p in original_pairs if p.get("goal", "") not in done_goals
                  and p.get("source", "ghost_trainer_spin") != "amplified"]

    print(f"\n+========================================+")
    print(f"|  SOVEREIGN SPIN AMPLIFIER              |")
    print(f"+========================================+")
    print(f"  Original pairs   : {len(original_pairs)}")
    print(f"  Already amplified: {len(done_goals)}")
    print(f"  To amplify now   : {len(to_amplify)}")
    print(f"  Variants/pair    : {variants_per_pair}")
    print(f"  Expected new     : ~{len(to_amplify) * variants_per_pair}")
    print(f"  Upload threshold : {UPLOAD_THRESHOLD}")
    print(f"  Current total    : {_count_spin()}")
    if dry_run:
        print("  MODE             : DRY RUN (nothing written)")
    print()

    if not to_amplify:
        print("  All pairs already amplified. Use --reset to re-run.")
        return

    templates = VARIANT_PROMPTS[:variants_per_pair]
    written = 0
    uploaded = False

    for i, pair in enumerate(to_amplify):
        # ── Demo mode: yield Ollama to live clients ──────────────────────────
        demo_flag = Path("data/demo_mode.flag")
        while demo_flag.exists():
            print("  [AMP DEMO MODE] Paused — Ollama reserved for live demo. Checking in 30s...")
            import time as _time; _time.sleep(30)

        goal     = pair.get("goal", "")
        rejected = pair.get("rejected", "")
        rej_score = pair.get("rejected_score", 0.5)
        domain   = pair.get("domain", "growth")

        print(f"  [{i+1}/{len(to_amplify)}] Amplifying: {goal[:70]}...")

        pair_variants = 0
        for vname, vprompt in templates:
            try:
                # Step 1: Generate variant goal
                var_goal = _generate_variant_goal(goal, vprompt)
                print(f"    [{vname}] -> {var_goal[:65]}...")

                if dry_run:
                    pair_variants += 1
                    continue

                # Step 2: Generate chosen response for variant goal
                chosen = _generate_chosen(var_goal)

                # Step 3: Write new SPIN pair
                new_pair = {
                    "goal":           var_goal,
                    "chosen":         chosen,
                    "rejected":       rejected,
                    "rejected_score": rej_score,
                    "domain":         domain,
                    "source":         "amplified",
                    "parent_goal":    goal,
                    "variant_type":   vname,
                }
                _append_spin(new_pair)
                pair_variants += 1
                written += 1

                # Check threshold
                current_total = _count_spin()
                if current_total >= UPLOAD_THRESHOLD and not uploaded:
                    print(f"\n  [AMP] THRESHOLD REACHED: {current_total} pairs!")
                    _upload_to_hf(skip=skip_upload)
                    uploaded = True

            except Exception as e:
                print(f"    [{vname}] ERROR: {e}")
                continue

        if not dry_run:
            amp_state["amplified_goals"].append(goal)
            amp_state["total_variants_written"] += pair_variants
            _save_amp_state(amp_state)

        print(f"    wrote {pair_variants} variants  |  total: {_count_spin()}\n")

    # Final upload check if threshold not yet hit mid-run
    final_count = _count_spin()
    print(f"\n+========================================+")
    print(f"|  AMPLIFIER COMPLETE                    |")
    print(f"+========================================+")
    print(f"  New variants written : {written}")
    print(f"  Total SPIN pairs     : {final_count}")
    print(f"  Upload threshold     : {UPLOAD_THRESHOLD}")

    if final_count >= UPLOAD_THRESHOLD and not uploaded:
        print(f"  Threshold met — uploading...")
        _upload_to_hf(skip=skip_upload)
    elif final_count < UPLOAD_THRESHOLD:
        remaining = UPLOAD_THRESHOLD - final_count
        print(f"  Need {remaining} more pairs to trigger upload")

    if _slack:
        try:
            msg = (f"*AMPLIFIER COMPLETE*\n"
                   f"Written: {written} new variants\n"
                   f"Total SPIN: {final_count}\n"
                   f"Upload triggered: {'YES' if (uploaded or final_count >= UPLOAD_THRESHOLD) else 'NO'}")
            _slack.send(msg)
        except Exception:
            pass


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.chdir(Path(__file__).parent)

    p = argparse.ArgumentParser(description="Sovereign SPIN Amplifier")
    p.add_argument("--variants",     type=int,  default=DEFAULT_VARIANTS,
                   help=f"Variants per pair (1-5, default {DEFAULT_VARIANTS})")
    p.add_argument("--dry-run",      action="store_true",
                   help="Show plan without writing anything")
    p.add_argument("--skip-upload",  action="store_true",
                   help="Amplify but skip HuggingFace upload")
    p.add_argument("--reset",        action="store_true",
                   help="Reset amplifier state and re-amplify all pairs")
    args = p.parse_args()

    amplify(
        variants_per_pair=min(max(args.variants, 1), 5),
        dry_run=args.dry_run,
        skip_upload=args.skip_upload,
        reset=args.reset,
    )
