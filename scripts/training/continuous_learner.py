"""
continuous_learner.py â€” GH05T3 autonomous business strategy learning loop.

What it does:
  1. Scans all leerobber repos â†’ updates capability map
  2. Rotates through training domains (business first, then cfo, content, ml_engineer, core)
  3. Runs ghost_trainer cycles to generate SPIN pairs
  4. When enough SPIN pairs accumulate, uploads to HuggingFace for RunPod training

Run: python continuous_learner.py
     python continuous_learner.py --cycles 50 --domain business
     python continuous_learner.py --rescan-every 20 --spin-threshold 200
"""
import argparse, json, os, sys, time
from pathlib import Path
try:
    import slack_notify as _slack
except ImportError:
    _slack = None
try:
    import economy_bridge as _eco
except ImportError:
    _eco = None

# â”€â”€ Config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DOMAIN_ROTATION = [
    "business",         # venture design: repo-grounded business creation
    "sales",            # pipeline, outreach, closing sovereign AI products
    "product_strategy", # PRDs, roadmap, user research, MVP design
    "growth",           # acquisition, viral loops, retention, SEO
    "cfo",              # financial modeling and capital strategy
    "content",          # marketing and content for sovereign businesses
    "legal_ip",         # IP protection, contracts, entity structure
    "ops",              # SOPs, hiring, vendor management, scaling systems
    "ml_engineer",      # technical deepening and MLOps
    "frontier",         # AI-native languages, agentic OS, post-human computing
    "core",             # self-improvement to sharpen the loop itself
]

DEFAULT_CYCLES_PER_DOMAIN = 30    # cycles before rotating
RESCAN_EVERY               = 20   # re-scan repos every N cycles
SPIN_UPLOAD_THRESHOLD      = 150  # upload SPIN pairs to HF when this many accumulate
STATE_FILE                 = Path("data/continuous_state.json")
SPIN_FILE                  = Path("data/spin_dataset.jsonl")
HF_DATASET_NAME            = "tastytator/sovereign-economy"


# â”€â”€ State persistence â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "total_cycles":        0,
        "domain_index":        0,
        "domain_cycles":       0,
        "last_scan_cycle":     -1,
        "last_upload_count":   0,
        "spin_uploads":        0,
        "session_start":       time.time(),
    }


def _save_state(state: dict):
    Path("data").mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


# â”€â”€ Repo scanning â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def rescan_repos(cycle: int = 0):
    print("\n  [SCAN] Updating repo capability map...")
    try:
        from repo_scanner import scan_all_repos
        result = scan_all_repos()
        active = sum(1 for v in result.values() if isinstance(v, dict) and v.get("exists"))
        print(f"  [SCAN] {active} repos active")
        if _slack:
            _slack.notify_scan(active, cycle)
        if _eco:
            _eco.on_repo_scan(active)
    except Exception as e:
        print(f"  [SCAN] ERROR: {e}")


# â”€â”€ SPIN upload â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _count_spin() -> int:
    if not SPIN_FILE.exists():
        return 0
    try:
        return sum(1 for _ in SPIN_FILE.open(encoding="utf-8"))
    except Exception:
        return 0


def upload_spin_to_hf(state: dict):
    """Upload new SPIN pairs to HuggingFace dataset for RunPod training."""
    count = _count_spin()
    new_pairs = count - state.get("last_upload_count", 0)
    if new_pairs <= 0:
        print("  [UPLOAD] No new SPIN pairs to upload")
        return

    print(f"\n  [UPLOAD] Pushing {new_pairs} new SPIN pairs to HuggingFace...")
    try:
        from datasets import load_dataset, Dataset
        import json as _json

        hf_token = os.environ.get("HF_TOKEN", "")
        if not hf_token:
            # Try reading from .env
            env_path = Path(__file__).parent / ".env"
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    if line.startswith("HF_TOKEN="):
                        hf_token = line.split("=", 1)[1].strip().strip('"\'')

        if not hf_token:
            print("  [UPLOAD] HF_TOKEN not set â€” skipping upload (set HF_TOKEN in .env)")
            return

        rows = []
        with SPIN_FILE.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(_json.loads(line))
                except Exception:
                    continue

        # Convert SPIN pairs to instruction-tuning format
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
                "source":      "ghost_trainer_spin",
                "domain":      row.get("domain", "business"),
            })

        if not examples:
            print("  [UPLOAD] No valid SPIN pairs found")
            return

        ds = Dataset.from_list(examples)
        # Push as a new split or append to existing
        ds.push_to_hub(
            HF_DATASET_NAME,
            split="spin_business",
            token=hf_token,
            private=False,
        )
        state["last_upload_count"] = count
        state["spin_uploads"]     += 1
        print(f"  [UPLOAD] Done. Total uploads: {state['spin_uploads']}")
        if _slack:
            _slack.notify_spin_upload(new_pairs, count, state["spin_uploads"])

    except ImportError:
        print("  [UPLOAD] datasets not installed â€” pip install datasets")
    except Exception as e:
        print(f"  [UPLOAD] ERROR: {e}")


# â”€â”€ Main loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run(cycles_per_domain: int = DEFAULT_CYCLES_PER_DOMAIN,
        total_cap: int | None = None,
        domain_override: str | None = None,
        rescan_every: int = RESCAN_EVERY,
        spin_threshold: int = SPIN_UPLOAD_THRESHOLD):

    # AUTO-DISABLED by GH05T3 aggressive engine: from ghost_trainer import Trainer
    pass  # safe placeholder

    state = _load_state()
    Path("data").mkdir(exist_ok=True)

    print("\n+=====================================================+")
    print("|  GH05T3 CONTINUOUS BUSINESS LEARNER                |")
    print("|  business -> sales -> product -> growth -> cfo      |")
    print("|  content -> legal -> ops -> ml_engineer -> core    |")
    print("|  Repo scanning: ON    SPIN upload: ON               |")
    print("+=====================================================+")
    print(f"  state: total_cycles={state['total_cycles']}  "
          f"domain_idx={state['domain_index']}  "
          f"spin_uploads={state['spin_uploads']}\n")

    # Initial scan
    rescan_repos(state["total_cycles"])

    trainer: Trainer | None = None
    current_domain = domain_override or DOMAIN_ROTATION[state["domain_index"] % len(DOMAIN_ROTATION)]

    global_cycle    = 0
    consecutive_err = 0
    MAX_CONSECUTIVE = 10   # stop only if 10 cycles in a row all fail

    import traceback as _tb

    def _heartbeat():
        """Write a timestamp so the supervisor / external monitor can detect hangs."""
        try:
            Path("data/learner_heartbeat.json").write_text(
                json.dumps({"ts": time.time(), "total_cycles": state["total_cycles"],
                            "domain": current_domain}),
                encoding="utf-8",
            )
        except Exception:
            pass

    try:
        while total_cap is None or global_cycle < total_cap:

            # Rotate domain when per-domain quota exhausted
            if not domain_override and state["domain_cycles"] >= cycles_per_domain:
                prev_domain = current_domain
                state["domain_index"]  = (state["domain_index"] + 1) % len(DOMAIN_ROTATION)
                state["domain_cycles"] = 0
                current_domain = DOMAIN_ROTATION[state["domain_index"]]
                trainer = None  # rebuild trainer for new domain
                print(f"\n  *** DOMAIN ROTATE -> {current_domain.upper()} ***\n")
                if _slack:
                    _slack.notify_domain_rotation(
                        prev_domain, current_domain,
                        state["total_cycles"], _count_spin()
                    )

            # â”€â”€ Demo mode: yield Ollama to live clients â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            demo_flag = Path("data/demo_mode.flag")
            if demo_flag.exists():
                print("  [DEMO MODE] Paused â€” Ollama reserved for live demo. Checking again in 30s...")
                # Write heartbeat while paused so supervisor doesn't kill/restart us
                try:
                    Path("data/learner_heartbeat.json").write_text(
                        json.dumps({"ts": time.time(), "total_cycles": state["total_cycles"],
                                    "status": "paused_demo_mode"}),
                        encoding="utf-8",
                    )
                except Exception:
                    pass
                time.sleep(30)
                continue

            # Build trainer for current domain
            if trainer is None or trainer.domain != current_domain:
                trainer = Trainer(domain=current_domain)

            # Rescan repos periodically
            if state["total_cycles"] - state["last_scan_cycle"] >= rescan_every:
                rescan_repos(state["total_cycles"])
                state["last_scan_cycle"] = state["total_cycles"]

            # â”€â”€ Run one training cycle (isolated â€” never kills the loop) â”€â”€â”€â”€â”€
            try:
                result = trainer.run_cycle()
                consecutive_err = 0   # reset on success

                state["total_cycles"]  += 1
                state["domain_cycles"] += 1
                global_cycle           += 1

                # Upload SPIN when threshold hit
                spin_count = _count_spin()
                if spin_count - state.get("last_upload_count", 0) >= spin_threshold:
                    upload_spin_to_hf(state)

                _save_state(state)
                _heartbeat()

            except KeyboardInterrupt:
                raise   # let outer handler catch this

            except Exception as cycle_err:
                consecutive_err += 1
                err_msg = f"Cycle error #{consecutive_err}: {cycle_err}"
                print(f"\n  [CYCLE ERR] {err_msg}")
                _tb.print_exc()

                # Still advance counters so state doesn't loop on same bad goal
                state["total_cycles"]  += 1
                state["domain_cycles"] += 1
                global_cycle           += 1
                _save_state(state)
                _heartbeat()

                if _slack:
                    try:
                        _slack.post("continuous-learner",
                                    f":warning: {err_msg} â€” continuing")
                    except Exception:
                        pass

                if consecutive_err >= MAX_CONSECUTIVE:
                    print(f"\n  [ABORT] {MAX_CONSECUTIVE} consecutive errors â€” "
                          f"rebuilding trainer and waiting 60s before retry")
                    trainer = None           # force fresh Trainer next cycle
                    consecutive_err = 0
                    time.sleep(60)
                else:
                    time.sleep(5)            # short pause before retrying
                continue

            # Brief breathing room between successful cycles
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n-- CONTINUOUS LEARNER STOPPED --")
        _save_state(state)

    spin_total = _count_spin()
    print(f"\n-- SESSION DONE --")
    print(f"  Cycles this session : {global_cycle}")
    print(f"  Total cycles ever   : {state['total_cycles']}")
    print(f"  SPIN pairs total    : {spin_total}")
    print(f"  HF uploads          : {state['spin_uploads']}")
    if _slack:
        _slack.notify_session_done(global_cycle, state["total_cycles"],
                                   spin_total, state["spin_uploads"])


# â”€â”€ Entry point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GH05T3 Continuous Business Learner")
    ap.add_argument("--cycles", type=int, default=None,
                    help="total cycles to run (default: infinite)")
    ap.add_argument("--cycles-per-domain", type=int, default=DEFAULT_CYCLES_PER_DOMAIN,
                    help=f"cycles before rotating domain (default: {DEFAULT_CYCLES_PER_DOMAIN})")
    ap.add_argument("--domain", default=None,
                    help="lock to one domain instead of rotating")
    ap.add_argument("--rescan-every", type=int, default=RESCAN_EVERY,
                    help=f"rescan repos every N cycles (default: {RESCAN_EVERY})")
    ap.add_argument("--spin-threshold", type=int, default=SPIN_UPLOAD_THRESHOLD,
                    help=f"upload to HF after N new SPIN pairs (default: {SPIN_UPLOAD_THRESHOLD})")
    ap.add_argument("--scan-only", action="store_true",
                    help="just scan repos and print capability map, then exit")
    args = ap.parse_args()

    if args.scan_only:
        rescan_repos()
        cap = Path("data/repo_capabilities.json")
        if cap.exists():
            data = json.loads(cap.read_text())
            print("\n--- Capability Summary ---")
            print(data.get("_summary", "(empty)"))
        sys.exit(0)

    run(
        cycles_per_domain=args.cycles_per_domain,
        total_cap=args.cycles,
        domain_override=args.domain,
        rescan_every=args.rescan_every,
        spin_threshold=args.spin_threshold,
    )
