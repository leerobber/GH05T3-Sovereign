"""
avery_flywheel.py — Sovereign Continuous Training Flywheel

Endless loop:
  1. DATA   — run continuous_learner until SPIN_THRESHOLD new pairs accumulate + upload to HF
  2. TRAIN  — launch RunPod, wait for LoRA to land on HuggingFace
  3. DEPLOY — merge LoRA -> GGUF -> ollama create avery-sovereign
  4. NOTIFY — Slack ping, then loop

Run:  python avery_flywheel.py
      python avery_flywheel.py --spin-threshold 200 --skip-deploy
"""
import argparse, gc, json, os, subprocess, sys, time
from pathlib import Path

# ── Paths / config ─────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
ENV_FILE    = ROOT / ".env"
STATE_FILE  = ROOT / "data" / "flywheel_state.json"
MERGED_DIR  = ROOT / "avery-merged"
GGUF_PATH   = ROOT / "avery-sovereign-q8.gguf"
MODELFILE   = ROOT / "Modelfile.avery"
LLAMA_CPP   = Path("C:/llama.cpp/convert_hf_to_gguf.py")

BASE_MODEL  = "Qwen/Qwen2-7B-Instruct"
LORA_REPO   = "tastytator/avery-sovereign-lora"
OLLAMA_NAME = "avery-sovereign"

SPIN_THRESHOLD_DEFAULT = 150  # new SPIN pairs before triggering a training run
BATCH_CYCLES           = 1    # check upload threshold every cycle

# All agents that can be trained — Avery runs every cycle, others run every N cycles
ALL_AGENTS = ["avery", "forge", "oracle", "codex", "sentinel", "nexus"]
AGENT_TRAIN_EVERY = 3  # train specialist agents every 3 Avery cycles


# ── Env loader ─────────────────────────────────────────────────────────────────
def _load_env():
    env = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"\'')
                os.environ.setdefault(k.strip(), v.strip().strip('"\''))
    return env


# ── State ──────────────────────────────────────────────────────────────────────
def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"runs": 0, "last_run_ts": 0, "total_spin_trained": 0}


def _save_state(s: dict):
    STATE_FILE.parent.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(s, indent=2), encoding="utf-8")


# ── Slack helper ───────────────────────────────────────────────────────────────
def _slack(msg: str):
    try:
        import slack_notify as sn
        sn.post("learner", msg)
    except Exception:
        pass


# ── Quality gate ──────────────────────────────────────────────────────────────
def _quality_gate(row: dict) -> bool:
    """Return True if the SPIN row is clean enough to train on."""
    chosen = row.get("chosen", "")
    goal   = row.get("goal", "")
    if "<think>" in chosen or "<|thinking|>" in chosen:
        return False
    if len(chosen) < 150 or len(goal) < 20:
        return False
    sovereign_terms = ["sovereign", "strategy", "proposal", "platform", "kairos",
                       "business", "revenue", "market", "user", "product", "cost",
                       "service", "customer", "pricing", "growth", "launch"]
    if not any(t in (goal + chosen).lower() for t in sovereign_terms):
        return False
    return True


# ── SPIN count helper ──────────────────────────────────────────────────────────
def _spin_count() -> int:
    spin_file = ROOT / "data" / "spin_dataset.jsonl"
    if not spin_file.exists():
        return 0
    try:
        return sum(1 for _ in spin_file.open(encoding="utf-8"))
    except Exception:
        return 0


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — DATA
# ══════════════════════════════════════════════════════════════════════════════
def phase_data(spin_threshold: int):
    """Run continuous_learner in BATCH_CYCLES chunks until one HF upload fires."""
    print("\n+--- FLYWHEEL: DATA PHASE ---+")
    print(f"  Target: {spin_threshold} new SPIN pairs -> upload to HF")

    # ── Mentor step: Claude teaches all agents before SPIN loop ──────────────
    try:
        from mentor_trainer import run_mentor_session
        _slack("[Flywheel] Mentor session starting — Claude teaching all agents...")
        new_pairs = run_mentor_session(pairs_per_agent=5)
        _slack(f"[Flywheel] Mentor done. {new_pairs} new teaching pairs added.")
    except Exception as e:
        print(f"  [MENTOR WARNING] {e}")

    # Import learner internals directly so we can check upload state
    sys.path.insert(0, str(ROOT))
    import continuous_learner as cl
    from ghost_trainer import Trainer

    cl_state = cl._load_state()
    baseline = cl_state.get("last_upload_count", 0)
    print(f"  Baseline upload count: {baseline}  Current SPIN: {_spin_count()}")

    trainer = None
    current_domain = cl.DOMAIN_ROTATION[cl_state["domain_index"] % len(cl.DOMAIN_ROTATION)]

    while True:
        # Rotate domain if needed
        if cl_state["domain_cycles"] >= cl.DEFAULT_CYCLES_PER_DOMAIN:
            prev = current_domain
            cl_state["domain_index"]  = (cl_state["domain_index"] + 1) % len(cl.DOMAIN_ROTATION)
            cl_state["domain_cycles"] = 0
            current_domain = cl.DOMAIN_ROTATION[cl_state["domain_index"]]
            trainer = None
            print(f"  *** DOMAIN ROTATE: {prev} -> {current_domain} ***")

        if trainer is None or trainer.domain != current_domain:
            trainer = Trainer(domain=current_domain)

        # Rescan repos periodically
        if cl_state["total_cycles"] - cl_state.get("last_scan_cycle", -1) >= cl.RESCAN_EVERY:
            cl.rescan_repos(cl_state["total_cycles"])
            cl_state["last_scan_cycle"] = cl_state["total_cycles"]

        # Run a batch of cycles
        for _ in range(BATCH_CYCLES):
            trainer.run_cycle()
            cl_state["total_cycles"]  += 1
            cl_state["domain_cycles"] += 1
            time.sleep(0.3)

        # Check if upload threshold reached
        spin_now = _spin_count()
        new_pairs = spin_now - cl_state.get("last_upload_count", 0)

        if new_pairs >= spin_threshold:
            # Quality gate: filter contaminated rows before upload
            spin_file = ROOT / "data" / "spin_dataset.jsonl"
            if spin_file.exists():
                raw_rows = [json.loads(l) for l in spin_file.open(encoding="utf-8") if l.strip()]
                clean_rows = [r for r in raw_rows if _quality_gate(r)]
                removed = len(raw_rows) - len(clean_rows)
                if removed:
                    spin_file.write_text(
                        "\n".join(json.dumps(r, ensure_ascii=False) for r in clean_rows) + "\n",
                        encoding="utf-8"
                    )
                    print(f"  [GATE] Removed {removed} contaminated rows ({len(clean_rows)} clean remain)")
                    spin_now = len(clean_rows)
                    cl_state["last_upload_count"] = max(0, cl_state.get("last_upload_count", 0) - removed)

            print(f"  SPIN: {spin_now} clean pairs ready for upload")
            cl.upload_spin_to_hf(cl_state)
            cl._save_state(cl_state)
            uploaded = cl_state.get("last_upload_count", 0)
            print(f"  Upload complete. Total on HF: {uploaded} pairs.")
            _slack(f"[Flywheel] Data phase done. {uploaded} clean SPIN pairs on HF. Starting training...")
            return

        print(f"  SPIN: {spin_now} total  ({new_pairs} new, need {spin_threshold})")

        cl._save_state(cl_state)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — TRAIN
# ══════════════════════════════════════════════════════════════════════════════
def phase_train(run_number: int = 1, train_all_agents: bool = False):
    """Launch RunPod, wait for training to complete.

    Every cycle: trains Avery on the latest SPIN + bootstrap data.
    Every AGENT_TRAIN_EVERY cycles: also trains all specialist agents.
    """
    print("\n+--- FLYWHEEL: TRAIN PHASE ---+")
    from runpod_launcher import launch

    # ── Always train Avery first ──────────────────────────────────────────
    print("  [1] Training AVERY (business strategist)...")
    _slack("[Flywheel] Launching RunPod training — Avery...")
    os.environ["TRAIN_AGENT"] = "avery"
    os.environ["TRAIN_MODE"]  = "sft"
    if "TRAIN_SPLIT" in os.environ:
        del os.environ["TRAIN_SPLIT"]
    launch(train_mode="sft", train_split=None)
    print("  Avery LoRA pushed to HuggingFace.")
    _slack(f"[Flywheel] Avery done. LoRA at {LORA_REPO}.")

    # ── Train specialist agents every N cycles OR if forced ───────────────
    should_train_specialists = train_all_agents or (run_number % AGENT_TRAIN_EVERY == 0)
    agents_data = ROOT / "data" / "agents_bootstrap.jsonl"

    if should_train_specialists and agents_data.exists():
        agent_count = sum(1 for _ in agents_data.open(encoding="utf-8"))
        print(f"\n  [2] Training ALL SPECIALIST AGENTS ({agent_count} total pairs)...")
        _slack(f"[Flywheel] Training all specialist agents (run #{run_number})...")

        for agent in [a for a in ALL_AGENTS if a != "avery"]:
            print(f"\n  --- {agent.upper()} ---")
            _slack(f"[Flywheel] Training {agent}...")
            os.environ["TRAIN_AGENT"] = agent
            os.environ["TRAIN_MODE"]  = "sft"
            if "TRAIN_SPLIT" in os.environ:
                del os.environ["TRAIN_SPLIT"]
            try:
                launch(train_mode="sft", train_split=None)
                print(f"  {agent} LoRA pushed to HuggingFace.")
                _slack(f"[Flywheel] {agent} done.")
            except Exception as e:
                print(f"  WARNING: {agent} training failed: {e}")
                _slack(f"[Flywheel] WARNING: {agent} training failed: {e}")

        print("\n  All specialist agents trained.")
        _slack("[Flywheel] All agents trained. Starting deploy...")
    elif should_train_specialists and not agents_data.exists():
        print("  NOTE: agents_bootstrap.jsonl not found — skipping specialist training.")
        print("  Run: python agents_bootstrap.py   to generate agent data.")
        _slack("[Flywheel] Specialist training skipped — no agents_bootstrap.jsonl.")
    else:
        cycles_until = AGENT_TRAIN_EVERY - (run_number % AGENT_TRAIN_EVERY)
        print(f"  NOTE: Specialist agents train every {AGENT_TRAIN_EVERY} cycles "
              f"({cycles_until} cycle(s) until next).")

    _slack(f"[Flywheel] Training phase done. Starting deploy...")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — DEPLOY
# ══════════════════════════════════════════════════════════════════════════════
def phase_deploy():
    """Merge LoRA into base model, convert to GGUF, reload Ollama."""
    print("\n+--- FLYWHEEL: DEPLOY PHASE ---+")

    env = _load_env()
    hf_token = env.get("HF_TOKEN", "")

    # ── Merge ──
    print("  [1/3] Merging LoRA into base model (CPU, may take ~5 min)...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    MERGED_DIR.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.float16,
        device_map="cpu", token=hf_token, low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, LORA_REPO, token=hf_token)
    model = model.merge_and_unload()
    model.save_pretrained(str(MERGED_DIR), safe_serialization=True)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, token=hf_token)
    tok.save_pretrained(str(MERGED_DIR))
    del model, tok
    gc.collect()
    print("  Merge done.")

    # ── Convert to GGUF ──
    print("  [2/3] Converting to GGUF Q8_0...")
    subprocess.run([
        sys.executable, str(LLAMA_CPP),
        str(MERGED_DIR),
        "--outfile", str(GGUF_PATH),
        "--outtype", "q8_0",
    ], check=True)
    print(f"  GGUF written: {GGUF_PATH} ({GGUF_PATH.stat().st_size / 1e9:.2f} GB)")

    # ── Reload Ollama ──
    print(f"  [3/3] Loading {OLLAMA_NAME} into Ollama...")
    subprocess.run(
        ["ollama", "create", OLLAMA_NAME, "-f", str(MODELFILE)],
        check=True
    )
    print(f"  {OLLAMA_NAME} is live.")
    _slack(f"[Flywheel] Deploy done. `ollama run {OLLAMA_NAME}` is updated.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main(spin_threshold: int, skip_deploy: bool, train_all_agents: bool):
    _load_env()
    state = _load_state()

    print("\n+============================================+")
    print("|   SOVEREIGN FLYWHEEL                       |")
    print("|   DATA -> TRAIN -> DEPLOY -> repeat        |")
    print("+============================================+")
    print(f"  Completed runs    : {state['runs']}")
    print(f"  SPIN threshold    : {spin_threshold}")
    print(f"  All agents        : {'every run' if train_all_agents else f'every {AGENT_TRAIN_EVERY} cycles'}")
    if skip_deploy:
        print("  Deploy phase      : SKIPPED")

    _slack(
        f"[Flywheel] Starting. Threshold={spin_threshold}. "
        f"Run #{state['runs']+1}. AllAgents={train_all_agents}"
    )

    try:
        while True:
            run_start  = time.time()
            run_number = state["runs"] + 1

            phase_data(spin_threshold)
            phase_train(run_number=run_number, train_all_agents=train_all_agents)

            if not skip_deploy:
                phase_deploy()

            elapsed = (time.time() - run_start) / 60
            state["runs"] += 1
            state["last_run_ts"] = time.time()
            state["total_spin_trained"] = _spin_count()
            _save_state(state)

            print(f"\n*** RUN {state['runs']} COMPLETE in {elapsed:.1f} min ***")
            _slack(
                f"[Flywheel] Run #{state['runs']} complete in {elapsed:.0f}min. "
                f"SPIN total: {state['total_spin_trained']}. Looping..."
            )

    except KeyboardInterrupt:
        _save_state(state)
        print("\n-- FLYWHEEL STOPPED --")
        _slack("[Flywheel] Stopped by user.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Sovereign Training Flywheel")
    ap.add_argument("--spin-threshold", type=int, default=SPIN_THRESHOLD_DEFAULT,
                    help=f"new SPIN pairs before training triggers (default: {SPIN_THRESHOLD_DEFAULT})")
    ap.add_argument("--skip-deploy", action="store_true",
                    help="skip the local merge/GGUF/ollama step (train only)")
    ap.add_argument("--train-all-agents", action="store_true",
                    help="train all 6 agents every cycle (default: every 3 cycles)")
    args = ap.parse_args()
    main(args.spin_threshold, args.skip_deploy, args.train_all_agents)
