"""
pre_train.py — Pre-flight checks before launching RunPod training.

Uploads any missing HF dataset splits, verifies column schemas,
and prints a summary of what will be trained on.

Run:  python pre_train.py            (check + upload if needed)
      python pre_train.py --dry-run  (just report, no uploads)
"""
import argparse, json, os, sys
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"

SYSTEM_PYTHON = r"C:\Users\leer4\AppData\Local\Programs\Python\Python312\python.exe"


def _load_env():
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))

_load_env()
# Disable HuggingFace XET transfer — large uploads fail mid-stream with DNS blips
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
HF_TOKEN  = os.environ.get("HF_TOKEN", "")
HF_REPO   = "tastytator/sovereign-economy"

if not HF_TOKEN:
    print("ERROR: HF_TOKEN not set in .env"); sys.exit(1)


def _to_str(val) -> str:
    """Convert any value (str, dict, list) to a clean training string."""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, dict):
        parts = []
        for k, v in val.items():
            if isinstance(v, (list, tuple)):
                parts.append(f"{k}:")
                for item in v:
                    parts.append(f"  - {item}")
            else:
                parts.append(f"{k}: {v}")
        return "\n".join(parts).strip()
    if isinstance(val, (list, tuple)):
        return "\n".join(str(x) for x in val).strip()
    return str(val).strip()


def _is_good(text: str, min_len: int = 100) -> bool:
    """Quality gate: filter out think-tagged, too-short, or empty responses."""
    if not text or len(text) < min_len:
        return False
    # Filter chain-of-thought leakage
    if "<think>" in text or "<|thinking|>" in text:
        return False
    return True


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for l in path.open(encoding="utf-8") if l.strip())


def _hf_splits() -> dict:
    """Return dict of {config: [splits]} currently on HuggingFace."""
    try:
        from datasets import get_dataset_split_names
        result = {}
        for config in ["dpo", "sft", "agents", "default"]:
            try:
                splits = get_dataset_split_names(HF_REPO, config_name=config, token=HF_TOKEN)
                if splits:
                    result[config] = splits
            except Exception:
                pass
        return result
    except Exception as e:
        print(f"  [HF] Could not list splits: {e}")
        return {}


def upload_spin_dpo(dry_run: bool) -> int:
    """Upload spin_dataset.jsonl as spin_business_dpo split. Returns row count."""
    spin_file = DATA / "spin_dataset.jsonl"
    rows = [json.loads(l) for l in spin_file.open(encoding="utf-8") if l.strip()]

    dpo_rows, skipped = [], 0
    for r in rows:
        goal     = _to_str(r.get("goal") or r.get("instruction") or "")
        prompt   = _to_str(r.get("prompt") or f"GOAL: {goal}\n\nProvide a detailed sovereign strategy:")
        chosen   = _to_str(r.get("chosen",   ""))
        rejected = _to_str(r.get("rejected", ""))
        if not _is_good(chosen) or not _is_good(rejected, min_len=50):
            skipped += 1
            continue
        dpo_rows.append({
            "prompt":   prompt,
            "chosen":   chosen,
            "rejected": rejected,
            "domain":   str(r.get("domain", "")),
        })
    if skipped:
        print(f"  Skipped {skipped} low-quality / think-tagged rows")

    print(f"  SPIN: {len(rows)} raw -> {len(dpo_rows)} clean DPO pairs")
    if not dpo_rows:
        print("  WARNING: No valid SPIN pairs to upload"); return 0

    if not dry_run:
        from datasets import Dataset
        ds = Dataset.from_list(dpo_rows)
        ds.push_to_hub(HF_REPO, config_name="dpo", split="spin_business_dpo",
                       token=HF_TOKEN, private=False)
        print(f"  Pushed {len(ds)} rows -> {HF_REPO} [dpo/spin_business_dpo]")

    return len(dpo_rows)


def upload_bootstrap_dpo(dry_run: bool) -> int:
    """Upload bootstrap_dataset.jsonl as bootstrap_dpo split. Returns row count."""
    bs_file = DATA / "bootstrap_dataset.jsonl"
    rows = [json.loads(l) for l in bs_file.open(encoding="utf-8") if l.strip()]

    dpo_rows, skipped = [], 0
    for r in rows:
        goal     = _to_str(r.get("goal") or r.get("instruction") or "")
        prompt   = _to_str(r.get("prompt") or f"GOAL: {goal}\n\nProvide a detailed sovereign strategy:")
        chosen   = _to_str(r.get("chosen",   ""))
        rejected = _to_str(r.get("rejected", ""))
        if not _is_good(chosen) or not _is_good(rejected, min_len=50):
            skipped += 1
            continue
        dpo_rows.append({
            "prompt":   prompt,
            "chosen":   chosen,
            "rejected": rejected,
            "domain":   str(r.get("domain", "")),
        })
    if skipped:
        print(f"  Skipped {skipped} low-quality rows")

    print(f"  Bootstrap: {len(rows)} raw -> {len(dpo_rows)} clean DPO pairs")
    if not dpo_rows:
        print("  WARNING: No valid bootstrap pairs to upload"); return 0

    if not dry_run:
        from datasets import Dataset
        ds = Dataset.from_list(dpo_rows)
        ds.push_to_hub(HF_REPO, config_name="dpo", split="bootstrap_dpo",
                       token=HF_TOKEN, private=False)
        print(f"  Pushed {len(ds)} rows -> {HF_REPO} [dpo/bootstrap_dpo]")

    return len(dpo_rows)


def _extract_messages_pair(r: dict):
    """Extract (instruction, response) from OpenAI-style messages format — first Q, last A."""
    msgs = r.get("messages", [])
    if not msgs:
        return None, None
    user_msgs = [m["content"] for m in msgs if m.get("role") == "user"]
    asst_msgs = [m["content"] for m in msgs if m.get("role") == "assistant"]
    if not user_msgs or not asst_msgs:
        return None, None
    return _to_str(user_msgs[0]), _to_str(asst_msgs[-1])


def _expand_multiturn(r: dict) -> list[tuple[str, str, str]]:
    """Expand a multi-turn conversation into one SFT pair per assistant turn.

    Turn 1: instruction = U1, response = A1
    Turn 2: instruction = "User: U1\nAssistant: A1\nUser: U2", response = A2
    ...
    """
    msgs = [m for m in r.get("messages", []) if m.get("role") in ("user", "assistant")]
    pairs = []
    history: list[dict] = []   # accumulates [U, A, U, A, ...]

    i = 0
    while i < len(msgs):
        if msgs[i]["role"] != "user":
            i += 1
            continue
        user_content = _to_str(msgs[i]["content"])
        i += 1
        if i >= len(msgs) or msgs[i]["role"] != "assistant":
            continue
        asst_content = _to_str(msgs[i]["content"])
        i += 1

        # Build instruction from prior history + this user turn
        if not history:
            instruction = user_content
        else:
            parts = []
            for h in history:
                role = "User" if h["role"] == "user" else "Avery"
                parts.append(f"{role}: {h['content']}")
            parts.append(f"User: {user_content}")
            instruction = "\n\n".join(parts)

        if len(instruction) >= 20 and _is_good(asst_content):
            pairs.append((instruction, asst_content, "multiturn"))

        history.append({"role": "user",      "content": user_content})
        history.append({"role": "assistant", "content": asst_content})

    return pairs


TRAINING_DATA_DIR = ROOT / "training_data"


def upload_sft_combined(dry_run: bool) -> int:
    """Merge bootstrap + SPIN + KAIROS + mentor data into a single SFT split."""
    rows = []
    for path in [DATA / "bootstrap_dataset.jsonl", DATA / "spin_dataset.jsonl",
                 DATA / "kairos_dataset.jsonl"]:
        if path.exists():
            rows += [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]

    # Mentor training files (messages format)
    SECURITY_DATA_DIR = ROOT / "backend" / "training" / "datasets"
    mentor_files = [
        TRAINING_DATA_DIR / "mentor_training.jsonl",
        TRAINING_DATA_DIR / "avery_mentor_training.jsonl",
        TRAINING_DATA_DIR / "web_research.jsonl",
        TRAINING_DATA_DIR / "iron_mesa_training.jsonl",
        TRAINING_DATA_DIR / "business_mesa_training.jsonl",
        TRAINING_DATA_DIR / "business_docs_training.jsonl",
        TRAINING_DATA_DIR / "domain_research" / "sovereign_nation.jsonl",
        SECURITY_DATA_DIR / "reasoning_chains.jsonl",
        SECURITY_DATA_DIR / "bug_bounty.jsonl",
        SECURITY_DATA_DIR / "adversarial_defense.jsonl",
        SECURITY_DATA_DIR / "cve_patterns.jsonl",
        DATA / "multiturn_dataset.jsonl",        # multi-turn conversations
        DATA / "mentor_pairs.jsonl",             # agent mentor pairs (from mentor_trainer.py)
    ]
    agent_training_dir = TRAINING_DATA_DIR / "agent_training"
    if agent_training_dir.exists():
        mentor_files += list(agent_training_dir.glob("*.jsonl"))

    mentor_rows = []
    for path in mentor_files:
        if path.exists():
            mentor_rows += [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]

    def _normalize_row(r: dict) -> tuple[str, str, str]:
        """Return (instruction, response, domain) from any row schema."""
        # Standard fields first
        if "messages" in r:
            inst, resp = _extract_messages_pair(r)
            return inst or "", resp or "", "mentor"

        # reasoning_chains schema
        if "question" in r and "final_answer" in r:
            inst = _to_str(r["question"])
            parts = []
            if r.get("synthesis"): parts.append(_to_str(r["synthesis"]))
            parts.append(_to_str(r["final_answer"]))
            return inst, "\n\n".join(parts), "security_reasoning"

        # bug_bounty schema
        if "vulnerability_found" in r and "remediation" in r:
            inst = f"Security assessment — {_to_str(r.get('target_system',''))}\n\n{_to_str(r['vulnerability_found'])}"
            resp = f"Impact Assessment:\n{_to_str(r.get('impact_assessment',''))}\n\nRemediation:\n{_to_str(r['remediation'])}"
            return inst, resp, "bug_bounty"

        # adversarial_defense schema
        if "threat_vector" in r and "mitigation_strategy" in r:
            inst = f"Threat: {_to_str(r['threat_vector'])}\n\nExploitation: {_to_str(r.get('exploitation_method',''))}"
            resp = f"Detection:\n{_to_str(r.get('detection_pattern',''))}\n\nMitigation:\n{_to_str(r['mitigation_strategy'])}"
            return inst, resp, "adversarial_defense"

        # cve_patterns schema
        if "vulnerability_pattern" in r and "defensive_lessons" in r:
            cve = r.get("source_cve", "")
            inst = f"Analyze vulnerability: {_to_str(r['vulnerability_pattern'])}" + (f" ({cve})" if cve else "")
            return inst, _to_str(r["defensive_lessons"]), "cve_patterns"

        # Generic fallback
        inst = _to_str(r.get("instruction") or r.get("prompt") or r.get("goal") or "")
        resp = _to_str(r.get("response") or r.get("chosen") or "")
        return inst, resp, str(r.get("domain", "mentor"))

    sft_rows, skipped = [], 0
    for r in rows:
        instruction = _to_str(r.get("goal") or r.get("instruction") or r.get("prompt") or "")
        response    = _to_str(r.get("chosen") or r.get("response") or "")
        if len(instruction) < 20 or not _is_good(response):
            skipped += 1
            continue
        sft_rows.append({"instruction": instruction, "response": response,
                          "domain": str(r.get("domain", ""))})

    for r in mentor_rows:
        # Multi-turn conversations get expanded into one pair per assistant turn
        if "messages" in r and r.get("turns", 0) > 1:
            expanded = _expand_multiturn(r)
            if expanded:
                for inst, resp, dom in expanded:
                    sft_rows.append({"instruction": inst, "response": resp, "domain": dom})
            else:
                skipped += 1
            continue
        instruction, response, domain = _normalize_row(r)
        if not instruction or len(instruction) < 20 or not _is_good(response):
            skipped += 1
            continue
        sft_rows.append({"instruction": instruction, "response": response,
                          "domain": domain})

    if skipped:
        print(f"  Skipped {skipped} low-quality rows")

    base_count = len(rows)
    mentor_count = len(mentor_rows)
    print(f"  SFT combined: {base_count} base + {mentor_count} mentor -> {len(sft_rows)} clean pairs")
    if not sft_rows:
        return 0

    if not dry_run:
        from datasets import Dataset
        ds = Dataset.from_list(sft_rows)
        ds.push_to_hub(HF_REPO, config_name="sft", split="train",
                       token=HF_TOKEN, private=False)
        print(f"  Pushed {len(ds)} rows -> {HF_REPO} [sft/train]")

    return len(sft_rows)


AGENT_NAMES = ["avery", "forge", "oracle", "codex", "sentinel", "nexus"]


def upload_agents_dpo(dry_run: bool) -> int:
    """Upload agents_bootstrap.jsonl as multi-agent DPO split. Returns row count."""
    agents_file = DATA / "agents_bootstrap.jsonl"
    if not agents_file.exists():
        print("  agents_bootstrap.jsonl not found — run: python agents_bootstrap.py")
        return 0

    rows = [json.loads(l) for l in agents_file.open(encoding="utf-8") if l.strip()]
    dpo_rows, skipped = [], 0

    for r in rows:
        prompt   = _to_str(r.get("prompt") or r.get("instruction") or r.get("goal") or "")
        chosen   = _to_str(r.get("chosen",   ""))
        rejected = _to_str(r.get("rejected", ""))
        agent    = str(r.get("agent", "unknown"))
        if not _is_good(chosen) or not _is_good(rejected, min_len=50):
            skipped += 1
            continue
        dpo_rows.append({
            "prompt":   prompt,
            "chosen":   chosen,
            "rejected": rejected,
            "agent":    agent,
            "domain":   agent,
        })
    if skipped:
        print(f"  Skipped {skipped} low-quality rows")

    per_agent = {a: sum(1 for r in dpo_rows if r["agent"] == a) for a in AGENT_NAMES}
    print(f"  Agents: {len(rows)} raw -> {len(dpo_rows)} clean DPO pairs")
    for a, n in per_agent.items():
        print(f"    {a:<10}: {n:>3} pairs")

    if not dpo_rows:
        print("  WARNING: No valid agent pairs to upload"); return 0

    if not dry_run:
        from datasets import Dataset
        ds = Dataset.from_list(dpo_rows)
        ds.push_to_hub(HF_REPO, config_name="agents", split="train",
                       token=HF_TOKEN, private=False)
        print(f"  Pushed {len(ds)} rows -> {HF_REPO} [agents/train]")

    return len(dpo_rows)


def main(dry_run: bool):
    print("\n+============================================+")
    print("|   SOVEREIGN PRE-TRAIN PREFLIGHT CHECK      |")
    print("+============================================+\n")
    print(f"  HF repo  : {HF_REPO}")
    print(f"  Dry run  : {dry_run}")
    print()

    # ── Local data inventory ──────────────────────────────────────────────────
    spin_count      = _count_jsonl(DATA / "spin_dataset.jsonl")
    bootstrap_count = _count_jsonl(DATA / "bootstrap_dataset.jsonl")
    kairos_count    = _count_jsonl(DATA / "kairos_dataset.jsonl")
    agents_count    = _count_jsonl(DATA / "agents_bootstrap.jsonl")
    mentor_count    = (
        _count_jsonl(ROOT / "training_data" / "mentor_training.jsonl") +
        _count_jsonl(ROOT / "training_data" / "avery_mentor_training.jsonl") +
        sum(_count_jsonl(p) for p in (ROOT / "training_data" / "agent_training").glob("*.jsonl")
            if (ROOT / "training_data" / "agent_training").exists()) +
        _count_jsonl(ROOT / "training_data" / "domain_research" / "sovereign_nation.jsonl")
    )
    web_count          = _count_jsonl(ROOT / "training_data" / "web_research.jsonl")
    iron_mesa_count    = _count_jsonl(ROOT / "training_data" / "iron_mesa_training.jsonl")
    business_mesa_count = _count_jsonl(ROOT / "training_data" / "business_mesa_training.jsonl")
    business_docs_count = _count_jsonl(ROOT / "training_data" / "business_docs_training.jsonl")
    total_local     = spin_count + bootstrap_count + kairos_count + agents_count + mentor_count + web_count + iron_mesa_count + business_mesa_count + business_docs_count
    print(f"[LOCAL DATA]")
    print(f"  spin_dataset.jsonl         : {spin_count:,} pairs")
    print(f"  bootstrap_dataset.jsonl    : {bootstrap_count:,} pairs")
    print(f"  kairos_dataset.jsonl       : {kairos_count:,} pairs")
    print(f"  agents_bootstrap.jsonl     : {agents_count:,} pairs")
    print(f"  mentor_training (real)     : {mentor_count:,} pairs")
    print(f"  web_research.jsonl         : {web_count:,} pairs")
    print(f"  iron_mesa_training.jsonl   : {iron_mesa_count:,} pairs")
    print(f"  business_mesa_training.jsonl : {business_mesa_count:,} pairs")
    print(f"  business_docs_training.jsonl : {business_docs_count:,} pairs")
    print(f"  Total                        : {total_local:,} pairs")
    print()

    if total_local < 50:
        print("ERROR: Not enough training data.")
        print("  For Avery: run the flywheel first.")
        print("  For agents: run python agents_bootstrap.py")
        sys.exit(1)

    # ── HF splits ────────────────────────────────────────────────────────────
    print("[HUGGINGFACE SPLITS]")
    existing = _hf_splits()
    for cfg, splits in existing.items():
        print(f"  {cfg}: {splits}")
    if not existing:
        print("  (none)")
    print()

    if not dry_run:
        print("[UPLOADING TO HUGGINGFACE...]")
    else:
        print("[DRY RUN — no uploads]")
    print()

    step = 1
    if bootstrap_count > 0:
        print(f"[{step}/4] bootstrap_dpo  (Avery)")
        upload_bootstrap_dpo(dry_run)
        step += 1

    if spin_count >= 50:
        print(f"[{step}/4] spin_business_dpo  (Avery)")
        upload_spin_dpo(dry_run)
        step += 1

    print(f"[{step}/4] train (combined SFT)")
    upload_sft_combined(dry_run)
    step += 1

    if agents_count > 0:
        print(f"[{step}/4] agents/train  (all 6 agents)")
        upload_agents_dpo(dry_run)

    print()
    print("[PREFLIGHT COMPLETE]")
    print(f"  Config 'dpo'    splits: bootstrap_dpo, spin_business_dpo")
    print(f"  Config 'sft'    splits: train")
    print(f"  Config 'agents' splits: train  (avery/forge/oracle/codex/sentinel/nexus)")
    if dry_run:
        print("  (dry-run: run without --dry-run to upload)")
    else:
        print("  Ready:")
        print("    Avery:   python runpod_launcher.py --mode orpo --split bootstrap_dpo")
        print("    FORGE:   TRAIN_AGENT=forge python runpod_launcher.py --mode orpo")
        print("    All:     TRAIN_AGENT=all python runpod_launcher.py --mode orpo")
    print()


def gen_datasets(dry_run: bool):
    """Run kairos + mentor generators to refresh training data before upload."""
    import subprocess
    generators = [
        [sys.executable, str(ROOT / "kairos_dataset_gen.py"), "--pairs", "100", "--append"],
        [sys.executable, str(ROOT / "mentor_trainer.py"), "--agent", "all", "--pairs", "3"],
    ]
    for cmd in generators:
        label = Path(cmd[1]).stem
        if dry_run:
            cmd = cmd + ["--dry-run"]
        print(f"\n[GEN] {label}...")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  WARNING: {label} exited {result.returncode} — continuing")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Pre-flight upload to HuggingFace before RunPod training."
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would happen without uploading or generating")
    ap.add_argument("--gen-datasets", action="store_true",
                    help="Run kairos + mentor generators first to refresh training data")
    args = ap.parse_args()
    if args.gen_datasets:
        gen_datasets(args.dry_run)
    main(args.dry_run)

