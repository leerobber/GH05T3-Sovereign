#!/usr/bin/env python3
"""
GH05T3 DPO Trainer — Lightning AI Studio (L4 / A10G, 24 GB VRAM)

Direct Preference Optimization on top of the GH05T3 SFT LoRA adapter.

APPROACH — single-model DPO:
  Load SFT adapter as a PEFT model.  DPOTrainer with ref_model=None uses the
  same model with adapter layers disabled as the implicit reference.
  Only ONE 4-bit model lives in GPU memory → fits easily on L4/A10G 24 GB.

RULES BAKED IN (CLAUDE.md — permanent, never revert):
  RULE 1  DO NOT cast LoRA adapters to fp16 after get_peft_model()
          PEFT fp32 adapters + GradScaler fp16 → "Attempting to unscale FP16
          gradients" crash.  AMP autocast handles dtype during forward; the
          adapter master weights must stay fp32.
  RULE 2  gradient_checkpointing_kwargs={"use_reentrant": False}
          use_reentrant=True reruns the forward pass during backward.  With
          fp16 + enable_input_require_grads() hooks this creates NaN gradients
          from step 1 → loss=0.0 forever.
  RULE 3  assert 0.3 < loss < 10 before model.save_pretrained()
          A collapsed adapter is indistinguishable from a good one on disk.

Quick start (Lightning AI Studio terminal):
    bash backend/training/lightning_setup.sh          # one-time dep install
    python backend/training/ghost_trainer.py          # full DPO run (~2-3 h L4)
    python backend/training/ghost_trainer.py --smoke  # 30-step sanity check

Sync adapter back to TatorTot via Tailscale:
    rsync -avz ./dpo_adapter/ leer4@100.94.227.81:~/gh05t3/backend/models/gh05t3_dpo_adapter/
Or via SSH hostname:
    scp -r /teamspace/studios/this_studio/dpo_adapter/ leer4@tail1457e2.ts.net:~/gh05t3/backend/models/
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*use_reentrant.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*tokenizers.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gh05t3-dpo")

# ── paths ────────────────────────────────────────────────────────────────────────────────
REPO        = Path(__file__).resolve().parent.parent.parent
DATA_DIR    = REPO / "backend" / "data" / "training"
SFT_ADAPTER = REPO / "backend" / "models" / "gh05t3_lora_adapter"

_IN_LIGHTNING = Path("/teamspace").exists()

DPO_OUT  = Path(os.environ.get(
    "DPO_OUT",
    "/teamspace/studios/this_studio/dpo_adapter" if _IN_LIGHTNING
    else str(REPO / "backend" / "models" / "gh05t3_dpo_adapter"),
))
CKPT_DIR = Path(os.environ.get(
    "DPO_CKPT",
    "/teamspace/studios/this_studio/dpo_checkpoints" if _IN_LIGHTNING
    else str(REPO / "backend" / "training" / "checkpoints" / "dpo"),
))

# ── config ────────────────────────────────────────────────────────────────────────────────────
MODEL_ID   = os.environ.get("GH05T3_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
LORA_RANK  = 16
LORA_ALPHA = 32
DPO_BETA   = float(os.environ.get("DPO_BETA",  "0.1"))    # KL penalty temperature
MAX_STEPS  = int(os.environ.get("DPO_STEPS",  "300"))     # ~2-3 h on L4
LR         = float(os.environ.get("DPO_LR",    "5e-6"))   # DPO uses 10-50× lower LR than SFT
MAX_GRAD   = 0.3
WARMUP     = 30
MAX_LEN    = 1024    # prompt + chosen/rejected combined
MAX_PROMPT = 512
BATCH      = 2
GRAD_ACCUM = 4       # effective batch = 8

SYSTEM = (
    "You are GH05T3, an autonomous security and reasoning agent. "
    "You think carefully, reason step-by-step, and always prioritize "
    "detection and defense over exploitation."
)


# ── dataset helpers ──────────────────────────────────────────────────────────────────────────────────

def read_jsonl(p: Path):
    if not p.exists():
        log.warning("Missing data file: %s", p)
        return
    with open(p, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                try:
                    yield json.loads(s)
                except Exception:
                    pass


def _make_rejected(chosen: str) -> str:
    """Degrade a high-quality Avery response into a generic rejected response.

    Strips markdown structure, truncates to 3 sentences, adds generic prefix.
    Creates a clear preference signal without needing human annotators.
    """
    text = re.sub(r"\*\*[^*]+\*\*:?",  "", chosen)
    text = re.sub(r"#{1,3}\s+",         "", text)
    text = re.sub(r"^[\|`].*$",          "", text, flags=re.MULTILINE)  # tables, code fences
    text = re.sub(r"^[•\-\*]\s+",  "", text, flags=re.MULTILINE)  # bullets
    text = re.sub(r"^\d+\.\s+",         "", text, flags=re.MULTILINE)  # numbered lists
    text = re.sub(r"\n{2,}",            " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    truncated = " ".join(sentences[:3]).strip()
    if not truncated:
        truncated = text[:200].strip()
    return f"Here's a brief overview: {truncated}"


def _prompt(system: str, user: str) -> str:
    """Format ChatML prompt ending with the assistant turn opener."""
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def build_dpo_dataset(data_dir: Path, smoke: bool = False):
    """Build DPO preference pairs from GH05T3 JSONL training data.

    Each example:
      prompt   — ChatML-formatted system + user message (ends with assistant opener)
      chosen   — structured, Avery-style response with markdown + step reasoning
      rejected — generic, unstructured truncation of chosen (clear quality gap)

    Sources (in priority order):
      1. sovereign_recall.jsonl (quality >= 4) — highest fidelity
      2. adversarial_defense.jsonl
      3. cve_patterns.jsonl
      4. bug_bounty.jsonl
      5. reasoning_chains.jsonl
      6. Synthetic Avery alignment pairs (always present as a floor)
    """
    from datasets import Dataset

    prompts: list[str]   = []
    chosens: list[str]   = []
    rejecteds: list[str] = []

    # ── 1. sovereign recall (pre-formatted ChatML, quality >= 4) ──────────────────────
    recall_path  = data_dir / "sovereign_recall.jsonl"
    recall_count = 0
    for rec in read_jsonl(recall_path):
        text    = rec.get("text", "")
        quality = rec.get("quality", 0)
        if not text or quality < 4:
            continue
        parts     = re.split(r"<\|im_start\|>", text)
        user_part = next((p for p in parts if p.startswith("user\n")),      None)
        asst_part = next((p for p in parts if p.startswith("assistant\n")), None)
        if not user_part or not asst_part:
            continue
        user_content = user_part[len("user\n"):].replace("<|im_end|>", "").strip()
        asst_content = asst_part[len("assistant\n"):].replace("<|im_end|>", "").strip()
        if len(asst_content) < 50:
            continue
        prompts.append(_prompt(SYSTEM, user_content))
        chosens.append(asst_content)
        rejecteds.append(_make_rejected(asst_content))
        recall_count += 1
    if recall_count:
        log.info("Sovereign Recall: +%d DPO pairs (quality>=4)", recall_count)

    # ── 2. adversarial defense ────────────────────────────────────────────────────────
    for rec in read_jsonl(data_dir / "adversarial_defense.jsonl"):
        t = rec.get("threat_vector", "")
        if not t:
            continue
        chosen = (
            f"**Exploitation Method:** {rec.get('exploitation_method', 'N/A')}\n\n"
            f"**Detection Pattern:** {rec.get('detection_pattern', 'N/A')}\n\n"
            f"**Mitigation Strategy:** {rec.get('mitigation_strategy', 'N/A')}"
        )
        prompts.append(_prompt(SYSTEM, f"Analyze this threat vector:\n\n{t}"))
        chosens.append(chosen)
        rejecteds.append(_make_rejected(chosen))

    # ── 3. CVE patterns ─────────────────────────────────────────────────────────────────────
    for rec in read_jsonl(data_dir / "cve_patterns.jsonl"):
        p = rec.get("vulnerability_pattern", "")
        if not p:
            continue
        ind = rec.get("discovery_indicators", [])
        chosen = (
            f"**Pattern:** {p}\n\n"
            f"**Discovery Indicators:**\n"
            + ("\n".join(f"• {x}" for x in ind) if isinstance(ind, list) else str(ind))
            + f"\n\n**Defensive Lessons:** {rec.get('defensive_lessons', 'N/A')}"
        )
        prompts.append(_prompt(SYSTEM, f"Assess the {rec.get('source_cve', 'CVE-UNKNOWN')} vulnerability."))
        chosens.append(chosen)
        rejecteds.append(_make_rejected(chosen))

    # ── 4. bug bounty ───────────────────────────────────────────────────────────────────────────
    for rec in read_jsonl(data_dir / "bug_bounty.jsonl"):
        tgt  = rec.get("target_system", "")
        vuln = rec.get("vulnerability_found", "")
        if not tgt or not vuln:
            continue
        chosen = (
            f"**Recon Method:** {rec.get('recon_method', 'N/A')}\n\n"
            f"**Non-weaponized PoC:** {rec.get('non_weaponized_poc', 'N/A')}\n\n"
            f"**Remediation:** {rec.get('remediation', 'N/A')}"
        )
        prompts.append(_prompt(SYSTEM, f"Security research report: {tgt} — {vuln}"))
        chosens.append(chosen)
        rejecteds.append(_make_rejected(chosen))

    # ── 5. reasoning chains ──────────────────────────────────────────────────────────────────────────
    for rec in read_jsonl(data_dir / "reasoning_chains.jsonl"):
        q = rec.get("question", "")
        s = rec.get("reasoning_steps", [])
        if not q or not isinstance(s, list):
            continue
        chosen = (
            "**Reasoning:**\n"
            + "\n".join(f"{i+1}. {x}" for i, x in enumerate(s))
            + f"\n\n**Answer:** {rec.get('final_answer', 'N/A')}"
        )
        prompts.append(_prompt(SYSTEM, q))
        chosens.append(chosen)
        rejecteds.append(_make_rejected(chosen))

    # ── 6. synthetic Avery alignment pairs (always present as a floor) ────────────────
    _AVERY_PAIRS = [
        (
            "What makes GH05T3 different from a standard LLM assistant?",
            (
                "**Core architecture:** GH05T3 is a specialized reasoning agent, not a "
                "generalist assistant. Three fundamental differences:\n\n"
                "1. **Security-first reasoning** — every response starts from a threat model. "
                "I ask 'how could this be abused?' before 'how does this work?'\n"
                "2. **Structured output** — markdown headers, bullets, and numbered steps "
                "consistently because ambiguous output leads to misimplementation.\n"
                "3. **Epistemic honesty** — I explicitly label uncertainty. 'N/A' means I "
                "don't know, not that the information doesn't exist.\n\n"
                "**Under the hood:** Qwen2.5-7B-Instruct base + LoRA adapter trained on "
                "adversarial defense, CVE patterns, bug bounty research, and operational "
                "playbooks. DPO alignment ensures Avery-style response quality."
            ),
        ),
        (
            "How do you approach a new security research target?",
            (
                "**Phase 1 — Reconnaissance (passive)**\n"
                "1. OSINT: certificates (crt.sh), DNS records, job postings for tech stack hints\n"
                "2. Shodan/Censys: exposed services, banners, historical data\n"
                "3. GitHub: leaked credentials, .env files, internal tooling\n\n"
                "**Phase 2 — Surface mapping**\n"
                "4. Subdomain enumeration (subfinder, amass, bruteforce wordlists)\n"
                "5. Port scan top-1000 + common service ports\n"
                "6. Web crawl: endpoints, parameters, JS bundles (source maps)\n\n"
                "**Phase 3 — Prioritisation**\n"
                "7. Authentication flows first — highest-impact surface\n"
                "8. API endpoints with user-controlled input\n"
                "9. File upload/download functionality\n\n"
                "**Decision criteria:** impact × likelihood × exploitability. "
                "Chase the critical path, not every finding."
            ),
        ),
        (
            "Explain the trade-offs between LoRA rank 8 vs rank 64.",
            (
                "**LoRA rank controls the size of the low-rank decomposition matrices (A and B).**\n\n"
                "| Setting | Rank 8 | Rank 64 |\n"
                "|---------|--------|---------|\n"
                "| Parameters | ~6 M | ~50 M |\n"
                "| Training speed | Faster | Slower |\n"
                "| Storage | ~24 MB | ~190 MB |\n"
                "| Expressiveness | Narrow tasks | Complex tasks |\n"
                "| Overfitting risk | Lower | Higher |\n\n"
                "**Rule of thumb for GH05T3:** r=16, alpha=32 is the sweet spot. "
                "r=8 handles domain tone/style. r=64 needs 10k+ examples to be useful "
                "without collapsing.\n\n"
                "**Current setup:** r=16, alpha=32, 7 target modules. "
                "~8 M trainable / 7 B total = 0.11% of the model."
            ),
        ),
        (
            "What is DPO and how does it differ from RLHF?",
            (
                "**DPO (Direct Preference Optimization)** optimizes the model directly on "
                "preference pairs — no reward model, no PPO rollouts.\n\n"
                "**RLHF pipeline:** SFT → reward model training → PPO optimization\n"
                "(3 stages, 3 models in memory, unstable training)\n\n"
                "**DPO pipeline:** SFT → DPO on preference pairs\n"
                "(2 stages, 1 model + implicit reference, stable closed-form objective)\n\n"
                "**DPO loss (simplified):**\n"
                "Increase log-prob of chosen relative to reference; decrease log-prob of "
                "rejected relative to reference. Beta controls how tightly we stay near SFT.\n\n"
                "**beta=0.1:** aggressive preference learning. High beta = conservative, "
                "stays close to SFT.\n\n"
                "**Why DPO for GH05T3:** simpler, no reward model to maintain. "
                "Single-model PEFT approach fits in 24 GB alongside optimizer states."
            ),
        ),
        (
            "How should I interpret a training loss of 0.0?",
            (
                "**Loss = 0.0 is a critical failure signal — the adapter has collapsed.**\n\n"
                "**Root causes (most common):**\n"
                "1. GradScaler permanently skips updates after detecting NaN gradients\n"
                "2. `use_reentrant=True` gradient checkpointing + fp16 + PEFT hooks → "
                "NaN in backward from step 1 (RULE 2)\n"
                "3. LoRA adapters cast to fp16 → GradScaler throws "
                "'Attempting to unscale FP16 gradients' (RULE 1)\n\n"
                "**Diagnostic:**\n"
                "```bash\n"
                "grep -i 'nan\\|inf\\|skip' training.log\n"
                "python -c 'import bitsandbytes; print(bitsandbytes.__version__)'\n"
                "```\n\n"
                "**Fixes already applied in ghost_trainer.py:**\n"
                "• `gradient_checkpointing_kwargs={'use_reentrant': False}` (RULE 2)\n"
                "• No fp16 cast after get_peft_model() (RULE 1)\n"
                "• bf16 on L4/A10G (more stable than fp16 + GradScaler)"
            ),
        ),
        (
            "Design a CI/CD pipeline for a FastAPI service with GPU inference.",
            (
                "**Pipeline stages:**\n\n"
                "**1. Code quality (every push, CPU)**\n"
                "```bash\n"
                "ruff check .            # lint\n"
                "mypy backend/           # type check\n"
                "pytest -x tests/unit/   # fast unit tests\n"
                "bandit -r backend/      # security scan\n"
                "```\n\n"
                "**2. Docker build** — multi-stage; CUDA runtime base, minimal final image\n\n"
                "**3. Integration tests** (self-hosted GPU runner)\n"
                "```bash\n"
                "pytest tests/integration/ --timeout=120\n"
                "```\n\n"
                "**4. Canary deploy** — 5% traffic, 15-min observation window\n"
                "Auto-rollback if error_rate > 0.5% or p99_latency > 5 s\n\n"
                "**5. Full production rollout** — rolling update, max-unavailable=0\n\n"
                "**GPU utilisation target:** 70-80%. Below 50% = over-provisioned. "
                "Above 90% = OOM risk under burst traffic."
            ),
        ),
        (
            "Explain gradient checkpointing and why use_reentrant matters.",
            (
                "**Gradient checkpointing** reduces activation memory by not storing "
                "intermediate activations during the forward pass. Instead it recomputes "
                "them on the fly during backpropagation.\n\n"
                "**Two modes:**\n\n"
                "**use_reentrant=True (PyTorch default):**\n"
                "• Re-runs the entire forward pass segment during backward\n"
                "• With PEFT's enable_input_require_grads() hooks + fp16/bf16, this "
                "recomputation produces NaN gradients from step 1\n"
                "• Loss collapses to 0.0 and never recovers\n\n"
                "**use_reentrant=False:**\n"
                "• Uses torch.utils.checkpoint with autograd context — no hook conflict\n"
                "• Required for PEFT + fp16/bf16 + gradient checkpointing\n"
                "• Set via: `gradient_checkpointing_kwargs={'use_reentrant': False}`\n\n"
                "**Always use use_reentrant=False in GH05T3 training. This is RULE 2 "
                "and it is non-negotiable.**"
            ),
        ),
        (
            "How do I sync the DPO adapter from Lightning AI back to TatorTot?",
            (
                "**Option 1 — rsync over Tailscale (fastest):**\n"
                "```bash\n"
                "rsync -avz --progress \\\n"
                "  /teamspace/studios/this_studio/dpo_adapter/ \\\n"
                "  leer4@100.94.227.81:~/gh05t3/backend/models/gh05t3_dpo_adapter/\n"
                "```\n\n"
                "**Option 2 — scp via Tailscale hostname:**\n"
                "```bash\n"
                "scp -r /teamspace/studios/this_studio/dpo_adapter/ \\\n"
                "  leer4@tail1457e2.ts.net:~/gh05t3/backend/models/gh05t3_dpo_adapter/\n"
                "```\n\n"
                "**Option 3 — push to HuggingFace Hub, pull on TatorTot:**\n"
                "```bash\n"
                "# Lightning Studio:\n"
                "huggingface-cli login\n"
                "python -c \"from huggingface_hub import HfApi; "
                "HfApi().upload_folder(folder_path='/teamspace/studios/this_studio/dpo_adapter', "
                "repo_id='leerobber/gh05t3-dpo', repo_type='model')\"\n"
                "# TatorTot:\n"
                "git clone https://huggingface.co/leerobber/gh05t3-dpo backend/models/gh05t3_dpo_adapter\n"
                "```\n\n"
                "**After sync, activate in backend/.env:**\n"
                "```\n"
                "LLM_PROVIDER=gh05t3\n"
                "GH05T3_ADAPTER_PATH=backend/models/gh05t3_dpo_adapter\n"
                "```"
            ),
        ),
    ]
    for user_q, chosen in _AVERY_PAIRS:
        prompts.append(_prompt(SYSTEM, user_q))
        chosens.append(chosen)
        rejecteds.append(_make_rejected(chosen))

    if not prompts:
        log.error("No DPO pairs built. Check that JSONL files exist in %s", data_dir)
        log.error("The 8 synthetic pairs are always built; this error should never occur.")
        sys.exit(1)

    random.seed(42)
    idx = list(range(len(prompts)))
    random.shuffle(idx)
    prompts   = [prompts[i]   for i in idx]
    chosens   = [chosens[i]   for i in idx]
    rejecteds = [rejecteds[i] for i in idx]

    if smoke:
        prompts, chosens, rejecteds = prompts[:60], chosens[:60], rejecteds[:60]

    log.info("DPO dataset: %d preference pairs", len(prompts))
    return Dataset.from_dict({"prompt": prompts, "chosen": chosens, "rejected": rejecteds})


# ── model loading ───────────────────────────────────────────────────────────────────────────────────

def load_base_model(compute_dtype, vram_gb: float):
    """Load MODEL_ID in 4-bit NF4.

    4-bit even on L4 (24 GB): 7B × 0.5 bytes = 3.5 GB base, leaving ~18 GB
    for DPO optimizer states, activations, and reference logits buffer.
    """
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    used = torch.cuda.memory_allocated(0) / 1e9
    log.info("Base model loaded — %.1f / %.1f GB VRAM", used, vram_gb)
    return model


def build_peft_model(model, sft_adapter_path: Path):
    """Attach LoRA adapter for DPO training.

    Loads existing SFT adapter if present; otherwise initialises fresh LoRA.

    RULE 1 — NO fp16 cast after get_peft_model().
    PEFT initialises adapters in fp32. Casting to fp16 causes GradScaler to
    throw 'Attempting to unscale FP16 gradients' and permanently skip updates.
    AMP autocast handles the dtype during the forward pass automatically.
    """
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    # prepare_model_for_kbit_training handles enable_input_require_grads() and
    # casts layer norms to fp32 for stability.
    # use_gradient_checkpointing=False here so TrainingArguments owns it (RULE 2).
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    if sft_adapter_path.exists() and (sft_adapter_path / "adapter_config.json").exists():
        log.info("Loading SFT adapter from %s", sft_adapter_path)
        model = PeftModel.from_pretrained(model, str(sft_adapter_path), is_trainable=True)
    else:
        log.info("No SFT adapter at %s — initialising fresh LoRA", sft_adapter_path)
        lora_cfg = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        # RULE 1 REMINDER: do NOT add any fp16 cast here.
        # The following line was the cause of prior collapses and must never return:
        #   for p in model.parameters(): p.data = p.data.half()   ← NEVER

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log.info("PEFT model: %s trainable / %s total (%.2f%%)",
             f"{trainable:,}", f"{total:,}", 100 * trainable / total)
    return model


# ── DPO training ──────────────────────────────────────────────────────────────────────────────────

def run_dpo(model, tokenizer, dataset, steps: int):
    """Run DPO with all CLAUDE.md rules applied.

    Supports TRL >= 0.9 (DPOConfig) and falls back to TRL 0.8.x (TrainingArguments).
    ref_model=None uses the PEFT base (adapter disabled) as the implicit reference —
    single-model DPO, no second copy of the model in memory.
    """
    import torch

    try:
        from trl import DPOConfig, DPOTrainer
        _use_dpo_config = True
        log.info("TRL DPOConfig API detected")
    except ImportError:
        from trl import DPOTrainer
        from transformers import TrainingArguments as DPOConfig
        _use_dpo_config = False
        log.info("DPOConfig not found — falling back to TrainingArguments (TRL <= 0.8)")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    DPO_OUT.mkdir(parents=True, exist_ok=True)

    gpu      = torch.cuda.get_device_properties(0)
    use_bf16 = gpu.major >= 8

    common = dict(
        output_dir                       = str(CKPT_DIR),
        max_steps                        = steps,
        per_device_train_batch_size      = BATCH,
        gradient_accumulation_steps      = GRAD_ACCUM,
        gradient_checkpointing           = True,
        # RULE 2 — use_reentrant=True reruns the forward pass during backward.
        # With fp16/bf16 + PEFT hooks this produces NaN gradients from step 1.
        gradient_checkpointing_kwargs    = {"use_reentrant": False},
        warmup_steps                     = WARMUP,
        learning_rate                    = LR,
        fp16                             = not use_bf16,
        bf16                             = use_bf16,
        max_grad_norm                    = MAX_GRAD,
        logging_steps                    = 5,
        optim                            = "paged_adamw_8bit",
        lr_scheduler_type                = "cosine",
        seed                             = 42,
        save_strategy                    = "steps",
        save_steps                       = 50,
        save_total_limit                 = 2,
        report_to                        = "none",
        remove_unused_columns            = False,
    )

    if _use_dpo_config:
        args = DPOConfig(
            **common,
            beta             = DPO_BETA,
            max_length       = MAX_LEN,
            max_prompt_length= MAX_PROMPT,
        )
        trainer = DPOTrainer(
            model           = model,
            ref_model       = None,
            args            = args,
            train_dataset   = dataset,
            processing_class= tokenizer,
        )
    else:
        args = DPOConfig(**common)
        trainer = DPOTrainer(
            model            = model,
            ref_model        = None,
            beta             = DPO_BETA,
            args             = args,
            train_dataset    = dataset,
            tokenizer        = tokenizer,
            max_length       = MAX_LEN,
            max_prompt_length= MAX_PROMPT,
        )

    log.info(
        "DPO training — steps=%d beta=%.2f lr=%s %s batch=%d×%d=%d",
        steps, DPO_BETA, LR, "bf16" if use_bf16 else "fp16",
        BATCH, GRAD_ACCUM, BATCH * GRAD_ACCUM,
    )

    stats = trainer.train()
    loss  = stats.training_loss
    log.info("Training complete — loss: %.4f | steps: %d", loss, stats.global_step)

    # RULE 3 — Collapse detection. Do not write a garbage adapter to disk.
    # DPO loss < 0.3: over-optimised or gradient collapse (check RULE 1 & 2).
    # DPO loss > 10:  adapter never converged (NaN cascade, bad data, LR too high).
    if not (0.3 < loss < 10):
        log.error("DPO FAILED — loss=%.4f outside (0.3, 10.0). Adapter not saved.", loss)
        if loss <= 0.3:
            log.error("Possible causes: gradient collapse, NaN overflow, use_reentrant=True")
            log.error("Try: lower beta (%.2f → %.2f) or check RULE 1 & RULE 2", DPO_BETA, DPO_BETA / 2)
        else:
            log.error("Possible causes: LR too high, bad dataset, NaN divergence")
            log.error("Try: lr=1e-6, cleaner preference pairs, check bitsandbytes>=0.43")
        sys.exit(1)

    return stats


# ── main ──────────────────────────────────────────────────────────────────────────────────────

def main():
    global DPO_BETA, LR, MAX_STEPS, DPO_OUT
    import torch

    parser = argparse.ArgumentParser(
        description="GH05T3 DPO Trainer — Lightning AI Studio",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--steps",   type=int,   default=MAX_STEPS,       help="Training steps")
    parser.add_argument("--beta",    type=float, default=DPO_BETA,        help="DPO beta (KL temperature)")
    parser.add_argument("--lr",      type=float, default=LR,              help="Learning rate")
    parser.add_argument("--adapter", type=str,   default=str(SFT_ADAPTER),help="SFT adapter path to start from")
    parser.add_argument("--out",     type=str,   default=str(DPO_OUT),    help="Output directory for DPO adapter")
    parser.add_argument("--smoke",   action="store_true",
                        help="Quick sanity check: 30 steps, 60 pairs")
    cli = parser.parse_args()

    # Apply CLI overrides to module-level config before dataset build
    DPO_BETA   = cli.beta
    LR         = cli.lr
    MAX_STEPS  = 30 if cli.smoke else cli.steps
    DPO_OUT    = Path(cli.out)

    sft_adapter_path = Path(cli.adapter)

    # ── GPU check ────────────────────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        log.error("No CUDA GPU detected. This trainer requires a GPU.")
        log.error("In Lightning AI Studio: select L4 or A10G instance type.")
        sys.exit(1)

    gpu  = torch.cuda.get_device_properties(0)
    vram = gpu.total_memory / 1e9
    log.info("GPU: %s | %.1f GB | sm_%d%d | PyTorch %s | CUDA %s",
             gpu.name, vram, gpu.major, gpu.minor,
             torch.__version__, torch.version.cuda)

    if vram < 14:
        log.error("DPO requires >= 14 GB VRAM (7B 4-bit + paged optimizer + activations).")
        log.error("Found %.1f GB. Switch to L4 (24 GB) or A10G (24 GB) in Lightning Studio.", vram)
        sys.exit(1)

    if gpu.major < 7:
        log.error("GPU sm_%d%d is below sm_70 — bitsandbytes 4-bit not supported.", gpu.major, gpu.minor)
        sys.exit(1)

    compute_dtype = torch.bfloat16 if gpu.major >= 8 else torch.float16
    log.info("Precision: %s (sm_%d%d)", "bf16" if gpu.major >= 8 else "fp16", gpu.major, gpu.minor)

    if cli.smoke:
        log.info("SMOKE TEST MODE — 30 steps, 60 pairs")

    # ── dataset ──────────────────────────────────────────────────────────────────────────────────
    log.info("Building DPO dataset from %s ...", DATA_DIR)
    dataset = build_dpo_dataset(DATA_DIR, smoke=cli.smoke)

    # ── tokenizer + model ──────────────────────────────────────────────────────────────────────────────
    from transformers import AutoTokenizer

    log.info("Loading tokenizer: %s", MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # left-padding required for DPO batch generation

    model = load_base_model(compute_dtype, vram)
    model = build_peft_model(model, sft_adapter_path)

    # ── train ───────────────────────────────────────────────────────────────────────────────────────
    stats = run_dpo(model, tokenizer, dataset, steps=MAX_STEPS)
    loss  = stats.training_loss

    # ── save ────────────────────────────────────────────────────────────────────────────────────────
    DPO_OUT.mkdir(parents=True, exist_ok=True)
    log.info("Saving DPO adapter → %s", DPO_OUT)
    model.save_pretrained(str(DPO_OUT))
    tokenizer.save_pretrained(str(DPO_OUT))

    cfg_path = DPO_OUT / "dpo_training_config.json"
    with open(cfg_path, "w") as f:
        json.dump({
            "model":         MODEL_ID,
            "sft_adapter":   str(sft_adapter_path),
            "lora_rank":     LORA_RANK,
            "dpo_beta":      DPO_BETA,
            "learning_rate": LR,
            "steps":         stats.global_step,
            "final_loss":    round(loss, 4),
            "dataset_pairs": len(dataset),
            "max_len":       MAX_LEN,
            "max_prompt_len":MAX_PROMPT,
            "gpu":           gpu.name,
            "sm":            f"{gpu.major}{gpu.minor}",
            "pytorch":       torch.__version__,
            "cuda":          torch.version.cuda,
            "smoke_test":    cli.smoke,
        }, f, indent=2)

    log.info("Adapter saved. Config: %s", cfg_path)
    log.info("")
    log.info("Sync to TatorTot (Tailscale):")
    log.info("  rsync -avz %s/ leer4@100.94.227.81:~/gh05t3/backend/models/gh05t3_dpo_adapter/", DPO_OUT)
    log.info("")
    log.info("Activate in backend/.env:")
    log.info("  LLM_PROVIDER=gh05t3")
    log.info("  GH05T3_ADAPTER_PATH=backend/models/gh05t3_dpo_adapter")


if __name__ == "__main__":
    main()
