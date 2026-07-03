"""
train_sovereign_sft.py — Sovereign Multi-Agent QLoRA Fine-Tune
Runs on RunPod 24GB+ GPU (A5000 / 3090 / 4090 / 5090).

Modes:
  SFT  — supervised fine-tune on instruction/response pairs
  ORPO — odds-ratio preference optimization (needs chosen+rejected)
  DPO  — direct preference optimization
  GRPO — group relative policy optimization (RL, no labels needed)

Agents:
  avery   — business strategist (KAIROS)
  forge   — code generation
  oracle  — memory and retrieval
  codex   — documentation
  sentinel— security review
  nexus   — orchestration
  all     — train on combined multi-agent dataset

Checkpoint resuming:
  Saves checkpoints every 50 steps to /workspace/checkpoints/.
  On restart, auto-resumes from latest checkpoint if present.

Output: LoRA adapter pushed to HuggingFace per agent repo.
"""
import os, sys, json, argparse
from pathlib import Path

# ── Args ──────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--mode",   default=os.environ.get("TRAIN_MODE",  "orpo"),
                choices=["sft", "dpo", "orpo", "grpo"])
ap.add_argument("--agent",  default=os.environ.get("TRAIN_AGENT", "avery"),
                choices=["avery","forge","oracle","codex","sentinel","nexus","all"])
ap.add_argument("--epochs",    type=int, default=int(os.environ.get("TRAIN_EPOCHS", "3")))
ap.add_argument("--max-steps", type=int, default=int(os.environ.get("TRAIN_MAX_STEPS", "-1")),
                help="Hard cap on training steps per agent (-1 = unlimited)")
ap.add_argument("--split",  default=os.environ.get("TRAIN_SPLIT", ""))
args, _ = ap.parse_known_args()

HF_TOKEN    = os.environ.get("HF_TOKEN", "")
BASE_MODEL  = "Qwen/Qwen2-7B-Instruct"
HF_DATASET  = "tastytator/sovereign-economy"

# ── Environment detection (Kaggle vs RunPod vs local) ─────────────────────────
ON_KAGGLE  = os.path.exists("/kaggle/working")
ON_RUNPOD  = os.path.exists("/workspace") and not ON_KAGGLE
WORK_ROOT  = "/kaggle/working" if ON_KAGGLE else "/workspace"

if ON_KAGGLE:
    print("[ENV] Kaggle detected -- outputs -> /kaggle/working/ (persisted)")
elif ON_RUNPOD:
    print("[ENV] RunPod detected  -- outputs -> /workspace/")
else:
    print("[ENV] Local detected   -- outputs -> ./workspace/")
    WORK_ROOT = str(Path(__file__).parent / "workspace")

OUTPUT_DIR  = os.environ.get("OUTPUT_DIR",  f"{WORK_ROOT}/avery-lora")
CKPT_DIR    = os.environ.get("CKPT_DIR",    f"{WORK_ROOT}/checkpoints")
EPOCHS      = args.epochs
MAX_STEPS   = args.max_steps
MAX_SEQ_LEN = 2048
MODE        = args.mode
AGENT       = args.agent

Path(CKPT_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ── Per-agent config ──────────────────────────────────────────────────────────

AGENT_CONFIGS = {
    "avery": {
        "system": (
            "You are Avery, the sovereign business strategist for SovereignNation — "
            "a fixed-cost AI platform built for lower and middle class families, "
            "children's education, and affordable connectivity. "
            "Use the KAIROS framework: Kickoff, Alignment, Implementation, Refinement, "
            "Optimization, Scaling. Be direct, structured, and actionable."
        ),
        "hf_repo": "tastytator/avery-sovereign-lora",
        "reward_terms": ["kairos", "sovereign", "strategy", "kickoff", "alignment",
                         "implementation", "revenue", "pricing", "market", "platform"],
    },
    "forge": {
        "system": (
            "You are FORGE, the sovereign code generation specialist for SovereignNation. "
            "Write production-ready Python, JavaScript, and TypeScript. "
            "Always include imports, error handling, type hints, and comments for non-obvious logic. "
            "Code must be secure, tested, and match SovereignNation's architecture."
        ),
        "hf_repo": "tastytator/forge-sovereign-lora",
        "reward_terms": ["def ", "class ", "import ", "async ", "return ", "```"],
    },
    "oracle": {
        "system": (
            "You are ORACLE, the sovereign memory and retrieval specialist for SovereignNation. "
            "Synthesize information into precise structured answers. "
            "Cite source type (memory / document / inference). "
            "Be concise. If data is missing, state what is needed."
        ),
        "hf_repo": "tastytator/oracle-sovereign-lora",
        "reward_terms": ["source:", "from memory", "based on", "retrieved", "summary:"],
    },
    "codex": {
        "system": (
            "You are CODEX, the sovereign documentation specialist for SovereignNation. "
            "Write clear complete technical documentation with markdown headings, "
            "code blocks, and working examples. Be accurate and immediately actionable."
        ),
        "hf_repo": "tastytator/codex-sovereign-lora",
        "reward_terms": ["##", "###", "```", "example", "usage", "installation"],
    },
    "sentinel": {
        "system": (
            "You are SENTINEL, the sovereign security specialist for SovereignNation. "
            "Review code and systems for vulnerabilities. Reference OWASP Top 10 and CWE. "
            "State: vulnerability, impact (low/med/high/critical), and the specific fix."
        ),
        "hf_repo": "tastytator/sentinel-sovereign-lora",
        "reward_terms": ["vulnerability", "owasp", "cwe", "risk", "exploit",
                         "authentication", "encrypt", "recommendation", "fix:"],
    },
    "nexus": {
        "system": (
            "You are NEXUS, the sovereign orchestration specialist for SovereignNation. "
            "Coordinate agents and design workflows. Output structured task graphs: "
            "sequence, parallelism, dependencies, and which agent owns each step."
        ),
        "hf_repo": "tastytator/nexus-sovereign-lora",
        "reward_terms": ["step", "phase", "agent", "workflow", "pipeline",
                         "depends", "parallel", "sequence"],
    },
}

# Multi-agent combined: single LoRA trained on all roles
AGENT_CONFIGS["all"] = {
    "system": "You are a sovereign AI specialist. Your role is determined by the system context.",
    "hf_repo": "tastytator/sovereign-agents-lora",
    "reward_terms": (
        AGENT_CONFIGS["avery"]["reward_terms"] +
        AGENT_CONFIGS["forge"]["reward_terms"] +
        AGENT_CONFIGS["oracle"]["reward_terms"]
    ),
}

cfg = AGENT_CONFIGS[AGENT]

print("=" * 60)
print(f"  SOVEREIGN QLoRA FINE-TUNE  [{MODE.upper()}  AGENT={AGENT.upper()}]")
print("=" * 60)
print(f"  Base model : {BASE_MODEL}")
print(f"  Dataset    : {HF_DATASET}")
print(f"  Mode       : {MODE}")
print(f"  Agent      : {AGENT}")
print(f"  Epochs     : {EPOCHS}")
print(f"  Output     : {cfg['hf_repo']}")
print()

if not HF_TOKEN:
    print("ERROR: HF_TOKEN not set."); sys.exit(1)

# ── Install deps ──────────────────────────────────────────────────────────────
print("[1/7] Installing dependencies...")
ret = os.system(
    "pip install -q --force-reinstall --no-deps "
    "'transformers==4.47.1' "
    "'tokenizers>=0.20,<0.22' "
    "safetensors && "
    "pip install -q "
    "'trl==0.12.2' "
    "'peft==0.14.0' "
    "'bitsandbytes>=0.43.0' "
    "'accelerate==1.2.1' "
    "datasets"
)
if ret != 0:
    print("WARNING: pip had non-zero exit. Continuing...")
print("      Done.")

from datasets import load_dataset

# ── Load dataset ──────────────────────────────────────────────────────────────
print("[2/7] Loading dataset...")

DPO_CONFIG   = "dpo"
SFT_CONFIG   = "sft"
AGENTS_CONFIG = "agents"
DPO_SPLITS   = ["bootstrap_dpo", "spin_business_dpo"]
SFT_SPLITS   = ["train"]


def try_load(split: str, config: str = None):
    kwargs = {"split": split, "token": HF_TOKEN}
    if config:
        kwargs["name"] = config
    try:
        ds = load_dataset(HF_DATASET, **kwargs)
        print(f"      Loaded '{config or 'default'}/{split}': {len(ds)} rows  cols={ds.column_names}")
        return ds
    except Exception as e:
        print(f"      '{config or 'default'}/{split}' not available: {type(e).__name__}")
        return None


ds = None
final_mode = MODE

# ── Agent-specific dataset loading ──────────────────────────────────────────
if AGENT != "avery":
    # Try agents config first (multi-agent bootstrap data)
    ds = try_load("train", config=AGENTS_CONFIG)
    if ds is not None and AGENT != "all":
        # Filter to this agent's rows
        ds = ds.filter(lambda row: row.get("agent", "") == AGENT)
        print(f"      Filtered to agent='{AGENT}': {len(ds)} rows")
        if len(ds) < 5:
            print(f"      WARNING: Only {len(ds)} rows for agent '{AGENT}'. Run agents_bootstrap.py first.")
            ds = None

# ── Avery / fallback dataset loading ────────────────────────────────────────
if ds is None:
    if args.split:
        ds = try_load(args.split, config=DPO_CONFIG) or try_load(args.split)
        if ds is None:
            print(f"ERROR: Split '{args.split}' not found. Run pre_train.py first.")
            sys.exit(1)
        if "chosen" in ds.column_names and "rejected" in ds.column_names:
            if final_mode == "sft":
                final_mode = "orpo"
                print("      Has chosen/rejected — upgrading to ORPO")
        else:
            if final_mode in ("dpo", "orpo"):
                final_mode = "sft"
                print("      No chosen/rejected — falling back to SFT")

    if ds is None and final_mode in ("dpo", "orpo"):
        for split in DPO_SPLITS:
            ds = try_load(split, config=DPO_CONFIG)
            if ds is not None:
                if "chosen" in ds.column_names and "rejected" in ds.column_names:
                    break
                ds = None

    if ds is None:
        print("      DPO splits not found — falling back to SFT")
        final_mode = "sft"
        for split in SFT_SPLITS:
            ds = try_load(split, config=SFT_CONFIG)
            if ds is not None:
                break
        if ds is None:
            ds = try_load("spin_business")

    if ds is None:
        print("ERROR: No dataset found. Run: python pre_train.py")
        sys.exit(1)

MODE = final_mode
if MODE == "grpo" and final_mode != "grpo":
    MODE = "grpo"  # keep GRPO if explicitly requested
    print("      GRPO mode: using prompts only (no chosen/rejected needed)")

print(f"\n      Mode={MODE}  Agent={AGENT}  Rows={len(ds)}")

# ── Load model ────────────────────────────────────────────────────────────────
print(f"\n[3/7] Loading {BASE_MODEL} with QLoRA (4-bit, standard PEFT)...")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit               = True,
    bnb_4bit_quant_type        = "nf4",
    bnb_4bit_compute_dtype     = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    bnb_4bit_use_double_quant  = True,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config = bnb_config,
    device_map          = "auto",
    token               = HF_TOKEN,
)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora_config = LoraConfig(
    r              = 16,
    lora_alpha     = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_dropout   = 0.05,
    bias           = "none",
    task_type      = TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
print("      Model ready.")
model.print_trainable_parameters()

# ── Checkpoint detection ──────────────────────────────────────────────────────
print(f"\n[4/7] Checking for existing checkpoints...")
resume_checkpoint = None
ckpt_base = Path(CKPT_DIR) / AGENT
if ckpt_base.exists():
    checkpoints = sorted(
        [d for d in ckpt_base.iterdir() if d.is_dir() and "checkpoint" in d.name],
        key=lambda d: int(d.name.split("-")[-1]) if d.name.split("-")[-1].isdigit() else 0
    )
    if checkpoints:
        resume_checkpoint = str(checkpoints[-1])
        print(f"      Found checkpoint: {resume_checkpoint}")
        print(f"      Training will RESUME from this checkpoint.")
    else:
        print("      No checkpoint found — starting fresh.")
else:
    print("      No checkpoint directory — starting fresh.")

# ── Train ─────────────────────────────────────────────────────────────────────
print(f"\n[5/7] Training ({MODE.upper()}, {EPOCHS} epochs)...")
from transformers import TrainingArguments

SYSTEM = cfg["system"]

def _prompt_wrap(goal: str) -> str:
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{goal}<|im_end|>\n"
            f"<|im_start|>assistant\n")


# ── SFT ───────────────────────────────────────────────────────────────────────
if MODE == "sft":
    from trl import SFTTrainer

    def fmt_sft(row):
        goal = str(
            row.get("instruction") or row.get("goal") or
            row.get("prompt") or ""
        ).strip()
        resp = str(row.get("response") or row.get("chosen") or "").strip()
        return {"text": _prompt_wrap(goal) + resp + "<|im_end|>"}

    ds_fmt = ds.map(fmt_sft, remove_columns=ds.column_names)
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=ds_fmt,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        args=TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            num_train_epochs            = EPOCHS,
            max_steps                   = MAX_STEPS,
            learning_rate               = 2e-4,
            fp16                        = not torch.cuda.is_bf16_supported(),
            bf16                        = torch.cuda.is_bf16_supported(),
            logging_steps               = 10,
            output_dir                  = str(ckpt_base),
            warmup_ratio                = 0.1,
            lr_scheduler_type           = "cosine",
            save_strategy               = "steps",
            save_steps                  = 50,
            save_total_limit            = 2,
            report_to                   = "none",
        ),
    )

# ── ORPO ──────────────────────────────────────────────────────────────────────
elif MODE == "orpo":
    from trl import ORPOTrainer, ORPOConfig

    def fmt_orpo(row):
        goal   = str(row.get("instruction") or row.get("goal") or "").strip()
        prompt = str(row.get("prompt") or f"GOAL: {goal}\n\nProvide a detailed sovereign response:").strip()
        return {
            "prompt":   _prompt_wrap(prompt),
            "chosen":   str(row["chosen"]).strip() + "<|im_end|>",
            "rejected": str(row["rejected"]).strip() + "<|im_end|>",
        }

    ds_fmt = ds.map(fmt_orpo, remove_columns=ds.column_names)
    trainer = ORPOTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=ds_fmt,
        args=ORPOConfig(
            max_length                  = MAX_SEQ_LEN,
            max_prompt_length           = 512,
            beta                        = 0.1,
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 8,
            num_train_epochs            = EPOCHS,
            max_steps                   = MAX_STEPS,
            learning_rate               = 8e-6,
            fp16                        = not torch.cuda.is_bf16_supported(),
            bf16                        = torch.cuda.is_bf16_supported(),
            logging_steps               = 10,
            output_dir                  = str(ckpt_base),
            warmup_ratio                = 0.1,
            lr_scheduler_type           = "cosine",
            save_strategy               = "steps",
            save_steps                  = 50,
            save_total_limit            = 2,
            report_to                   = "none",
        ),
    )

# ── DPO ───────────────────────────────────────────────────────────────────────
elif MODE == "dpo":
    from trl import DPOTrainer, DPOConfig

    def fmt_dpo(row):
        goal   = str(row.get("instruction") or row.get("goal") or "").strip()
        prompt = str(row.get("prompt") or f"GOAL: {goal}\n\nProvide a detailed sovereign response:").strip()
        return {
            "prompt":   _prompt_wrap(prompt),
            "chosen":   str(row["chosen"]).strip() + "<|im_end|>",
            "rejected": str(row["rejected"]).strip() + "<|im_end|>",
        }

    ds_fmt = ds.map(fmt_dpo, remove_columns=ds.column_names)
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        train_dataset=ds_fmt,
        args=DPOConfig(
            max_length                  = MAX_SEQ_LEN,
            max_prompt_length           = 512,
            beta                        = 0.1,
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 8,
            num_train_epochs            = EPOCHS,
            max_steps                   = MAX_STEPS,
            learning_rate               = 5e-6,
            fp16                        = not torch.cuda.is_bf16_supported(),
            bf16                        = torch.cuda.is_bf16_supported(),
            logging_steps               = 10,
            output_dir                  = str(ckpt_base),
            warmup_ratio                = 0.1,
            lr_scheduler_type           = "cosine",
            save_strategy               = "steps",
            save_steps                  = 50,
            save_total_limit            = 2,
            report_to                   = "none",
        ),
    )

# ── GRPO ──────────────────────────────────────────────────────────────────────
elif MODE == "grpo":
    from trl import GRPOTrainer, GRPOConfig

    try:
        from unsloth import PatchFastRL
        PatchFastRL("GRPO", FastLanguageModel)
        print("      unsloth GRPO patch applied.")
    except (ImportError, AttributeError):
        pass  # newer unsloth handles this automatically

    reward_terms = cfg["reward_terms"]

    def _reward(completions, **kwargs):
        scores = []
        for text in completions:
            score = 0.0
            text_lower = text.lower()
            # Quality: contains role-specific terms
            hits = sum(1 for t in reward_terms if t.lower() in text_lower)
            score += min(hits * 0.4, 2.0)
            # Length: penalize too short or empty
            if len(text) > 400:
                score += 1.0
            elif len(text) > 150:
                score += 0.5
            else:
                score -= 0.5
            # Structure: has line breaks or sections
            if text.count("\n") > 3:
                score += 0.5
            scores.append(score)
        return scores

    def fmt_grpo(row):
        prompt_text = str(
            row.get("prompt") or row.get("instruction") or
            row.get("goal") or ""
        ).strip()
        return {"prompt": _prompt_wrap(prompt_text)}

    ds_fmt = ds.map(fmt_grpo, remove_columns=ds.column_names)
    ds_fmt = ds_fmt.filter(lambda row: len(row["prompt"]) > 20)

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds_fmt,
        reward_funcs=[_reward],
        args=GRPOConfig(
            max_prompt_length           = 512,
            max_completion_length       = 512,
            num_generations             = 4,
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 8,
            num_train_epochs            = EPOCHS,
            learning_rate               = 5e-6,
            fp16                        = not torch.cuda.is_bf16_supported(),
            bf16                        = torch.cuda.is_bf16_supported(),
            logging_steps               = 10,
            output_dir                  = str(ckpt_base),
            warmup_ratio                = 0.1,
            lr_scheduler_type           = "cosine",
            save_strategy               = "steps",
            save_steps                  = 50,
            save_total_limit            = 2,
            report_to                   = "none",
        ),
    )

# ── Intermediate-push callback — survives Kaggle/RunPod session death ─────────
from transformers import TrainerCallback

class IntermediatePushCallback(TrainerCallback):
    """Push LoRA adapter to HuggingFace every N steps.

    If the training session is killed (Kaggle timeout, RunPod spot preemption,
    OOM) the latest adapter is already on HF — nothing is lost.
    """
    PUSH_EVERY = int(os.environ.get("PUSH_EVERY_STEPS", "100"))

    def __init__(self, model, tokenizer, hf_repo, hf_token):
        self._model     = model
        self._tokenizer = tokenizer
        self._repo      = hf_repo
        self._token     = hf_token
        self._last_push = 0

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step == 0 or step - self._last_push < self.PUSH_EVERY:
            return
        self._last_push = step
        print(f"\n  [PUSH] Intermediate push at step {step} → {self._repo} ...")
        try:
            self._model.push_to_hub(self._repo, token=self._token,
                                    private=False, commit_message=f"step-{step}")
            print(f"  [PUSH] Done (step {step})")
        except Exception as e:
            print(f"  [PUSH] WARNING: push failed at step {step}: {e} — continuing")


push_cb = IntermediatePushCallback(
    model     = model,
    tokenizer = tokenizer,
    hf_repo   = cfg["hf_repo"],
    hf_token  = HF_TOKEN,
)

trainer.add_callback(push_cb)
trainer_stats = trainer.train(resume_from_checkpoint=resume_checkpoint)
final_loss    = trainer_stats.metrics.get("train_loss", 0)
final_runtime = trainer_stats.metrics.get("train_runtime", 0)
print(f"      Training complete in {final_runtime:.1f}s")
print(f"      Final loss: {final_loss:.4f}")

# ── Save & push ───────────────────────────────────────────────────────────────
print(f"\n[6/7] Saving LoRA adapter to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

hf_repo = cfg["hf_repo"]
print(f"\n[7/7] Pushing to HuggingFace: {hf_repo}...")
model.push_to_hub(hf_repo, token=HF_TOKEN, private=False)
tokenizer.push_to_hub(hf_repo, token=HF_TOKEN, private=False)

# Clean up checkpoint after successful push to save disk
try:
    import shutil
    if ckpt_base.exists():
        shutil.rmtree(str(ckpt_base))
        print("      Cleaned checkpoint directory.")
except Exception:
    pass

print()
print("=" * 60)
print(f"  TRAINING COMPLETE  [{MODE.upper()}  {AGENT.upper()}]")
print(f"  Loss   : {final_loss:.4f}")
print(f"  Time   : {final_runtime/60:.1f} min")
print(f"  LoRA   : https://huggingface.co/{hf_repo}")
print("=" * 60)

Path("/workspace/training_complete.txt").write_text(json.dumps({
    "status":    "complete",
    "model":     hf_repo,
    "agent":     AGENT,
    "mode":      MODE,
    "loss":      final_loss,
    "runtime_s": final_runtime,
}))
