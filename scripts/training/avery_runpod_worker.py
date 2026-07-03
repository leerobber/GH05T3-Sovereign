#!/usr/bin/env python3
"""
avery_runpod_worker.py
======================
Runs INSIDE a RunPod pod. Single-shot, no interaction.
Steps: install → train QLoRA → merge → GGUF Q8_0 → push GGUF to HF → sentinel

Launched by avery_pipeline.py via SSH.
Do NOT run locally — requires GPU + Linux /workspace.
"""
import argparse, gc, json, os, subprocess, sys, time
from pathlib import Path

# ── Args (injected by avery_pipeline.py) ────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--hf_token",    required=True)
ap.add_argument("--base_model",  default="Qwen/Qwen2.5-3B-Instruct")
ap.add_argument("--lora_repo",   default="tastytator/avery-sovereign-lora")
ap.add_argument("--dataset",     default="tastytator/sovereign-economy")
ap.add_argument("--gguf_repo",   default="tastytator/avery-sovereign-gguf")
ap.add_argument("--gguf_name",   default="avery-3b-q8.gguf")
ap.add_argument("--epochs",      type=int,   default=3)
ap.add_argument("--batch_size",  type=int,   default=4)
ap.add_argument("--grad_acc",    type=int,   default=4)
ap.add_argument("--max_seq",     type=int,   default=2048)
ap.add_argument("--lora_r",      type=int,   default=16)
ap.add_argument("--workspace",   default="/workspace")
args = ap.parse_args()

WORKSPACE   = Path(args.workspace)
LORA_DIR    = WORKSPACE / "avery-lora"
MERGED_DIR  = WORKSPACE / "avery-merged"
GGUF_PATH   = WORKSPACE / args.gguf_name
SENTINEL    = WORKSPACE / "avery_done.txt"
LOG         = WORKSPACE / "avery_worker.log"

os.environ["HF_TOKEN"]            = args.hf_token
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SYSTEM = (
    "You are Avery, the sovereign business strategist for SovereignNation — "
    "a fixed-cost AI platform for lower and middle class families, "
    "professional services firms, children's education, and affordable connectivity. "
    "Use the KAIROS framework: Kickoff, Alignment, Implementation, Refinement, "
    "Optimization, Scaling. Be direct, structured, and actionable."
)

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def run(cmd, **kwargs):
    result = subprocess.run(cmd, shell=True, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")
    return result

# ─────────────────────────────────────────────────────────────────────────────

def step_install():
    log("=" * 55)
    log("STEP 1/5: Installing dependencies")
    log("=" * 55)
    run("pip install -q --force-reinstall --no-deps "
        '"transformers==4.47.1" "tokenizers>=0.20,<0.22" safetensors')
    run("pip install -q "
        '"trl==0.12.2" "peft==0.14.0" "bitsandbytes>=0.43.0" '
        '"accelerate==1.2.1" datasets huggingface_hub gguf')
    log("Deps ready.")


def step_train():
    log("=" * 55)
    log("STEP 2/5: Training QLoRA")
    log("=" * 55)

    import torch
    gpu  = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    log(f"GPU: {gpu}  VRAM: {vram:.1f}GB")

    # Scale batch for available VRAM
    batch = args.batch_size
    if vram >= 70:   batch = 8   # H100/A100 80GB
    elif vram >= 35: batch = 6   # A100 40GB
    elif vram >= 20: batch = 4   # 3090/A5000
    else:            batch = 2   # T4/V100

    has_bf16  = torch.cuda.is_bf16_supported()
    dtype     = torch.bfloat16 if has_bf16 else torch.float16
    log(f"batch={batch}  bf16={has_bf16}  dtype={dtype}")

    from datasets import load_dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                               BitsAndBytesConfig, TrainingArguments)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    # Load dataset
    log(f"Loading {args.dataset} (sft/train)...")
    ds_raw = load_dataset(args.dataset, name="sft", split="train", token=args.hf_token)
    log(f"Raw rows: {len(ds_raw):,}")

    # Format for Qwen2.5 chat template
    def fmt(row):
        instr = str(row.get("instruction", "")).strip()
        resp  = str(row.get("response",    "")).strip()
        if len(instr) < 20 or len(resp) < 20:
            return {"text": ""}
        text = (
            "<|im_start|>system\n" + SYSTEM + "<|im_end|>\n"
            "<|im_start|>user\n"   + instr  + "<|im_end|>\n"
            "<|im_start|>assistant\n" + resp + "<|im_end|>"
        )
        return {"text": text}

    ds = ds_raw.map(fmt, remove_columns=ds_raw.column_names, num_proc=4)
    ds = ds.filter(lambda r: len(r["text"]) > 60)
    log(f"Formatted: {len(ds):,} examples")

    # Load model 4-bit QLoRA
    log(f"Loading {args.base_model} in 4-bit NF4...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        device_map="auto",
        token=args.hf_token,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, token=args.hf_token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # LoRA
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    trainable, total = model.get_nb_trainable_parameters()
    log(f"Trainable params: {trainable/1e6:.1f}M / {total/1e6:.1f}M "
        f"({100*trainable/total:.2f}%)")

    # Train
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=args.max_seq,
            packing=True,
            per_device_train_batch_size=batch,
            gradient_accumulation_steps=max(1, args.grad_acc // (batch // 2)) if batch > 2 else args.grad_acc,
            num_train_epochs=args.epochs,
            learning_rate=2e-4,
            bf16=has_bf16, fp16=not has_bf16,
            logging_steps=50,
            output_dir=str(LORA_DIR),
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            save_strategy="no",
            report_to="none",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        ),
    )

    t0 = time.time()
    log(f"Training {args.epochs} epochs...")
    stats  = trainer.train()
    elapsed = (time.time() - t0) / 60
    loss   = stats.metrics.get("train_loss", 0)
    log(f"Training done: loss={loss:.4f}  time={elapsed:.1f}min")

    # Save LoRA adapter + push to HF
    log(f"Saving LoRA to {LORA_DIR}...")
    LORA_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(LORA_DIR))
    tokenizer.save_pretrained(str(LORA_DIR))

    log(f"Pushing LoRA to {args.lora_repo}...")
    model.push_to_hub(args.lora_repo, token=args.hf_token, private=False)
    tokenizer.push_to_hub(args.lora_repo, token=args.hf_token, private=False)
    log(f"LoRA pushed: https://huggingface.co/{args.lora_repo}")

    # Free GPU memory before merge
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    log("GPU memory freed.")


def step_merge():
    log("=" * 55)
    log("STEP 3/5: Merging LoRA into base model")
    log("=" * 55)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    log(f"Loading base model {args.base_model} on CPU (fp16)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
        token=args.hf_token,
        low_cpu_mem_usage=True,
    )

    log(f"Loading LoRA from {args.lora_repo}...")
    model = PeftModel.from_pretrained(model, args.lora_repo, token=args.hf_token)

    log("Merging weights...")
    model = model.merge_and_unload()

    log(f"Saving merged model to {MERGED_DIR}...")
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MERGED_DIR), safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=args.hf_token)
    tokenizer.save_pretrained(str(MERGED_DIR))

    size_gb = sum(f.stat().st_size for f in MERGED_DIR.glob("*.safetensors")) / 1e9
    log(f"Merged model saved: {size_gb:.2f} GB")

    del model, tokenizer
    gc.collect()
    log("Merge complete.")


def step_gguf():
    log("=" * 55)
    log("STEP 4/5: Converting to GGUF Q8_0")
    log("=" * 55)

    llama_dir = WORKSPACE / "llama.cpp"
    if not llama_dir.exists():
        log("Cloning llama.cpp...")
        run("git clone --depth 1 https://github.com/ggerganov/llama.cpp "
            f"{llama_dir}", capture_output=True)
    else:
        log("llama.cpp already cloned.")

    run("pip install -q gguf")

    converter = llama_dir / "convert_hf_to_gguf.py"
    log(f"Converting {MERGED_DIR} → {GGUF_PATH}...")
    run(f'python "{converter}" "{MERGED_DIR}" --outfile "{GGUF_PATH}" --outtype q8_0')

    size_gb = GGUF_PATH.stat().st_size / 1e9
    log(f"GGUF created: {size_gb:.2f} GB (expected ~3.1-3.4 GB for 3B Q8)")
    if not (2.5 < size_gb < 4.5):
        raise RuntimeError(f"Unexpected GGUF size {size_gb:.2f}GB — check conversion")


def step_push_gguf():
    log("=" * 55)
    log("STEP 5/5: Uploading GGUF to HuggingFace")
    log("=" * 55)

    from huggingface_hub import HfApi, login
    login(token=args.hf_token, add_to_git_credential=False)
    api = HfApi()

    # Create repo if needed
    try:
        api.create_repo(repo_id=args.gguf_repo, repo_type="model",
                        token=args.hf_token, exist_ok=True)
    except Exception as e:
        log(f"  Repo: {e}")

    log(f"Uploading {args.gguf_name} ({GGUF_PATH.stat().st_size/1e9:.2f}GB)...")
    api.upload_file(
        path_or_fileobj=str(GGUF_PATH),
        path_in_repo=args.gguf_name,
        repo_id=args.gguf_repo,
        token=args.hf_token,
        commit_message="Avery 3B Q8_0 — Qwen2.5-3B-Instruct + SovereignNation LoRA",
    )
    url = f"https://huggingface.co/{args.gguf_repo}/resolve/main/{args.gguf_name}"
    log(f"GGUF uploaded: {url}")
    return url


# ─────────────────────────────────────────────────────────────────────────────

def main():
    log("\n" + "=" * 55)
    log("  AVERY 3B RUNPOD WORKER")
    log(f"  Base  : {args.base_model}")
    log(f"  LoRA  : {args.lora_repo}")
    log(f"  GGUF  : {args.gguf_repo}/{args.gguf_name}")
    log(f"  Epochs: {args.epochs}  MaxSeq: {args.max_seq}")
    log("=" * 55 + "\n")

    t_start = time.time()

    try:
        step_install()
        step_train()
        step_merge()
        step_gguf()
        url = step_push_gguf()

        elapsed = (time.time() - t_start) / 60
        summary = json.dumps({
            "status":   "complete",
            "gguf_url": url,
            "gguf_repo": args.gguf_repo,
            "gguf_name": args.gguf_name,
            "elapsed_min": round(elapsed, 1),
        })
        SENTINEL.write_text(summary)

        log("\n" + "=" * 55)
        log("  ALL STEPS COMPLETE")
        log(f"  Time    : {elapsed:.1f} min")
        log(f"  GGUF    : {url}")
        log(f"  Download: huggingface-cli download {args.gguf_repo} {args.gguf_name}")
        log("=" * 55)

    except Exception as e:
        log(f"\nERROR: {e}")
        import traceback
        log(traceback.format_exc())
        error_sentinel = WORKSPACE / "avery_error.txt"
        error_sentinel.write_text(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
