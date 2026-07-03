"""
GH05T3 LoRA Fine-Tuning — Qwen2.5-Coder-7B-Instruct via unsloth.

Run via API:  POST /api/training/finetune
Run directly: python -m training.finetune

Heavy deps (install once — handled by install.ps1 on Windows):
    pip install "unsloth[cu124]" trl>=0.9.0 datasets accelerate

After training, merge + export to GGUF for Ollama:
    python -m training.finetune --merge   (creates models/gh05t3-merged-v{n}/)
    Then: ollama create gh05t3 -f models/gh05t3-merged-v{n}/Modelfile

Env vars:
    FINETUNE_BASE_MODEL   unsloth/Qwen2.5-Coder-7B-Instruct
    FINETUNE_MAX_STEPS    500
    FINETUNE_BATCH_SIZE   2
    FINETUNE_LORA_RANK    16
    FINETUNE_GRAD_ACCUM   4   (effective batch = BATCH_SIZE * GRAD_ACCUM = 8)
    FINETUNE_LR           2e-4
    FINETUNE_MAX_SEQ      2048
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from threading import Thread

LOG = logging.getLogger("ghost.training.finetune")

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

BASE_MODEL    = os.environ.get("FINETUNE_BASE_MODEL", "unsloth/Qwen2.5-Coder-7B-Instruct")
MAX_STEPS     = int(os.environ.get("FINETUNE_MAX_STEPS",  "500"))
BATCH_SIZE    = int(os.environ.get("FINETUNE_BATCH_SIZE", "2"))
LORA_RANK     = int(os.environ.get("FINETUNE_LORA_RANK",  "16"))
GRAD_ACCUM    = int(os.environ.get("FINETUNE_GRAD_ACCUM", "4"))
LEARNING_RATE = float(os.environ.get("FINETUNE_LR",       "2e-4"))
MAX_SEQ_LEN   = int(os.environ.get("FINETUNE_MAX_SEQ",    "2048"))

_state: dict = {
    "running":      False,
    "step":         0,
    "total_steps":  0,
    "loss":         None,
    "phase":        "idle",
    "error":        None,
    "output_dir":   None,
    "version":      None,
    "started_at":   None,
    "finished_at":  None,
    "dataset_size": 0,
}


def finetune_status() -> dict:
    return dict(_state)


def _next_version() -> int:
    existing = sorted(MODELS_DIR.glob("gh05t3-lora-v*/"))
    if not existing:
        return 1
    try:
        return int(existing[-1].name.split("-v")[-1]) + 1
    except ValueError:
        return len(existing) + 1


class _StepCallback:
    """Minimal transformers TrainerCallback that updates _state."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            _state["step"] = state.global_step
            _state["loss"] = round(logs["loss"], 4)
            try:
                from integrations.wandb_logger import log_training_step
                log_training_step(state.global_step, logs["loss"],
                                  _state.get("dataset_size", 0))
            except Exception:
                pass

    def on_save(self, args, state, control, **kwargs):
        LOG.info("checkpoint saved at step %d", state.global_step)


def _run_training(output_dir: Path, ws_broadcast=None):
    global _state
    try:
        # ── 1. Load model ──────────────────────────────────────
        _state["phase"] = "loading_model"
        LOG.info("Loading base model: %s (4-bit)", BASE_MODEL)

        try:
            from unsloth import FastLanguageModel
        except ImportError:
            raise RuntimeError(
                "unsloth not installed.\n"
                "Run:  pip install \"unsloth[cu124]\" trl>=0.9.0 datasets accelerate\n"
                "Then restart the backend and retry."
            )
        from trl import SFTTrainer
        from transformers import TrainingArguments, TrainerCallback

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL,
            max_seq_length=MAX_SEQ_LEN,
            dtype=None,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_RANK,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
            lora_alpha=LORA_RANK,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

        # ── 2. Load data ───────────────────────────────────────
        _state["phase"] = "loading_data"
        LOG.info("Loading training datasets...")
        from training.formatter import load_all_as_chatml, to_chatml_text
        examples = load_all_as_chatml()

        if not examples:
            raise RuntimeError(
                "No training data found in training/datasets/.\n"
                "Run POST /api/training/run first, then retry."
            )

        texts = [to_chatml_text(ex) for ex in examples]
        LOG.info("Loaded %d training examples", len(texts))
        _state["dataset_size"] = len(texts)

        from datasets import Dataset as HFDataset
        hf_ds = HFDataset.from_dict({"text": texts})

        # ── 3. Train ───────────────────────────────────────────
        _state["phase"]       = "training"
        _state["total_steps"] = MAX_STEPS

        cb = _StepCallback()

        class _CB(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                cb.on_log(args, state, control, logs=logs)
            def on_save(self, args, state, control, **kwargs):
                cb.on_save(args, state, control)

        args = TrainingArguments(
            output_dir=str(output_dir),
            max_steps=MAX_STEPS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            warmup_steps=20,
            learning_rate=LEARNING_RATE,
            fp16=True,
            bf16=False,
            logging_steps=10,
            save_steps=100,
            optim="adamw_8bit",
            lr_scheduler_type="cosine",
            seed=42,
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=hf_ds,
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LEN,
            args=args,
            callbacks=[_CB()],
        )

        LOG.info("Starting LoRA training: %d steps, lr=%s, rank=%d",
                 MAX_STEPS, LEARNING_RATE, LORA_RANK)
        trainer.train()

        # ── 4. Save adapter ────────────────────────────────────
        _state["phase"] = "saving"
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        _write_modelfile(output_dir)

        # ── 5. W&B finish + notifications ──────────────────────
        final_loss = _state.get("loss") or 0.0
        try:
            from integrations.wandb_logger import log_training_complete
            log_training_complete(
                version=_state["version"],
                final_loss=final_loss,
                steps=MAX_STEPS,
                dataset_size=len(texts),
                model=BASE_MODEL,
            )
        except Exception:
            pass

        try:
            import asyncio
            from integrations.notifier import notify_finetune_complete
            asyncio.run(notify_finetune_complete(
                _state["version"], MAX_STEPS, str(output_dir)))
        except Exception:
            pass

        _state["phase"]       = "complete"
        _state["output_dir"]  = str(output_dir)
        _state["finished_at"] = time.time()
        LOG.info("Fine-tune complete → %s", output_dir)
        LOG.info("To deploy: ollama create gh05t3 -f \"%s/Modelfile\"", output_dir)

    except Exception as e:
        LOG.exception("Fine-tune failed")
        _state["phase"]       = "error"
        _state["error"]       = str(e)
        _state["finished_at"] = time.time()
        try:
            import asyncio
            from integrations.notifier import notify_error
            asyncio.run(notify_error("finetune", str(e)[:200]))
        except Exception:
            pass
    finally:
        _state["running"] = False


def _write_modelfile(output_dir: Path):
    """Write an Ollama Modelfile so the user can load with 'ollama create'."""
    content = f"""\
FROM ./model.gguf
SYSTEM \"\"\"You are GH05T3, an autonomous security and reasoning agent.
You think carefully, reason step-by-step, and always prioritize defense over exploitation.\"\"\"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"
"""
    (output_dir / "Modelfile.template").write_text(content)
    instructions = f"""\
# GH05T3 LoRA adapter — deployment instructions

## 1. Merge adapter with base model (in your backend venv):
python -m training.finetune --merge --version {_state.get('version', 1)}

## 2. Convert to GGUF (requires llama.cpp):
python llama.cpp/convert_hf_to_gguf.py models/gh05t3-merged-v{_state.get('version', 1)} --outfile models/gh05t3-merged-v{_state.get('version', 1)}/model.gguf --outtype q4_k_m

## 3. Load into Ollama:
ollama create gh05t3 -f models/gh05t3-merged-v{_state.get('version', 1)}/Modelfile.template

## 4. Update .env to use the fine-tuned model:
OLLAMA_PROPOSER=gh05t3
LLM_PROVIDER=ollama
"""
    (output_dir / "DEPLOY.md").write_text(instructions)


async def run_finetune(ws_broadcast=None) -> dict:
    """Start fine-tuning in a background thread. Returns immediately."""
    if _state["running"]:
        return {"status": "already_running", "state": finetune_status()}

    version    = _next_version()
    output_dir = MODELS_DIR / f"gh05t3-lora-v{version}"
    output_dir.mkdir(parents=True, exist_ok=True)

    _state.update({
        "running":      True,
        "step":         0,
        "total_steps":  MAX_STEPS,
        "loss":         None,
        "phase":        "starting",
        "error":        None,
        "output_dir":   str(output_dir),
        "version":      version,
        "started_at":   time.time(),
        "finished_at":  None,
        "dataset_size": 0,
        "base_model":   BASE_MODEL,
        "lora_rank":    LORA_RANK,
        "max_steps":    MAX_STEPS,
    })

    try:
        from integrations.wandb_logger import init_run
        init_run(run_name=f"gh05t3-lora-v{version}", job_type="finetune")
    except Exception:
        pass

    thread = Thread(target=_run_training, args=(output_dir, ws_broadcast), daemon=True)
    thread.start()
    LOG.info("Fine-tune thread started → v%d", version)
    return {"status": "started", "version": version, "output_dir": str(output_dir)}


def merge_adapter(version: int | None = None):
    """
    Merge LoRA adapter into base model and save to models/gh05t3-merged-v{n}/.
    Run from CLI: python -m training.finetune --merge
    Requires the same unsloth deps as training.
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        raise RuntimeError("unsloth not installed — run: pip install \"unsloth[cu124]\"")

    if version is None:
        candidates = sorted(MODELS_DIR.glob("gh05t3-lora-v*/"))
        if not candidates:
            raise RuntimeError("No LoRA adapter found. Train first.")
        adapter_dir = candidates[-1]
        version = int(adapter_dir.name.split("-v")[-1])
    else:
        adapter_dir = MODELS_DIR / f"gh05t3-lora-v{version}"

    merged_dir = MODELS_DIR / f"gh05t3-merged-v{version}"
    merged_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Loading adapter from %s ...", adapter_dir)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=False,
    )
    LOG.info("Merging and saving to %s ...", merged_dir)
    model.save_pretrained_merged(str(merged_dir), tokenizer,
                                 save_method="merged_16bit")
    _write_modelfile(merged_dir)
    LOG.info("Merge complete: %s", merged_dir)
    LOG.info("Next: convert to GGUF with llama.cpp, then: ollama create gh05t3 -f %s/Modelfile.template", merged_dir)
    return str(merged_dir)


# ── CLI entry point ────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import asyncio
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if "--merge" in sys.argv:
        v = None
        for a in sys.argv:
            if a.startswith("--version="):
                v = int(a.split("=")[1])
        merge_adapter(version=v)
    else:
        asyncio.run(run_finetune())
