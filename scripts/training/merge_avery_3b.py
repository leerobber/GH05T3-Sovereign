"""
merge_avery_3b.py
=================
Post-Kaggle rebuild for Avery on Qwen2.5-3B-Instruct base.

Steps:
  1. Pull LoRA from tastytator/avery-sovereign-lora (HuggingFace)
  2. Merge LoRA into Qwen2.5-3B-Instruct base weights (CPU, float16)
  3. Convert merged model to GGUF Q8_0 via llama.cpp
  4. Write Modelfile pointing to new 3B GGUF
  5. ollama create avery-sovereign:latest
  6. Smoke test â€” verify Avery responds correctly

Run: python merge_avery_3b.py
     python merge_avery_3b.py --skip-merge   (if merged dir already exists)
     python merge_avery_3b.py --test-only    (smoke test current ollama model)
"""

import argparse
from gc import gc
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

# â”€â”€ Config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ROOT        = Path(__file__).parent
WORK_DIR    = ROOT / "avery-3b-merged"        # merged safetensors output
GGUF_PATH   = ROOT / "avery-3b-sovereign-q8.gguf"
MODELFILE   = ROOT / "Modelfile.avery-3b"
LLAMA_CPP   = Path("C:/llama.cpp/convert_hf_to_gguf.py")
HF_TOKEN    = Path("C:/Users/leer4/.cache/huggingface/token").read_text().strip()
BASE_MODEL  = "Qwen/Qwen2.5-3B-Instruct"
LORA_REPO   = "tastytator/avery-sovereign-lora"
OLLAMA_NAME = "avery-sovereign"
LOG_FILE    = ROOT / "merge_avery_3b.log"

SYSTEM = (
    "You are Avery, the sovereign business strategist for SovereignNation â€” "
    "a fixed-cost AI platform built for professional services firms, lower and "
    "middle class families, children's education, and affordable connectivity. "
    "Use the KAIROS framework: Kickoff, Alignment, Implementation, Refinement, "
    "Optimization, Scaling. Be direct, structured, and actionable."
)

# â”€â”€ Logging â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("merge_avery")


# â”€â”€ Step 1+2: Merge LoRA into base model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def step_merge() -> bool:
    log.info("=" * 60)
    log.info("  STEP 1-2: Merge LoRA -> %s", BASE_MODEL)
    log.info("=" * 60)

    # Skip if already merged
    safetensors = list(WORK_DIR.glob("*.safetensors"))
    if safetensors and WORK_DIR.exists():
        total_size = sum(f.stat().st_size for f in safetensors)
        log.info("Merged weights already present (%d files, %.1f GB) â€” skipping.",
                 len(safetensors), total_size / 1e9)
        return True

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as e:
        log.error("Missing dependency: %s  (pip install transformers peft torch)", e)
        return False

    t0 = time.time()
    log.info("Loading base model %s on CPU (float16)...", BASE_MODEL)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="cpu",
            token=HF_TOKEN,
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        log.error("Failed to load base model: %s", e)
        return False

    log.info("Loading LoRA adapter from %s ...", LORA_REPO)
    try:
        model = PeftModel.from_pretrained(model, LORA_REPO, token=HF_TOKEN)
    except Exception as e:
        log.error("Failed to load LoRA: %s", e)
        log.error("Has Kaggle training finished? Check: https://huggingface.co/%s", LORA_REPO)
        return False

    log.info("Merging LoRA into base weights...")
    model = model.merge_and_unload()

    log.info("Saving merged model to %s ...", WORK_DIR)
    model.save_pretrained(str(WORK_DIR), safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
    tokenizer.save_pretrained(str(WORK_DIR))

    elapsed = (time.time() - t0) / 60
    log.info("Merge complete in %.1f min. Freeing memory...", elapsed)
    del model, tokenizer
    gc.collect()

    safetensors_after = list(WORK_DIR.glob("*.safetensors"))
    log.info("Saved %d safetensors files.", len(safetensors_after))
    return True


# â”€â”€ Step 3: GGUF conversion â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def step_gguf() -> bool:
    log.info("=" * 60)
    log.info("  STEP 3: Convert to GGUF Q8_0")
    log.info("=" * 60)

    if GGUF_PATH.exists() and GGUF_PATH.stat().st_size > 100_000_000:
        log.info("GGUF already exists (%.2f GB) â€” skipping.", GGUF_PATH.stat().st_size / 1e9)
        return True

    if not LLAMA_CPP.exists():
        log.error("llama.cpp not found at %s", LLAMA_CPP)
        return False

    log.info("Running convert_hf_to_gguf.py...")
    result = subprocess.run(
        [sys.executable, str(LLAMA_CPP),
         str(WORK_DIR),
         "--outfile", str(GGUF_PATH),
         "--outtype", "q8_0"],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        log.error("GGUF conversion FAILED:\n%s", result.stderr[-2000:])
        return False

    size_gb = GGUF_PATH.stat().st_size / 1e9
    log.info("GGUF created: %.2f GB at %s", size_gb, GGUF_PATH)

    # Sanity: 3B Q8 should be ~3.1-3.4 GB
    if size_gb < 2.0 or size_gb > 5.0:
        log.warning("Unexpected GGUF size %.2f GB (expected ~3.1-3.4 GB for 3B Q8)", size_gb)

    return True


# â”€â”€ Step 4: Modelfile â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def step_modelfile() -> bool:
    log.info("=" * 60)
    log.info("  STEP 4: Write Modelfile")
    log.info("=" * 60)

    MODELFILE.write_text(
        f"FROM {GGUF_PATH.as_posix()}\n"
        f'SYSTEM """{SYSTEM}"""\n'
        "PARAMETER temperature 0.7\n"
        "PARAMETER top_p 0.9\n"
        "PARAMETER num_ctx 4096\n",
        encoding="utf-8"
    )
    log.info("Modelfile written: %s", MODELFILE)
    return True


# â”€â”€ Step 5: ollama create â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def step_ollama_create() -> bool:
    log.info("=" * 60)
    log.info("  STEP 5: ollama create %s", OLLAMA_NAME)
    log.info("=" * 60)

    result = subprocess.run(
        ["ollama", "create", OLLAMA_NAME, "-f", str(MODELFILE)],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        log.error("ollama create FAILED:\n%s", result.stderr)
        return False

    log.info("ollama model '%s' created successfully.", OLLAMA_NAME)

    # Verify it appears in ollama list
    check = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if OLLAMA_NAME in check.stdout:
        # Get size from list
        for line in check.stdout.splitlines():
            if OLLAMA_NAME in line:
                log.info("Ollama confirms: %s", line.strip())
    else:
        log.warning("'%s' not found in ollama list after create.", OLLAMA_NAME)

    return True


# â”€â”€ Step 6: Smoke test â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def step_smoke_test() -> bool:
    log.info("=" * 60)
    log.info("  STEP 6: Smoke test")
    log.info("=" * 60)

    import urllib.request

    prompt = (
        "A CPA firm has 12 staff and handles 200 business clients. "
        "They want to use AI for tax analysis but are worried about client data privacy. "
        "Give me a 3-step plan using the KAIROS framework."
    )

    payload = json.dumps({
        "model":  OLLAMA_NAME,
        "prompt": prompt,
        "stream": False,
    }).encode()

    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        log.info("Sending test prompt to %s (may take 10-30s)...", OLLAMA_NAME)
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())

        response = data.get("response", "")
        word_count = len(response.split())

        if word_count < 30:
            log.error("SMOKE TEST FAILED â€” response too short (%d words): %s",
                      word_count, response[:200])
            return False

        log.info("SMOKE TEST PASSED â€” %d words", word_count)
        log.info("Response preview:\n%s", response[:500])

        # Check for KAIROS in response (Avery should use her framework)
        if any(k in response for k in ["KAIROS", "Kickoff", "Alignment", "Implementation"]):
            log.info("KAIROS framework detected in response â€” avery is aligned!")

        return True

    except Exception as e:
        log.error("SMOKE TEST ERROR: %s", e)
        return False


# â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def main(skip_merge: bool = False, test_only: bool = False):
    log.info("\n" + "=" * 60)
    log.info("  Avery 3B Rebuild Pipeline")
    log.info("  Base   : %s", BASE_MODEL)
    log.info("  LoRA   : %s", LORA_REPO)
    log.info("  Output : %s", GGUF_PATH)
    log.info("=" * 60 + "\n")

    if test_only:
        ok = step_smoke_test()
        sys.exit(0 if ok else 1)

    steps = [
        ("merge",        step_merge,        skip_merge),
        ("gguf",         step_gguf,         False),
        ("modelfile",    step_modelfile,    False),
        ("ollama_create",step_ollama_create,False),
        ("smoke_test",   step_smoke_test,   False),
    ]

    for step_name, step_fn, skip in steps:
        if skip:
            log.info("Skipping step: %s (--skip-merge)", step_name)
            continue
        ok = step_fn()
        if not ok:
            log.error("\nPIPELINE FAILED at step: %s", step_name)
            log.error("Check log: %s", LOG_FILE)
            sys.exit(1)
        gc.collect()

    log.info("\n" + "=" * 60)
    log.info("  AVERY 3B REBUILD COMPLETE")
    log.info("  Model  : %s (in Ollama)", OLLAMA_NAME)
    log.info("  GGUF   : %s", GGUF_PATH)
    log.info("  Log    : %s", LOG_FILE)
    log.info("=" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-merge", action="store_true",
                    help="Skip merge step if avery-3b-merged dir already exists")
    ap.add_argument("--test-only", action="store_true",
                    help="Only run smoke test on current avery-sovereign model")
    args = ap.parse_args()
    main(skip_merge=args.skip_merge, test_only=args.test_only)
