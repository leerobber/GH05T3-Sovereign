"""
SovereignNation — Merge All Agent LoRA Adapters
================================================
Merges each fine-tuned LoRA adapter with its base model,
converts to GGUF (Q4_K_M), writes a Modelfile, and
re-creates the Ollama model — replacing the base model
placeholder with the actual fine-tuned sovereign agent.

Agents processed: forge, oracle, sentinel, codex
(avery and nexus will be added after their retrains complete)

Run: python merge_all_agents.py
Logs to: C:/Users/leer4/GH05T3/merge_all_agents.log
"""

import os, sys, gc, subprocess, time, logging
from pathlib import Path

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_FILE = Path("C:/Users/leer4/GH05T3/merge_all_agents.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("merge_all")

# ── Config ─────────────────────────────────────────────────────────────────────
HF_TOKEN   = open("C:/Users/leer4/.cache/huggingface/token").read().strip()
BASE_MODEL = "Qwen/Qwen2-1.5B-Instruct"
WORK_DIR   = Path("C:/Users/leer4/GH05T3/sovereign-merged")
LLAMA_CPP  = "C:/llama.cpp/convert_hf_to_gguf.py"
WORK_DIR.mkdir(parents=True, exist_ok=True)

# ── Agent definitions ───────────────────────────────────────────────────────────
AGENTS = {
    "forge": {
        "lora_repo": "tastytator/forge-sovereign-lora",
        "ollama_name": "forge-sovereign",
        "system": (
            "You are FORGE, the sovereign code generation specialist for SovereignNation. "
            "Write production-ready Python, JavaScript, and TypeScript with imports, "
            "error handling, type hints, and comments for non-obvious logic."
        ),
    },
    "oracle": {
        "lora_repo": "tastytator/oracle-sovereign-lora",
        "ollama_name": "oracle-sovereign",
        "system": (
            "You are ORACLE, the sovereign memory and retrieval specialist. "
            "Synthesize information into precise structured answers. "
            "Cite source type (memory / document / inference). Be concise."
        ),
    },
    "sentinel": {
        "lora_repo": "tastytator/sentinel-sovereign-lora",
        "ollama_name": "sentinel-sovereign",
        "system": (
            "You are SENTINEL, the sovereign security specialist. "
            "Review code for vulnerabilities. Reference OWASP Top 10 and CWE. "
            "State: vulnerability, impact (low/med/high/critical), specific fix."
        ),
    },
    "codex": {
        "lora_repo": "tastytator/codex-sovereign-lora",
        "ollama_name": "codex-sovereign",
        "system": (
            "You are CODEX, the sovereign documentation specialist. "
            "Write clear complete technical documentation with markdown, "
            "code blocks, and examples."
        ),
    },
}


def merge_agent(name: str, cfg: dict) -> bool:
    """Merge one agent's LoRA, convert to GGUF, load into Ollama. Returns True on success."""
    log.info(f"\n{'='*60}")
    log.info(f"  STARTING AGENT: {name.upper()}")
    log.info(f"{'='*60}")

    merged_dir = WORK_DIR / f"{name}-merged"
    gguf_path  = WORK_DIR / f"{name}-sovereign-q8.gguf"
    mf_path    = WORK_DIR / f"Modelfile.{name}"

    # ── Skip GGUF if it already exists ─────────────────────────────────────────
    if gguf_path.exists() and gguf_path.stat().st_size > 100_000_000:
        log.info(f"  GGUF already exists ({gguf_path.stat().st_size/1e9:.2f} GB) — skipping merge+convert.")
    else:
        merged_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Merge (skip if safetensors already present from prior run)
        safetensors_files = list(merged_dir.glob("*.safetensors"))
        if safetensors_files:
            log.info(f"[1/4] Merged weights already present ({len(safetensors_files)} safetensors files) — skipping merge.")
        else:
            log.info(f"[1/4] Loading base model {BASE_MODEL} on CPU...")
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            t0 = time.time()
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.float16,
                device_map="cpu",
                token=HF_TOKEN,
                low_cpu_mem_usage=True,
            )
            log.info(f"[1/4] Loading LoRA adapter from {cfg['lora_repo']} ...")
            model = PeftModel.from_pretrained(model, cfg["lora_repo"], token=HF_TOKEN)

            log.info("[1/4] Merging LoRA into base weights...")
            model = model.merge_and_unload()

            log.info(f"[1/4] Saving merged model to {merged_dir} ...")
            model.save_pretrained(str(merged_dir), safe_serialization=True)

            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
            tokenizer.save_pretrained(str(merged_dir))

            del model, tokenizer
            gc.collect()
            log.info(f"      Merge done in {(time.time()-t0)/60:.1f} min. Memory freed.")

        # Step 2: Convert to GGUF Q8_0
        log.info(f"[2/4] Converting to GGUF Q8_0 -> {gguf_path} ...")
        result = subprocess.run([
            sys.executable, LLAMA_CPP,
            str(merged_dir),
            "--outfile", str(gguf_path),
            "--outtype", "q8_0",
        ], capture_output=True, text=True)
        if result.returncode != 0:
            log.error(f"      GGUF conversion FAILED:\n{result.stderr}")
            return False
        log.info(f"      GGUF done: {gguf_path.stat().st_size/1e9:.2f} GB")

    # Step 3: Modelfile
    log.info("[3/4] Writing Ollama Modelfile...")
    mf_path.write_text(
        f"FROM {gguf_path.as_posix()}\n"
        f'SYSTEM """{cfg["system"]}"""\n'
        "PARAMETER temperature 0.7\n"
        "PARAMETER top_p 0.9\n"
        "PARAMETER num_ctx 4096\n"
    )
    log.info(f"      Written to {mf_path}")

    # Step 4: Ollama create
    log.info(f"[4/4] Creating Ollama model '{cfg['ollama_name']}' ...")
    result = subprocess.run(
        ["ollama", "create", cfg["ollama_name"], "-f", str(mf_path)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        log.error(f"      ollama create FAILED:\n{result.stderr}")
        return False
    log.info(f"      [OK] {cfg['ollama_name']} is live in Ollama!")
    return True


def test_agent(name: str, ollama_name: str) -> bool:
    """Quick smoke test — send a single prompt and check for non-empty response."""
    log.info(f"[TEST] Smoke-testing {ollama_name}...")
    test_prompts = {
        "forge":    "Write a Python function that validates an email address.",
        "oracle":   "What is the capital of France? Cite source type.",
        "sentinel": "Review this code for security issues: query = 'SELECT * FROM users WHERE id=' + user_input",
        "codex":    "Write a one-paragraph description of what a REST API is.",
    }
    prompt = test_prompts.get(name, "Say hello and introduce yourself briefly.")
    try:
        import requests, json
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": ollama_name, "prompt": prompt, "stream": False},
            timeout=120,
        )
        data = resp.json()
        response_text = data.get("response", "")
        if len(response_text) > 20:
            log.info(f"[TEST] PASS {ollama_name} responded ({len(response_text)} chars)")
            log.info(f"[TEST]    Preview: {response_text[:200]}...")
            return True
        else:
            log.error(f"[TEST] FAIL {ollama_name} response too short: {response_text!r}")
            return False
    except Exception as e:
        log.error(f"[TEST] FAIL {ollama_name} test FAILED: {e}")
        return False


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("SovereignNation — Agent Merge Pipeline")
    log.info(f"Base model : {BASE_MODEL}")
    log.info(f"Work dir   : {WORK_DIR}")
    log.info(f"Agents     : {list(AGENTS.keys())}")
    log.info("")

    results = {}
    for agent_name, agent_cfg in AGENTS.items():
        try:
            ok = merge_agent(agent_name, agent_cfg)
            if ok:
                ok = test_agent(agent_name, agent_cfg["ollama_name"])
            results[agent_name] = "DONE" if ok else "FAILED"
        except Exception as e:
            log.exception(f"Unhandled error for {agent_name}: {e}")
            results[agent_name] = f"❌ ERROR: {e}"
        gc.collect()

    log.info("\n" + "="*60)
    log.info("  FINAL RESULTS")
    log.info("="*60)
    for name, status in results.items():
        log.info(f"  {name:12s} → {status}")
    log.info("="*60)
    log.info(f"Log saved to: {LOG_FILE}")
