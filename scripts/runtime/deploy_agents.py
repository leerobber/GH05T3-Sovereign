"""
deploy_agents.py â€” Merge all 6 sovereign LoRAs and load into Ollama.

Usage:
    python deploy_agents.py              # deploy all 6 agents
    python deploy_agents.py avery        # deploy one agent
    python deploy_agents.py --list       # show what would be deployed

Each agent:
  1. Downloads LoRA from HuggingFace
  2. Merges into base model on CPU
  3. Converts to GGUF Q8_0 (best quality supported by convert_hf_to_gguf.py)
  4. Writes Modelfile
  5. Runs: ollama create <agent>-sovereign
"""

import argparse
from gc import gc
import io
import os
import subprocess
import sys
from pathlib import Path

# Force UTF-8 output on Windows to avoid charmap encoding errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / "backend" / ".env")
load_dotenv(Path(__file__).parent / ".env")

HF_TOKEN   = os.environ.get("HF_TOKEN", "")
BASE_MODEL = "Qwen/Qwen2-1.5B-Instruct"   # all Kaggle runs used 1.5B (P100 fallback)
LLAMA_CPP  = Path("C:/llama.cpp/convert_hf_to_gguf.py")
OUT_DIR    = Path("C:/Users/leer4/GH05T3/ollama_models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

AGENTS = {
    "avery": {
        "lora":   "tastytator/avery-sovereign-lora",
        "system": (
            "You are Avery, the sovereign business strategist for SovereignNation. "
            "Use the KAIROS framework: Kickoff, Alignment, Implementation, Refinement, "
            "Optimization, Scaling. Be direct, structured, actionable."
        ),
    },
    "forge": {
        "lora":   "tastytator/forge-sovereign-lora",
        "system": (
            "You are FORGE, the sovereign code generation specialist for SovereignNation. "
            "Write production-ready Python, JavaScript, and TypeScript with imports, "
            "error handling, type hints, and comments for non-obvious logic."
        ),
    },
    "oracle": {
        "lora":   "tastytator/oracle-sovereign-lora",
        "system": (
            "You are ORACLE, the sovereign memory and retrieval specialist. "
            "Synthesize information into precise structured answers. "
            "Cite source type (memory / document / inference). Be concise."
        ),
    },
    "codex": {
        "lora":   "tastytator/codex-sovereign-lora",
        "system": (
            "You are CODEX, the sovereign documentation specialist. "
            "Write clear complete technical documentation with markdown, code blocks, examples."
        ),
    },
    "sentinel": {
        "lora":   "tastytator/sentinel-sovereign-lora",
        "system": (
            "You are SENTINEL, the sovereign security specialist. "
            "Review code for vulnerabilities. Reference OWASP Top 10 and CWE. "
            "State: vulnerability, impact (low/med/high/critical), specific fix."
        ),
    },
    "nexus": {
        "lora":   "tastytator/nexus-sovereign-lora",
        "system": (
            "You are NEXUS, the sovereign orchestration specialist. "
            "Coordinate agents and design structured task workflows with sequence, "
            "parallelism, dependencies, and which agent owns each step."
        ),
    },
}


def deploy_agent(name: str, cfg: dict):
    print(f"\n{'='*60}")
    print(f"  DEPLOYING {name.upper()}")
    print(f"{'='*60}")

    merged_dir = OUT_DIR / f"{name}-merged"
    gguf_path  = OUT_DIR / f"{name}-sovereign-q8.gguf"
    modelfile  = OUT_DIR / f"Modelfile.{name}"
    ollama_name = f"{name}-sovereign"

    # â”€â”€ 1. Merge â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if merged_dir.exists() and any(merged_dir.iterdir()):
        print(f"[1/4] Merged model already exists at {merged_dir} â€” skipping merge.")
    else:
        print(f"[1/4] Loading base model {BASE_MODEL} on CPU...")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            dtype=torch.float16,
            device_map="cpu",
            token=HF_TOKEN,
        )
        print(f"[1/4] Loading LoRA {cfg['lora']}...")
        model = PeftModel.from_pretrained(model, cfg["lora"], token=HF_TOKEN)
        print("[1/4] Merging weights...")
        model = model.merge_and_unload()
        merged_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(merged_dir), safe_serialization=True)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
        tokenizer.save_pretrained(str(merged_dir))
        del model, tokenizer
        gc.collect()
        print("      Merge done.")

    # â”€â”€ 2. Convert to GGUF â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if gguf_path.exists():
        print(f"[2/4] GGUF already exists at {gguf_path} â€” skipping conversion.")
    else:
        print(f"[2/4] Converting to GGUF Q8_0 -> {gguf_path} ...")
        subprocess.run([
            sys.executable, str(LLAMA_CPP),
            str(merged_dir),
            "--outfile", str(gguf_path),
            "--outtype", "q8_0",
        ], check=True)
        print("      Conversion done.")

    # â”€â”€ 3. Modelfile â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("[3/4] Writing Modelfile...")
    modelfile.write_text(
        f'FROM {gguf_path.as_posix()}\n'
        f'SYSTEM """{cfg["system"]}"""\n'
        'PARAMETER temperature 0.7\n'
        'PARAMETER top_p 0.9\n'
        'PARAMETER num_ctx 4096\n',
        encoding="utf-8",
    )

    # â”€â”€ 4. Load into Ollama â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"[4/4] Loading into Ollama as '{ollama_name}'...")
    result = subprocess.run(
        ["ollama", "create", ollama_name, "-f", str(modelfile)],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"      Done. Run with: ollama run {ollama_name}")
    else:
        print(f"      WARNING: ollama create failed:\n{result.stderr}")

    print(f"\n  {name.upper()} deployed as '{ollama_name}'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agent", nargs="?", help="agent name (or 'all')")
    ap.add_argument("--list", action="store_true", help="show agents without deploying")
    args = ap.parse_args()

    if args.list:
        print("\nAgents available for deployment:")
        for name, cfg in AGENTS.items():
            gguf = OUT_DIR / f"{name}-sovereign-q8.gguf"
            status = "GGUF exists" if gguf.exists() else "needs merge+convert"
            print(f"  {name:<10}  {cfg['lora']:<42}  [{status}]")
        return

    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set. Add it to backend/.env or .env")
        sys.exit(1)

    if not LLAMA_CPP.exists():
        print(f"ERROR: llama.cpp not found at {LLAMA_CPP}")
        sys.exit(1)

    targets = list(AGENTS.items())
    if args.agent and args.agent != "all":
        if args.agent not in AGENTS:
            print(f"ERROR: unknown agent '{args.agent}'. Choose from: {', '.join(AGENTS)}")
            sys.exit(1)
        targets = [(args.agent, AGENTS[args.agent])]

    print(f"\nDeploying {len(targets)} agent(s): {', '.join(n for n, _ in targets)}")
    print(f"Base model: {BASE_MODEL}")
    print(f"Output:     {OUT_DIR}")

    for name, cfg in targets:
        try:
            deploy_agent(name, cfg)
        except Exception as e:
            print(f"\nERROR deploying {name}: {e}")
            print("Continuing with next agent...")

    print("\n" + "="*60)
    print("  ALL DONE")
    print("  Test with:  ollama run avery-sovereign")
    print("="*60)


if __name__ == "__main__":
    main()
