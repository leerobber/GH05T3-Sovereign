"""
Merge avery-sovereign-lora into Qwen2-7B-Instruct and convert to GGUF.
"""
import os, sys, gc, subprocess
from pathlib import Path

HF_TOKEN   = os.environ.get("HF_TOKEN", "")
BASE_MODEL = "Qwen/Qwen2-7B-Instruct"
LORA_REPO  = "tastytator/avery-sovereign-lora"
MERGED_DIR = Path("C:/Users/leer4/GH05T3/avery-merged")
GGUF_PATH  = Path("C:/Users/leer4/GH05T3/avery-sovereign-q8.gguf")
MODELFILE  = Path("C:/Users/leer4/GH05T3/Modelfile.avery")
LLAMA_CPP  = "C:/llama.cpp/convert_hf_to_gguf.py"

MERGED_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Merge ──────────────────────────────────────────────────────────────────
print("[1/4] Loading base model on CPU (this takes a few minutes)...")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    device_map="cpu",
    token=HF_TOKEN,
    low_cpu_mem_usage=True,
)

print("[1/4] Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, LORA_REPO, token=HF_TOKEN)

print("[1/4] Merging LoRA weights...")
model = model.merge_and_unload()

print(f"[1/4] Saving merged model to {MERGED_DIR} ...")
model.save_pretrained(str(MERGED_DIR), safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
tokenizer.save_pretrained(str(MERGED_DIR))

del model, tokenizer
gc.collect()
print("      Done. Memory freed.")

# ── 2. Convert to GGUF ────────────────────────────────────────────────────────
print(f"[2/4] Converting to GGUF Q8_0 → {GGUF_PATH} ...")
subprocess.run([
    sys.executable, LLAMA_CPP,
    str(MERGED_DIR),
    "--outfile", str(GGUF_PATH),
    "--outtype", "q8_0",
], check=True)
print("      Conversion done.")

# ── 3. Modelfile ──────────────────────────────────────────────────────────────
print("[3/4] Writing Ollama Modelfile...")
MODELFILE.write_text(
    f'FROM {GGUF_PATH.as_posix()}\n'
    'SYSTEM """You are Avery, a sovereign business strategist for SovereignNation — '
    'a fixed-cost AI platform built for lower and middle class users, children\'s '
    'education, and affordable connectivity. Be direct, structured, and strategic."""\n'
    'PARAMETER temperature 0.7\n'
    'PARAMETER top_p 0.9\n'
)
print(f"      Written to {MODELFILE}")

# ── 4. Summary ────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  DONE — next steps:")
print(f"  ollama create avery-sovereign -f {MODELFILE}")
print( "  ollama run avery-sovereign")
print("=" * 60)
