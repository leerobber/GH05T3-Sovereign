"""
remote_merge.py — Runs on RunPod pod.
Merges a sovereign LoRA into Qwen2-7B-Instruct, converts to GGUF Q8_0,
and uploads the GGUF to HuggingFace.

Controlled by env vars (set by runpod_merge.py launcher):
  HF_TOKEN       — HuggingFace write token
  AGENT_NAME     — e.g. avery, forge
  LORA_REPO      — e.g. tastytator/avery-sovereign-lora
  HF_REPO        — repo to upload GGUF to (usually same as LORA_REPO)
  GGUF_FILENAME  — e.g. avery-sovereign-q8.gguf
"""
import gc, json, os, subprocess, sys
from pathlib import Path

HF_TOKEN      = os.environ.get("HF_TOKEN", "")
AGENT_NAME    = os.environ.get("AGENT_NAME", "avery")
BASE_MODEL    = "Qwen/Qwen2-7B-Instruct"
LORA_REPO     = os.environ.get("LORA_REPO", f"tastytator/{AGENT_NAME}-sovereign-lora")
HF_REPO       = os.environ.get("HF_REPO", LORA_REPO)
GGUF_FILENAME = os.environ.get("GGUF_FILENAME", f"{AGENT_NAME}-sovereign-q8.gguf")
MERGED_DIR    = Path(f"/workspace/{AGENT_NAME}-merged")
GGUF_PATH     = Path(f"/workspace/{GGUF_FILENAME}")
LLAMA_DIR     = Path("/workspace/llama.cpp")
DONE_FILE     = Path("/workspace/merge_complete.txt")


def run(cmd):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed (exit {r.returncode}): {cmd}")


print(f"\n=== SOVEREIGN MERGE: {AGENT_NAME.upper()} ===")
print(f"  LoRA   : {LORA_REPO}")
print(f"  Output : {GGUF_FILENAME}")
print(f"  HF repo: {HF_REPO}")

# ── 1. Install deps ──────────────────────────────────────────────────────────
print("[1/5] Installing Python dependencies...")
run('pip install -q "transformers>=4.40.0,<5.0.0" "peft>=0.10.0,<0.15.0" accelerate huggingface_hub')

# ── 2. Merge LoRA ────────────────────────────────────────────────────────────
print("[2/5] Loading base model + LoRA on GPU...")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MERGED_DIR.mkdir(parents=True, exist_ok=True)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    device_map="cuda",
    token=HF_TOKEN,
    low_cpu_mem_usage=True,
)
model = PeftModel.from_pretrained(model, LORA_REPO, token=HF_TOKEN)
print("[2/5] Merging weights...")
model = model.merge_and_unload()
print(f"[2/5] Saving merged model to {MERGED_DIR} ...")
model.save_pretrained(str(MERGED_DIR), safe_serialization=True)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
tokenizer.save_pretrained(str(MERGED_DIR))
del model, tokenizer
gc.collect()
torch.cuda.empty_cache()
print("      Done. VRAM freed.")

# ── 3. Clone llama.cpp ───────────────────────────────────────────────────────
print("[3/5] Setting up llama.cpp...")
if not LLAMA_DIR.exists():
    run("git clone --depth 1 https://github.com/ggerganov/llama.cpp /workspace/llama.cpp")
run("pip install -q -r /workspace/llama.cpp/requirements.txt")

# ── 4. Convert to GGUF ──────────────────────────────────────────────────────
print("[4/5] Converting to GGUF Q8_0...")
run(f"python /workspace/llama.cpp/convert_hf_to_gguf.py {MERGED_DIR} "
    f"--outfile {GGUF_PATH} --outtype q8_0")
size_gb = GGUF_PATH.stat().st_size / 1e9
print(f"      GGUF size: {size_gb:.2f} GB")

# ── 5. Upload to HuggingFace ─────────────────────────────────────────────────
print("[5/5] Uploading GGUF to HuggingFace...")
from huggingface_hub import HfApi
api = HfApi(token=HF_TOKEN)
api.upload_file(
    path_or_fileobj=str(GGUF_PATH),
    path_in_repo=GGUF_FILENAME,
    repo_id=HF_REPO,
    repo_type="model",
    commit_message=f"Add Q8_0 GGUF (merged Qwen2-7B + {AGENT_NAME}-sovereign-lora)",
)
gguf_url = f"https://huggingface.co/{HF_REPO}/blob/main/{GGUF_FILENAME}"
print(f"      Uploaded: {gguf_url}")

# ── Done ─────────────────────────────────────────────────────────────────────
result = {"agent": AGENT_NAME, "gguf_url": gguf_url, "gguf_size_gb": round(size_gb, 2)}
DONE_FILE.write_text(json.dumps(result))
print("\n=== MERGE COMPLETE ===")
print(json.dumps(result, indent=2))
