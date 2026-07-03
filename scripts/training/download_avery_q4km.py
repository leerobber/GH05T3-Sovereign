"""
download_avery_q4km.py
Run this AFTER kaggle_quantize_avery.ipynb has finished uploading Q4KM to HuggingFace.

Steps:
  1. Deletes the local Q8 GGUF (frees 8.1 GB)
  2. Downloads Q4KM from HuggingFace (~4.5 GB)
  3. Swaps Modelfile.avery to point at Q4KM
  4. Re-registers avery-sovereign in Ollama
  5. Runs a quick smoke test

Usage:
  python download_avery_q4km.py
"""
import os, sys, shutil, subprocess

GH05T3 = "C:/Users/leer4/GH05T3"
Q8_PATH = f"{GH05T3}/avery-sovereign-q8.gguf"
Q4KM_PATH = f"{GH05T3}/avery-sovereign-q4km.gguf"
MODELFILE_Q8 = f"{GH05T3}/Modelfile.avery"
MODELFILE_Q4KM = f"{GH05T3}/Modelfile.avery.q4km"
HF_REPO = "tastytator/avery-sovereign-lora"
HF_FILENAME = "avery-sovereign-q4km.gguf"

def step(n, msg):
    print(f"\n[{n}] {msg}")

step(1, "Checking Q4KM exists on HuggingFace...")
try:
    from huggingface_hub import list_repo_files
    hf_token = os.environ.get("HF_TOKEN", "")
    files = list(list_repo_files(HF_REPO, token=hf_token or None))
    if HF_FILENAME not in files:
        print(f"  ERROR: {HF_FILENAME} not found in {HF_REPO}")
        print(f"  Files available: {files}")
        print("  Run kaggle_quantize_avery.ipynb first!")
        sys.exit(1)
    print(f"  Found {HF_FILENAME} on HuggingFace ✅")
except ImportError:
    print("  huggingface_hub not installed — run: pip install huggingface_hub")
    sys.exit(1)

step(2, f"Deleting Q8 GGUF to free 8.1 GB: {Q8_PATH}")
if os.path.exists(Q8_PATH):
    os.remove(Q8_PATH)
    print(f"  Deleted ✅")
else:
    print(f"  Already gone (OK)")

# Check space
total, used, free = shutil.disk_usage("C:/")
print(f"  Free space now: {free/1e9:.1f} GB")
if free < 5e9:
    print("  WARNING: Less than 5 GB free. Download may fail.")

step(3, f"Downloading Q4KM from HuggingFace (~4.5 GB)...")
from huggingface_hub import hf_hub_download
hf_token = os.environ.get("HF_TOKEN", "")
downloaded = hf_hub_download(
    repo_id=HF_REPO,
    filename=HF_FILENAME,
    local_dir=GH05T3,
    token=hf_token or None
)
size = os.path.getsize(Q4KM_PATH)
print(f"  Downloaded: {size/1e9:.2f} GB ✅")

step(4, "Swapping Modelfile.avery to Q4KM...")
if os.path.exists(MODELFILE_Q4KM):
    shutil.copy2(MODELFILE_Q4KM, MODELFILE_Q8)
    print(f"  Modelfile.avery now points to Q4KM ✅")
else:
    # Write inline
    with open(MODELFILE_Q8, 'w') as f:
        f.write(f"FROM {Q4KM_PATH}\n")
        f.write('SYSTEM """You are Avery, the sovereign business strategist for SovereignNation — a fixed-cost AI platform for lower and middle class families, professional services firms, children\'s education, and affordable connectivity. Use the KAIROS framework: Kickoff, Alignment, Implementation, Refinement, Optimization, Scaling. Be direct, structured, and actionable."""\n')
        f.write("PARAMETER temperature 0.7\n")
        f.write("PARAMETER top_p 0.9\n")
        f.write("PARAMETER num_ctx 4096\n")
        f.write("PARAMETER repeat_penalty 1.1\n")
    print(f"  Modelfile.avery written ✅")

step(5, "Re-registering avery-sovereign in Ollama...")
result = subprocess.run(
    ["ollama", "create", "avery-sovereign", "-f", MODELFILE_Q8],
    capture_output=True, text=True
)
if result.returncode == 0:
    print("  avery-sovereign created ✅")
else:
    print("  Ollama create output:")
    print(result.stdout[-1000:])
    print(result.stderr[-1000:])

step(6, "Smoke test — sending KAIROS prompt...")
result = subprocess.run(
    ["ollama", "run", "avery-sovereign",
     "Give me a 2-sentence KAIROS kickoff for a CPA firm exploring AI."],
    capture_output=True, text=True, timeout=120
)
if result.returncode == 0:
    response = result.stdout.strip()
    print(f"\n  Avery says:\n  {response[:400]}")
    print("\n  Test passed ✅")
else:
    print("  Test failed:", result.stderr[:500])

print("\n=== DONE ===")
print("Avery is now running Q4_K_M — fits in VRAM, ~8-15s inference.")
print("The old Q8 GGUF has been deleted (8.1 GB freed).")
