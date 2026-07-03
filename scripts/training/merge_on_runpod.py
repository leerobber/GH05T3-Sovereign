"""
merge_on_runpod.py — Merge avery-sovereign-lora on a RunPod GPU pod,
convert to GGUF Q8, upload to HuggingFace, then stop the pod.

Usage:  python merge_on_runpod.py
"""
import os, sys, time, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import runpod_launcher as rl

HF_TOKEN  = os.environ.get("HF_TOKEN", "")
LORA_REPO = "tastytator/avery-sovereign-lora"
HF_REPO   = "tastytator/avery-sovereign-gguf"
GGUF_NAME = "avery-sovereign-q8.gguf"

# ── Worker script that runs on the pod ──────────────────────────────────────
WORKER_PY = f'''import os, gc, subprocess, sys

HF_TOKEN   = "{HF_TOKEN}"
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
LORA_REPO  = "{LORA_REPO}"
HF_REPO    = "{HF_REPO}"
GGUF_NAME  = "{GGUF_NAME}"
MERGED_DIR = "/workspace/avery-merged"
GGUF_PATH  = f"/workspace/{{GGUF_NAME}}"

print("[1/4] Installing dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.40", "peft>=0.10", "accelerate", "huggingface_hub"], check=True)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi

print("[2/4] Loading base model on GPU...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    token=HF_TOKEN,
)
print("[2/4] Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, LORA_REPO, token=HF_TOKEN)
print("[2/4] Merging weights...")
model = model.merge_and_unload()

import os as _os
_os.makedirs(MERGED_DIR, exist_ok=True)
print(f"[2/4] Saving merged model to {{MERGED_DIR}}...")
model.save_pretrained(MERGED_DIR, safe_serialization=True)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
tokenizer.save_pretrained(MERGED_DIR)
del model, tokenizer
gc.collect()
print("      Done.")

print("[3/4] Installing llama.cpp converter...")
if not _os.path.exists("/workspace/llama.cpp"):
    subprocess.run(["git", "clone", "--depth", "1",
        "https://github.com/ggerganov/llama.cpp", "/workspace/llama.cpp"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gguf"], check=True)

print(f"[3/4] Converting to GGUF Q8_0 -> {{GGUF_PATH}} ...")
subprocess.run([
    sys.executable, "/workspace/llama.cpp/convert_hf_to_gguf.py",
    MERGED_DIR, "--outfile", GGUF_PATH, "--outtype", "q8_0",
], check=True)
print("      Conversion done.")

print(f"[4/4] Uploading GGUF to HuggingFace {{HF_REPO}}...")
api = HfApi()
try:
    api.create_repo(repo_id=HF_REPO, token=HF_TOKEN, exist_ok=True)
except Exception as e:
    print(f"      (repo create: {{e}})")
api.upload_file(
    path_or_fileobj=GGUF_PATH,
    path_in_repo=GGUF_NAME,
    repo_id=HF_REPO,
    token=HF_TOKEN,
)
print()
print("=" * 60)
print("  MERGE + CONVERT + UPLOAD COMPLETE")
print(f"  https://huggingface.co/{{HF_REPO}}/resolve/main/{{GGUF_NAME}}")
print("=" * 60)

# Signal completion
with open("/workspace/merge_complete.txt", "w") as f:
    f.write("done")
'''


def main():
    rl._load_env()

    print("+==========================================+")
    print("|  AVERY MERGE + CONVERT + UPLOAD           |")
    print("+==========================================+")
    print()

    # ── Find GPU ──────────────────────────────────────────────────────────────
    print("[1/6] Checking for orphan pods...")
    rl._kill_orphan_pods()
    print("  Clean.")

    print("[2/6] Finding available GPU...")
    gpus = rl.find_all_gpus_sorted()
    if not gpus:
        print("  ERROR: No 24GB+ GPU available.")
        sys.exit(1)

    pod = None
    for g in gpus:
        name  = g["displayName"]
        price = g.get("communityPrice") or g.get("securePrice")
        cloud = "COMMUNITY" if g.get("communityPrice") else "SECURE"
        print(f"  [{cloud}] {name} @ ${price:.3f}/hr ... ", end="", flush=True)
        try:
            pod = rl.start_pod(g["id"], cloud_type=cloud)
            if pod:
                print(f"SUCCESS")
                break
            print("unavailable")
        except Exception as e:
            print(f"error: {e}")

    if not pod:
        print("  ERROR: Could not start a pod.")
        sys.exit(1)

    pod_id = pod["id"]
    print(f"[3/6] Pod started: {pod_id}")

    # ── Wait for SSH ──────────────────────────────────────────────────────────
    print("[4/6] Waiting for SSH...")
    ip, port = None, None
    for i in range(60):
        time.sleep(5)
        info = rl.get_pod(pod_id)
        ip, port = rl.get_ssh_info(info)
        print(f"  [{i+1}/60] SSH: {ip}:{port}")
        if ip and port:
            break

    if not ip:
        print("  ERROR: SSH never ready. Stopping pod.")
        rl.stop_pod(pod_id)
        sys.exit(1)

    print(f"  SSH ready: {ip}:{port}. Waiting 60s for sshd init...")
    time.sleep(60)

    # ── Upload worker ─────────────────────────────────────────────────────────
    print("[5/6] Uploading merge script...")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                     delete=False, encoding="utf-8") as f:
        f.write(WORKER_PY)
        worker_path = f.name

    for attempt in range(3):
        try:
            rl._scp_upload(ip, port, worker_path, "/workspace/merge_worker.py")
            print("  Uploaded.")
            break
        except Exception as e:
            print(f"  SCP attempt {attempt+1} failed: {e}. Retrying in 15s...")
            time.sleep(15)
    else:
        print("  ERROR: SCP failed 3 times. Stopping pod.")
        rl.stop_pod(pod_id)
        sys.exit(1)

    # ── Launch merge ──────────────────────────────────────────────────────────
    print("[6/6] Launching merge on pod (10-20 min)...")
    rl._ssh_run(ip, port,
        "nohup python /workspace/merge_worker.py "
        "> /workspace/merge.log 2>&1 &",
        timeout=30)

    start = time.time()
    while True:
        time.sleep(30)
        elapsed = int(time.time() - start)

        # Check completion sentinel
        done_check = rl._ssh_run(ip, port,
            "test -f /workspace/merge_complete.txt && echo DONE || echo PENDING",
            capture=True, timeout=15)
        if done_check and "DONE" in (done_check.stdout or ""):
            tail = rl._ssh_run(ip, port, "tail -10 /workspace/merge.log",
                               capture=True, timeout=15)
            print(f"\n{tail.stdout if tail else ''}")
            break

        # Print last log line as progress
        tail = rl._ssh_run(ip, port, "tail -1 /workspace/merge.log",
                           capture=True, timeout=15)
        last = (tail.stdout or "").strip() if tail else "..."

        # Check for errors
        err_check = rl._ssh_run(ip, port,
            "grep -c 'Traceback\\|Error' /workspace/merge.log 2>/dev/null || echo 0",
            capture=True, timeout=15)
        err_count = int((err_check.stdout or "0").strip()) if err_check else 0
        if err_count > 0:
            print(f"\n  ERROR detected in log:")
            full = rl._ssh_run(ip, port, "tail -30 /workspace/merge.log",
                               capture=True, timeout=15)
            print(full.stdout if full else "")
            break

        print(f"  [{elapsed//60}m{elapsed%60:02d}s] {last}")

    print("\n  Stopping pod to save credits...")
    rl.stop_pod(pod_id)
    print(f"  Pod {pod_id} stopped.")
    print()
    print("  Next step — create Ollama model:")
    print(f"  ollama pull hf.co/tastytator/avery-sovereign-gguf:Q8_0")
    print(f"  OR: download GGUF then run: ollama create avery-sovereign -f Modelfile.avery")


if __name__ == "__main__":
    main()

