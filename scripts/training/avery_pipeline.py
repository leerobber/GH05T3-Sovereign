#!/usr/bin/env python3
"""
avery_pipeline.py
=================
Full Avery 3B training pipeline. Runs end-to-end.

Steps:
  1. Push 115K training records to HuggingFace
  2. Start RunPod pod (A100/3090 — cheapest available)
  3. Upload + run on-pod worker (train → merge → GGUF → push GGUF to HF)
  4. Download GGUF from HuggingFace
  5. Update Modelfile and create new Ollama avery-sovereign model
  6. Smoke test

Usage:
  python avery_pipeline.py                      # full pipeline
  python avery_pipeline.py --skip-push          # skip HF data push
  python avery_pipeline.py --skip-push --skip-train  # local rebuild only (GGUF already on HF)
  python avery_pipeline.py --ollama-only        # download GGUF + rebuild Ollama model only
  python avery_pipeline.py --smoke-test         # test current avery-sovereign in Ollama

Env vars required:
  HF_TOKEN       — HuggingFace token
  RUNPOD_API_KEY — RunPod API key
"""
import argparse, gc, json, os, subprocess, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).parent

# ── Config ────────────────────────────────────────────────────────────────────
HF_TOKEN     = os.environ.get("HF_TOKEN") or (
    lambda p: p.read_text().strip() if p.exists() else ""
)(Path("C:/Users/leer4/.cache/huggingface/token"))
RUNPOD_KEY   = os.environ.get("RUNPOD_API_KEY", "")

BASE_MODEL   = "Qwen/Qwen2.5-3B-Instruct"
LORA_REPO    = "tastytator/avery-sovereign-lora"
GGUF_REPO    = "tastytator/avery-sovereign-gguf"
GGUF_NAME    = "avery-3b-q8.gguf"
HF_DATASET   = "tastytator/sovereign-economy"
OLLAMA_NAME  = "avery-sovereign"

GGUF_LOCAL   = ROOT / GGUF_NAME
MODELFILE    = ROOT / "Modelfile.avery"
WORKER_SCRIPT = ROOT / "avery_runpod_worker.py"
STATE_FILE   = ROOT / "data" / "avery_pipeline_state.json"

# GPU preference (cheapest first)
GPU_PRIORITY = [
    "NVIDIA GeForce RTX 3090",
    "NVIDIA GeForce RTX 3090 Ti",
    "NVIDIA RTX A5000",
    "NVIDIA RTX A6000",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA A100-SXM4-40GB",
    "NVIDIA A100 80GB PCIe",
]

POD_IMAGE    = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
POD_NAME     = "avery-3b-train"
SSH_KEY_PUB  = None  # auto-loaded from runpod_launcher.py

SYSTEM = (
    "You are Avery, the sovereign business strategist for SovereignNation — "
    "a fixed-cost AI platform for lower and middle class families, "
    "professional services firms, children's education, and affordable connectivity. "
    "Use the KAIROS framework: Kickoff, Alignment, Implementation, Refinement, "
    "Optimization, Scaling. Be direct, structured, and actionable."
)

# ─────────────────────────────────────────────────────────────────────────────

def banner(msg):
    w = max(len(msg) + 4, 55)
    print("\n" + "=" * w)
    print(f"  {msg}")
    print("=" * w)

def ts():
    return time.strftime("[%H:%M:%S]")

# ── Step 1: Push training data ────────────────────────────────────────────────

def push_training_data():
    banner("STEP 1/6  Push training data to HuggingFace")
    result = subprocess.run(
        [sys.executable, str(ROOT / "push_avery_training.py")],
        env={**os.environ, "HF_TOKEN": HF_TOKEN}
    )
    if result.returncode != 0:
        raise RuntimeError("push_avery_training.py failed")
    print(f"{ts()} Data push complete.")


# ── Step 2-3: RunPod train + merge + GGUF ─────────────────────────────────────

def _load_runpod():
    """Import runpod_launcher helpers."""
    sys.path.insert(0, str(ROOT))
    import runpod_launcher as rl
    rl._load_env()
    return rl


def launch_runpod_training():
    banner("STEP 2-3/6  RunPod: train → merge → GGUF → push")

    rl = _load_runpod()

    # Kill any orphan pods
    print(f"{ts()} Checking for orphan pods...")
    rl._kill_orphan_pods()

    # Find cheapest available GPU
    print(f"{ts()} Finding GPU...")
    gpus = rl.find_all_gpus_sorted()
    if not gpus:
        raise RuntimeError("No GPU available on RunPod right now.")

    pod = None
    for g in gpus:
        name  = g["displayName"]
        price = g.get("communityPrice") or g.get("securePrice") or 0
        cloud = "COMMUNITY" if g.get("communityPrice") else "SECURE"
        if not any(p in name for p in GPU_PRIORITY):
            continue
        print(f"{ts()} [{cloud}] {name} @ ${price:.3f}/hr ...", end=" ", flush=True)
        try:
            pod = rl.start_pod(g["id"], cloud_type=cloud)
            if pod:
                print("✓ started")
                break
            print("unavailable")
        except Exception as e:
            print(f"error: {e}")

    # If no preferred GPU, take anything with enough VRAM
    if not pod:
        print(f"{ts()} No preferred GPU found, trying any 24GB+ GPU...")
        for g in gpus:
            price = g.get("communityPrice") or g.get("securePrice") or 0
            cloud = "COMMUNITY" if g.get("communityPrice") else "SECURE"
            print(f"{ts()} Trying {g['displayName']} @ ${price:.3f}/hr ...", end=" ", flush=True)
            try:
                pod = rl.start_pod(g["id"], cloud_type=cloud)
                if pod:
                    print("✓")
                    break
                print("unavailable")
            except Exception as e:
                print(f"error: {e}")

    if not pod:
        raise RuntimeError("Could not start any RunPod pod.")

    pod_id = pod["id"]
    price  = pod.get("costPerHr", "?")
    print(f"{ts()} Pod started: {pod_id}  cost=${price}/hr")
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps({"pod_id": pod_id, "started": time.time()}))

    # Wait for SSH
    print(f"{ts()} Waiting for SSH...")
    ip, port = None, None
    for i in range(90):
        time.sleep(10)
        info = rl.get_pod(pod_id)
        ip, port = rl.get_ssh_info(info)
        if ip and port:
            print(f"{ts()} SSH ready: {ip}:{port}  (after {(i+1)*10}s)")
            break
        if i % 6 == 5:
            print(f"{ts()} Still waiting... {(i+1)*10}s")

    if not ip:
        rl.stop_pod(pod_id)
        raise RuntimeError("SSH never became ready. Pod stopped.")

    print(f"{ts()} Waiting 45s for sshd init...")
    time.sleep(45)

    # Upload worker script
    print(f"{ts()} Uploading worker script...")
    for attempt in range(5):
        try:
            rl._scp_upload(ip, port, str(WORKER_SCRIPT), "/workspace/avery_runpod_worker.py")
            print(f"{ts()} Worker uploaded.")
            break
        except Exception as e:
            if attempt < 4:
                print(f"{ts()} SCP attempt {attempt+1} failed: {e}. Retry in 15s...")
                time.sleep(15)
            else:
                rl.stop_pod(pod_id)
                raise RuntimeError(f"SCP failed 5 times: {e}")

    # Build worker command
    worker_cmd = (
        f"nohup python /workspace/avery_runpod_worker.py "
        f"--hf_token '{HF_TOKEN}' "
        f"--base_model '{BASE_MODEL}' "
        f"--lora_repo '{LORA_REPO}' "
        f"--dataset '{HF_DATASET}' "
        f"--gguf_repo '{GGUF_REPO}' "
        f"--gguf_name '{GGUF_NAME}' "
        f"--epochs 3 --batch_size 4 --max_seq 2048 "
        f"> /workspace/avery_worker.log 2>&1 &"
    )

    print(f"{ts()} Launching worker on pod...")
    rl._ssh_run(ip, port, worker_cmd, timeout=30)
    print(f"{ts()} Worker launched. Monitoring...")

    # Monitor
    t_start = time.time()
    last_line = ""
    while True:
        time.sleep(30)
        elapsed = int(time.time() - t_start)

        # Check for completion sentinel
        check = rl._ssh_run(ip, port,
            "test -f /workspace/avery_done.txt && cat /workspace/avery_done.txt || echo PENDING",
            capture=True, timeout=15)
        if check:
            out = (check.stdout or "").strip()
            if out != "PENDING" and out:
                try:
                    result = json.loads(out)
                    print(f"\n{ts()} COMPLETE: {result}")
                    STATE_FILE.write_text(json.dumps({
                        "pod_id": pod_id, "result": result,
                        "completed": time.time()
                    }))
                    break
                except json.JSONDecodeError:
                    if "complete" in out.lower():
                        break

        # Check for error
        err = rl._ssh_run(ip, port,
            "test -f /workspace/avery_error.txt && cat /workspace/avery_error.txt || echo OK",
            capture=True, timeout=15)
        if err:
            err_out = (err.stdout or "").strip()
            if err_out != "OK" and err_out:
                # Get full log
                log_tail = rl._ssh_run(ip, port, "tail -40 /workspace/avery_worker.log",
                                       capture=True, timeout=15)
                print(f"\n{ts()} ERROR on pod:\n{err_out}")
                if log_tail:
                    print(log_tail.stdout or "")
                rl.stop_pod(pod_id)
                raise RuntimeError(f"Worker failed: {err_out}")

        # Progress line
        tail = rl._ssh_run(ip, port, "tail -1 /workspace/avery_worker.log",
                           capture=True, timeout=15)
        if tail:
            line = (tail.stdout or "").strip()
            if line and line != last_line:
                last_line = line
                print(f"{ts()} [{elapsed//60}m] {line}")

    print(f"{ts()} Stopping pod...")
    rl.stop_pod(pod_id)
    print(f"{ts()} Pod {pod_id} stopped. ✓")


# ── Step 4: Download GGUF ─────────────────────────────────────────────────────

def download_gguf():
    banner("STEP 4/6  Download GGUF from HuggingFace")

    if GGUF_LOCAL.exists() and GGUF_LOCAL.stat().st_size > 2_000_000_000:
        print(f"{ts()} {GGUF_LOCAL} already exists ({GGUF_LOCAL.stat().st_size/1e9:.2f}GB), skipping download.")
        return

    print(f"{ts()} Downloading {GGUF_REPO}/{GGUF_NAME}...")

    try:
        from huggingface_hub import hf_hub_download, login
        login(token=HF_TOKEN, add_to_git_credential=False)
        path = hf_hub_download(
            repo_id=GGUF_REPO,
            filename=GGUF_NAME,
            local_dir=str(ROOT),
            token=HF_TOKEN,
        )
        print(f"{ts()} Downloaded to: {path}")
    except Exception:
        # Fallback: huggingface-cli
        print(f"{ts()} Trying huggingface-cli...")
        subprocess.run(
            f'huggingface-cli download {GGUF_REPO} {GGUF_NAME} --local-dir "{ROOT}"',
            shell=True, env={**os.environ, "HF_TOKEN": HF_TOKEN}, check=True
        )

    size = GGUF_LOCAL.stat().st_size / 1e9
    print(f"{ts()} GGUF local: {GGUF_LOCAL}  ({size:.2f} GB)")
    if not (2.5 < size < 4.5):
        raise RuntimeError(f"Unexpected GGUF size {size:.2f}GB — expected 3.1-3.4GB for 3B Q8")


# ── Step 5: Create Ollama model ───────────────────────────────────────────────

def create_ollama_model():
    banner("STEP 5/6  Create Ollama model")

    modelfile_content = (
        f"FROM {GGUF_LOCAL.as_posix()}\n"
        f'SYSTEM """{SYSTEM}"""\n'
        "PARAMETER temperature 0.7\n"
        "PARAMETER top_p 0.9\n"
        "PARAMETER num_ctx 4096\n"
        "PARAMETER repeat_penalty 1.1\n"
    )

    MODELFILE.write_text(modelfile_content, encoding="utf-8")
    print(f"{ts()} Modelfile written: {MODELFILE}")

    print(f"{ts()} Running: ollama create {OLLAMA_NAME}...")
    result = subprocess.run(
        ["ollama", "create", OLLAMA_NAME, "-f", str(MODELFILE)],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"ollama create failed:\n{result.stderr}")

    # Verify
    check = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    for line in check.stdout.splitlines():
        if OLLAMA_NAME in line:
            print(f"{ts()} Ollama confirmed: {line.strip()}")
            break

    print(f"{ts()} Model '{OLLAMA_NAME}' created. ✓")


# ── Step 6: Smoke test ────────────────────────────────────────────────────────

def smoke_test():
    banner("STEP 6/6  Smoke test")
    import urllib.request

    prompt = (
        "A CPA firm has 12 staff and handles 200 business clients. "
        "They want to use AI for tax analysis but are worried about client data privacy. "
        "Give me a 3-step go-to-market plan using the KAIROS framework."
    )

    payload = json.dumps({
        "model": OLLAMA_NAME, "prompt": prompt, "stream": False
    }).encode()

    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    print(f"{ts()} Sending test prompt (may take 15-30s)...")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())

        response = data.get("response", "")
        words = len(response.split())

        if words < 30:
            raise RuntimeError(f"Response too short ({words} words): {response[:200]}")

        # Check KAIROS
        kairos_hit = any(k in response for k in ["KAIROS","Kickoff","Alignment","Implementation"])
        print(f"\n{ts()} SMOKE TEST PASSED — {words} words, KAIROS={kairos_hit}")
        print(f"\n--- Response preview ---\n{response[:600]}\n---")
        return True

    except Exception as e:
        print(f"\n{ts()} SMOKE TEST FAILED: {e}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-push",   action="store_true", help="Skip HF data push")
    ap.add_argument("--skip-train",  action="store_true", help="Skip RunPod training")
    ap.add_argument("--skip-download", action="store_true", help="Skip GGUF download")
    ap.add_argument("--ollama-only", action="store_true", help="Download GGUF + rebuild Ollama only")
    ap.add_argument("--smoke-test",  action="store_true", help="Smoke test current model only")
    args = ap.parse_args()

    if args.smoke_test:
        smoke_test()
        return

    if args.ollama_only:
        args.skip_push = True
        args.skip_train = True

    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set.")
        sys.exit(1)

    banner("AVERY 3B FULL PIPELINE")
    print(f"  Base model : {BASE_MODEL}")
    print(f"  LoRA repo  : {LORA_REPO}")
    print(f"  GGUF repo  : {GGUF_REPO}")
    print(f"  Ollama name: {OLLAMA_NAME}")
    print(f"  Skip push  : {args.skip_push}")
    print(f"  Skip train : {args.skip_train}")

    t_total = time.time()

    try:
        if not args.skip_push:
            push_training_data()

        if not args.skip_train:
            if not RUNPOD_KEY:
                print("ERROR: RUNPOD_API_KEY not set.")
                sys.exit(1)
            launch_runpod_training()

        if not args.skip_download:
            download_gguf()

        create_ollama_model()
        smoke_test()

        elapsed = (time.time() - t_total) / 60
        banner(f"AVERY 3B PIPELINE COMPLETE  ({elapsed:.0f} min total)")
        print(f"  Ollama model : {OLLAMA_NAME}")
        print(f"  GGUF         : {GGUF_LOCAL}")
        print(f"  LoRA         : https://huggingface.co/{LORA_REPO}")
        print(f"  GGUF (HF)    : https://huggingface.co/{GGUF_REPO}")

    except KeyboardInterrupt:
        print("\nInterrupted. Pod may still be running — check RunPod dashboard.")
        if STATE_FILE.exists():
            state = json.loads(STATE_FILE.read_text())
            pod_id = state.get("pod_id")
            if pod_id:
                print(f"Stop pod manually: pod_id={pod_id}")
        sys.exit(1)
    except Exception as e:
        print(f"\nPIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
