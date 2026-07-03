"""
runpod_merge.py — Launch a RunPod pod to merge a sovereign LoRA -> GGUF and upload to HuggingFace.

Usage:
  python runpod_merge.py                    # merge avery (default)
  python runpod_merge.py --agent forge      # merge forge
  python runpod_merge.py --agent forge --status
  python runpod_merge.py --agent forge --stop
  python runpod_merge.py --agent forge --download

After completion, run:
  ollama create <agent>-sovereign -f <Modelfile>
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import runpod_launcher as rl

# ── Agent registry ─────────────────────────────────────────────────────────
AGENT_CONFIGS = {
    "avery": {
        "lora_repo":    "tastytator/avery-sovereign-lora",
        "hf_repo":      "tastytator/avery-sovereign-lora",
        "gguf_file":    "avery-sovereign-q8.gguf",
        "local_gguf":   Path(__file__).parent / "avery-sovereign-q8.gguf",
        "modelfile":    Path(__file__).parent / "Modelfile.avery",
        "ollama_name":  "avery-sovereign",
        "system_prompt": (
            "You are Avery, a sovereign business strategist for SovereignNation — "
            "a fixed-cost AI platform built for lower and middle class users, children's "
            "education, and affordable connectivity. Be direct, structured, and strategic."
        ),
    },
    "forge": {
        "lora_repo":    "tastytator/forge-sovereign-lora",
        "hf_repo":      "tastytator/forge-sovereign-lora",
        "gguf_file":    "forge-sovereign-q8.gguf",
        "local_gguf":   Path(__file__).parent / "ollama_models" / "forge-sovereign-q8.gguf",
        "modelfile":    Path(__file__).parent / "ollama_models" / "Modelfile.forge",
        "ollama_name":  "forge-sovereign",
        "system_prompt": (
            "You are FORGE, the sovereign code generation specialist for SovereignNation. "
            "Write production-ready Python, JavaScript, and TypeScript with imports, error "
            "handling, type hints, and comments for non-obvious logic."
        ),
    },
}

MERGE_SCRIPT = Path(__file__).parent / "remote_merge.py"


def _state_file(agent: str) -> Path:
    p = Path(__file__).parent / "data" / f"merge_state_{agent}.json"
    p.parent.mkdir(exist_ok=True)
    return p


def _save_state(agent: str, state: dict):
    _state_file(agent).write_text(json.dumps(state, indent=2), encoding="utf-8")


def _load_state(agent: str) -> dict:
    f = _state_file(agent)
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _check_merge_done(ip: str, port: int):
    try:
        result = rl._ssh_run(ip, port,
            "cat /workspace/merge_complete.txt 2>/dev/null || echo NOT_DONE",
            capture=True, timeout=20)
        out = result.stdout.strip() if result.returncode == 0 else "NOT_DONE"
        if "NOT_DONE" in out:
            return False, None
        try:
            return True, json.loads(out)
        except Exception:
            return True, {"raw": out}
    except Exception:
        return False, None


def _write_modelfile(cfg: dict):
    gguf_posix = cfg["local_gguf"].as_posix()
    cfg["modelfile"].parent.mkdir(parents=True, exist_ok=True)
    cfg["modelfile"].write_text(
        f'FROM {gguf_posix}\n'
        f'SYSTEM """{cfg["system_prompt"]}"""\n'
        'PARAMETER temperature 0.7\n'
        'PARAMETER top_p 0.9\n'
        'PARAMETER num_ctx 4096\n',
        encoding="utf-8",
    )
    print(f"  Written: {cfg['modelfile']}")


def launch(agent: str):
    cfg = AGENT_CONFIGS[agent]
    rl._load_env()
    hf_token = rl.HF_TOKEN
    if not rl.API_KEY:
        print("ERROR: RUNPOD_API_KEY not set in .env"); sys.exit(1)
    if not hf_token:
        print("ERROR: HF_TOKEN not set in .env"); sys.exit(1)

    orig_name = rl.POD_NAME
    rl.POD_NAME = f"{agent}-sovereign-merge"

    print("\n+==========================================+")
    print(f"|  SOVEREIGN RUNPOD MERGE: {agent.upper():<16}  |")
    print("+==========================================+\n")
    print(f"  LoRA  : {cfg['lora_repo']}")
    print(f"  GGUF  : {cfg['gguf_file']}")
    print(f"  SSH   : {rl._ssh_key_path()}")
    print()

    print("[0/6] Checking for orphan merge pods...")
    rl._kill_orphan_pods()
    print("  Clean.")

    print("\n[1/6] Finding available GPU (24GB+ VRAM)...")
    gpus = rl.find_all_gpus_sorted()
    pod = None
    for cloud_type in ["COMMUNITY", "SECURE"]:
        if pod:
            break
        for g in gpus:
            price = (g.get("communityPrice") if cloud_type == "COMMUNITY"
                     else g.get("securePrice")) or 99
            print(f"  [{cloud_type}] {g['displayName']} ({g['memoryInGb']}GB) @ ${price:.3f}/hr...")
            try:
                pod = rl.start_pod(g["id"], cloud_type=cloud_type)
                print(f"  SUCCESS: {g['displayName']} @ ${price:.3f}/hr")
                break
            except Exception as e:
                msg = str(e)
                if any(x in msg for x in ["SUPPLY_CONSTRAINT", "no longer any instances",
                                           "does not have the resources", "resources are unavailable"]):
                    print("  Unavailable -- trying next...")
                    continue
                raise

    rl.POD_NAME = orig_name

    if pod is None:
        print("ERROR: No GPUs available."); sys.exit(1)

    pod_id = pod["id"]
    print(f"\n[2/6] Pod started: {pod_id}  Status: {pod['desiredStatus']}")
    _save_state(agent, {"pod_id": pod_id, "started_at": time.time()})

    print("[3/6] Waiting for SSH...")
    ip, port = None, None
    for attempt in range(60):
        time.sleep(15)
        try:
            p = rl.get_pod(pod_id)
            ip, port = rl.get_ssh_info(p)
            status = p.get("desiredStatus", "?")
            print(f"  [{attempt+1}/60] Status: {status}  SSH: {ip}:{port}")
            if ip and port:
                break
        except Exception as e:
            print(f"  [{attempt+1}/60] API error: {e}")
    else:
        print("  ERROR: Pod never got SSH."); sys.exit(1)

    print("  Waiting 25s for sshd...")
    time.sleep(25)

    print(f"\n[4/6] Uploading merge script to {ip}:{port}...")
    rl._scp_upload(ip, port, str(MERGE_SCRIPT), "/workspace/remote_merge.py")
    print("  Uploaded.")
    _save_state(agent, {"pod_id": pod_id, "ip": ip, "port": port, "started_at": time.time()})

    print("\n[5/6] Launching merge on pod...")
    merge_cmd = (
        f"export HF_TOKEN={hf_token} "
        f"AGENT_NAME={agent} "
        f"LORA_REPO={cfg['lora_repo']} "
        f"HF_REPO={cfg['hf_repo']} "
        f"GGUF_FILENAME={cfg['gguf_file']}; "
        "nohup python /workspace/remote_merge.py "
        "> /workspace/merge.log 2>&1 & echo LAUNCHED"
    )
    rl._ssh_run(ip, port, merge_cmd, timeout=30)
    print(f"""
  Merge is running on the pod (nohup -- survives terminal close).

  To watch live:
    ssh -i {rl._ssh_key_path()} -p {port} root@{ip} "tail -f /workspace/merge.log"

  To check status:
    python runpod_merge.py --agent {agent} --status
""")

    print("[6/6] Monitoring for completion (Ctrl+C safe -- pod keeps running)...")
    _monitor(agent, pod_id, ip, port)


def _monitor(agent: str, pod_id: str, ip: str, port: int, max_minutes: int = 150):
    for minute in range(1, max_minutes + 1):
        time.sleep(60)

        try:
            p = rl.get_pod(pod_id)
            new_ip, new_port = rl.get_ssh_info(p)
            if new_ip:
                ip, port = new_ip, new_port
        except Exception:
            pass

        done, info = _check_merge_done(ip, port)

        if minute % 5 == 0:
            try:
                result = rl._ssh_run(ip, port, "tail -3 /workspace/merge.log",
                                     capture=True, timeout=15)
                snippet = result.stdout.strip().replace("\n", " | ")
                print(f"  [{minute}m] {snippet[:120]}")
            except Exception:
                print(f"  [{minute}m] (log unavailable)")
        else:
            print(f"  [{minute}m] {'DONE' if done else 'merging...'}")

        if done:
            _handle_completion(agent, pod_id, info)
            return

    print(f"\n  TIMEOUT: {max_minutes} minutes reached.")
    print(f"  python runpod_merge.py --agent {agent} --status")
    state = _load_state(agent)
    state.update({"ip": ip, "port": port, "pod_id": pod_id})
    _save_state(agent, state)


def _handle_completion(agent: str, pod_id: str, info: dict):
    cfg = AGENT_CONFIGS[agent]
    print("\n  *** MERGE COMPLETE ***")
    if info:
        print(f"  GGUF URL : {info.get('gguf_url', '?')}")
        print(f"  Size     : {info.get('gguf_size_gb', '?')} GB")

    print("\n  Stopping pod to save credits...")
    try:
        rl.stop_pod(pod_id)
        print(f"  Pod {pod_id} stopped.")
    except Exception as e:
        print(f"  Stop error: {e}")
    _save_state(agent, {})

    print(f"\n  GGUF uploaded to HuggingFace.")
    print(f"  Next: python runpod_merge.py --agent {agent} --download")
    print(f"  Then: ollama create {cfg['ollama_name']} -f {cfg['modelfile']}")


def download(agent: str):
    cfg = AGENT_CONFIGS[agent]
    rl._load_env()
    hf_token = rl.HF_TOKEN

    print(f"Downloading {cfg['gguf_file']} from {cfg['hf_repo']} ...")
    from huggingface_hub import hf_hub_download
    local = hf_hub_download(
        repo_id=cfg["hf_repo"],
        filename=cfg["gguf_file"],
        local_dir=str(cfg["local_gguf"].parent),
        token=hf_token,
    )
    print(f"  Saved to: {local}")
    print(f"  Size: {Path(local).stat().st_size / 1e9:.2f} GB")

    _write_modelfile(cfg)

    print("\n  Next steps:")
    print(f"    ollama create {cfg['ollama_name']} -f {cfg['modelfile']}")
    print(f"    ollama run {cfg['ollama_name']}")


def status(agent: str):
    rl._load_env()
    state  = _load_state(agent)
    pod_id = state.get("pod_id")
    if not pod_id:
        print(f"No merge pod in state for {agent}."); return

    print(f"  Agent : {agent}")
    print(f"  Pod   : {pod_id}")
    try:
        p = rl.get_pod(pod_id)
        ip, port = rl.get_ssh_info(p)
        print(f"  Status : {p.get('desiredStatus')}")
        print(f"  SSH    : {ip}:{port}")
        if ip and port:
            done, info = _check_merge_done(ip, port)
            print(f"  Done   : {done}")
            if done:
                print(f"  Info   : {info}")
            else:
                result = rl._ssh_run(ip, port, "tail -20 /workspace/merge.log",
                                     capture=True, timeout=20)
                print("\n--- Last 20 lines of merge.log ---")
                print(result.stdout.decode("utf-8", errors="replace")
                      if isinstance(result.stdout, bytes) else result.stdout or "(empty)")
    except Exception as e:
        print(f"  API error: {e}")


def stop(agent: str):
    rl._load_env()
    state  = _load_state(agent)
    pod_id = state.get("pod_id")
    if pod_id:
        try:
            rl.stop_pod(pod_id)
            print(f"Stopped pod {pod_id}")
        except Exception as e:
            print(f"Error: {e}")
        _save_state(agent, {})
    else:
        print(f"No merge pod in state for {agent}.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="RunPod LoRA Merge Launcher")
    p.add_argument("--agent",    default="avery",
                   choices=list(AGENT_CONFIGS), help="Which agent to merge (default: avery)")
    p.add_argument("--status",   action="store_true", help="Check merge pod status")
    p.add_argument("--stop",     action="store_true", help="Stop merge pod")
    p.add_argument("--download", action="store_true", help="Download GGUF from HuggingFace")
    args = p.parse_args()

    if args.agent not in AGENT_CONFIGS:
        print(f"Unknown agent: {args.agent}. Choices: {list(AGENT_CONFIGS)}")
        sys.exit(1)

    if args.status:
        status(args.agent)
    elif args.stop:
        stop(args.agent)
    elif args.download:
        download(args.agent)
    else:
        launch(args.agent)
