"""
runpod_launcher.py — Sovereign RunPod Training Launcher

Usage:
  python runpod_launcher.py              # launch + train + auto-stop
  python runpod_launcher.py --status    # check running pod + tail log
  python runpod_launcher.py --stop      # stop running pod(s)
  python runpod_launcher.py --tail      # attach to live training log
  python runpod_launcher.py --cleanup   # stop ALL sovereign training pods
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

import requests

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

SERVERLESS_API = "https://api.runpod.io/v2"

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY           = os.environ.get("RUNPOD_API_KEY", "")
HF_TOKEN          = os.environ.get("HF_TOKEN", "")
NETWORK_VOLUME_ID = os.environ.get("RUNPOD_VOLUME_ID", "")
SSH_KEY_PUB       = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIARR9yuDQlP7BJ8VKXq3o/bZlPLov71iDTRb2HBtfMQl claude-avery-training"
TRAIN_SCRIPT  = Path(__file__).parent / "runpod_train.py"
STATE_FILE    = Path(__file__).parent / "data" / "runpod_state.json"
LOG_FILE      = Path(__file__).parent / "data" / "train_run.log"
ERR_FILE      = Path(__file__).parent / "data" / "train_run_err.log"

# GPU preference order — cheapest first, then by price
GPU_PRIORITY  = [
    "NVIDIA RTX A5000",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA GeForce RTX 3090 Ti",
    "NVIDIA RTX A6000",
    "NVIDIA GeForce RTX 4090",
]
POD_IMAGE     = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
POD_NAME      = "avery-sovereign-training"

GQL_URL = ""  # set by _load_env


# ── Env loader ────────────────────────────────────────────────────────────────
def _load_env():
    global API_KEY, HF_TOKEN, GQL_URL, NETWORK_VOLUME_ID
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                v = v.strip().strip("'\"")
                os.environ.setdefault(k.strip(), v)
                if k.strip() == "RUNPOD_API_KEY" and not API_KEY:
                    API_KEY = v
                if k.strip() == "HF_TOKEN" and not HF_TOKEN:
                    HF_TOKEN = v
                if k.strip() == "RUNPOD_VOLUME_ID" and not NETWORK_VOLUME_ID:
                    NETWORK_VOLUME_ID = v
    GQL_URL = f"https://api.runpod.io/graphql?api_key={API_KEY}"


def _write_env_key(key: str, value: str):
    """Append or update a key in the .env file."""
    env_path = Path(__file__).parent / ".env"
    text = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    lines = text.splitlines()
    updated = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[i] = f"{key}={value}"
            updated = True
            break
    if not updated:
        lines.append(f"{key}={value}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _gql(query: str, variables: dict = None) -> dict:
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    r = requests.post(GQL_URL, json=payload,
                      headers={"Content-Type": "application/json"}, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL errors: {data['errors']}")
    return data["data"]


# ── SSH key ───────────────────────────────────────────────────────────────────
def _ssh_key_path() -> str:
    candidates = [
        Path.home() / ".ssh" / "avery_training",
        Path.home() / ".ssh" / "runpod",
        Path.home() / ".ssh" / "id_ed25519",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    print(f"WARNING: No SSH private key found. Tried: {[str(c) for c in candidates]}")
    return str(Path.home() / ".ssh" / "avery_training")


# ── RunPod API ────────────────────────────────────────────────────────────────
def find_all_gpus_sorted() -> list:
    data = _gql("{ gpuTypes { id displayName memoryInGb communityPrice securePrice } }")
    eligible = [g for g in data["gpuTypes"]
                if (g.get("memoryInGb") or 0) >= 24
                and (g.get("communityPrice") or g.get("securePrice"))]
    priority_ids = {name: i for i, name in enumerate(GPU_PRIORITY)}
    def sort_key(g):
        pri   = priority_ids.get(g["displayName"], 99)
        price = g.get("communityPrice") or g.get("securePrice") or 99
        return (pri, price)
    eligible.sort(key=sort_key)
    return eligible


def start_pod(gpu_type_id: str, cloud_type: str = "COMMUNITY",
              network_volume_id: str = "") -> dict:
    mutation = """
    mutation CreatePod($input: PodFindAndDeployOnDemandInput!) {
      podFindAndDeployOnDemand(input: $input) {
        id name desiredStatus
        runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } }
      }
    }
    """
    vol_id = network_volume_id or NETWORK_VOLUME_ID
    inp = {
        "cloudType":         cloud_type,
        "gpuCount":          1,
        "volumeInGb":        30,
        "containerDiskInGb": 30,
        "minVcpuCount":      4,
        "minMemoryInGb":     15,
        "gpuTypeId":         gpu_type_id,
        "name":              POD_NAME,
        "imageName":         POD_IMAGE,
        "ports":             "22/tcp",
        "volumeMountPath":   "/workspace",
        "env": [e for e in [
            {"key": "PUBLIC_KEY",   "value": SSH_KEY_PUB},
            {"key": "HF_TOKEN",     "value": HF_TOKEN},
            {"key": "TRAIN_MODE",   "value": os.environ.get("TRAIN_MODE",  "orpo")},
            {"key": "TRAIN_AGENT",  "value": os.environ.get("TRAIN_AGENT", "avery")},
            ({"key": "TRAIN_SPLIT", "value": os.environ["TRAIN_SPLIT"]}
             if os.environ.get("TRAIN_SPLIT") else None),
        ] if e],
    }
    if vol_id:
        inp["networkVolumeId"] = vol_id
    data = _gql(mutation, {"input": inp})
    return data["podFindAndDeployOnDemand"]


def get_pod(pod_id: str) -> dict:
    query = """
    query GetPod($podId: String!) {
      pod(input: { podId: $podId }) {
        id name desiredStatus
        runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } }
      }
    }
    """
    return _gql(query, {"podId": pod_id})["pod"]


def list_all_pods() -> list:
    q = "{myself{pods{id name desiredStatus runtime{uptimeInSeconds ports{ip isIpPublic privatePort publicPort type}}}}}"
    return _gql(q)["myself"]["pods"]


def stop_pod(pod_id: str):
    mutation = """
    mutation StopPod($podId: String!) {
      podStop(input: { podId: $podId }) { id desiredStatus }
    }
    """
    return _gql(mutation, {"podId": pod_id})


# ── Network Volume ─────────────────────────────────────────────────────────────
def list_volumes() -> list:
    q = "{ myself { networkVolumes { id name size dataCenterId } } }"
    return _gql(q)["myself"]["networkVolumes"]


def create_volume(name: str, size_gb: int = 50, datacenter_id: str = "US-TX-3") -> dict:
    mutation = """
    mutation CreateVolume($input: CreateNetworkVolumeInput!) {
      createNetworkVolume(input: $input) { id name size dataCenterId }
    }
    """
    return _gql(mutation, {"input": {
        "name":         name,
        "size":         size_gb,
        "dataCenterId": datacenter_id,
    }})["createNetworkVolume"]


def delete_volume(volume_id: str) -> dict:
    mutation = """
    mutation DeleteVolume($id: String!) {
      deleteNetworkVolume(input: { id: $id })
    }
    """
    return _gql(mutation, {"id": volume_id})


# ── Serverless Endpoints ───────────────────────────────────────────────────────
def list_endpoints() -> list:
    q = "{ myself { endpoints { id name workersMin workersMax gpuIds templateId networkVolumeId } } }"
    try:
        return _gql(q)["myself"]["endpoints"]
    except Exception:
        return []


def create_endpoint(
    name: str,
    image_name: str,
    gpu_ids: str = "AMPERE_24,AMPERE_16",
    workers_min: int = 0,
    workers_max: int = 3,
    idle_timeout: int = 5,
    volume_id: str = "",
    env_vars: list = None,
) -> dict:
    """Create a RunPod serverless endpoint from a Docker image."""
    mutation = """
    mutation CreateTemplate($input: SaveTemplateInput!) {
      saveTemplate(input: $input) { id name imageName }
    }
    """
    template = _gql(mutation, {"input": {
        "name":        f"{name}-template",
        "imageName":   image_name,
        "isServerless": True,
        "env": env_vars or [],
        "containerDiskInGb": 20,
    }})["saveTemplate"]

    ep_mutation = """
    mutation CreateEndpoint($input: EndpointInput!) {
      saveEndpoint(input: $input) { id name workersMin workersMax }
    }
    """
    ep_input = {
        "name":            name,
        "templateId":      template["id"],
        "gpuIds":          gpu_ids,
        "workersMin":      workers_min,
        "workersMax":      workers_max,
        "idleTimeout":     idle_timeout,
        "scalerType":      "QUEUE_DELAY",
        "scalerValue":     4,
    }
    if volume_id:
        ep_input["networkVolumeId"] = volume_id
    return _gql(ep_mutation, {"input": ep_input})["saveEndpoint"]


def run_serverless_job(endpoint_id: str, payload: dict, timeout: int = 300) -> dict:
    """Submit a job to a serverless endpoint and poll until done."""
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    r = requests.post(f"{SERVERLESS_API}/{endpoint_id}/run", json=payload,
                      headers=headers, timeout=30)
    r.raise_for_status()
    job = r.json()
    job_id = job["id"]
    for _ in range(timeout // 2):
        time.sleep(2)
        r2 = requests.get(f"{SERVERLESS_API}/{endpoint_id}/status/{job_id}",
                          headers=headers, timeout=10)
        r2.raise_for_status()
        s = r2.json()
        if s["status"] in ("COMPLETED", "FAILED", "CANCELLED"):
            return s
    return {"status": "TIMEOUT", "id": job_id}


def get_ssh_info(pod: dict):
    ports = (pod.get("runtime") or {}).get("ports") or []
    for p in ports:
        if p.get("privatePort") == 22 and p.get("isIpPublic"):
            return p["ip"], p["publicPort"]
    return None, None


# ── State ─────────────────────────────────────────────────────────────────────
def _save_state(state: dict):
    STATE_FILE.parent.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


# ── SSH helpers ───────────────────────────────────────────────────────────────
def _scp_upload(ip: str, port: int, local: str, remote: str, timeout: int = 60):
    key = _ssh_key_path()
    cmd = ["scp", "-i", key, "-P", str(port),
           "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15",
           local, f"root@{ip}:{remote}"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"SCP failed: {result.stderr}")


def _ssh_run(ip: str, port: int, command: str, capture: bool = False, timeout: int = 30):
    key = _ssh_key_path()
    cmd = ["ssh", "-i", key, "-p", str(port),
           "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15",
           "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
           f"root@{ip}", command]
    if capture:
        return subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout)
    return subprocess.run(cmd, timeout=timeout)


def _check_training_done(ip: str, port: int) -> tuple[bool, dict | None]:
    """Return (done, info_dict). done=True means /workspace/training_complete.txt exists."""
    try:
        result = _ssh_run(ip, port,
                          "cat /workspace/training_complete.txt 2>/dev/null || echo NOT_DONE",
                          capture=True, timeout=20)
        out = result.stdout.strip() if result.returncode == 0 else "NOT_DONE"
        if "NOT_DONE" in out:
            return False, None
        try:
            return True, json.loads(out)
        except Exception:
            return True, {"raw": out}
    except subprocess.TimeoutExpired:
        return False, None
    except Exception as e:
        print(f"    [SSH check error: {e}]")
        return False, None


# ── Leak check: kill orphan pods before launching ─────────────────────────────
def _kill_orphan_pods():
    """Stop any sovereign training pods that are RUNNING without a local state."""
    pods = list_all_pods()
    running = [p for p in pods
               if p["name"] == POD_NAME and p["desiredStatus"] == "RUNNING"]
    if not running:
        return
    print(f"  [LEAK] {len(running)} orphan pod(s) found — stopping before launch:")
    for p in running:
        uptime = (p.get("runtime") or {}).get("uptimeInSeconds", "?")
        print(f"    Pod {p['id']} (uptime {uptime}s) -> stopping")
        try:
            stop_pod(p["id"])
        except Exception as e:
            print(f"    [stop error: {e}]")
    print("  [LEAK] Orphans cleared. Sleeping 10s...")
    time.sleep(10)


# ── Main: launch ──────────────────────────────────────────────────────────────
def launch(train_mode: str = None, train_split: str = None):
    _load_env()
    if not API_KEY:
        print("ERROR: RUNPOD_API_KEY not set in .env"); sys.exit(1)
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set in .env"); sys.exit(1)

    train_mode  = train_mode  or os.environ.get("TRAIN_MODE",  "orpo")
    # SFT mode trains on the full sft/train split — no split arg needed
    if train_mode == "sft":
        train_split = train_split or os.environ.get("TRAIN_SPLIT", None)
    else:
        train_split = train_split or os.environ.get("TRAIN_SPLIT", "bootstrap_dpo")

    print("\n+==========================================+")
    print("|  SOVEREIGN RUNPOD TRAINING LAUNCHER      |")
    print("+==========================================+\n")
    print(f"  Mode  : {train_mode.upper()}")
    print(f"  Split : {train_split}")
    print(f"  SSH   : {_ssh_key_path()}")
    print()

    # ── Kill any orphan pods ──────────────────────────────────────────────────
    print("[0/6] Checking for orphan pods...")
    _kill_orphan_pods()
    print("  Clean.")

    # ── Find GPU ──────────────────────────────────────────────────────────────
    print("\n[1/6] Finding available GPU...")
    gpus = find_all_gpus_sorted()
    pod = None
    for cloud_type in ["COMMUNITY", "SECURE"]:
        if pod:
            break
        if cloud_type == "SECURE":
            print("  Community cloud dry — trying Secure cloud...")
        for g in gpus:
            price = (g.get("communityPrice") if cloud_type == "COMMUNITY"
                     else g.get("securePrice")) or 99
            print(f"  [{cloud_type}] {g['displayName']} ({g['memoryInGb']}GB) @ ${price:.3f}/hr...")
            try:
                # Inject mode into env before start
                os.environ["TRAIN_MODE"]  = train_mode
                if train_split:
                    os.environ["TRAIN_SPLIT"] = train_split
                else:
                    os.environ.pop("TRAIN_SPLIT", None)
                pod = start_pod(g["id"], cloud_type=cloud_type)
                print(f"  SUCCESS: {g['displayName']} @ ${price:.3f}/hr")
                break
            except Exception as e:
                msg = str(e)
                if any(x in msg for x in ["SUPPLY_CONSTRAINT", "no longer any instances",
                                           "does not have the resources", "RUNPOD",
                                           "resources are unavailable"]):
                    print("  Unavailable — trying next...")
                    continue
                raise

    if pod is None:
        print("ERROR: No GPUs available. Try again in a few minutes.")
        sys.exit(1)

    pod_id = pod["id"]
    print(f"\n[2/6] Pod started: {pod_id}  Status: {pod['desiredStatus']}")
    _save_state({"pod_id": pod_id, "started_at": time.time(),
                 "train_mode": train_mode, "train_split": train_split})

    # ── Wait for SSH ──────────────────────────────────────────────────────────
    print("[3/6] Waiting for SSH...")
    ip, port = None, None
    for attempt in range(60):
        time.sleep(15)
        try:
            pod = get_pod(pod_id)
            ip, port = get_ssh_info(pod)
            status = pod.get("desiredStatus", "?")
            print(f"  [{attempt+1}/60] Status: {status}  SSH: {ip}:{port}")
            if ip and port:
                break
        except Exception as e:
            print(f"  [{attempt+1}/60] API error: {e}")
    else:
        print("  ERROR: Pod never got SSH. Run: python runpod_launcher.py --cleanup")
        sys.exit(1)

    print("  Waiting 60s for sshd + authorized_keys to initialize...")
    time.sleep(60)

    # ── Upload (with retry) ───────────────────────────────────────────────────
    print(f"\n[4/6] Uploading training script to {ip}:{port}...")
    for scp_attempt in range(1, 6):
        try:
            _scp_upload(ip, port, str(TRAIN_SCRIPT), "/workspace/runpod_train.py")
            print("  Uploaded.")
            break
        except RuntimeError as e:
            if scp_attempt < 5:
                print(f"  SCP attempt {scp_attempt}/5 failed — waiting 20s: {e}")
                time.sleep(20)
            else:
                print(f"  SCP failed after 5 attempts. Pod may need longer to boot.")
                raise
    _save_state({"pod_id": pod_id, "ip": ip, "port": port,
                 "started_at": time.time(), "train_mode": train_mode,
                 "train_split": train_split,
                 "train_agent": os.environ.get("TRAIN_AGENT", "avery")})

    # ── Launch ────────────────────────────────────────────────────────────────
    print(f"\n[5/6] Launching training on pod...")
    train_cmd = (
        f"nohup python /workspace/runpod_train.py "
        f"--hf_token {HF_TOKEN} "
        f"> /workspace/train.log 2>&1 & echo LAUNCHED"
    )
    _ssh_run(ip, port, train_cmd, timeout=300)

    print(f"""
  Training is running on the pod (nohup — survives terminal close).

  To watch live:
    ssh -i {_ssh_key_path()} -p {port} root@{ip} "tail -f /workspace/train.log"
  Or:
    python runpod_launcher.py --tail

  To check status:
    python runpod_launcher.py --status
""")

    # ── Monitor ───────────────────────────────────────────────────────────────
    print("[6/6] Monitoring for completion (Ctrl+C safe — pod keeps running)...")
    _monitor(pod_id, ip, port)


def _monitor(pod_id: str, ip: str, port: int, max_minutes: int = 180):
    """Poll for training_complete.txt. Safe to kill — pod keeps running."""
    log_lines = []
    for minute in range(1, max_minutes + 1):
        time.sleep(60)

        # Re-fetch SSH info in case of pod restart
        try:
            pod = get_pod(pod_id)
            new_ip, new_port = get_ssh_info(pod)
            if new_ip:
                ip, port = new_ip, new_port
        except Exception:
            pass

        done, info = _check_training_done(ip, port)

        # Also tail log every 5 min
        if minute % 5 == 0:
            try:
                result = _ssh_run(ip, port, "tail -3 /workspace/train.log", capture=True, timeout=15)
                snippet = result.stdout.strip().replace("\n", " | ")
                print(f"  [{minute}m] {snippet[:120]}")
            except Exception:
                print(f"  [{minute}m] (log unavailable)")
        else:
            print(f"  [{minute}m] {'DONE' if done else 'training...'}")

        if done:
            _handle_completion(pod_id, info)
            return

    # Timeout reached
    print(f"\n  TIMEOUT: {max_minutes} minutes reached.")
    print("  Training may still be running. Check with: python runpod_launcher.py --status")
    print(f"  SSH: ssh -i {_ssh_key_path()} -p {port} root@{ip} 'tail -50 /workspace/train.log'")
    # Save current state so --status can reconnect
    state = _load_state()
    state.update({"ip": ip, "port": port, "pod_id": pod_id})
    _save_state(state)


def _handle_completion(pod_id: str, info: dict):
    print("\n  *** TRAINING COMPLETE ***")
    if info:
        print(f"  Model  : {info.get('model', '?')}")
        loss = info.get('loss')
        rt   = info.get('runtime_s', 0)
        print(f"  Loss   : {loss:.4f}" if loss else "  Loss   : ?")
        print(f"  Time   : {rt/60:.1f} min" if rt else "  Time   : ?")

    print("\n  Stopping pod to save credits...")
    try:
        stop_pod(pod_id)
        print(f"  Pod {pod_id} stopped.")
    except Exception as e:
        print(f"  Stop error: {e}")
    _save_state({})

    print("\n  Next step: python merge_and_convert.py")
    print("  LoRA: https://huggingface.co/tastytator/avery-sovereign-lora\n")


# ── CLI: --status ─────────────────────────────────────────────────────────────
def status():
    _load_env()
    state = _load_state()

    # Try state-based lookup first
    pod_id = state.get("pod_id")
    if not pod_id:
        print("No pod in local state. Checking RunPod for running pods...")
        pods = [p for p in list_all_pods()
                if p["name"] == POD_NAME and p["desiredStatus"] == "RUNNING"]
        if not pods:
            print("  No running sovereign pods found.")
            return
        print(f"  Found {len(pods)} running pod(s):")
        for p in pods:
            ip, port = get_ssh_info(p)
            uptime   = (p.get("runtime") or {}).get("uptimeInSeconds", "?")
            print(f"  Pod {p['id']}  uptime={uptime}s  SSH={ip}:{port}")
            if ip and port:
                result = _ssh_run(ip, port, "tail -20 /workspace/train.log", capture=True, timeout=20)
                print("\n--- Last 20 lines of train.log ---")
                print(result.stdout or "(empty)")
        return

    print(f"  Pod    : {pod_id}")
    print(f"  Mode   : {state.get('train_mode', '?')}")
    print(f"  Split  : {state.get('train_split', '?')}")
    try:
        pod  = get_pod(pod_id)
        ip, port = get_ssh_info(pod)
        print(f"  Status : {pod.get('desiredStatus')}")
        print(f"  SSH    : {ip}:{port}")
        if ip and port:
            done, info = _check_training_done(ip, port)
            print(f"  Done   : {done}")
            if done:
                print(f"  Info   : {info}")
                print("\n  Run: python runpod_launcher.py --stop")
            else:
                result = _ssh_run(ip, port, "tail -20 /workspace/train.log", capture=True, timeout=20)
                print("\n--- Last 20 lines of train.log ---")
                print(result.stdout or "(empty)")
    except Exception as e:
        print(f"  API error: {e}")


# ── CLI: --tail ───────────────────────────────────────────────────────────────
def tail():
    _load_env()
    state = _load_state()
    ip, port = state.get("ip"), state.get("port")
    pod_id   = state.get("pod_id")

    if not ip and pod_id:
        pod = get_pod(pod_id)
        ip, port = get_ssh_info(pod)
    if not ip:
        pods = [p for p in list_all_pods()
                if p["name"] == POD_NAME and p["desiredStatus"] == "RUNNING"]
        if pods:
            ip, port = get_ssh_info(pods[0])
    if not ip:
        print("No running pod found."); return

    key = _ssh_key_path()
    print(f"Tailing train.log on {ip}:{port}  (Ctrl+C to detach)\n")
    subprocess.run(["ssh", "-i", key, "-p", str(port),
                    "-o", "StrictHostKeyChecking=no",
                    f"root@{ip}", "tail -f /workspace/train.log"])


# ── CLI: --stop ───────────────────────────────────────────────────────────────
def stop():
    _load_env()
    state  = _load_state()
    pod_id = state.get("pod_id")
    if pod_id:
        try:
            stop_pod(pod_id)
            print(f"Stopped pod {pod_id}")
        except Exception as e:
            print(f"Error stopping {pod_id}: {e}")
        _save_state({})
    else:
        print("No pod in local state.")


# ── CLI: --cleanup ────────────────────────────────────────────────────────────
def cleanup():
    _load_env()
    pods = list_all_pods()
    running = [p for p in pods
               if p["name"] == POD_NAME and p["desiredStatus"] == "RUNNING"]
    if not running:
        print("No running sovereign pods."); return
    print(f"Stopping {len(running)} running pod(s):")
    for p in running:
        try:
            stop_pod(p["id"])
            print(f"  Stopped {p['id']}")
        except Exception as e:
            print(f"  Error {p['id']}: {e}")
    _save_state({})
    print("Done.")


# ── Setup: Network Volume ──────────────────────────────────────────────────────
def setup_volume(size_gb: int = 50, datacenter_id: str = "US-TX-3"):
    """Create a network volume for model caching and write ID to .env."""
    _load_env()
    global NETWORK_VOLUME_ID

    print("+==========================================+")
    print("|  SOVEREIGN NETWORK VOLUME SETUP          |")
    print("+==========================================+\n")

    # Check if volume already exists
    try:
        existing = list_volumes()
        sovereign_vols = [v for v in existing if "sovereign" in v["name"].lower()]
        if sovereign_vols:
            v = sovereign_vols[0]
            print(f"  Found existing volume: {v['name']} ({v['size']}GB) — ID: {v['id']}")
            NETWORK_VOLUME_ID = v["id"]
            _write_env_key("RUNPOD_VOLUME_ID", v["id"])
            print(f"  Saved to .env: RUNPOD_VOLUME_ID={v['id']}")
            print(f"\n  Mount path on pods: /runpod-volume")
            print(f"  HF cache will auto-populate on first training run.")
            return v
    except Exception as e:
        print(f"  (Could not list existing volumes: {e})")

    print(f"  Creating {size_gb}GB network volume in {datacenter_id}...")
    try:
        vol = create_volume("sovereign-model-cache", size_gb=size_gb,
                            datacenter_id=datacenter_id)
        NETWORK_VOLUME_ID = vol["id"]
        _write_env_key("RUNPOD_VOLUME_ID", vol["id"])
        print(f"\n  Volume created:  {vol['name']}")
        print(f"  ID:              {vol['id']}")
        print(f"  Size:            {vol['size']}GB")
        print(f"  Datacenter:      {vol['dataCenterId']}")
        print(f"\n  Saved to .env: RUNPOD_VOLUME_ID={vol['id']}")
        print(f"\n  Next training run will mount this volume and cache the base model.")
        print(f"  After first run: model downloads never happen again.")
        return vol
    except Exception as e:
        print(f"\n  ERROR: {e}")
        print("  Tip: Make sure your RunPod account has network volumes available.")
        print("  Fallback: Training still works without a volume (just re-downloads each time).")


# ── Setup: Serverless Endpoint ─────────────────────────────────────────────────
def setup_serverless(image: str = "", gpu_ids: str = "AMPERE_24,AMPERE_16"):
    """Create an Avery serverless inference endpoint."""
    _load_env()

    # Default image — the pre-built sovereign handler (GHCR or Docker Hub)
    if not image:
        image = os.environ.get("SERVERLESS_IMAGE",
                               "tastytator/avery-serverless:latest")

    print("+==========================================+")
    print("|  SOVEREIGN SERVERLESS ENDPOINT SETUP     |")
    print("+==========================================+\n")

    # Check for existing endpoint
    try:
        existing = list_endpoints()
        avery_eps = [e for e in existing if "avery" in e["name"].lower()]
        if avery_eps:
            ep = avery_eps[0]
            print(f"  Found existing endpoint: {ep['name']} — ID: {ep['id']}")
            _write_env_key("RUNPOD_ENDPOINT_ID", ep["id"])
            print(f"  URL: https://api.runpod.io/v2/{ep['id']}/run")
            return ep
    except Exception as e:
        print(f"  (Could not list endpoints: {e})")

    print(f"  Image:   {image}")
    print(f"  GPUs:    {gpu_ids}")
    print(f"  Workers: 0 min / 3 max (scale-to-zero)\n")

    env_vars = [
        {"key": "HF_TOKEN",       "value": HF_TOKEN},
        {"key": "LORA_REPO",      "value": "tastytator/avery-sovereign-lora"},
        {"key": "BASE_MODEL",     "value": "Qwen/Qwen2-7B-Instruct"},
    ]
    if NETWORK_VOLUME_ID:
        env_vars.append({"key": "HF_HOME", "value": "/runpod-volume/hf-cache"})

    try:
        ep = create_endpoint(
            name="avery-sovereign-inference",
            image_name=image,
            gpu_ids=gpu_ids,
            workers_min=0,
            workers_max=3,
            idle_timeout=5,
            volume_id=NETWORK_VOLUME_ID,
            env_vars=env_vars,
        )
        _write_env_key("RUNPOD_ENDPOINT_ID", ep["id"])
        print(f"  Endpoint created: {ep['name']}")
        print(f"  ID:               {ep['id']}")
        print(f"\n  API endpoint: https://api.runpod.io/v2/{ep['id']}/run")
        print(f"  Saved to .env: RUNPOD_ENDPOINT_ID={ep['id']}")
        print(f"\n  Usage:")
        print(f"    python serverless_deploy.py --ask 'Build a SaaS pricing strategy'")
        return ep
    except Exception as e:
        print(f"\n  ERROR creating endpoint: {e}")
        print("  You can create it manually at: https://www.runpod.io/serverless")
        print(f"  Use image: {image}")


# ── Volume info ────────────────────────────────────────────────────────────────
def volume_info():
    _load_env()
    print("+==========================================+")
    print("|  RUNPOD STORAGE STATUS                   |")
    print("+==========================================+\n")
    try:
        vols = list_volumes()
        if not vols:
            print("  No network volumes. Run: python runpod_launcher.py --setup-volume")
            return
        print(f"  {'ID':<20} {'Name':<30} {'Size':>6}  {'DC'}")
        print(f"  {'-'*20} {'-'*30} {'------':>6}  {'--'}")
        for v in vols:
            marker = " ← active" if v["id"] == NETWORK_VOLUME_ID else ""
            print(f"  {v['id']:<20} {v['name']:<30} {v['size']:>5}GB  {v['dataCenterId']}{marker}")
    except Exception as e:
        print(f"  Error: {e}")

    try:
        eps = list_endpoints()
        if eps:
            print(f"\n  Serverless Endpoints:")
            for ep in eps:
                print(f"    {ep['id']:<20} {ep['name']}")
    except Exception:
        pass


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Sovereign RunPod Launcher")
    p.add_argument("--status",         action="store_true", help="Check running pod status + log")
    p.add_argument("--stop",           action="store_true", help="Stop the tracked pod")
    p.add_argument("--tail",           action="store_true", help="Attach to live training log")
    p.add_argument("--cleanup",        action="store_true", help="Stop ALL sovereign pods")
    p.add_argument("--setup-volume",   action="store_true", help="Create/find network volume for model caching")
    p.add_argument("--volume-info",    action="store_true", help="Show volume and endpoint status")
    p.add_argument("--setup-serverless", action="store_true", help="Create Avery serverless inference endpoint")
    p.add_argument("--image",          default="", help="Docker image for serverless endpoint")
    p.add_argument("--gpu-ids",        default="AMPERE_24,AMPERE_16", help="GPU types for serverless")
    p.add_argument("--volume-size",    type=int, default=50, help="Volume size in GB (default: 50)")
    p.add_argument("--datacenter",     default="US-TX-3", help="RunPod datacenter ID")
    p.add_argument("--mode",           default="",  help="Training mode: sft / orpo / dpo")
    p.add_argument("--split",          default="",  help="HF split to train on")
    args = p.parse_args()

    if args.status:
        status()
    elif args.stop:
        stop()
    elif args.tail:
        tail()
    elif args.cleanup:
        cleanup()
    elif args.setup_volume:
        setup_volume(size_gb=args.volume_size, datacenter_id=args.datacenter)
    elif args.volume_info:
        volume_info()
    elif args.setup_serverless:
        setup_serverless(image=args.image, gpu_ids=args.gpu_ids)
    else:
        launch(train_mode=args.mode or None, train_split=args.split or None)
