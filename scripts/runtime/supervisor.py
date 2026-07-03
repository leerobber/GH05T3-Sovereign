"""
GH05T3 + Sovereign Economy — Process Supervisor
================================================
Single entry point for everything. No terminal windows.
All output → logs/  Auto-restarts crashed services.
Status API → http://localhost:8090/status

Usage:
    python supervisor.py            # start everything
    python supervisor.py --stop     # kill everything
    python supervisor.py --status   # print status and exit
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional
import urllib.request

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[2]  # scripts/runtime/supervisor.py → GH05T3 root
ECO_DIR = Path(r"C:\Users\leer4\Documents\agent-economy")
LOGS    = ROOT / "logs"
LOGS.mkdir(exist_ok=True)

def _find_py(*candidates) -> Path:
    """Return the first existing Python executable, fall back to sys.executable."""
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    return Path(sys.executable)

GH_PY  = _find_py(
    ROOT / "backend" / ".venv" / "Scripts" / "python.exe",
    ROOT / ".venv"           / "Scripts" / "python.exe",
    ROOT / "venv"            / "Scripts" / "python.exe",
)
ECO_PY = _find_py(
    ECO_DIR / "venv"    / "Scripts" / "python.exe",
    ECO_DIR / ".venv"   / "Scripts" / "python.exe",
)
# System Python 3.12 — has stripe, fastapi, uvicorn installed globally
SYS_PY = _find_py(
    r"C:\Users\leer4\AppData\Local\Programs\Python\Python312\python.exe",
)
# RyzenAI conda env — has onnxruntime_genai, sentence_transformers, BGE ONNX
NPU_PY = _find_py(
    r"C:\Users\leer4\.conda\envs\ryzen-ai-1.7.0\python.exe",
)
# Phi env — onnxruntime-genai-cuda + nvidia-*-cu12 for Phi-3.5-mini on RTX 5050
PHI_PY = _find_py(
    r"C:\Users\leer4\phi_dml_env\Scripts\python.exe",
)

STATUS_PORT = 8090

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[
        logging.FileHandler(LOGS / "supervisor.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("supervisor")
log.info("GH05T3 Python : %s", GH_PY)
log.info("Economy Python: %s", ECO_PY)

# ── Service definitions ────────────────────────────────────────────────────────
# order matters — services start in list order, each waits for the previous
SERVICES = [
    {
        "name":    "economy-api",
        "cmd":     [str(SYS_PY), "-m", "uvicorn", "main:app",
                    "--host", "0.0.0.0", "--port", "8081", "--log-level", "info"],
        "cwd":     str(ECO_DIR),
        "port":    8081,
        "health":  "http://localhost:8081/",           # /health is slow; root / is fast
        "timeout": 30,
        "restart": True,
    },
    {
        "name":    "mongo",
        "cmd":     ["mongod", "--dbpath", str(ROOT / "mongo-data"),
                    "--port", "27017", "--bind_ip", "127.0.0.1"],
        "cwd":     str(ROOT),
        "port":    27017,
        "health":  None,           # TCP check on 27017
        "timeout": 20,
        "restart": True,
        # kill_port=False: if Windows MongoDB service is running on 27017, skip start.
        # If port is free, supervisor starts its own mongod and manages it.
        "kill_port": False,
    },
    {
        "name":    "backend",
        "cmd":     [str(SYS_PY), "-m", "uvicorn", "server:app",
                    "--host", "0.0.0.0", "--port", "8001"],
        "cwd":     str(ROOT / "backend"),
        "port":    8001,
        "health":  "http://localhost:8001/api/health",
        "timeout": 20,
        "restart": True,
    },
    {
        "name":    "gateway",
        "cmd":     [str(SYS_PY), "-m", "uvicorn", "gateway_v3:app",
                    "--host", "0.0.0.0", "--port", "8002"],
        "cwd":     str(ROOT / "backend"),
        "port":    8002,
        "health":  "http://localhost:8002/health",
        "timeout": 20,
        "restart": True,
    },
    {
        "name":    "frontend",
        "cmd":     [str(SYS_PY), "-m", "http.server", "3210",
                    "--directory", str(ROOT / "frontend" / "build")],
        "cwd":     str(ROOT),
        "port":    3210,
        "health":  "http://localhost:3210",
        "timeout": 5,
        "restart": True,
    },
    # ── GH05T3 learning pipeline ───────────────────────────────────────────────
    {
        "name":    "continuous-learner",
        "cmd":     [str(SYS_PY), str(ROOT / "scripts" / "training" / "continuous_learner.py")],
        "cwd":     str(ROOT),
        "port":    None,
        "health":  None,        # uses heartbeat file check below
        "timeout": 15,
        "restart": True,
        "heartbeat_file": str(ROOT / "data" / "learner_heartbeat.json"),
        "heartbeat_max_age": 900,   # seconds — restart if silent > 15 min (cycles can take 10 min)
    },
    {
        "name":    "cmd-listener",
        "cmd":     [str(SYS_PY), str(ROOT / "scripts" / "runtime" / "cmd_listener.py")],
        "cwd":     str(ROOT),
        "port":    None,
        "health":  None,
        "timeout": 5,
        "restart": True,
    },
    {
        "name":    "amplifier",
        "cmd":     [str(SYS_PY), str(ROOT / "scripts" / "training" / "amplifier.py"), "--variants", "5"],
        "cwd":     str(ROOT),
        "port":    None,
        "health":  None,
        "timeout": 5,
        "restart": True,
    },
    # ── SovereignNation client proxy ───────────────────────────────────────────
    {
        "name":    "serve",
        "cmd":     [str(SYS_PY), "sovereignnation/serve.py"],
        "cwd":     str(ROOT),
        "port":    7861,
        "health":  "http://localhost:7861/health",
        "timeout": 10,
        "restart": True,
    },
    # ── SovereignNation payments service ──────────────────────────────────────
    {
        "name":    "payments",
        "cmd":     [str(SYS_PY), "-m", "uvicorn", "payments:app",
                    "--host", "0.0.0.0", "--port", "7862", "--log-level", "info"],
        "cwd":     str(ROOT / "sovereignnation"),
        "port":    7862,
        "health":  "http://localhost:7862/health",
        "timeout": 15,
        "restart": True,
    },
    # ── Phi-3.5-mini GPU inference service (phi_dml_env, RTX 5050 CUDA EP) ──────
    {
        "name":    "phi",
        "cmd":     [str(PHI_PY), "phi_service.py"],
        "cwd":     str(ROOT / "sovereignnation"),
        "port":    8112,
        "health":  "http://127.0.0.1:8112/health",  # 200 once uvicorn is up; model loads ~5s after
        "timeout": 60,
        "restart": True,
    },
    # ── Agent pipeline CORS proxy (48hr_poc_agent_pipeline.html -> Ollama) ──────
    {
        "name":    "pipeline-backend",
        "cmd":     [str(SYS_PY), "-m", "uvicorn", "pipeline_backend:app",
                    "--host", "0.0.0.0", "--port", "8099", "--log-level", "warning"],
        "cwd":     str(ROOT / "sovereignnation"),
        "port":    8099,
        "health":  "http://localhost:8099/health",
        "timeout": 15,
        "restart": True,
    },
    # ── Sovereign Interface — Genomic dispatch + Theory Lab (port 8100) ──────────
    {
        "name":    "sovereign-interface",
        "cmd":     [str(SYS_PY), "-m", "uvicorn", "sovereign_interface:app",
                    "--host", "0.0.0.0", "--port", "8100", "--log-level", "warning"],
        "cwd":     str(ROOT / "sovereignnation"),
        "port":    8100,
        "health":  "http://localhost:8100/health",
        "timeout": 20,
        "restart": True,
    },
    # ── Genome Lab UI — enterprise dashboard static server (port 7720) ───────────
    {
        "name":    "genome-lab-ui",
        "cmd":     [str(SYS_PY), "-m", "http.server", "7720",
                    "--directory", str(ROOT / "frontend")],
        "cwd":     str(ROOT),
        "port":    7720,
        "health":  "http://localhost:7720/genome_dashboard.html",
        "timeout": 5,
        "restart": True,
    },
    # ── NPU Embedding Service (RyzenAI ryzen-ai-1.7.0 env, BGE-large ONNX) ──────
    {
        "name":    "npu-embed",
        "cmd":     [str(NPU_PY), "-m", "uvicorn", "npu_embedding_service:app",
                    "--host", "127.0.0.1", "--port", "8111", "--log-level", "warning"],
        "cwd":     str(ROOT / "sovereignnation"),
        "port":    8111,
        "health":  "http://localhost:8111/health",
        "timeout": 300,  # BGE-large ONNX + MiniLM cold-load takes 2-3min on first run
        "restart": True,
    },
    # ── SAGE / KAIROS Self-Improvement Engine ────────────────────────────────────
    {
        "name":    "sage-engine",
        "cmd":     [str(SYS_PY), "-m", "uvicorn", "start_sage:app",
                    "--host", "0.0.0.0", "--port", "8098", "--log-level", "warning"],
        "cwd":     str(ROOT / "scripts" / "runtime"),
        "port":    8098,
        "health":  "http://localhost:8098/health",
        "timeout": 20,
        "restart": True,
    },
    # ── Landing page static server ─────────────────────────────────────────────
    {
        "name":    "landing-server",
        "cmd":     [str(SYS_PY), "-m", "http.server", "8765",
                    "--directory", str(ROOT / "sovereignnation" / "landing")],
        "cwd":     str(ROOT),
        "port":    8765,
        "health":  "http://localhost:8765/",
        "timeout": 5,
        "restart": True,
    },
    # ── Cloudflare tunnels ────────────────────────────────────────────────────
    {
        "name":    "tunnel-chat",
        "cmd":     [r"C:\Program Files (x86)\cloudflared\cloudflared.exe",
                    "tunnel", "--url", "http://localhost:7861",
                    "--logfile", str(ROOT / "data" / "tunnel_chat.log"),
                    "--loglevel", "info"],
        "cwd":     str(ROOT),
        "port":    None,
        "health":  None,
        "timeout": 15,
        "restart": True,
        "backoff": [5, 10, 20, 40, 60],
    },
    {
        "name":    "tunnel-landing",
        "cmd":     [r"C:\Program Files (x86)\cloudflared\cloudflared.exe",
                    "tunnel", "--url", "http://localhost:8765",
                    "--logfile", str(ROOT / "data" / "tunnel_landing.log"),
                    "--loglevel", "info"],
        "cwd":     str(ROOT),
        "port":    None,
        "health":  None,
        "timeout": 15,
        "restart": True,
        "backoff": [5, 10, 20, 40, 60],
    },
    # ── Tunnel URL watcher ────────────────────────────────────────────────────
    {
        "name":    "tunnel-watcher",
        "cmd":     [str(SYS_PY), str(ROOT / "scripts" / "runtime" / "tunnel_watcher.py")],
        "cwd":     str(ROOT),
        "port":    None,
        "health":  None,
        "timeout": 3,
        "restart": True,
    },
]

# ── State ──────────────────────────────────────────────────────────────────────
_state: dict[str, dict] = {
    s["name"]: {
        "status":        "stopped",   # stopped|starting|running|degraded|restarting
        "pid":           None,
        "restarts":      0,
        "last_restart":  0.0,
        "last_healthy":  0.0,
    }
    for s in SERVICES
}

_procs:  dict[str, Optional[subprocess.Popen]] = {s["name"]: None for s in SERVICES}
_stop_event = threading.Event()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _log_path(name: str) -> Path:
    return LOGS / f"{name}.log"


def _tcp_open(port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


def _http_ok(url: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status < 500
    except Exception:
        return False


def _is_healthy(svc: dict) -> bool:
    health = svc.get("health")
    port   = svc.get("port")
    if health:
        return _http_ok(health)
    if port:
        return _tcp_open(port)

    # Heartbeat file check — detect hung (alive but silent) processes
    hb_file = svc.get("heartbeat_file")
    hb_max  = svc.get("heartbeat_max_age", 300)
    if hb_file:
        hb_path = Path(hb_file)
        if hb_path.exists():
            try:
                data = json.loads(hb_path.read_text(encoding="utf-8"))
                age  = time.time() - data.get("ts", 0)
                if age > hb_max:
                    log.warning("%-18s  heartbeat stale (%.0fs) — marking unhealthy",
                                svc["name"], age)
                    return False
                return True
            except Exception:
                pass
        # File doesn't exist yet — process may still be starting
        proc = _procs.get(svc["name"])
        return proc is not None and proc.poll() is None

    # No check defined — process alive = healthy
    proc = _procs.get(svc["name"])
    return proc is not None and proc.poll() is None


def _kill_port(port: int):
    """Kill whatever is listening on a port (Windows)."""
    try:
        out = subprocess.check_output(
            f'netstat -ano 2>nul | findstr "0.0.0.0:{port} "',
            shell=True, text=True, stderr=subprocess.DEVNULL,
        )
        for line in out.strip().splitlines():
            parts = line.split()
            if parts:
                pid = parts[-1]
                if pid.isdigit() and pid != "0":
                    subprocess.run(f"taskkill /F /PID {pid}", shell=True,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def _kill_tree(pid: int):
    """Kill a process and all its descendants (Windows taskkill /T)."""
    try:
        subprocess.run(
            f"taskkill /F /T /PID {pid}",
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def _start_proc(svc: dict) -> subprocess.Popen:
    """Spawn a service, redirect output to its log file."""
    log_file = open(_log_path(svc["name"]), "a", encoding="utf-8", buffering=1)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["AETHYRO_SKIP_LICENSE"] = "1"  # owner bypass — no external license check needed
    # Merge any per-service env overrides
    for k, v in svc.get("env", {}).items():
        env[k] = v

    proc = subprocess.Popen(
        svc["cmd"],
        cwd=svc["cwd"],
        stdout=log_file,
        stderr=log_file,
        env=env,
        creationflags=subprocess.CREATE_NO_WINDOW,  # no terminal window on Windows
    )
    return proc


# ── Start / stop ───────────────────────────────────────────────────────────────

def start_service(svc: dict, wait: bool = True):
    name = svc["name"]
    st   = _state[name]

    # If kill_port=False and port is already occupied → externally managed, skip start
    if svc.get("port") and not svc.get("kill_port", True):
        if _tcp_open(svc["port"]):
            log.info("%-18s  port %d already occupied (external service) — skipping start",
                     name, svc["port"])
            st["status"]       = "running"
            st["last_healthy"] = time.time()
            return

    # Run optional pre-start hook (e.g. clear stale lock files)
    pre_start = svc.get("pre_start")
    if pre_start:
        try:
            pre_start()
        except Exception as e:
            log.warning("%-18s  pre_start hook error: %s", name, e)

    # Clear port if occupied — skip for services that flag kill_port=False
    if svc.get("port") and svc.get("kill_port", True):
        _kill_port(svc["port"])

    log.info("Starting %-18s ...", name)
    st["status"] = "starting"

    try:
        proc = _start_proc(svc)
    except FileNotFoundError as e:
        log.error("%-18s  FAILED to start: %s", name, e)
        st["status"] = "stopped"
        return

    _procs[name] = proc
    st["pid"]    = proc.pid

    if not wait:
        st["status"] = "running"
        return

    # wait for health check
    deadline = time.time() + svc["timeout"]
    while time.time() < deadline:
        if proc.poll() is not None:
            log.error("%-18s  exited during startup (code %s) — see logs/%s.log",
                      name, proc.returncode, name)
            st["status"] = "stopped"
            return
        if _is_healthy(svc):
            st["status"]       = "running"
            st["last_healthy"] = time.time()
            log.info("%-18s  UP  (pid=%d)", name, proc.pid)
            return
        time.sleep(1)

    # timed out — still mark running if process is alive
    if proc.poll() is None:
        log.warning("%-18s  health check timed out but process alive — continuing", name)
        st["status"] = "degraded"
    else:
        log.error("%-18s  failed to start within %ds", name, svc["timeout"])
        st["status"] = "stopped"


def stop_all():
    log.info("Stopping all services...")
    for svc in reversed(SERVICES):
        name = svc["name"]
        proc = _procs.get(name)
        if proc and proc.poll() is None:
            log.info("  Stopping %s (pid=%d)", name, proc.pid)
            _kill_tree(proc.pid)   # kill entire subprocess tree, not just direct child
            try:
                proc.wait(timeout=3)
            except Exception:
                pass
        if svc.get("port"):
            _kill_port(svc["port"])
        _state[name]["status"] = "stopped"
        _state[name]["pid"]    = None
    log.info("All stopped.")


# ── Monitor thread ─────────────────────────────────────────────────────────────

def _monitor():
    BACKOFF = [5, 10, 20, 40, 60]   # seconds between restart attempts

    while not _stop_event.wait(10):  # check every 10 seconds
      try:
        for svc in SERVICES:
            name = svc["name"]
            st   = _state[name]

            if st["status"] in ("stopped", "starting", "restarting"):
                continue
            if not svc.get("restart"):
                continue

            proc = _procs.get(name)

            # Externally managed: kill_port=False + no proc we started.
            # Skip process-death check; just do a health/port check.
            if not svc.get("kill_port", True) and proc is None:
                if _is_healthy(svc):
                    st["status"]       = "running"
                    st["last_healthy"] = time.time()
                else:
                    age = time.time() - st["last_healthy"]
                    if age > 30:
                        st["status"] = "degraded"
                continue

            dead = (proc is None or proc.poll() is not None)

            if dead:
                since = time.time() - st["last_restart"]
                idx   = min(st["restarts"], len(BACKOFF) - 1)
                wait  = BACKOFF[idx]
                if since < wait:
                    continue
                log.warning("%-18s  died — restarting (attempt %d, waited %ds)",
                            name, st["restarts"] + 1, wait)
                st["status"]       = "restarting"
                st["restarts"]    += 1
                st["last_restart"] = time.time()
                threading.Thread(target=start_service, args=(svc,), daemon=True).start()
                continue

            # process alive — run health check
            if _is_healthy(svc):
                st["status"]       = "running"
                st["last_healthy"] = time.time()
            else:
                age = time.time() - st["last_healthy"]
                if age > 30:
                    st["status"] = "degraded"
                # Heartbeat-based hung-process kill: if the svc has a heartbeat
                # file and it's stale by 2x the max age, force-kill so the
                # monitor's dead-process branch will restart it next tick.
                hb_max = svc.get("heartbeat_max_age", 0)
                if hb_max and age > hb_max * 2:
                    log.warning("%-18s  appears hung (no heartbeat >%ds) — force-killing",
                                svc["name"], hb_max * 2)
                    p = _procs.get(svc["name"])
                    if p and p.poll() is None:
                        _kill_tree(p.pid)  # kill entire subprocess tree
      except Exception as exc:
          log.error("Monitor loop error (will retry): %s", exc, exc_info=True)


# ── Status HTTP server ─────────────────────────────────────────────────────────

class _StatusHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):  # silence request logs
        pass

    def do_GET(self):
        if self.path not in ("/status", "/status/"):
            self.send_response(404)
            self.end_headers()
            return
        payload = json.dumps({
            "services": {
                name: {
                    "status":    st["status"],
                    "pid":       st["pid"],
                    "restarts":  st["restarts"],
                }
                for name, st in _state.items()
            },
            "all_ok": all(st["status"] == "running" for st in _state.values()),
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)


def _run_status_server():
    srv = HTTPServer(("127.0.0.1", STATUS_PORT), _StatusHandler)
    while not _stop_event.is_set():
        srv.handle_request()


# ── Main ───────────────────────────────────────────────────────────────────────

PIDFILE = ROOT / "data" / "supervisor.pid"


def _acquire_lock() -> bool:
    """Write pidfile. Return False if another instance appears to be running."""
    PIDFILE.parent.mkdir(exist_ok=True)
    if PIDFILE.exists():
        try:
            existing_pid = int(PIDFILE.read_text().strip())
            # Check if that PID is actually a running supervisor
            import psutil
            proc = psutil.Process(existing_pid)
            cmdline = " ".join(proc.cmdline())
            if "supervisor.py" in cmdline:
                log.warning("Supervisor already running (pid=%d) — exiting.", existing_pid)
                return False
        except Exception:
            pass  # stale pidfile, dead PID, psutil.NoSuchProcess, etc — proceed
    PIDFILE.write_text(str(os.getpid()))
    return True


def _release_lock():
    try:
        PIDFILE.unlink(missing_ok=True)
    except Exception:
        pass


def _ensure_mongo_data():
    (ROOT / "mongo-data").mkdir(exist_ok=True)


def run():
    if not _acquire_lock():
        sys.exit(0)

    import atexit
    atexit.register(_release_lock)

    _ensure_mongo_data()

    # Status API thread (widget polls this)
    threading.Thread(target=_run_status_server, daemon=True).start()
    log.info("Status API: http://localhost:%d/status", STATUS_PORT)

    # Start services in order, each waiting for health before next
    for svc in SERVICES:
        if _stop_event.is_set():
            break
        start_service(svc, wait=True)

    log.info("All services started. Monitoring...")

    # Monitor thread for auto-restart
    threading.Thread(target=_monitor, daemon=True).start()

    # Handle Ctrl-C / SIGTERM
    def _handle_shutdown(*_):
        log.info("Shutdown signal received.")
        _stop_event.set()
        stop_all()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    # Block main thread
    while not _stop_event.is_set():
        time.sleep(1)


def cmd_stop():
    """Send stop signal by killing all ports."""
    print("Stopping all services...")
    for svc in SERVICES:
        if svc.get("port"):
            _kill_port(svc["port"])
            print(f"  Killed port {svc['port']}")
    print("Done.")


def cmd_status():
    """Print current status from the running supervisor."""
    try:
        with urllib.request.urlopen(f"http://localhost:{STATUS_PORT}/status", timeout=3) as r:
            data = json.loads(r.read())
        print(f"\n{'SERVICE':<20}  {'STATUS':<12}  {'PID':<8}  RESTARTS")
        print("-" * 52)
        for name, info in data["services"].items():
            print(f"{name:<20}  {info['status']:<12}  {str(info['pid'] or ''):<8}  {info['restarts']}")
        print()
        print("Overall:", "OK" if data["all_ok"] else "DEGRADED")
    except Exception as e:
        print(f"Supervisor not running (can't reach port {STATUS_PORT}): {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop",   action="store_true", help="kill all services")
    ap.add_argument("--status", action="store_true", help="print status")
    args = ap.parse_args()

    if args.stop:
        cmd_stop()
    elif args.status:
        cmd_status()
    else:
        run()
