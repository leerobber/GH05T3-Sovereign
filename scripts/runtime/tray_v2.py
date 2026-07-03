"""
GH05T3 Sovereign Stack - System Tray Host v2
Starts supervisor, shows live service status, opens dashboard.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import webbrowser
import urllib.request
from pathlib import Path

try:
    import pystray
    from PIL import Image, ImageDraw
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pystray", "pillow"])
    import pystray
    from PIL import Image, ImageDraw

ROOT           = Path(__file__).resolve().parents[2]
SUPERVISOR     = ROOT / "scripts" / "runtime" / "supervisor.py"
LOGS           = ROOT / "logs"

SYS_PY = Path(r"C:\Users\leer4\AppData\Local\Programs\Python\Python312\python.exe")
if not SYS_PY.exists():
    SYS_PY = Path(sys.executable)

DASH_URL         = "http://localhost:3210"
GENOME_URL       = "http://localhost:7720/genome_dashboard.html"
SOVEREIGN_URL    = "http://localhost:8000/docs"
SUPERVISOR_API   = "http://localhost:8090/status"
GATEWAY_HEALTH   = "http://localhost:8002/health"
BACKEND_HEALTH   = "http://localhost:8001/api/health"
SOVEREIGN_HEALTH = "http://localhost:8000/health"

_supervisor_proc = None
_status_cache    = {}
_status_lock     = threading.Lock()


def _make_icon(healthy=True):
    im = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d  = ImageDraw.Draw(im)
    color = (245, 158, 11, 255) if healthy else (220, 38, 38, 255)
    d.ellipse((4, 4, 60, 60),   fill=color)
    d.ellipse((10, 10, 54, 54), fill=(10, 10, 10, 255))
    d.ellipse((20, 22, 28, 30), fill=color)
    d.ellipse((36, 22, 44, 30), fill=color)
    d.arc((18, 32, 46, 52), 10, 170, fill=color, width=3)
    return im


def _get_json(url, timeout=3.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def _http_ok(url, timeout=2.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status < 500
    except Exception:
        return False


def _poll_status():
    while True:
        sup = _get_json(SUPERVISOR_API) or {}
        with _status_lock:
            _status_cache["supervisor"]   = sup
            _status_cache["gateway_ok"]   = _http_ok(GATEWAY_HEALTH)
            _status_cache["backend_ok"]   = _http_ok(BACKEND_HEALTH)
            _status_cache["sovereign_ok"] = _http_ok(SOVEREIGN_HEALTH)
            _status_cache["last_poll"]    = time.time()
        time.sleep(10)


def _all_ok():
    with _status_lock:
        return _status_cache.get("supervisor", {}).get("all_ok", False)


def _start_supervisor():
    global _supervisor_proc
    if _supervisor_proc and _supervisor_proc.poll() is None:
        return
    if _get_json(SUPERVISOR_API):
        return
    LOGS.mkdir(exist_ok=True)
    log_file = open(LOGS / "tray_supervisor.log", "a", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONIOENCODING"]    = "utf-8"
    env["AETHYRO_SKIP_LICENSE"] = "1"
    _supervisor_proc = subprocess.Popen(
        [str(SYS_PY), str(SUPERVISOR)],
        cwd=str(ROOT),
        stdout=log_file,
        stderr=log_file,
        env=env,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


def _stop_supervisor():
    global _supervisor_proc
    if _supervisor_proc and _supervisor_proc.poll() is None:
        _supervisor_proc.terminate()
        try:
            _supervisor_proc.wait(timeout=5)
        except Exception:
            _supervisor_proc.kill()
    subprocess.Popen(
        [str(SYS_PY), str(SUPERVISOR), "--stop"],
        cwd=str(ROOT),
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


def open_dashboard(*_):
    webbrowser.open(DASH_URL)

def open_genome(*_):
    webbrowser.open(GENOME_URL)

def open_sovereign(*_):
    webbrowser.open(SOVEREIGN_URL)

def open_logs(*_):
    os.startfile(str(LOGS))

def restart_stack(icon, *_):
    icon.notify("Restarting stack...", "GH05T3")
    def _do():
        _stop_supervisor()
        time.sleep(3)
        _start_supervisor()
        time.sleep(5)
        icon.notify("Stack restarted", "GH05T3")
    threading.Thread(target=_do, daemon=True).start()

def stop_stack(icon, *_):
    icon.notify("Stopping stack...", "GH05T3")
    _stop_supervisor()

def quit_app(icon, *_):
    icon.notify("Shutting down GH05T3...", "GH05T3")
    _stop_supervisor()
    time.sleep(2)
    icon.stop()


def _build_menu(icon):
    with _status_lock:
        svcs = _status_cache.get("supervisor", {}).get("services", {})

    important = [
        "backend", "gateway", "economy-api", "frontend",
        "sovereign-interface", "genome-lab-ui", "sage-engine",
        "payments", "serve", "phi",
    ]

    status_items = []
    for svc in important:
        info   = svcs.get(svc, {})
        status = info.get("status", "not started")
        if status == "running":
            dot = "[OK]"
        elif status in ("stopped", "degraded"):
            dot = "[--]"
        else:
            dot = "[ ?]"
        restarts = info.get("restarts", 0)
        label = "  {}  {:<22} {}".format(dot, svc, status)
        if restarts > 5:
            label += "  (restarts:{})".format(restarts)
        status_items.append(pystray.MenuItem(label, None, enabled=False))

    return pystray.Menu(
        pystray.MenuItem("Open Dashboard  (port 3210)",   open_dashboard, default=True),
        pystray.MenuItem("Genome Lab  (port 7720)",       open_genome),
        pystray.MenuItem("Sovereign Core  (port 8000)",  open_sovereign),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("--- Service Status ---",        None, enabled=False),
        *status_items,
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Restart Full Stack",  restart_stack),
        pystray.MenuItem("Stop All Services",   stop_stack),
        pystray.MenuItem("Open Logs Folder",    open_logs),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit GH05T3",         quit_app),
    )


def main():
    threading.Thread(target=_start_supervisor, daemon=True).start()
    threading.Thread(target=_poll_status,      daemon=True).start()

    def _delayed_open():
        for _ in range(30):
            time.sleep(2)
            if _http_ok(DASH_URL) or _http_ok(GENOME_URL):
                webbrowser.open(GENOME_URL)
                return
        webbrowser.open(DASH_URL)

    threading.Thread(target=_delayed_open, daemon=True).start()

    icon = pystray.Icon(
        "gh05t3",
        _make_icon(True),
        "GH05T3 Sovereign Stack",
        menu=pystray.Menu(pystray.MenuItem("Loading...", None, enabled=False)),
    )

    def _refresh():
        while True:
            time.sleep(15)
            try:
                ok = _all_ok()
                icon.icon  = _make_icon(ok)
                icon.menu  = _build_menu(icon)
                icon.title = "GH05T3 - All OK" if ok else "GH05T3 - DEGRADED"
            except Exception:
                pass

    threading.Thread(target=_refresh, daemon=True).start()
    time.sleep(2)
    icon.menu = _build_menu(icon)
    icon.run()


if __name__ == "__main__":
    main()
