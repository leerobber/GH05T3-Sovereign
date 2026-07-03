"""GH05T3 tray icon — background host.

Runs in the system tray. Menu:
    - Open Dashboard       → http://localhost:3210
    - Toggle GhostEye
    - Toggle Voice         → starts/stops the Hey-GH05T3 listener
    - Pause Ghost          → scheduler off, chat offline
    - Quit                 → graceful shutdown of backend + frontend + mongo
"""
from __future__ import annotations
import os
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

try:
    import pystray
    from PIL import Image, ImageDraw
except ImportError:
    print("pip install pystray pillow  (auto-installed by install.ps1)")
    sys.exit(1)

try:
    import httpx
except ImportError:
    import urllib.request as _ur
    httpx = None

ROOT = Path(__file__).parent
DASH = "http://localhost:3210"
API = "http://localhost:8001/api"

voice_proc: subprocess.Popen | None = None
sched_on = True


def _icon() -> Image.Image:
    im = Image.new("RGBA", (64, 64), (5, 5, 5, 0))
    d = ImageDraw.Draw(im)
    d.ellipse((6, 6, 58, 58), fill=(245, 158, 11, 255))
    d.ellipse((22, 22, 30, 30), fill=(5, 5, 5, 255))
    d.ellipse((38, 22, 46, 30), fill=(5, 5, 5, 255))
    d.arc((18, 34, 46, 54), 0, 180, fill=(5, 5, 5, 255), width=3)
    return im


def _post(path: str, params: dict | None = None, body: dict | None = None) -> dict | None:
    try:
        if httpx:
            with httpx.Client(timeout=10) as c:
                r = c.post(f"{API}{path}", params=params, json=body)
                return r.json() if r.headers.get("content-type", "").startswith("application/json") else {"text": r.text}
        else:
            import json, urllib.parse, urllib.request
            url = f"{API}{path}"
            if params:
                url += "?" + urllib.parse.urlencode(params)
            data = json.dumps(body).encode() if body else b""
            req = urllib.request.Request(url, data=data, method="POST",
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                return json.loads(r.read().decode())
    except Exception as e:
        return {"error": str(e)}


def open_dashboard(_=None, __=None):
    webbrowser.open(DASH)


def toggle_scheduler(icon, item):
    global sched_on
    sched_on = not sched_on
    _post("/scheduler/toggle", params={"enable": sched_on})
    icon.notify(f"Scheduler {'RESUMED' if sched_on else 'PAUSED'}", "GH05T3")


def toggle_voice(icon, item):
    global voice_proc
    if voice_proc and voice_proc.poll() is None:
        voice_proc.terminate()
        voice_proc = None
        icon.notify("Voice loop stopped", "GH05T3")
        return
    voice_script = ROOT / "voice.py"
    py = ROOT / "backend" / ".venv" / "Scripts" / "python.exe"
    if not py.exists():
        py = sys.executable
    voice_proc = subprocess.Popen([str(py), str(voice_script)], cwd=str(ROOT))
    icon.notify("Voice loop started — say 'hey GH05T3'", "GH05T3")


def toggle_ghosteye(icon, item):
    state = _post("/companion/ghosteye/toggle", params={"enabled": not item.checked})
    icon.notify(f"GhostEye {'ON' if state and state.get('enabled') else 'OFF'}", "GH05T3")


def quit_app(icon, item):
    if voice_proc and voice_proc.poll() is None:
        voice_proc.terminate()
    # kill mongo + backend + frontend companion processes (best-effort)
    os.system('taskkill /FI "WINDOWTITLE eq gh05t3-backend*" /F >nul 2>&1')
    os.system('taskkill /FI "WINDOWTITLE eq gh05t3-frontend*" /F >nul 2>&1')
    os.system('taskkill /FI "WINDOWTITLE eq gh05t3-mongo*" /F >nul 2>&1')
    icon.stop()


def main():
    # Wait for backend to come up, then open dashboard
    def _bring_up():
        for _ in range(30):
            time.sleep(1)
            try:
                if httpx:
                    httpx.get(f"{API}/state", timeout=2)
                else:
                    import urllib.request as _ur
                    _ur.urlopen(f"{API}/state", timeout=2)
                break
            except Exception:
                continue
        open_dashboard()
    threading.Thread(target=_bring_up, daemon=True).start()

    menu = pystray.Menu(
        pystray.MenuItem("Open Dashboard", open_dashboard, default=True),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Hey GH05T3 (voice)", toggle_voice),
        pystray.MenuItem("GhostEye", toggle_ghosteye, checked=lambda _i: True),
        pystray.MenuItem("Pause Ghost (scheduler)", toggle_scheduler),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit GH05T3", quit_app),
    )
    icon = pystray.Icon("gh05t3", _icon(), "GH05T3 · The Ghost", menu)
    icon.run()


if __name__ == "__main__":
    main()
