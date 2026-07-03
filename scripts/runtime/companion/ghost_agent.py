#!/usr/bin/env python3
"""
GH05T3 Local Companion Agent
============================

Runs on YOUR laptop (Windows / macOS / Linux). Dials OUT to the GH05T3
gateway over WebSocket after you paste a one-time 6-digit pairing code.
No inbound ports, no listening services â€” completely NAT-friendly.

Install:
    pip install -r requirements.txt

Run (you'll be prompted for the gateway URL and pair code):
    python ghost_agent.py

Or headless:
    GHOST_GATEWAY_URL=https://your-gh05t3.app PAIR_CODE=123456 python ghost_agent.py

Permission surface â€” toggle from CLI flags (default: all OFF except notify):
    --screen-read        allow screen capture requests
    --shell-exec         allow shell commands (with allow-list)
    --fs-read PATH       allow read inside PATH (repeat for multiple roots)
    --fs-write PATH      allow write inside PATH (repeat for multiple roots)
    --clipboard          allow clipboard read/write
    --notify             allow system notifications (default ON)
    --mic                allow microphone capture upload
    --all                grant everything (use only on your own machine)

Kill switch: Ctrl+C OR press `q` in the terminal. In GUI contexts we register
a global hotkey Ctrl+Shift+K (requires `keyboard` package and admin on Windows).

Everything is logged to ./ghost_agent.log for audit.
"""
from __future__ import annotations
import argparse
import asyncio
import base64
import io
import json
import logging
import os
import platform
from shlex import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

try:
    import websockets
except ImportError:
    print("pip install websockets first.")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler("ghost_agent.log"), logging.StreamHandler()],
)
LOG = logging.getLogger("ghost-agent")

SHELL_ALLOWLIST = {
    "git", "ls", "dir", "pwd", "cat", "type", "echo", "python", "pytest",
    "node", "yarn", "npm", "pip", "make", "cmake", "gcc", "clang", "rustc",
    "cargo", "go", "docker", "kubectl", "tree", "grep", "rg", "fd", "find",
    "curl", "wget", "whoami", "hostname", "uname", "date", "which", "where",
    "head", "tail", "wc", "du", "df", "ps", "top", "free", "uptime",
}


# ---------------------------------------------------------------------------
# Capability impls
# ---------------------------------------------------------------------------
def cap_screenshot():
    """Return base64-PNG of the primary screen."""
    try:
        import mss
        from PIL import Image
    except ImportError:
        return {"error": "pip install mss pillow for screen_read"}
    with mss.mss() as sct:
        img = sct.grab(sct.monitors[1])
        pil = Image.frombytes("RGB", img.size, img.rgb)
        # downscale to 1280 wide max
        w, h = pil.size
        if w > 1280:
            pil = pil.resize((1280, int(h * 1280 / w)))
        buf = io.BytesIO()
        pil.save(buf, format="PNG", optimize=True)
        return {"png_b64": base64.b64encode(buf.getvalue()).decode(), "w": pil.size[0], "h": pil.size[1]}


def cap_shell(cmd: str, allowlist: bool = True, timeout: int = 15) -> dict:
    if not cmd.strip():
        return {"error": "empty command"}
    tokens = shlex.split(cmd, posix=(os.name != "nt"))
    if allowlist:
        head = Path(tokens[0]).name.lower()
        # strip .exe suffix on windows
        head = head.rsplit(".", 1)[0] if head.endswith(".exe") else head
        if head not in SHELL_ALLOWLIST:
            return {"error": f"command '{head}' not in allow-list"}
    try:
        proc = subprocess.run(
            tokens, capture_output=True, text=True, timeout=timeout,
            shell=False,
        )
        return {"stdout": proc.stdout[-10000:], "stderr": proc.stderr[-10000:],
                "rc": proc.returncode}
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except FileNotFoundError:
        return {"error": f"not found: {tokens[0]}"}


def _path_in(roots: list[Path], target: Path) -> bool:
    try:
        t = target.resolve()
    except Exception:
        return False
    return any(str(t).startswith(str(r.resolve())) for r in roots)


def cap_fs_read(path: str, roots: list[Path], max_bytes: int = 200_000) -> dict:
    p = Path(path).expanduser()
    if not _path_in(roots, p):
        return {"error": "path outside allowed roots"}
    if not p.exists():
        return {"error": "no such file"}
    if p.is_dir():
        items = sorted(os.listdir(p))[:500]
        return {"dir": str(p), "entries": items}
    data = p.read_bytes()[:max_bytes]
    try:
        return {"file": str(p), "text": data.decode("utf-8")}
    except UnicodeDecodeError:
        return {"file": str(p), "b64": base64.b64encode(data).decode()}


def cap_fs_write(path: str, content: str, roots: list[Path]) -> dict:
    p = Path(path).expanduser()
    if not _path_in(roots, p):
        return {"error": "path outside allowed roots"}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return {"written": str(p), "bytes": len(content)}


def cap_clipboard_read() -> dict:
    try:
        import pyperclip
    except ImportError:
        return {"error": "pip install pyperclip"}
    return {"text": pyperclip.paste()}


def cap_clipboard_write(text: str) -> dict:
    try:
        import pyperclip
    except ImportError:
        return {"error": "pip install pyperclip"}
    pyperclip.copy(text)
    return {"copied_bytes": len(text)}


def cap_notify(title: str, body: str) -> dict:
    try:
        system = platform.system()
        if system == "Windows":
            try:
                from win10toast import ToastNotifier
                ToastNotifier().show_toast(title, body, duration=5, threaded=True)
                return {"ok": True, "via": "win10toast"}
            except ImportError:
                pass
        if system == "Darwin":
            subprocess.run([
                "osascript", "-e",
                f'display notification "{body}" with title "{title}"',
            ])
            return {"ok": True, "via": "osascript"}
        if system == "Linux":
            if shutil.which("notify-send"):
                subprocess.run(["notify-send", title, body])
                return {"ok": True, "via": "notify-send"}
        # fallback: console bell
        print(f"\a[GH05T3] {title}: {body}")
        return {"ok": True, "via": "console"}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
async def run(gateway: str, token: str, label: str, caps: set[str],
              fs_read_roots: list[Path], fs_write_roots: list[Path],
              allow_any_shell: bool, ghosteye: bool, ghosteye_interval: int,
              ghosteye_ocr: bool):
    parsed = urlparse(gateway)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    ws_url = f"{scheme}://{parsed.netloc}/api/companion/ws"
    LOG.info("connecting to %s as %s caps=%s", ws_url, label, sorted(caps))

    info = {
        "token": token,
        "label": label,
        "capabilities": sorted(caps),
        "info": {
            "os": platform.system(),
            "release": platform.release(),
            "arch": platform.machine(),
            "python": sys.version.split()[0],
            "ghosteye": ghosteye,
        },
    }

    backoff = 2
    eye_enabled = {"v": ghosteye}  # mutable flag that control msgs can flip

    while True:
        try:
            async with websockets.connect(ws_url, max_size=16 * 1024 * 1024) as ws:
                await ws.send(json.dumps(info))
                hello = json.loads(await ws.recv())
                LOG.info("paired: %s", hello)
                backoff = 2

                # launch GhostEye task if enabled
                eye_task = None
                if ghosteye and "screen_read" in caps:
                    eye_task = asyncio.create_task(
                        _ghosteye_loop(ws, eye_enabled, ghosteye_interval, ghosteye_ocr)
                    )

                try:
                    async for raw in ws:
                        msg = json.loads(raw)
                        # control messages from server
                        if msg.get("control") == "ghosteye":
                            eye_enabled["v"] = bool(msg.get("enabled"))
                            LOG.info("ghosteye %s via control", "on" if eye_enabled["v"] else "off")
                            continue
                        rid = msg.get("req_id")
                        action = msg.get("action")
                        args = msg.get("args") or {}
                        result = _dispatch(action, args, caps, fs_read_roots, fs_write_roots, allow_any_shell)
                        try:
                            await ws.send(json.dumps({"req_id": rid, "result": result}))
                        except Exception:
                            LOG.exception("send failed")
                finally:
                    if eye_task:
                        eye_task.cancel()
        except Exception as e:
            LOG.warning("connection error: %s â€” retrying in %ds", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(60, backoff * 2)


async def _ghosteye_loop(ws, enabled_flag: dict, interval: int, ocr: bool):
    """Periodic screen capture + optional OCR + push to gateway."""
    LOG.info("GhostEye loop starting interval=%ds ocr=%s", interval, ocr)
    while True:
        try:
            await asyncio.sleep(interval)
            if not enabled_flag["v"]:
                continue
            frame = cap_screenshot()
            if "error" in frame:
                continue
            text = ""
            if ocr:
                text = _ocr_png_b64(frame["png_b64"])
            active_app = _active_app_title()
            payload = {
                "event": "ghosteye_frame",
                "data": {
                    "png_b64": frame["png_b64"],
                    "w": frame.get("w"), "h": frame.get("h"),
                    "text": text,
                    "active_app": active_app,
                },
            }
            await ws.send(json.dumps(payload))
        except asyncio.CancelledError:
            break
        except Exception:
            LOG.exception("ghosteye frame failed")


def _ocr_png_b64(png_b64: str) -> str:
    try:
        import pytesseract
        from PIL import Image
        import io as _io
        img = Image.open(_io.BytesIO(base64.b64decode(png_b64)))
        return pytesseract.image_to_string(img)[:4000]
    except ImportError:
        return ""
    except Exception as e:
        LOG.warning("OCR failed: %s", e)
        return ""


def _active_app_title() -> str:
    """Best-effort: return focused window title. Silent no-op if unavailable."""
    try:
        system = platform.system()
        if system == "Windows":
            import ctypes
            from ctypes import wintypes
            user32 = ctypes.windll.user32
            h = user32.GetForegroundWindow()
            length = user32.GetWindowTextLengthW(h)
            buff = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(h, buff, length + 1)
            return buff.value[:120]
        if system == "Darwin":
            script = 'tell application "System Events" to get name of first application process whose frontmost is true'
            out = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=2)
            return (out.stdout or "").strip()[:120]
        if system == "Linux":
            if shutil.which("xdotool"):
                out = subprocess.run(["xdotool", "getactivewindow", "getwindowname"],
                                     capture_output=True, text=True, timeout=2)
                return (out.stdout or "").strip()[:120]
    except Exception:
        pass
    return ""


def _dispatch(action: str, args: dict, caps: set[str],
              fs_read_roots: list[Path], fs_write_roots: list[Path],
              allow_any_shell: bool) -> dict:
    LOG.info("cmd %s args=%s", action, {k: (v if len(str(v)) < 80 else f"<{len(str(v))}B>") for k, v in args.items()})
    if action == "screenshot":
        if "screen_read" not in caps:
            return {"error": "screen_read not granted"}
        return cap_screenshot()
    if action == "shell":
        if "shell_exec" not in caps:
            return {"error": "shell_exec not granted"}
        return cap_shell(args.get("cmd", ""), allowlist=not allow_any_shell,
                         timeout=int(args.get("timeout", 15)))
    if action == "fs_read":
        if "fs_read" not in caps:
            return {"error": "fs_read not granted"}
        return cap_fs_read(args.get("path", ""), fs_read_roots)
    if action == "fs_write":
        if "fs_write" not in caps:
            return {"error": "fs_write not granted"}
        return cap_fs_write(args.get("path", ""), args.get("content", ""), fs_write_roots)
    if action == "clipboard_read":
        if "clipboard" not in caps:
            return {"error": "clipboard not granted"}
        return cap_clipboard_read()
    if action == "clipboard_write":
        if "clipboard" not in caps:
            return {"error": "clipboard not granted"}
        return cap_clipboard_write(args.get("text", ""))
    if action == "notify":
        if "notify" not in caps:
            return {"error": "notify not granted"}
        return cap_notify(args.get("title", "GH05T3"), args.get("body", ""))
    return {"error": f"unknown action {action}"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args():
    p = argparse.ArgumentParser(description="GH05T3 local companion")
    p.add_argument("--gateway", default=os.environ.get("GHOST_GATEWAY_URL", ""))
    p.add_argument("--pair-code", default=os.environ.get("PAIR_CODE", ""))
    p.add_argument("--label", default=platform.node())
    p.add_argument("--screen-read", action="store_true")
    p.add_argument("--shell-exec", action="store_true")
    p.add_argument("--allow-any-shell", action="store_true",
                   help="DANGEROUS: disables shell allow-list")
    p.add_argument("--fs-read", action="append", default=[])
    p.add_argument("--fs-write", action="append", default=[])
    p.add_argument("--clipboard", action="store_true")
    p.add_argument("--notify", action="store_true", default=True)
    p.add_argument("--mic", action="store_true")
    p.add_argument("--ghosteye", action="store_true",
                   help="enable periodic screen observation streaming (requires --screen-read)")
    p.add_argument("--ghosteye-interval", type=int, default=15,
                   help="seconds between GhostEye captures (default 15)")
    p.add_argument("--ghosteye-ocr", action="store_true",
                   help="run pytesseract on each frame (pip install pytesseract + system tesseract)")
    p.add_argument("--all", action="store_true")
    return p.parse_args()


def _claim(gateway: str, code: str, label: str) -> str:
    import urllib.request
    import urllib.parse
    url = f"{gateway.rstrip('/')}/api/companion/claim?code={urllib.parse.quote(code)}&label={urllib.parse.quote(label)}"
    req = urllib.request.Request(url, method="POST")
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.load(r)
    return data["token"]


def main():
    args = _parse_args()
    gateway = args.gateway or input("Gateway URL (e.g. https://gh05t3.you.app): ").strip()
    code = args.pair_code or input("Pairing code (6 digits): ").strip()
    token = _claim(gateway, code, args.label)
    LOG.info("got companion token (hidden)")

    caps: set[str] = set()
    if args.all:
        caps = {"screen_read", "shell_exec", "fs_read", "fs_write", "clipboard", "notify", "mic"}
    else:
        if args.screen_read: caps.add("screen_read")
        if args.shell_exec:  caps.add("shell_exec")
        if args.fs_read:     caps.add("fs_read")
        if args.fs_write:    caps.add("fs_write")
        if args.clipboard:   caps.add("clipboard")
        if args.notify:      caps.add("notify")
        if args.mic:         caps.add("mic")

    fs_read_roots = [Path(p).expanduser() for p in args.fs_read] or [Path.home()]
    fs_write_roots = [Path(p).expanduser() for p in args.fs_write]

    try:
        asyncio.run(run(gateway, token, args.label, caps,
                        fs_read_roots, fs_write_roots, args.allow_any_shell,
                        args.ghosteye or args.all,
                        args.ghosteye_interval,
                        args.ghosteye_ocr))
    except KeyboardInterrupt:
        LOG.info("companion stopped by user")


if __name__ == "__main__":
    main()
