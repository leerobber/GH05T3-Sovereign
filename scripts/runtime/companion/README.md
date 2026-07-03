# GH05T3 Companion — install & run

You're on a **LOQ Lenovo 15** (Windows · Ryzen 7 · NVIDIA GeForce + AMD).
This agent turns that laptop into GH05T3's hands and eyes, opt-in only,
NAT-friendly, no inbound ports.

## One-time install (Windows PowerShell)

```powershell
# 1. Clone or copy the /app/companion folder to your laptop:
#    Download ghost_agent.py + requirements.txt anywhere, e.g. C:\ghost\

# 2. Make a virtualenv (optional but recommended):
python -m venv ghost-venv
.\ghost-venv\Scripts\Activate.ps1

# 3. Install deps:
pip install -r requirements.txt
```

## Get a pairing code

Open the GH05T3 dashboard → **Companion** panel → **"Pair new device"**.
You'll get a 6-digit code (expires in 10 min).

## Run with the permissions YOU want

### Minimal (notifications only, totally safe):
```powershell
python ghost_agent.py --gateway https://your-gh05t3.app --pair-code 123456
```

### Everything (her eyes, hands, voice — use on your own machine only):
```powershell
python ghost_agent.py --gateway https://your-gh05t3.app --pair-code 123456 `
    --all --fs-read C:\Users\You\projects --fs-write C:\Users\You\ghost-inbox
```

### Recommended daily driver for you (LOQ):
```powershell
python ghost_agent.py `
    --gateway https://your-gh05t3.app `
    --pair-code 123456 `
    --screen-read `
    --shell-exec `
    --clipboard `
    --notify `
    --fs-read C:\Users\You\projects `
    --fs-write C:\Users\You\ghost-inbox
```

## Permission flags

| Flag | Grants | Risk |
|------|--------|------|
| `--screen-read` | Dashboard can request screenshots of your primary monitor | Medium (screens may contain secrets) |
| `--shell-exec` | Run commands from the shell allow-list (git, python, pytest, …) | Medium |
| `--allow-any-shell` | Disables the allow-list — **any** command | **HIGH — use with caution** |
| `--fs-read PATH` | Read files inside PATH (repeatable) | Low–Medium |
| `--fs-write PATH` | Write files inside PATH (repeatable) | Medium |
| `--clipboard` | Read & write your OS clipboard | Low |
| `--notify` | System toast notifications (default ON) | None |
| `--mic` | Capture microphone and upload short clips | Medium |
| `--all` | Everything above at once | Max |

## Kill switch

* `Ctrl+C` in the terminal stops the agent instantly.
* Revoke from the dashboard: **Companion → Revoke** (severs the WebSocket).
* Agent also honors `DEEP_FREEZE` broadcast if the gateway ever sends it.

## Logs

All commands, arguments, and responses are written to `./ghost_agent.log`
in the working directory. Useful for audit / replay.

## Android

For your Android phone use the **mobile PWA** served by the GH05T3 dashboard:
open the site in Chrome/Edge, hit **"Add to Home Screen"**, then use the
dashboard normally. Voice-in uses your phone's built-in SpeechRecognition.
Full screen capture on Android requires a native app (not yet shipped).

## TatorTot integration

When you stand up the Ollama gateway on this same machine (ports 8001/8002/8003),
set `OLLAMA_GATEWAY_URL=http://localhost:8000` in the dashboard's deploy env
so nightly KAIROS cycles route through your local GPUs instead of the cloud.
The companion agent and the Ollama gateway are independent — run either or both.
