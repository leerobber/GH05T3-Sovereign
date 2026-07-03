"""
cmd_listener.py â€” Sovereign #gh05t3-cmd Slack command listener.

Polls #gh05t3-cmd every 5s for new messages. Executes commands and replies.

Commands:
  !status          â€” economy + training status
  !herald          â€” send full training briefing
  !amplify         â€” start amplifier if not running
  !train           â€” launch RunPod training
  !spin            â€” show SPIN dataset stats
  !cycles          â€” show KAIROS cycle counts per domain
  !help            â€” list commands

Run: python cmd_listener.py
"""
import json, os, subprocess, sys, time
from pathlib import Path

BASE = Path(__file__).parent
os.chdir(BASE)

# â”€â”€ Load env â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _load_env():
    env_path = BASE / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))

_load_env()

import requests
import slack_notify as _slack

POLL_SEC     = 5
POLL_MAX_SEC = 120   # cap backoff at 2 minutes
CMD_CHANNEL  = "gh05t3-cmd"
SEEN_FILE    = BASE / "data" / "cmd_seen.json"
PYTHON       = sys.executable


def _load_seen() -> set:
    if SEEN_FILE.exists():
        try:
            return set(json.loads(SEEN_FILE.read_text(encoding="utf-8")))
        except Exception:
            pass
    return set()


def _save_seen(seen: set):
    SEEN_FILE.parent.mkdir(exist_ok=True)
    SEEN_FILE.write_text(json.dumps(list(seen)[-500:]), encoding="utf-8")


def _get_channel_id(name: str) -> str | None:
    return None  # Slack API disabled — configure SLACK_BOT_TOKEN to enable


def _get_messages(channel_id: str, oldest: str = "") -> list:
    token = os.environ.get("SLACK_BOT_TOKEN", "")
    params = {"channel": channel_id, "limit": 10}
    if oldest:
        params["oldest"] = oldest
    try:
        r = requests.get(
            "https://slack.com/api/conversations.history",
            headers={"Authorization": f"Bearer {token}"},
            params=params,
            timeout=10,
        )
        return r.json().get("messages", [])
    except requests.exceptions.ConnectionError:
        raise   # let main loop's backoff handle it
    except requests.exceptions.Timeout:
        raise   # same
    except Exception:
        return []


def _reply(text: str):
    _slack.post(CMD_CHANNEL, text)


# â”€â”€ Command handlers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def cmd_status() -> str:
    lines = ["*STATUS*"]
    # Economy
    try:
        r = requests.get("http://localhost:8081/", timeout=3)
        eco = r.json()
        lines.append(f"  Economy API: :white_check_mark: {eco.get('name','?')} v{eco.get('version','?')}")
    except Exception:
        lines.append("  Economy API: :x: offline")

    # Continuous learner running?
    import subprocess as sp
    out = sp.run(["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV"],
                 capture_output=True, text=True).stdout
    learner_running = "continuous_learner" in " ".join(
        sp.run(["wmic", "process", "where", "name='python.exe'",
                "get", "commandline", "/format:csv"],
               capture_output=True, text=True).stdout.split()
    )
    lines.append(f"  Continuous learner: {'running' if learner_running else 'stopped'}")

    # SPIN count
    spin_file = BASE / "data" / "spin_dataset.jsonl"
    spin = sum(1 for _ in spin_file.open(encoding="utf-8")) if spin_file.exists() else 0
    lines.append(f"  SPIN pairs: {spin}")

    # Continuous state
    state_file = BASE / "data" / "continuous_state.json"
    if state_file.exists():
        s = json.loads(state_file.read_text(encoding="utf-8"))
        lines.append(f"  KAIROS cycles: {s.get('total_cycles',0)}")
        domains = ["business","sales","product_strategy","growth","cfo","content","legal_ip","ops","ml_engineer","frontier","core"]
        active = domains[s.get('domain_index',0) % len(domains)]
        lines.append(f"  Active domain: {active} (cycle {s.get('domain_cycles',0)})")

    return "\n".join(lines)


def cmd_herald() -> str:
    result = subprocess.run(
        [PYTHON, str(BASE / "herald.py"), "--console"],
        capture_output=True, text=True, cwd=str(BASE), timeout=10,
    )
    return result.stdout[:2000] if result.stdout else "Herald failed."


def cmd_spin() -> str:
    spin_file = BASE / "data" / "spin_dataset.jsonl"
    if not spin_file.exists():
        return "No SPIN data yet."
    rows = [json.loads(l) for l in spin_file.open(encoding="utf-8") if l.strip()]
    domains: dict[str, int] = {}
    amplified = 0
    for r in rows:
        d = r.get("domain", "?")
        domains[d] = domains.get(d, 0) + 1
        if r.get("source") == "amplified":
            amplified += 1
    lines = [f"*SPIN DATASET* ({len(rows)} total | {amplified} amplified)"]
    for d, n in sorted(domains.items(), key=lambda x: -x[1])[:8]:
        lines.append(f"  {d}: {n}")
    return "\n".join(lines)


def cmd_cycles() -> str:
    lines = ["*KAIROS CYCLES*"]
    for ckpt in sorted((BASE / "data").glob("ckpt_*.json")):
        try:
            d = json.loads(ckpt.read_text(encoding="utf-8"))
            domain = ckpt.stem.replace("ckpt_", "")
            lines.append(f"  {domain}: {d.get('cycle',0)} cycles  "
                         f"baseline={d.get('baseline',0):.3f}  "
                         f"passes={d.get('passes',0)}")
        except Exception:
            pass
    return "\n".join(lines)


def cmd_amplify() -> str:
    # Check if already running
    import subprocess as sp
    procs = sp.run(
        ["wmic", "process", "where", "name='python.exe'", "get", "commandline", "/format:csv"],
        capture_output=True, text=True,
    ).stdout
    if "amplifier.py" in procs:
        spin_file = BASE / "data" / "spin_dataset.jsonl"
        current = sum(1 for _ in spin_file.open(encoding="utf-8")) if spin_file.exists() else 0
        return f"Amplifier already running. Current SPIN total: {current}"
    subprocess.Popen(
        [PYTHON, str(BASE / "scripts" / "training" / "amplifier.py"), "--variants", "5"],  # canonical path
        cwd=str(BASE),
        stdout=open(BASE / "logs" / "amplifier.log", "a"),
        stderr=open(BASE / "logs" / "amplifier_err.log", "a"),
    )
    return "Amplifier started (--variants 5). Watch #continuous-learner for upload notifications."


def cmd_train() -> str:
    subprocess.Popen(
        [PYTHON, str(BASE / "runpod_launcher.py"), "--mode", "sft"],
        cwd=str(BASE),
        stdout=open(BASE / "logs" / "train_cmd.log", "a"),
        stderr=open(BASE / "logs" / "train_cmd_err.log", "a"),
    )
    return "RunPod training launched (SFT mode). Watch #avery-training for progress."


COMMANDS = {
    "!status":   cmd_status,
    "!herald":   cmd_herald,
    "!amplify":  cmd_amplify,
    "!train":    cmd_train,
    "!spin":     cmd_spin,
    "!cycles":   cmd_cycles,
    "!help":     lambda: (
        "*Available commands:*\n"
        "  `!status`   â€” economy + training status\n"
        "  `!herald`   â€” full training briefing\n"
        "  `!amplify`  â€” start SPIN amplifier\n"
        "  `!train`    â€” launch RunPod training\n"
        "  `!spin`     â€” SPIN dataset stats\n"
        "  `!cycles`   â€” KAIROS cycle counts\n"
        "  `!help`     â€” this message"
    ),
}


# â”€â”€ Main loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main():
    Path("logs").mkdir(exist_ok=True)
    seen = _load_seen()

    channel_id = _get_channel_id(CMD_CHANNEL)
    if not channel_id:
        print(f"ERROR: Could not find #{CMD_CHANNEL} channel.")
        sys.exit(1)

    print(f"GH05T3 CMD Listener â€” polling #{CMD_CHANNEL} every {POLL_SEC}s")
    print(f"Channel ID: {channel_id}")
    _reply(":robot_face: *GH05T3 command listener online.* Type `!help` to see commands.")

    oldest = str(time.time())
    backoff = POLL_SEC   # current sleep duration; grows on network failure

    while True:
        try:
            messages = _get_messages(channel_id, oldest=oldest)
            # Success â€” reset backoff
            backoff = POLL_SEC

            for msg in reversed(messages):
                ts  = msg.get("ts", "")
                txt = msg.get("text", "").strip()

                if ts in seen or not txt:
                    continue
                if msg.get("subtype"):   # joins, leaves, etc.
                    seen.add(ts)
                    continue

                seen.add(ts)
                oldest = ts

                # find command
                cmd_key = txt.split()[0].lower() if txt else ""
                handler = COMMANDS.get(cmd_key)
                if handler:
                    print(f"CMD: {txt}")
                    try:
                        result = handler()
                    except Exception as e:
                        result = f":x: Error: {e}"
                    _reply(result)

            _save_seen(seen)

        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            # Network failure â€” exponential backoff, suppress repeated stack traces
            err_short = str(e)[:120]
            print(f"Poll error (network, retrying in {backoff}s): {err_short}")
            time.sleep(backoff)
            backoff = min(backoff * 2, POLL_MAX_SEC)
            continue   # skip the normal sleep below

        except Exception as e:
            print(f"Poll error: {e}")

        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
