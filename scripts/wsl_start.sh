#!/usr/bin/env bash
# Start GH05T3 backend services in WSL
# Usage: bash /mnt/c/Users/leer4/GH05T3/scripts/wsl_start.sh [--gateway-only|--backend-only]
set -uo pipefail

GH05T3="/mnt/c/Users/leer4/GH05T3"
VENV_DIR="$HOME/gh05t3-wsl"
ENV_FILE="$HOME/.gh05t3-wsl.env"
LOG_DIR="/tmp/gh05t3-wsl"
PID_DIR="/tmp/gh05t3-wsl-pids"

mkdir -p "$LOG_DIR" "$PID_DIR"

# Load environment
if [[ -f "$ENV_FILE" ]]; then
    # shellcheck source=/dev/null
    source "$ENV_FILE"
else
    echo "ERROR: ~/.gh05t3-wsl.env not found. Run wsl_setup.sh first."
    exit 1
fi

# Activate venv
source "$VENV_DIR/bin/activate"

ONLY="${1:-all}"

# ── Refresh Windows host IP (it can change on WSL restart) ──────────────────
WIN_HOST_IP=$(ip route show default 2>/dev/null | awk '/default/ {print $3; exit}')
export WIN_HOST_IP="${WIN_HOST_IP:-172.30.96.1}"

# Prefer local Ollama in WSL; fall back to Windows host
OLLAMA_URL=""
for CANDIDATE in "http://127.0.0.1:11434" "http://localhost:11434" "http://${WIN_HOST_IP}:11434"; do
    if curl -sf --max-time 2 "${CANDIDATE}/api/tags" > /dev/null 2>&1; then
        OLLAMA_URL="$CANDIDATE"
        break
    fi
done
OLLAMA_URL="${OLLAMA_URL:-http://${WIN_HOST_IP}:11434}"
export OLLAMA_HOST="$OLLAMA_URL"
export OLLAMA_BASE_URL="$OLLAMA_URL"
export OLLAMA_GATEWAY_URL="$OLLAMA_URL"

# MongoDB runs on Windows; WSL reaches it via the host gateway IP
export MONGO_URL="mongodb://${WIN_HOST_IP}:27017"

echo ""
echo "  ╔══════════════════════════════════════════════════════════╗"
echo "  ║       GH05T3 WSL Launcher                               ║"
echo "  ╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  Windows host: $WIN_HOST_IP"
echo "  Ollama:       $OLLAMA_HOST"
echo "  MongoDB:      $MONGO_URL"
echo "  Mode:         $ONLY"
echo ""

# ── Ensure Windows MongoDB is reachable from WSL ─────────────────────────────
_ensure_windows_mongo() {
    if timeout 2 bash -c "echo > /dev/tcp/${WIN_HOST_IP}/27017" 2>/dev/null; then
        echo "  [ok] MongoDB reachable at $MONGO_URL"
        return 0
    fi
    echo "  [mongo] Starting Windows MongoDB (bind 0.0.0.0 for WSL)..."
    powershell.exe -NoProfile -Command "
        Stop-Process -Name mongod -Force -ErrorAction SilentlyContinue
        Start-Sleep 1
        New-NetFirewallRule -DisplayName 'MongoDB WSL' -Direction Inbound -LocalPort 27017 -Protocol TCP -Action Allow -ErrorAction SilentlyContinue | Out-Null
        Start-Process -FilePath 'C:\Program Files\MongoDB\Server\8.2\bin\mongod.exe' -ArgumentList '--dbpath','C:\Users\leer4\GH05T3\mongo-data','--bind_ip','0.0.0.0','--port','27017','--quiet' -WindowStyle Hidden
    " >/dev/null 2>&1 || true
    for _ in {1..10}; do
        if timeout 2 bash -c "echo > /dev/tcp/${WIN_HOST_IP}/27017" 2>/dev/null; then
            echo "  [ok] MongoDB reachable at $MONGO_URL"
            return 0
        fi
        sleep 1
    done
    echo "  [warn] MongoDB not reachable — backend will start in degraded mode"
    return 1
}
_ensure_windows_mongo
echo ""

# ── Conflict check: Windows supervisor on same ports ───────────────────────────
WIN_HOST_IP="${WIN_HOST_IP:-$(ip route show default 2>/dev/null | awk '/default/ {print $3; exit}')}"
if timeout 1 bash -c "echo > /dev/tcp/${WIN_HOST_IP}/8090" 2>/dev/null; then
    echo "  [warn] Windows supervisor may be on :8090 — stop it before WSL stack:"
    echo "         python /mnt/c/Users/leer4/GH05T3/scripts/runtime/supervisor.py --stop"
fi

# ── Kill existing WSL processes on our ports ─────────────────────────────────
PORTS_TO_KILL=(8001 8081 8090)
if [[ "$ONLY" == "all" || "$ONLY" == "--gateway-only" ]]; then
    PORTS_TO_KILL+=(8002)
fi
for PORT in "${PORTS_TO_KILL[@]}"; do
    PID=$(lsof -ti ":$PORT" 2>/dev/null || true)
    if [[ -n "$PID" ]]; then
        echo "  [cleanup] Killing PID $PID on port $PORT"
        kill -9 "$PID" 2>/dev/null || true
    fi
done
sleep 1

# ── Check Ollama reachable ────────────────────────────────────────────────────
if curl -sf --max-time 2 "${OLLAMA_URL}/api/tags" > /dev/null 2>&1; then
    echo "  [ok] Ollama reachable at $OLLAMA_URL"
else
    echo "  [warn] Ollama not reachable — AI calls will fail"
    echo "         Start Ollama in WSL (ollama serve) or on Windows"
fi
echo ""

# ── Start backend API (port 8001) ────────────────────────────────────────────
if [[ "$ONLY" == "all" || "$ONLY" == "--backend-only" ]]; then
    echo "  [1/2] Starting backend API on :8001..."
    cd "$GH05T3/backend"
    AETHYRO_SKIP_LICENSE=1 WSL_MODE=1 MONGO_URL="$MONGO_URL" uvicorn server:app \
        --host 0.0.0.0 --port 8001 \
        --log-level info \
        > "$LOG_DIR/backend.log" 2>&1 &
    BACKEND_PID=$!
    echo "$BACKEND_PID" > "$PID_DIR/backend.pid"
    sleep 2
    if kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo "  [1/2] Backend started (PID $BACKEND_PID)"
    else
        echo "  [1/2] Backend failed — check $LOG_DIR/backend.log"
    fi
fi

# ── Start gateway (port 8002) ─────────────────────────────────────────────────
if [[ "$ONLY" == "all" || "$ONLY" == "--gateway-only" ]]; then
    echo "  [2/2] Starting gateway on :8002..."
    cd "$GH05T3/backend"
    AETHYRO_SKIP_LICENSE=1 WSL_MODE=1 uvicorn gateway_v3:app \
        --host 0.0.0.0 --port 8002 \
        --log-level info \
        > "$LOG_DIR/gateway.log" 2>&1 &
    GW_PID=$!
    echo "$GW_PID" > "$PID_DIR/gateway.pid"
    sleep 3
    if kill -0 "$GW_PID" 2>/dev/null; then
        echo "  [2/2] Gateway started (PID $GW_PID)"
    else
        echo "  [2/2] Gateway failed — check $LOG_DIR/gateway.log"
        tail -20 "$LOG_DIR/gateway.log" 2>/dev/null
    fi
fi

echo ""
echo "  ──────────────────────────────────────────────────────────"

# Wait for gateway ready
if [[ "$ONLY" == "all" || "$ONLY" == "--gateway-only" ]]; then
    echo "  Waiting for gateway health check..."
    for i in {1..15}; do
        if curl -sf --max-time 2 "http://localhost:8002/health" > /dev/null 2>&1; then
            echo "  [ok] Gateway ready at http://localhost:8002"
            break
        fi
        sleep 1
    done
fi

echo ""
echo "  Services:"
[[ "$ONLY" == "all" || "$ONLY" == "--backend-only" ]] && \
    echo "    Backend API:  http://localhost:8001/docs"
[[ "$ONLY" == "all" || "$ONLY" == "--gateway-only" ]] && \
    echo "    Gateway:      http://localhost:8002/docs"
echo ""
echo "  Logs: tail -f $LOG_DIR/*.log"
echo "  Stop: bash /mnt/c/Users/leer4/GH05T3/scripts/wsl_stop.sh"
echo ""
