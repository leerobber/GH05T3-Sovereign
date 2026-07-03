#!/usr/bin/env bash
# One-time WSL setup for GH05T3 backend
# Run from WSL: bash /mnt/c/Users/leer4/GH05T3/scripts/wsl_setup.sh
set -euo pipefail

GH05T3="/mnt/c/Users/leer4/GH05T3"
VENV_DIR="$HOME/gh05t3-wsl"

echo "=== GH05T3 WSL Setup ==="
echo ""

# ── System packages ──────────────────────────────────────────────────────────
echo "[1/5] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3 python3-pip python3-venv python3-dev \
    build-essential libssl-dev libffi-dev \
    git curl wget \
    libsqlite3-dev \
    portaudio19-dev \
    ffmpeg \
    2>/dev/null || true

echo "[1/5] Done."

# ── Python venv ──────────────────────────────────────────────────────────────
echo "[2/5] Creating Python venv at $VENV_DIR..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel setuptools -q
echo "[2/5] Done. Python: $(python --version)"

# ── Install requirements ─────────────────────────────────────────────────────
echo "[3/5] Installing Python requirements (this takes a few minutes)..."
pip install -r "$GH05T3/backend/requirements-wsl.txt" -q
echo "[3/5] Done."

# ── Environment config ────────────────────────────────────────────────────────
echo "[4/5] Writing WSL environment config..."
WIN_HOST_IP=$(ip route show default 2>/dev/null | awk '/default/ {print $3; exit}')
WIN_HOST_IP="${WIN_HOST_IP:-172.30.96.1}"

OLLAMA_URL=""
for CANDIDATE in "http://127.0.0.1:11434" "http://localhost:11434" "http://${WIN_HOST_IP}:11434"; do
    if curl -sf --max-time 2 "${CANDIDATE}/api/tags" > /dev/null 2>&1; then
        OLLAMA_URL="$CANDIDATE"
        break
    fi
done
OLLAMA_URL="${OLLAMA_URL:-http://${WIN_HOST_IP}:11434}"

cat > "$HOME/.gh05t3-wsl.env" << EOF
# GH05T3 WSL environment — sourced by wsl_start.sh
export GH05T3_ROOT="$GH05T3"
export BACKEND_DIR="$GH05T3/backend"
export PYTHONPATH="$GH05T3/backend:$GH05T3:$VENV_DIR/lib/python3.12/site-packages"
export VENV_DIR="$VENV_DIR"

# Windows host services
export WIN_HOST_IP="$WIN_HOST_IP"
export OLLAMA_HOST="$OLLAMA_URL"
export OLLAMA_BASE_URL="$OLLAMA_URL"
export OLLAMA_GATEWAY_URL="$OLLAMA_URL"

# Gateway config
export GW3_PORT=8002
export BACKEND_PORT=8001
export SUPERVISOR_PORT=8090

# Force WSL-safe mode (disables Windows-only features)
export WSL_MODE=1
export SKIP_RIBS_GPU=1
export LLM_PROVIDER=ollama

# Data paths (use Windows filesystem via /mnt/c)
export DATA_DIR="$GH05T3/backend/data"
export DB_PATH="$GH05T3/backend/data/sovereign.db"

# MongoDB on Windows (WSL reaches via host gateway IP)
export MONGO_URL="mongodb://${WIN_HOST_IP}:27017"
export DB_NAME="gh05t3"
EOF

echo "[4/5] Config written to ~/.gh05t3-wsl.env"
echo "      Windows host IP: $WIN_HOST_IP"

# ── Shell aliases ─────────────────────────────────────────────────────────────
echo "[5/5] Adding shell aliases..."
ALIAS_BLOCK='
# GH05T3 WSL aliases
alias gh05t3-start="bash /mnt/c/Users/leer4/GH05T3/scripts/wsl_start.sh"
alias gh05t3-stop="bash /mnt/c/Users/leer4/GH05T3/scripts/wsl_stop.sh"
alias gh05t3-logs="tail -f /tmp/gh05t3-wsl/*.log 2>/dev/null || echo no logs yet"
alias gh05t3-activate="source $HOME/gh05t3-wsl/bin/activate && source $HOME/.gh05t3-wsl.env"
'
if ! grep -q "gh05t3-start" ~/.bashrc 2>/dev/null; then
    echo "$ALIAS_BLOCK" >> ~/.bashrc
fi

echo "[5/5] Done."
echo ""
echo "=== Setup complete! ==="
echo ""
echo "  To start GH05T3 in WSL:  gh05t3-start"
echo "  To activate the venv:     gh05t3-activate"
echo "  Config:                   ~/.gh05t3-wsl.env"
echo ""
echo "  Next: open a new terminal or run: source ~/.bashrc"
