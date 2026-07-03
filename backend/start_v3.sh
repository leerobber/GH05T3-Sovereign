#!/usr/bin/env bash
# GH05T3 — gateway v3 startup
# Runs alongside the existing server.py (which owns port 8001).
# Set GATEWAY_PORT in .env to override.
set -e

cd "$(dirname "$0")"

if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

PORT="${GATEWAY_PORT:-8002}"

echo "🖤 Starting GH05T3 gateway_v3 on port $PORT"
exec uvicorn gateway_v3:app \
  --host 0.0.0.0 \
  --port "$PORT" \
  --log-level info
