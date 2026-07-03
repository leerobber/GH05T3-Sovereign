#!/usr/bin/env bash
# Stop GH05T3 WSL services
PID_DIR="/tmp/gh05t3-wsl-pids"

echo "Stopping GH05T3 WSL services..."

for f in "$PID_DIR"/*.pid; do
    [[ -f "$f" ]] || continue
    PID=$(cat "$f")
    NAME=$(basename "$f" .pid)
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID" 2>/dev/null && echo "  stopped $NAME (PID $PID)"
    fi
    rm -f "$f"
done

# Also kill by port in case pid files are stale
for PORT in 8001 8002 8081 8090; do
    PID=$(lsof -ti ":$PORT" 2>/dev/null || true)
    [[ -n "$PID" ]] && kill -9 "$PID" 2>/dev/null && echo "  killed port $PORT (PID $PID)"
done

echo "Done."
