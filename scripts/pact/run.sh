#!/usr/bin/env bash
# Canonical Pact workflow entrypoint (WSL / Linux / Git Bash).
# All broker logic lives in scripts/*.py — this script only orchestrates.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PY="${PY:-python3}"
if [[ -f "$ROOT/backend/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/backend/.venv/bin/activate"
elif [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/.venv/bin/activate"
fi

export PYTHONPATH="${PYTHONPATH:-}:$ROOT/backend"
export AETHYRO_SKIP_LICENSE="${AETHYRO_SKIP_LICENSE:-1}"
export PACT_DO_NOT_TRACK="${PACT_DO_NOT_TRACK:-true}"

CMD="${1:-help}"
shift || true

case "$CMD" in
  consumer)
    exec "$PY" -m pytest tests/test_oss_pact.py -q --tb=line "$@"
    ;;
  provider)
    exec "$PY" -m pytest tests/test_oss_provider_verify.py -q --tb=line "$@"
    ;;
  publish)
    VERSION="${1:-$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo local)}"
    TAG="${2:-ci}"
    exec "$PY" scripts/publish_pacts.py pacts/ "$VERSION" "$TAG"
    ;;
  can-i-deploy)
    exec "$PY" scripts/can_i_deploy.py \
      --consumer gh05t3-gateway \
      --provider gh05t3-oss \
      --version "${1:-$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo local)}" \
      --tag "${2:-ci}"
    ;;
  health)
    exec "$PY" scripts/broker_health.py
    ;;
  all)
    "$0" consumer
    "$0" provider
    "$0" publish
    "$0" can-i-deploy
    ;;
  help|*)
    cat <<'EOF'
Usage: scripts/pact/run.sh <command>

  consumer       Generate pacts (pytest tests/test_oss_pact.py)
  provider       Verify provider (pytest tests/test_oss_provider_verify.py)
  publish [ver] [tag]  Publish pacts/ to broker (no-op without secrets)
  can-i-deploy [ver] [tag]
  health         Broker health check
  all            consumer → provider → publish → can-i-deploy

Env: PACT_BROKER_BASE_URL or PACT_BROKER_URL, PACT_BROKER_TOKEN
Windows: use scripts/pact/run.ps1
EOF
    ;;
esac