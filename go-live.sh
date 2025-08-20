#!/usr/bin/env bash
set -euo pipefail

# Unified script to bring the stack LIVE (build + up + status)
# Safe to re-run. Adds helpful diagnostics for common Docker permission issues.

if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: ./go-live.sh [--rebuild] [--fresh]"
  echo "  --rebuild  Force rebuild images"
  echo "  --fresh    Remove containers + volumes before starting (DANGEROUS)"
  exit 0
fi

REBUILD=false
FRESH=false
for arg in "$@"; do
  case "$arg" in
    --rebuild) REBUILD=true ;;
    --fresh) FRESH=true ;;
  esac
done

if ! command -v docker >/dev/null 2>&1; then
  echo "[ERR] Docker not installed or not on PATH" >&2; exit 1; fi

if ! groups | grep -q '\bdocker\b'; then
  echo "[WARN] Current shell not in docker group. Trying newgrp docker..." >&2
  exec newgrp docker <<'EOF'
./go-live.sh "$@"
EOF
fi

if [[ ! -f .env ]]; then
  echo "[INFO] No .env found. Creating from .env.example"
  cp .env.example .env
fi

if $FRESH; then
  echo "[ACTION] Stopping and removing existing stack (including volumes)"
  docker compose down -v || true
fi

if $REBUILD; then
  echo "[ACTION] Rebuilding images"
  docker compose build --no-cache
else
  echo "[ACTION] Building images (cache allowed)"
  docker compose build
fi

echo "[ACTION] Starting stack"
docker compose up -d

echo "[INFO] Waiting for API readiness (health/ready)"
ATTEMPTS=30
until curl -fsS http://localhost:8000/health/ready >/dev/null 2>&1; do
  ((ATTEMPTS--)) || { echo "[ERR] API not ready after timeout"; docker compose logs --tail=200 api; exit 1; }
  sleep 2
  echo -n '.'
done

echo -e "\n[SUCCESS] API ready at http://localhost:8000"

cat <<SUMMARY
------------------------------------------------------------------------
Live Stack Summary:
  API Docs:      http://localhost:8000/docs
  Health:        http://localhost:8000/health
  Readiness:     http://localhost:8000/health/ready
  Metrics:       http://localhost:8000/metrics
  Frontend (if built): http://localhost:8000/
------------------------------------------------------------------------
Use 'docker compose logs -f api' to tail logs.
Use './go-live.sh --rebuild' after changing dependencies.
Use './go-live.sh --fresh' for a clean slate (drops DB volume!).
SUMMARY
