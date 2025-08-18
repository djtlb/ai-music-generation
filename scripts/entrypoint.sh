#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] Starting AI Music Generation Platform (ENV=${ENV:-production})"

# Apply database migrations (placeholder)
if command -v alembic >/dev/null 2>&1; then
  echo "[entrypoint] Running alembic migrations"
  alembic upgrade head || echo "[entrypoint] Alembic migrations skipped"
fi

# Extra initialization hooks can be placed here

if [ "${USE_GUNICORN:-1}" = "1" ]; then
  echo "[entrypoint] Launching gunicorn with uvicorn workers"
  exec gunicorn backend.main:app \
    --workers "${WORKERS:-2}" \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:${PORT:-8000} \
    --log-level ${LOG_LEVEL:-info}
else
  echo "[entrypoint] Launching uvicorn directly"
  exec python -m uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000} --log-level ${LOG_LEVEL:-info}
fi
