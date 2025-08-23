#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] Starting AI Music Generation Platform (ENV=${ENV:-production})"

# Create necessary directories
mkdir -p /app/generated_audio /app/uploads

echo "[entrypoint] Launching uvicorn with app.py"
exec python -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --log-level ${LOG_LEVEL:-info}
