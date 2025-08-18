#!/usr/bin/env bash
set -euo pipefail

# Simple launch script: start backend (uvicorn) and frontend (vite) concurrently.
# Usage: ./dev.sh [--no-fast]

FAST=1
for arg in "$@"; do
  case "$arg" in
    --no-fast) FAST=0 ; shift ;;
  esac
done

if [ ! -d .venv ]; then
  echo "[dev] Creating virtual environment (.venv)";
  python -m venv .venv
fi

source .venv/bin/activate

if [ ! -f .venv/.deps_installed ]; then
  echo "[dev] Installing core backend dependencies (fast mode)";
  if [ -f core-requirements.txt ]; then
    pip install -q -r core-requirements.txt
  else
    pip install -q -r backend/requirements.txt
  fi
  touch .venv/.deps_installed
fi

export FAST_DEV=$FAST

BACKEND_PORT=${BACKEND_PORT:-8000}
FRONTEND_PORT=${FRONTEND_PORT:-5173}

BACKEND_APP="backend.main:app"
if [ "$FAST_DEV" = "1" ] && [ -f backend/dev_app.py ]; then
  BACKEND_APP="backend.dev_app:app"
fi
echo "[dev] Starting backend ($BACKEND_APP) on :$BACKEND_PORT (FAST_DEV=$FAST_DEV)"
uvicorn $BACKEND_APP --reload --port $BACKEND_PORT &
BACK_PID=$!

echo "[dev] Installing node modules if needed"
if [ ! -d node_modules ]; then
  npm install --no-audit --no-fund --legacy-peer-deps >/dev/null 2>&1 || npm install
fi

echo "[dev] Starting frontend on :$FRONTEND_PORT"
npm run vite -- --port $FRONTEND_PORT &
FRONT_PID=$!

cleanup() {
  echo "\n[dev] Shutting down (pids: $BACK_PID $FRONT_PID)";
  kill $BACK_PID 2>/dev/null || true
  kill $FRONT_PID 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[dev] Both services launched. Backend: http://localhost:$BACKEND_PORT  Frontend: http://localhost:$FRONTEND_PORT"
wait
