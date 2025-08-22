#!/usr/bin/env bash
# Simple deployment helper for ai-music-generation
# Copies code to remote host and launches app.

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-}"
if [ -z "$REMOTE_HOST" ]; then
  echo "REMOTE_HOST is not set. Please set it before running."
  echo "Usage: REMOTE_HOST=\"user@your.server.com\" ./deploy_remote.sh <command>"
  exit 1
fi

REMOTE_DIR="/opt/ai-music-generation"
PYTHON_BIN="python3"
VENV_DIR=".venv"
APP_ENTRY="backend/main.py"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-backend/requirements-prod.txt}"

color() { local c="$1"; shift; printf "\033[%sm%s\033[0m\n" "$c" "$*"; }
info() { color 36 "$*"; }
warn() { color 33 "$*"; }
err()  { color 31 "$*"; }

remote_exec() {
  ssh -o StrictHostKeyChecking=accept-new "$REMOTE_HOST" "$@"
}

cmd_health() {
  local port="${PORT:-8000}";
  local host_only="${REMOTE_HOST##*@}";
  info "Health check http://$host_only:$port/health";
  remote_exec bash -lc "curl -fsS http://127.0.0.1:$port/health || curl -fsS http://localhost:$port/health || echo 'Health endpoint not reachable'"
}

cmd_logs() {
  remote_exec bash -lc "cd '$REMOTE_DIR'; tail -n 200 -f run.log"
}

cmd_status() {
  remote_exec bash -lc "ps -ef | grep -v grep | grep '$APP_ENTRY' || echo 'Not running'"
}

cmd_restart() {
  info "Restarting application..."
  remote_exec bash -lc "cd '$REMOTE_DIR'; pkill -f '$APP_ENTRY' || true; sleep 2"
  remote_exec bash -lc "cd '$REMOTE_DIR'; source '$REMOTE_DIR/$VENV_DIR/bin/activate'; PORT=8000 nohup $PYTHON_BIN '$APP_ENTRY' > run.log 2>&1 & echo PID:\$!"
  sleep 3
  cmd_health || true
}

usage() {
  cat <<EOF
Usage: $(basename "$0") <command>
Commands:
  health           Check /health endpoint
  logs             Tail run.log
  status           Show running process
  restart          Restart the application
  help             Show this help
EOF
}

main() {
  local cmd="${1:-help}"; shift || true
  case "$cmd" in
    health) cmd_health "$@";;
    logs) cmd_logs "$@";;
    status) cmd_status "$@";;
    restart) cmd_restart "$@";;
    help|--help|-h) usage;;
    *) err "Unknown command: $cmd"; usage; exit 1;;
  esac
}

main "$@"
