#!/usr/bin/env bash
set -euo pipefail

###############
# AI Music Generation Dev Stack Launcher
# Automates: docker permission sanity, build, up, health wait, summary
###############

PROJECT_ROOT="$(cd -- "${BASH_SOURCE[0]%/*}"/.. && pwd)"
cd "$PROJECT_ROOT"

COMPOSE_FILE=${COMPOSE_FILE:-docker-compose.yml}
HEALTH_URL=${HEALTH_URL:-http://localhost:8000/health}
WAIT_SECS=${WAIT_SECS:-60}
USE_SUDO=""

log() { printf "\e[36m[dev-stack]\e[0m %s\n" "$*"; }
warn() { printf "\e[33m[warn]\e[0m %s\n" "$*"; }
err() { printf "\e[31m[err]\e[0m %s\n" "$*"; }

need_cmd() { command -v "$1" >/dev/null 2>&1 || { err "Required command '$1' not found"; exit 1; }; }
need_cmd docker
need_cmd grep
need_cmd awk

# Detect docker daemon accessibility
if ! docker info >/dev/null 2>&1; then
  warn "Docker daemon not accessible as current user ($USER). Will fallback to sudo if possible."
  USE_SUDO=sudo
fi

# Offer to add user to docker group (only if we had to fallback to sudo)
if [[ -n "$USE_SUDO" ]]; then
  if ! id -nG "$USER" | grep -qw docker; then
    warn "Adding user '$USER' to docker group (requires sudo). You must log out/in afterwards for it to take effect."
    if $USE_SUDO groupadd docker 2>/dev/null; then :; fi
    $USE_SUDO usermod -aG docker "$USER" || warn "Failed to add user to docker group."
    ADDED_GROUP=1
  fi
fi

log "Using compose file: $COMPOSE_FILE"
if [[ ! -f $COMPOSE_FILE ]]; then
  err "Compose file '$COMPOSE_FILE' not found"; exit 1; fi

log "Building images (this may take a while on first run)"
$USE_SUDO docker compose -f "$COMPOSE_FILE" build --pull

log "Starting stack detached"
$USE_SUDO docker compose -f "$COMPOSE_FILE" up -d

log "Listing services"
$USE_SUDO docker compose -f "$COMPOSE_FILE" ps

log "Waiting for health: $HEALTH_URL (timeout ${WAIT_SECS}s)"
start_ts=$(date +%s)
ok=0
while true; do
  if curl -fsS "$HEALTH_URL" -o /tmp/health.json; then
    if grep -q 'healthy' /tmp/health.json; then
      log "Health endpoint reports healthy"
      ok=1
      break
    fi
  fi
  now=$(date +%s)
  if (( now - start_ts > WAIT_SECS )); then
    break
  fi
  sleep 2
done

if (( ok == 0 )); then
  warn "Service did not become fully healthy within timeout. Showing last 50 backend log lines."
  $USE_SUDO docker compose -f "$COMPOSE_FILE" logs --tail=50 api || true
  exit 1
fi

cat <<EOF
-------------------------------------------------
Dev stack up.
API:       http://localhost:8000
Health:    $HEALTH_URL
Docs:      http://localhost:8000/docs
To tail:   ${USE_SUDO} docker compose logs -f api
To stop:   ${USE_SUDO} docker compose down
EOF

if [[ -n "${ADDED_GROUP:-}" ]]; then
  warn "User added to docker group. Log out and log back in to drop need for sudo next time."
fi

exit 0
