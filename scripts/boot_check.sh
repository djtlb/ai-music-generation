#!/bin/bash
# Simple boot verification: checks frontend, backend, proxy health
set -e
FRONTEND=${FRONTEND:-http://localhost:5173}
BACKEND=${BACKEND:-http://localhost:8000}
PROXY=${PROXY:-http://localhost:3000}

function check() {
  local name=$1
  local url=$2
  status=$(curl -s -o /dev/null -w "%{http_code}" "$url" || true)
  if [ "$status" = "200" ]; then
    echo "✅ $name reachable ($status)"
  else
    echo "⚠️  $name not ready (HTTP $status)"
  fi
}

check Backend "$BACKEND/health"
check BackendDocs "$BACKEND/docs"
check Frontend "$FRONTEND"
check Proxy "$PROXY"

