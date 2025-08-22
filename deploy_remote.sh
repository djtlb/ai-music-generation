#!/usr/bin/env bash
# Simple deployment helper for ai-music-generation
# Copies code to remote host and launches app.
# NOTE: Running services as root is discouraged; consider creating a non-root user.
# Customize variables below as needed.

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-}"              # Remote SSH target (must be set as environment variable)
if [ -z "$REMOTE_HOST" ]; then
  err "REMOTE_HOST is not set. Please set it before running."
  echo "Usage: REMOTE_HOST=\"user@your.server.com\" ./deploy_remote.sh <command>"
  exit 1
fi

REMOTE_DIR="/opt/ai-music-generation"      # Destination directory on remote host (override: REMOTE_DIR=...)
PYTHON_BIN="python3"                       # Remote python command
VENV_DIR=".venv"                           # Virtualenv directory name under REMOTE_DIR
APP_ENTRY="backend/main.py"                         # Entry point script

# You can force which requirements file to install first via REQUIREMENTS_FILE env var, e.g.
# REQUIREMENTS_FILE=backend/requirements-prod.txt ./deploy_remote.sh deploy

REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-backend/requirements-prod.txt}"

# Additional requirements files to install if present (after REQUIREMENTS_FILE)
EXTRA_REQUIREMENTS=(core-requirements.txt requirements-arrangement.txt requirements-audio.txt requirements-faiss.txt requirements-pipeline.txt requirements-planner.txt)
RSYNC_EXCLUDES=(
  .git
  __pycache__
  '*.pyc'
  .mypy_cache
  .pytest_cache
  node_modules
  pids
  '*.wav'
  generated_audio
  .DS_Store
  .venv
  venv
  '*.so'
  '*.so.*'
  nvidia
  cuda
  torch
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

color() { local c="$1"; shift; printf "\033[%sm%s\033[0m\n" "$c" "$*"; }
info() { color 36 "$*"; }
warn() { color 33 "$*"; }
err()  { color 31 "$*"; }

die() { err "ERROR: $*"; exit 1; }

require_cmd() { command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"; }

build_rsync_excludes() {
  local args=()
  for e in "${RSYNC_EXCLUDES[@]}"; do
    args+=(--exclude "$e")
  done
  printf '%s\n' "${args[@]}"
}

cmd_sync() {
  require_cmd rsync
  info "Syncing project to $REMOTE_HOST:$REMOTE_DIR";
  local excludes; IFS=$'\n' read -r -d '' -a excludes < <(build_rsync_excludes && printf '\0')
  # Clean remote venv first to avoid conflicts
  remote_exec bash -c "rm -rf '$REMOTE_DIR/.venv' '$REMOTE_DIR/venv' || true"
  rsync -avz --progress "${excludes[@]}" "$PROJECT_DIR/" "$REMOTE_HOST:$REMOTE_DIR/"
}

remote_exec() {
  ssh -o StrictHostKeyChecking=accept-new "$REMOTE_HOST" "$@"
}

cmd_env() {
  info "Creating / updating virtual environment on remote";
  remote_exec bash -lc "set -e; mkdir -p '$REMOTE_DIR'; cd '$REMOTE_DIR'; if [ ! -d '$VENV_DIR' ]; then $PYTHON_BIN -m venv '$VENV_DIR'; fi; source '$VENV_DIR/bin/activate'; python -m pip install --upgrade pip; \
    if [ -f '$REQUIREMENTS_FILE' ]; then echo 'Installing primary requirements file: $REQUIREMENTS_FILE'; pip install -r '$REQUIREMENTS_FILE'; else echo 'Primary requirements file not found: $REQUIREMENTS_FILE'; fi; \
    for f in core-requirements.txt requirements-arrangement.txt requirements-audio.txt requirements-faiss.txt requirements-pipeline.txt requirements-planner.txt; do [ -f \"\$f\" ] && echo \"Installing optional requirements: \$f\" && pip install -r \"\$f\" || true; done"
}

wait_for_port() {
  local host="$1" port="$2" timeout="${3:-30}" start now
  start=$(date +%s)
  info "Waiting for $host:$port (timeout ${timeout}s)";
  while true; do
    if nc -z "$host" "$port" 2>/dev/null; then info "Port $port is open"; return 0; fi
    now=$(date +%s)
    if (( now-start > timeout )); then warn "Timeout waiting for $host:$port"; return 1; fi
    sleep 1
  done
}

cmd_start() {
  local port="${PORT:-8000}"
  info "Starting $APP_ENTRY (nohup) on remote (PORT=$port)";
  # Ensure run.log exists before starting
  remote_exec bash -lc "cd '$REMOTE_DIR'; touch run.log"
  remote_exec bash -lc "cd '$REMOTE_DIR'; source '$REMOTE_DIR/$VENV_DIR/bin/activate'; PORT=$port nohup $PYTHON_BIN '$APP_ENTRY' > run.log 2>&1 & echo PID:\$!"
  # Best-effort wait for port (assumes remote host is same as REMOTE_HOST host portion)
  local host_only="${REMOTE_HOST##*@}"; wait_for_port "$host_only" "$port" 25 || true
}

cmd_stop() {
  info "Stopping remote app (matching $APP_ENTRY)";
  remote_exec bash -lc "pkill -f '$APP_ENTRY' || true"
}

cmd_logs() {
  # Ensure run.log exists before tailing
  remote_exec bash -lc "cd '$REMOTE_DIR'; touch run.log"
  remote_exec bash -lc "cd '$REMOTE_DIR'; tail -n 200 -f run.log"
}

cmd_status() {
  remote_exec bash -lc "ps -ef | grep -v grep | grep '$APP_ENTRY' || echo 'Not running'"
}

cmd_health() {
  local port="${PORT:-8000}";
  local host_only="${REMOTE_HOST##*@}";
  info "Health check http://$host_only:$port/health";
  remote_exec bash -lc "curl -fsS http://127.0.0.1:$port/health || curl -fsS http://localhost:$port/health || echo 'Health endpoint not reachable'"
}

cmd_restart() {
  cmd_stop || true
  cmd_start
  cmd_health || true
}

cmd_deploy() {
  cmd_sync
  cmd_env
  cmd_stop || true
  cmd_start
  cmd_status
}logs() {
  remote_exec journalctl -u ai-music.service -n 200 -f --no-pager
}

cmd_gpu_setup() {
  info "Setting up AMD GPU support on remote";
  remote_exec bash -lc "
    echo 'Installing AMD GPU drivers and ROCm (no DKMS)...'
    cd '$REMOTE_DIR'
    
    # Update system first
    apt update
    
    # Install kernel headers if needed
    apt install -y linux-headers-\$(uname -r) dkms build-essential || true
    
    # Install AMD GPU drivers without DKMS
    if [ -f amdgpu-install_6.4.60403-1_all.deb ]; then
      echo 'Installing AMD GPU driver 6.4 (no DKMS)...'
      dpkg -i amdgpu-install_6.4.60403-1_all.deb || apt-get install -f -y
      # Skip DKMS and use precompiled packages only
      amdgpu-install --usecase=graphics,opencl,hip,rocm --no-dkms -y || true
    elif [ -f amdgpu-install_6.0.60002-1_all.deb ]; then
      echo 'Installing AMD GPU driver 6.0 (no DKMS)...'
      dpkg -i amdgpu-install_6.0.60002-1_all.deb || apt-get install -f -y
      amdgpu-install --usecase=graphics,opencl,hip,rocm --no-dkms -y || true
    else
      echo 'No AMD GPU installer found, installing ROCm from repos...'
      # Add ROCm repository
      wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -
      echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.4.3 ubuntu main' > /etc/apt/sources.list.d/rocm.list
      apt update
      apt install -y rocm-dev hip-runtime-amd rocm-device-libs || true
    fi
    
    # Alternative: Install OpenCL and basic compute stack
    apt install -y mesa-opencl-icd opencl-headers clinfo || true
    
    # Add user to render and video groups
    usermod -a -G render,video root || true
    
    # Set environment variables
    echo 'export PATH=\$PATH:/opt/rocm/bin:/opt/rocm/opencl/bin' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/rocm/lib' >> ~/.bashrc
    
    echo 'AMD GPU setup complete (DKMS skipped due to EFI issues).'
    echo 'Note: Some features may not work without proper kernel modules.'
  "
}

cmd_gpu_status() {
  info "Checking GPU status on remote";
  remote_exec bash -lc "
    echo '=== AMD GPU Hardware ==='
    lspci | grep -i amd || echo 'No AMD GPU detected'
    echo
    echo '=== DRI Devices ==='
    ls -la /dev/dri/ 2>/dev/null || echo 'No DRI devices found'
    echo
    echo '=== ROCm Status ==='
    rocm-smi 2>/dev/null || echo 'ROCm not available'
    echo
    echo '=== OpenCL Status ==='
    clinfo 2>/dev/null | head -20 || echo 'OpenCL not available'
    echo
    echo '=== PyTorch ROCm Test ==='
    cd '$REMOTE_DIR' && source '$VENV_DIR/bin/activate' && python3 -c '
import torch
print(f\\\"PyTorch version: {torch.__version__}\\\")
print(f\\\"CUDA available: {torch.cuda.is_available()}\\\")
print(f\\\"ROCm available: {torch.cuda.is_available() if hasattr(torch.cuda, \\\"is_available\\\") else False}\\\")
if torch.cuda.is_available():
    print(f\\\"GPU count: {torch.cuda.device_count()}\\\")
    for i in range(torch.cuda.device_count()):
        print(f\\\"GPU {i}: {torch.cuda.get_device_name(i)}\\\")
' 2>/dev/null || echo 'PyTorch not available or GPU not detected'
  "
}

cmd_cpu_setup() {
  info "Setting up CPU-only mode for AI workloads";
  remote_exec bash -lc "
    echo 'Configuring CPU-only AI environment...'
    cd '$REMOTE_DIR' && source '$VENV_DIR/bin/activate'
    
    # Install CPU-only PyTorch
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    
    # Install other CPU optimizations
    pip3 install accelerate transformers optimum[onnxruntime]
    
    # Set environment variables for CPU optimization
    echo 'export OMP_NUM_THREADS=\$(nproc)' >> ~/.bashrc
    echo 'export MKL_NUM_THREADS=\$(nproc)' >> ~/.bashrc
    echo 'export OPENBLAS_NUM_THREADS=\$(nproc)' >> ~/.bashrc
    
    echo 'CPU-only setup complete. AI models will run on CPU.'
  "
}

usage() {
  cat <<EOF
Usage: $(basename "$0") <command>
Commands:
  sync             Rsync project to remote
  env              Create/update virtualenv & install deps
  start            Start app (nohup)
  stop             Stop app
  restart          Stop then start app + health check
  logs             Tail run.log
  status           Show running process
  health           Curl /health endpoint
  deploy           sync + env + restart app
  gpu-setup        Install AMD GPU drivers and ROCm (skip if EFI errors)
  gpu-status       Check GPU hardware and driver status
  cpu-setup        Configure CPU-only AI environment
  service-logs     Tail systemd service logs
  help             Show this help
Environment variables you can override:
  REMOTE_HOST, REMOTE_DIR, PYTHON_BIN, VENV_DIR, APP_ENTRY
EOF
}

main() {
  local cmd="${1:-help}"; shift || true
  case "$cmd" in
    sync) cmd_sync "$@";;
    env) cmd_env "$@";;
    start) cmd_start "$@";;
    stop) cmd_stop "$@";;
    logs) cmd_logs "$@";;
    restart) cmd_restart "$@";;
    health) cmd_health "$@";;
    status) cmd_status "$@";;
    deploy) cmd_deploy "$@";;
    gpu-setup) cmd_gpu_setup "$@";;
    gpu-status) cmd_gpu_status "$@";;
    cpu-setup) cmd_cpu_setup "$@";;
    service-logs) cmd_service_logs "$@";;
    help|--help|-h) usage;;
    *) err "Unknown command: $cmd"; usage; exit 1;;
  esac
}

main "$@"
