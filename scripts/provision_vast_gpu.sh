#!/usr/bin/env bash
set -euo pipefail

# Cumulus Vast.ai GPU VM Provisioning Script
# - Installs system deps
# - Builds Chronos (GPU partitioner)
# - Sets up Python venv and installs Cumulus worker
# - Configures S3/env and starts the worker via systemd (fallback to nohup)

# Defaults
REPO_URL=""
INSTALL_DIR="/opt/cumulus"
VENV_DIR="/opt/cumulus-env"
ENV_DIR="/opt/cumulus-distributed"
ENV_FILE="${ENV_DIR}/.env"
PORT="8081"
WORKERS="4"
PYTHON_BIN="python3"

# Optional S3 config (will create ENV_FILE if provided)
S3_BUCKET=""
S3_REGION="us-east-1"
AWS_KEY_ID=""
AWS_SECRET_KEY=""

usage() {
  cat << USAGE
Usage: $0 \\
  --repo-url <git_url> [--port 8081] [--workers 4] \\
  [--s3-bucket <bucket>] [--s3-region us-east-1] \\
  [--aws-key-id <id>] [--aws-secret-key <secret>]

Notes:
  - Run as root on the Vast.ai GPU VM
  - Example public repo: https://github.com/5ury44/cumulus.git
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-url) REPO_URL="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --s3-bucket) S3_BUCKET="$2"; shift 2 ;;
    --s3-region) S3_REGION="$2"; shift 2 ;;
    --aws-key-id) AWS_KEY_ID="$2"; shift 2 ;;
    --aws-secret-key) AWS_SECRET_KEY="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ "${EUID}" -ne 0 ]]; then
  echo "This script must be run as root." >&2
  exit 1
fi

if [[ -z "${REPO_URL}" ]]; then
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    REPO_URL=$(git config --get remote.origin.url || true)
  fi
fi

if [[ -z "${REPO_URL}" ]]; then
  echo "--repo-url is required (e.g. https://github.com/5ury44/cumulus.git) or run from inside a git clone." >&2
  exit 1
fi

echo "[1/8] Installing system dependencies..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y \
  git build-essential cmake \
  ocl-icd-opencl-dev opencl-headers \
  python3 python3-venv python3-pip \
  curl ca-certificates pkg-config \
  wget unzip

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "Warning: nvidia-smi not found. Ensure this is a GPU image."
fi

echo "[2/8] Cloning repository to ${INSTALL_DIR}..."
mkdir -p "${INSTALL_DIR%/*}"
if [[ -d "${INSTALL_DIR}/.git" ]]; then
  echo "Repo already present at ${INSTALL_DIR}, pulling latest..."
  git -C "${INSTALL_DIR}" fetch --all || true
  git -C "${INSTALL_DIR}" pull || true
else
  rm -rf "${INSTALL_DIR}"
  git clone "${REPO_URL}" "${INSTALL_DIR}"
fi

echo "[3/8] Building Chronos (GPU partitioner)..."
pushd "${INSTALL_DIR}/chronos_core" >/dev/null
bash scripts/install-quick.sh
popd >/dev/null

if ! command -v chronos_cli >/dev/null 2>&1 && [[ ! -x "/usr/local/bin/chronos_cli" ]]; then
  echo "Error: chronos_cli not found after build." >&2
  exit 1
fi

echo "[4/8] Creating Python venv at ${VENV_DIR} and installing worker..."
${PYTHON_BIN} -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip wheel setuptools
pip install -e "${INSTALL_DIR}"

# Worker runtime deps (ensures presence if not covered by setup.py)
if [[ -f "${INSTALL_DIR}/cumulus/worker/requirements.txt" ]]; then
  pip install -r "${INSTALL_DIR}/cumulus/worker/requirements.txt"
fi

# Optional GPU-enabled PyTorch (generic channels); user can adjust for specific CUDA
pip install --upgrade torch torchvision || true

echo "[5/8] Writing environment (.env) if S3 info provided..."
mkdir -p "${ENV_DIR}"
if [[ -n "${S3_BUCKET}" || -n "${AWS_KEY_ID}" || -n "${AWS_SECRET_KEY}" ]]; then
  cat > "${ENV_FILE}" <<EOF
CUMULUS_S3_BUCKET=${S3_BUCKET}
CUMULUS_S3_REGION=${S3_REGION}
AWS_ACCESS_KEY_ID=${AWS_KEY_ID}
AWS_SECRET_ACCESS_KEY=${AWS_SECRET_KEY}

# L1 local cache (checkpoints)
CUMULUS_LOCAL_CACHE_DIR=/tmp/cumulus/checkpoints
CUMULUS_CACHE_SIZE_LIMIT_GB=10.0
CUMULUS_KEEP_CHECKPOINTS=5
CUMULUS_CHECKPOINT_EVERY_STEPS=100
CUMULUS_CHECKPOINT_EVERY_SECONDS=300
CUMULUS_AUTO_CLEANUP=true
CUMULUS_ENABLE_JOB_METADATA=true
EOF
  echo "Wrote ${ENV_FILE}"
else
  echo "Skipping S3 env write; using local-only checkpointing unless ${ENV_FILE} provided later."
fi

echo "[6/8] Creating systemd service for Cumulus worker on port ${PORT}..."
SERVICE_PATH="/etc/systemd/system/cumulus-worker.service"
cat > "${SERVICE_PATH}" <<SERVICE
[Unit]
Description=Cumulus Worker (GPU Partitioned Execution)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=${INSTALL_DIR}
Environment=LD_LIBRARY_PATH=/usr/local/lib
Environment=MAX_CONCURRENT_JOBS=${WORKERS}
EnvironmentFile=-${ENV_FILE}
ExecStart=${VENV_DIR}/bin/python -m uvicorn worker.server:create_app --factory --host 0.0.0.0 --port ${PORT}
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
SERVICE

systemctl daemon-reload || true
if command -v systemctl >/dev/null 2>&1 && systemctl is-system-running >/dev/null 2>&1; then
  systemctl enable cumulus-worker.service || true
  systemctl restart cumulus-worker.service || true
else
  echo "systemd not available or inactive; starting with nohup fallback..."
  # Kill any existing uvicorn processes
  pkill -f "uvicorn.*worker.server" || true
  sleep 1
  
  # Start worker with proper environment and correct module path
  (
    cd "${INSTALL_DIR}" && \
    set -a; [ -f "${ENV_FILE}" ] && . "${ENV_FILE}"; set +a; \
    export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"; \
    nohup "${VENV_DIR}/bin/python" -m uvicorn worker.server:create_app --factory --host 0.0.0.0 --port "${PORT}" \
      > /tmp/cumulus-worker.log 2>&1 &
  )
fi

echo "[7/8] Verifying Chronos and worker health..."
export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"
set +e
chronos_cli stats || /usr/local/bin/chronos_cli stats || true

# Wait for worker to start and verify health
echo "Waiting for worker to start..."
for i in {1..20}; do
  if curl -fsS "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo "✅ Worker is healthy on port ${PORT}!"
    break
  fi
  echo "Attempt $i/20: Worker not ready yet..."
  sleep 2
done

# Show final status
if curl -fsS "http://localhost:${PORT}/health" >/dev/null 2>&1; then
  echo "✅ Worker verification successful"
else
  echo "❌ Worker failed to start - checking logs:"
  tail -n 20 /tmp/cumulus-worker.log 2>/dev/null || echo "No log file found"
fi
set -e

echo "[8/8] Done. Next steps:"
cat << NEXT

- If behind a firewall or private network, create an SSH tunnel from your local machine:
  ssh -p <vast_ssh_port> -N -f -L 8080:localhost:${PORT} root@<vast_public_ip>

- From your local environment, verify:
  curl -s http://localhost:8080/health
  curl -s http://localhost:8080/api/info | jq .

- Run local tests (in this repo on your laptop):
  python cumulus/tests/test_complete_nn.py
  python cumulus/tests/test_artifact_store.py   # requires S3 in ${ENV_FILE}

Logs:
  journalctl -u cumulus-worker -f       # if systemd
  tail -f /tmp/cumulus-worker.log       # nohup fallback

Config:
  Edit ${ENV_FILE} and then restart:
    systemctl restart cumulus-worker || (pkill -f "uvicorn.*worker.server" && cd ${INSTALL_DIR} && set -a; [ -f ${ENV_FILE} ] && . ${ENV_FILE}; set +a; export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"; nohup ${VENV_DIR}/bin/python -m uvicorn worker.server:create_app --factory --host 0.0.0.0 --port ${PORT} > /tmp/cumulus-worker.log 2>&1 &)

NEXT

echo "Provisioning complete."


