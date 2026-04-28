#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
LEADER_PORT="${LEADER_PORT:-8787}"

log() {
  printf '[leader:macos] %s\n' "$*"
}

python_bin() {
  if command -v python3.11 >/dev/null 2>&1; then
    command -v python3.11
  elif command -v python3 >/dev/null 2>&1; then
    command -v python3
  else
    return 1
  fi
}

main() {
  cd "${PROJECT_DIR}"
  local py
  py="$(python_bin)" || {
    log "Python is missing. Run bootstrap_macos.sh once, or install Python 3.11."
    exit 1
  }

  if [ ! -x "${VENV_DIR}/bin/python" ]; then
    log "Creating virtual environment at ${VENV_DIR}"
    "${py}" -m venv "${VENV_DIR}"
  fi

  log "Installing project package into virtual environment"
  "${VENV_DIR}/bin/python" -m pip install --upgrade pip
  "${VENV_DIR}/bin/python" -m pip install -r "${PROJECT_DIR}/requirements.txt"
  if ! "${VENV_DIR}/bin/python" -c "import torch, torchvision" >/dev/null 2>&1; then
    log "Installing PyTorch build selected for this machine"
    "${VENV_DIR}/bin/python" -m dml_cluster.torch_install --install
  fi
  "${VENV_DIR}/bin/python" -m pip install -e "${PROJECT_DIR}"

  log "Starting leader"
  exec "${VENV_DIR}/bin/python" -m dml_cluster.leader \
    --port "${LEADER_PORT}" \
    --project-dir "${PROJECT_DIR}" \
    "$@"
}

main "$@"
