#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
LEADER_PORT="${LEADER_PORT:-8787}"

log() {
  printf '[leader:macos] %s\n' "$*"
}

effective_port() {
  local port="${LEADER_PORT}"
  local previous=""
  for arg in "$@"; do
    if [ "${previous}" = "--port" ]; then
      port="${arg}"
      previous=""
      continue
    fi
    case "${arg}" in
      --port=*)
        port="${arg#--port=}"
        ;;
      --port)
        previous="--port"
        ;;
      *)
        previous=""
        ;;
    esac
  done
  printf '%s\n' "${port}"
}

kill_stale_leader_on_port() {
  local port="$1"
  local pids
  pids="$(lsof -nP -tiTCP:"${port}" -sTCP:LISTEN 2>/dev/null || true)"
  [ -n "${pids}" ] || return 0

  local pid command matched=0 killed_pids=""
  while IFS= read -r pid; do
    [ -n "${pid}" ] || continue
    command="$(ps -p "${pid}" -o command= 2>/dev/null || true)"
    case "${command}" in
      *dml_cluster.leader*|*dml-leader*|*"${PROJECT_DIR}/leader_macos.sh"*)
        matched=1
        killed_pids="${killed_pids} ${pid}"
        log "Stopping stale leader process ${pid} on port ${port}"
        kill "${pid}" 2>/dev/null || true
        ;;
      *)
        log "Port ${port} is already used by another process:"
        log "  pid=${pid} ${command}"
        log "Stop that process or choose another port with --port."
        exit 1
        ;;
    esac
  done <<< "${pids}"

  [ "${matched}" -eq 1 ] || return 0
  local attempt
  for attempt in 1 2 3 4 5; do
    local any_alive=0
    for pid in ${killed_pids}; do
      if kill -0 "${pid}" 2>/dev/null; then
        any_alive=1
      fi
    done
    [ "${any_alive}" -eq 0 ] && break
    sleep 0.2
  done

  for pid in ${killed_pids}; do
    if kill -0 "${pid}" 2>/dev/null; then
      command="$(ps -p "${pid}" -o command= 2>/dev/null || true)"
      case "${command}" in
        *dml_cluster.leader*|*dml-leader*|*"${PROJECT_DIR}/leader_macos.sh"*)
          log "Force-stopping stale leader process ${pid} on port ${port}"
          kill -9 "${pid}" 2>/dev/null || true
          ;;
      esac
    fi
  done

  pids="$(lsof -nP -tiTCP:"${port}" -sTCP:LISTEN 2>/dev/null || true)"
  [ -n "${pids}" ] || return 0

  while IFS= read -r pid; do
    [ -n "${pid}" ] || continue
    command="$(ps -p "${pid}" -o command= 2>/dev/null || true)"
    case "${command}" in
      *dml_cluster.leader*|*dml-leader*|*"${PROJECT_DIR}/leader_macos.sh"*)
        log "Force-stopping stale leader process ${pid} still listening on port ${port}"
        kill -9 "${pid}" 2>/dev/null || true
        ;;
      *)
        log "Port ${port} is still used by another process:"
        log "  pid=${pid} ${command}"
        exit 1
        ;;
    esac
  done <<< "${pids}"
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
  local port
  port="$(effective_port "$@")"
  kill_stale_leader_on_port "${port}"

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
