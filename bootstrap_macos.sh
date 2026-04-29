#!/usr/bin/env bash
set -euo pipefail

LEADER_HOST="${LEADER_HOST:-ruturajs-macbook-pro.taila5426e.ts.net}"
LEADER_PORT="${LEADER_PORT:-8787}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"

log() {
  printf '[bootstrap:macos] %s\n' "$*"
}

have_command() {
  command -v "$1" >/dev/null 2>&1
}

ensure_xcode_clt() {
  if xcode-select -p >/dev/null 2>&1; then
    log "Xcode Command Line Tools already present"
    return
  fi
  log "Installing Xcode Command Line Tools. Approve the macOS prompt, then this script will continue."
  xcode-select --install >/dev/null 2>&1 || true
  until xcode-select -p >/dev/null 2>&1; do
    sleep 10
  done
  log "Xcode Command Line Tools installed"
}

load_homebrew_shellenv() {
  if have_command brew; then
    return
  fi
  if [ -x /opt/homebrew/bin/brew ]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  elif [ -x /usr/local/bin/brew ]; then
    eval "$(/usr/local/bin/brew shellenv)"
  fi
}

ensure_homebrew() {
  load_homebrew_shellenv
  if have_command brew; then
    log "Homebrew already present"
    return
  fi
  log "Installing Homebrew. Homebrew may ask for your macOS password to create its install directory."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  load_homebrew_shellenv
  if ! have_command brew; then
    log "Homebrew installed but is not on PATH. Open a new terminal and re-run this script."
    exit 1
  fi
}

ensure_brew_command() {
  local command_name="$1"
  local package_name="$2"
  if have_command "${command_name}"; then
    log "${command_name} already present"
    return
  fi
  log "Installing ${package_name} via Homebrew"
  brew install "${package_name}"
}

python311_bin() {
  if have_command python3.11; then
    command -v python3.11
    return
  fi
  local brew_python
  brew_python="$(brew --prefix python@3.11 2>/dev/null || true)"
  if [ -n "${brew_python}" ] && [ -x "${brew_python}/bin/python3.11" ]; then
    printf '%s\n' "${brew_python}/bin/python3.11"
    return
  fi
  return 1
}

ensure_python311() {
  if python311_bin >/dev/null 2>&1; then
    log "Python 3.11 already present"
    return
  fi
  log "Installing Python 3.11 via Homebrew"
  brew install python@3.11
}

tailscale_bin() {
  if have_command tailscale; then
    command -v tailscale
    return
  fi
  local brew_prefix
  brew_prefix="$(brew --prefix tailscale 2>/dev/null || true)"
  if [ -n "${brew_prefix}" ] && [ -x "${brew_prefix}/bin/tailscale" ]; then
    printf '%s\n' "${brew_prefix}/bin/tailscale"
    return
  fi
  return 1
}

tailscale_is_running() {
  local ts
  ts="$(tailscale_bin 2>/dev/null || true)"
  [ -n "${ts}" ] || return 1
  "${ts}" status --json 2>/dev/null | grep -q '"BackendState"[[:space:]]*:[[:space:]]*"Running"'
}

ensure_tailscale() {
  local installed_now=0
  if tailscale_bin >/dev/null 2>&1; then
    log "Tailscale CLI already present"
  else
    log "Installing Tailscale via Homebrew"
    brew install tailscale
    installed_now=1
  fi

  log "Starting Tailscale service if needed"
  brew services start tailscale >/dev/null 2>&1 || true

  if tailscale_is_running; then
    log "Tailscale is authenticated"
    return
  fi

  if [ "${installed_now}" -eq 1 ]; then
    log "Tailscale was just installed."
  fi
  log "Authenticate Tailscale in the browser. If the browser does not open, use the URL printed below."
  local auth_output
  auth_output="$("$(tailscale_bin)" up --timeout=1s 2>&1 || true)"
  printf '%s\n' "${auth_output}"
  local auth_url
  auth_url="$(printf '%s\n' "${auth_output}" | grep -Eo 'https://[^[:space:]]+' | head -n 1 || true)"
  if [ -n "${auth_url}" ]; then
    open "${auth_url}" >/dev/null 2>&1 || true
  else
    "$(tailscale_bin)" up || true
  fi

  log "Waiting for Tailscale authentication to finish"
  until tailscale_is_running; do
    sleep 5
  done
  log "Tailscale is authenticated"
}

ensure_venv() {
  local py311
  py311="$(python311_bin)"
  if [ ! -x "${VENV_DIR}/bin/python" ]; then
    log "Creating virtual environment at ${VENV_DIR}"
    "${py311}" -m venv "${VENV_DIR}"
  else
    log "Virtual environment already present"
  fi

  log "Installing project package into virtual environment"
  "${VENV_DIR}/bin/python" -m pip install --upgrade pip
  "${VENV_DIR}/bin/python" -m pip install -r "${PROJECT_DIR}/requirements.txt"
  ensure_torch
  "${VENV_DIR}/bin/python" -m pip install -e "${PROJECT_DIR}"
}

ensure_torch() {
  if [ "${SKIP_TORCH_INSTALL:-0}" = "1" ]; then
    log "Skipping PyTorch install because SKIP_TORCH_INSTALL=1"
    return
  fi
  if "${VENV_DIR}/bin/python" -c "import torch, torchvision" >/dev/null 2>&1; then
    log "PyTorch and torchvision already present"
    return
  fi
  log "Installing PyTorch build selected for this machine"
  "${VENV_DIR}/bin/python" -m dml_cluster.torch_install --install
}

main() {
  cd "${PROJECT_DIR}"
  ensure_xcode_clt
  ensure_homebrew
  ensure_brew_command curl curl
  ensure_brew_command git git
  ensure_python311
  ensure_tailscale
  ensure_venv

  log "Detected hardware:"
  "${VENV_DIR}/bin/python" -m dml_cluster.hardware
  log "Starting worker inside virtual environment"
  exec "${VENV_DIR}/bin/python" -m dml_cluster.worker \
    --leader "${LEADER_HOST}" \
    --port "${LEADER_PORT}" \
    --project-dir "${PROJECT_DIR}"
}

main "$@"
