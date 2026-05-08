#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXPERIMENT_STARTED_AT="$(date +%s)"

cd "${PROJECT_DIR}"

MODEL="${MODEL:-resnet101}"
DATASET="${DATASET:-tiny-imagenet-200}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
SAMPLES="${SAMPLES:-128}"
BATCHES="${BATCHES:-2}"
EVAL_BATCHES="${EVAL_BATCHES:-1}"
EPOCHS="${EPOCHS:-1}"
BASE_BATCH="${BASE_BATCH:-2}"
HOST="${HOST:-0.0.0.0}"
LEADER_PORT="${LEADER_PORT:-8787}"
DIST_PORT="${DIST_PORT:-29521}"
REQUIRED_WORKERS="${REQUIRED_WORKERS:-1}"
START_DELAY="${START_DELAY:-2}"
AUTO_TIMEOUT="${AUTO_TIMEOUT:-900}"
DIST_TIMEOUT="${DIST_TIMEOUT:-300}"

leader_base=(
  ./installations/leader_macos.sh
  --mode distributed
  --distributed-parallel parameter-server
  --distributed-dataset "${DATASET}"
  --distributed-model "${MODEL}"
  --distributed-image-size "${IMAGE_SIZE}"
  --distributed-samples "${SAMPLES}"
  --distributed-base-batch "${BASE_BATCH}"
  --distributed-batches-per-epoch "${BATCHES}"
  --distributed-eval-batches "${EVAL_BATCHES}"
  --distributed-timeout "${DIST_TIMEOUT}"
  --epochs "${EPOCHS}"
  --host "${HOST}"
  --port "${LEADER_PORT}"
  --dist-port "${DIST_PORT}"
  --auto-start
  --exit-after-run
  --start-delay-seconds "${START_DELAY}"
  --auto-start-timeout "${AUTO_TIMEOUT}"
)

run_exp() {
  local label="$1"
  shift
  printf '\n=== %s ===\n' "${label}"
  "${leader_base[@]}" "$@"
}

echo "[experiments] smoke ResNet101 TinyImageNet parameter-server matrix"
echo "[experiments] expected workers for distributed runs: ${REQUIRED_WORKERS}"

run_exp "EXP 1: solo parameter-server baseline" \
  --required-workers 0 \
  --distributed-leader-only \
  --distributed-optimizations none

run_exp "EXP 2: parameter-server, no optimization" \
  --required-workers "${REQUIRED_WORKERS}" \
  --distributed-optimizations none

run_exp "EXP 3: parameter-server, fp16 updates" \
  --required-workers "${REQUIRED_WORKERS}" \
  --distributed-optimizations fp16

printf '\n=== Building experiment comparison summary ===\n'
"${PROJECT_DIR}/.venv/bin/python" "${PROJECT_DIR}/experiments/summarize_runs.py" \
  --project-dir "${PROJECT_DIR}" \
  --started-at "${EXPERIMENT_STARTED_AT}"

printf '\n=== Parameter-server smoke complete. Results are in runs/ ===\n'
