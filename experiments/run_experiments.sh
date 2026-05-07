#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXPERIMENT_STARTED_AT="$(date +%s)"

MODEL="resnet50"
DATASET="cifar10"
IMAGE_SIZE=64
SAMPLES=10000
BATCHES=100
EVAL_BATCHES=20
EPOCHS=5
BASE_BATCH=32
MICROBATCH=2
COMPRESS_RATIO=0.01
STRAGGLER_RANK=1
STRAGGLER_DELAY=3.0
MASTER_ADDR=""
HOST=""
LEADER_PORT=8787
DIST_PORT=29501
DIST_TIMEOUT=300
REQUIRED_WORKERS=1
START_DELAY=2
AUTO_TIMEOUT=900
LOOPBACK=0
RUN_DATA=1
RUN_PIPELINE=1

usage() {
  cat <<'EOF'
Usage: experiments/run_experiments.sh [options]

Options:
  --loopback                 Run a small local smoke matrix and start a local worker.
  --model NAME               Model name. Default: resnet50.
  --dataset NAME             Dataset name. Default: cifar10.
  --image-size N             Input image size. Default: 64.
  --samples N                Training sample limit. Default: 10000.
  --batches N                Batches per epoch. Default: 100.
  --eval-batches N           Validation batches. Default: 20.
  --epochs N                 Epoch count. Default: 5.
  --base-batch N             Data-parallel per-rank batch and pipeline batch. Default: 32.
  --microbatch N             Pipeline microbatch size. Default: 2.
  --host HOST                Leader bind host. Use 0.0.0.0 for real multi-machine runs.
  --master-addr HOST         torch.distributed master address.
  --leader-port N            Leader control port. Default: 8787.
  --dist-port N              torch.distributed port. Default: 29501.
  --straggler-delay SEC      Straggler sleep per batch. Default: 3.0.
  --dist-timeout SEC         Gloo collective timeout in seconds. Default: 300 (120 in loopback).
  --required-workers N       Required workers for distributed experiments. Default: 1.
  --only-pipeline            Run only the pipeline experiment.
  --skip-pipeline            Skip pipeline experiment.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --loopback)
      LOOPBACK=1
      MASTER_ADDR="127.0.0.1"
      HOST="127.0.0.1"
      LEADER_PORT=8893
      DIST_PORT=29661
      DIST_TIMEOUT=120
      SAMPLES=64
      BATCHES=2
      EVAL_BATCHES=1
      EPOCHS=1
      BASE_BATCH=4
      MICROBATCH=2
      STRAGGLER_DELAY=0.1
      shift
      ;;
    --model) MODEL="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --image-size) IMAGE_SIZE="$2"; shift 2 ;;
    --samples) SAMPLES="$2"; shift 2 ;;
    --batches) BATCHES="$2"; shift 2 ;;
    --eval-batches) EVAL_BATCHES="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --base-batch) BASE_BATCH="$2"; shift 2 ;;
    --microbatch) MICROBATCH="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --master-addr) MASTER_ADDR="$2"; shift 2 ;;
    --leader-port) LEADER_PORT="$2"; shift 2 ;;
    --dist-port) DIST_PORT="$2"; shift 2 ;;
    --straggler-delay) STRAGGLER_DELAY="$2"; shift 2 ;;
    --dist-timeout) DIST_TIMEOUT="$2"; shift 2 ;;
    --required-workers) REQUIRED_WORKERS="$2"; shift 2 ;;
    --only-pipeline) RUN_DATA=0; RUN_PIPELINE=1; shift ;;
    --skip-pipeline) RUN_PIPELINE=0; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

cd "${PROJECT_DIR}"

LOCAL_WORKER_PID=""
cleanup() {
  if [[ -n "${LOCAL_WORKER_PID}" ]] && kill -0 "${LOCAL_WORKER_PID}" 2>/dev/null; then
    kill "${LOCAL_WORKER_PID}" 2>/dev/null || true
    wait "${LOCAL_WORKER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

leader_base=(
  ./installations/leader_macos.sh
  --mode distributed
  --distributed-dataset "${DATASET}"
  --distributed-model "${MODEL}"
  --distributed-image-size "${IMAGE_SIZE}"
  --distributed-samples "${SAMPLES}"
  --distributed-base-batch "${BASE_BATCH}"
  --distributed-batches-per-epoch "${BATCHES}"
  --distributed-eval-batches "${EVAL_BATCHES}"
  --epochs "${EPOCHS}"
  --distributed-timeout "${DIST_TIMEOUT}"
  --auto-start
  --exit-after-run
  --start-delay-seconds "${START_DELAY}"
  --auto-start-timeout "${AUTO_TIMEOUT}"
)
if [[ -n "${MASTER_ADDR}" ]]; then
  leader_base+=(--dist-master-addr "${MASTER_ADDR}")
fi
if [[ -n "${HOST}" ]]; then
  leader_base+=(--host "${HOST}")
fi
leader_base+=(--port "${LEADER_PORT}")

run_exp() {
  local label="$1"
  shift
  printf '\n=== %s ===\n' "${label}"
  "${leader_base[@]}" "$@"
}

start_loopback_worker() {
  if [[ "${LOOPBACK}" -ne 1 ]]; then
    return
  fi
  if [[ -n "${LOCAL_WORKER_PID}" ]] && kill -0 "${LOCAL_WORKER_PID}" 2>/dev/null; then
    return
  fi
  printf '\n=== Starting local loopback worker ===\n'
  "${PROJECT_DIR}/.venv/bin/python" -m dml_cluster.worker \
    --leader 127.0.0.1 \
    --port "${LEADER_PORT}" \
    --project-dir "${PROJECT_DIR}" &
  LOCAL_WORKER_PID="$!"
}

if [[ "${RUN_DATA}" -eq 1 ]]; then
  # Each experiment gets its own dist port (+2 per step) so TCP TIME_WAIT from the
  # previous run cannot block the next one.
  run_exp "EXP 1: solo data-parallel baseline" \
    --dist-port "$((DIST_PORT + 0))" \
    --required-workers 0 \
    --distributed-leader-only \
    --distributed-parallel data \
    --distributed-optimizations none

  start_loopback_worker

  run_exp "EXP 2: two-node data parallel, no optimization" \
    --dist-port "$((DIST_PORT + 2))" \
    --required-workers "${REQUIRED_WORKERS}" \
    --distributed-parallel data \
    --distributed-optimizations none

  run_exp "EXP 3: two-node data parallel, TopK" \
    --dist-port "$((DIST_PORT + 4))" \
    --required-workers "${REQUIRED_WORKERS}" \
    --distributed-parallel data \
    --distributed-optimizations topk \
    --distributed-compress-ratio "${COMPRESS_RATIO}"

  run_exp "EXP 4: two-node data parallel, straggler" \
    --dist-port "$((DIST_PORT + 6))" \
    --required-workers "${REQUIRED_WORKERS}" \
    --distributed-parallel data \
    --distributed-optimizations straggler \
    --distributed-straggler-rank "${STRAGGLER_RANK}" \
    --distributed-straggler-delay "${STRAGGLER_DELAY}"

  run_exp "EXP 5: two-node data parallel, TopK + straggler" \
    --dist-port "$((DIST_PORT + 8))" \
    --required-workers "${REQUIRED_WORKERS}" \
    --distributed-parallel data \
    --distributed-optimizations topk-straggler \
    --distributed-compress-ratio "${COMPRESS_RATIO}" \
    --distributed-straggler-rank "${STRAGGLER_RANK}" \
    --distributed-straggler-delay "${STRAGGLER_DELAY}"
else
  start_loopback_worker
fi

if [[ "${RUN_PIPELINE}" -eq 1 ]]; then
  run_exp "EXP 6: two-node pipeline parallel" \
    --dist-port "$((DIST_PORT + 10))" \
    --required-workers "${REQUIRED_WORKERS}" \
    --distributed-parallel pipeline \
    --distributed-microbatch-size "${MICROBATCH}" \
    --distributed-optimizations none
fi

printf '\n=== Building experiment comparison summary ===\n'
"${PROJECT_DIR}/.venv/bin/python" "${PROJECT_DIR}/experiments/summarize_runs.py" \
  --project-dir "${PROJECT_DIR}" \
  --started-at "${EXPERIMENT_STARTED_AT}"

printf '\n=== All experiments complete. Results are in runs/ ===\n'
