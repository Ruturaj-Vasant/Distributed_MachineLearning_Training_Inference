#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[experiments] smoke ViT-B/16 TinyImageNet data-parallel matrix"
echo "[experiments] expected world size: leader + ${REQUIRED_WORKERS:-1} worker(s)"

exec "${SCRIPT_DIR}/run_experiments.sh" \
  --model vit_b_16 \
  --dataset tiny-imagenet-200 \
  --image-size 224 \
  --samples "${SAMPLES:-128}" \
  --batches "${BATCHES:-2}" \
  --eval-batches "${EVAL_BATCHES:-1}" \
  --epochs "${EPOCHS:-1}" \
  --base-batch "${BASE_BATCH:-2}" \
  --microbatch "${MICROBATCH:-1}" \
  --host "${HOST:-0.0.0.0}" \
  --master-addr "${MASTER_ADDR:-leader-macbook-pro.taila5426e.ts.net}" \
  --leader-port "${LEADER_PORT:-8787}" \
  --dist-port "${DIST_PORT:-29501}" \
  --required-workers "${REQUIRED_WORKERS:-1}" \
  --straggler-delay "${STRAGGLER_DELAY:-0.1}" \
  --skip-pipeline
