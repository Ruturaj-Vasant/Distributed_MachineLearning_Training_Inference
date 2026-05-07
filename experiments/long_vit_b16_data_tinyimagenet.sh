#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[experiments] long ViT-B/16 TinyImageNet data-parallel matrix"
echo "[experiments] default global batch target: 16 with leader + 1 worker via base batch 8"

exec "${SCRIPT_DIR}/run_experiments.sh" \
  --model vit_b_16 \
  --dataset tiny-imagenet-200 \
  --image-size 224 \
  --samples "${SAMPLES:-100000}" \
  --batches "${BATCHES:-6250}" \
  --eval-batches "${EVAL_BATCHES:-20}" \
  --epochs "${EPOCHS:-1}" \
  --base-batch "${BASE_BATCH:-8}" \
  --microbatch "${MICROBATCH:-2}" \
  --host "${HOST:-0.0.0.0}" \
  --master-addr "${MASTER_ADDR:-leader-macbook-pro.taila5426e.ts.net}" \
  --leader-port "${LEADER_PORT:-8787}" \
  --dist-port "${DIST_PORT:-29501}" \
  --required-workers "${REQUIRED_WORKERS:-1}" \
  --straggler-delay "${STRAGGLER_DELAY:-3.0}" \
  --skip-pipeline
