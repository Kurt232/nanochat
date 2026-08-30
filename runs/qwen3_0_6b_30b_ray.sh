#!/usr/bin/env bash
set -eo pipefail

# Load the cluster's outbound proxy settings used for PyPI and Hugging Face.
# shellcheck disable=SC1090
source "${HOME}/.bashrc" >/dev/null 2>&1 || true
set -u

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/opt/tiger/nanochat_runtime}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
storage_uri="${NANOCHAT_STORAGE_URI:-hdfs://harunavaali/home/byte_search_aisearch_strategy/wenjiedu/nanochat}"

python -m scripts.ray_train \
    --address "${RAY_ADDRESS:-auto}" \
    --num-workers 32 \
    --cpus-per-worker "${CPUS_PER_WORKER:-2}" \
    --run-name qwen3-0.6b-30b \
    --prepare-data-shards "${PREPARE_DATA_SHARDS:-900}" \
    --download-workers "${DOWNLOAD_WORKERS:-8}" \
    --local-base-dir "${NANOCHAT_BASE_DIR}" \
    --storage-uri "${storage_uri}" \
    -- \
    --model-architecture qwen3-0.6b \
    --model-tag qwen3-0.6b-30b \
    --run qwen3-0.6b-30b \
    --max-seq-len 4096 \
    --window-pattern L \
    --device-batch-size "${DEVICE_BATCH_SIZE:-2}" \
    --total-batch-size 2097152 \
    --target-tokens 30000000000 \
    --embedding-lr 0.0003 \
    --scalar-lr 0.0003 \
    --matrix-lr 0.02 \
    --warmup-steps 200 \
    --eval-every 500 \
    --eval-tokens 4194304 \
    --core-metric-every -1 \
    --sample-every -1 \
    --save-every 500
