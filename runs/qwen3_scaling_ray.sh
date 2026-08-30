#!/usr/bin/env bash
set -uo pipefail

# Small Qwen3 miniseries at approximately 50 training tokens per parameter.
# Each point uses the same tokenizer, data, optimizer, context length and global
# batch as the completed 0.6B/30B and 1B/50B runs. Only final model+meta files
# are retained; optimizer shards are intentionally omitted.

# shellcheck disable=SC1090
source "${HOME}/.bashrc" >/dev/null 2>&1 || true

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/opt/tiger/nanochat_runtime}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
storage_uri="${NANOCHAT_STORAGE_URI:-hdfs://harunavaali/home/byte_search_aisearch_strategy/wenjiedu/nanochat}"
results_dir="${NANOCHAT_BASE_DIR}/qwen3_scaling_results"
mkdir -p "${results_dir}"

variants=(qwen3-scale-48m qwen3-scale-131m qwen3-scale-285m)
tags=(qwen3-scale-48m-2.4b qwen3-scale-131m-6.6b qwen3-scale-285m-14.3b)
target_tokens=(2412262400 6567820800 14262528000)

remote_complete() {
    local tag="$1"
    hdfs dfs -ls "${storage_uri}/base_checkpoints/${tag}/meta_*.json" >/dev/null 2>&1 &&
        hdfs dfs -ls "${storage_uri}/base_checkpoints/${tag}/model_*.pt" >/dev/null 2>&1
}

for index in "${!variants[@]}"; do
    variant="${variants[$index]}"
    tag="${tags[$index]}"
    tokens="${target_tokens[$index]}"
    log_file="${results_dir}/${tag}.log"

    if remote_complete "${tag}"; then
        echo "Skipping completed ${tag}"
        continue
    fi

    echo "Starting ${tag}: ${variant}, target_tokens=${tokens}"
    set +e
    python -m scripts.ray_train \
        --address "${RAY_ADDRESS:-auto}" \
        --num-workers 32 \
        --cpus-per-worker "${CPUS_PER_WORKER:-2}" \
        --run-name "${tag}" \
        --prepare-data-shards 0 \
        --local-base-dir "${NANOCHAT_BASE_DIR}" \
        --storage-uri "${storage_uri}" \
        -- \
        --model-architecture "${variant}" \
        --model-tag "${tag}" \
        --run "${tag}" \
        --max-seq-len 4096 \
        --window-pattern L \
        --device-batch-size "${DEVICE_BATCH_SIZE:-2}" \
        --total-batch-size 2097152 \
        --target-tokens "${tokens}" \
        --embedding-lr 0.0003 \
        --scalar-lr 0.0003 \
        --matrix-lr 0.02 \
        --warmup-steps 200 \
        --eval-every 999999 \
        --eval-tokens 4194304 \
        --core-metric-every -1 \
        --sample-every -1 \
        --save-every -1 \
        --no-save-optimizer \
        2>&1 | tee "${log_file}"
    status=${PIPESTATUS[0]}
    set -e

    if ! remote_complete "${tag}"; then
        echo "${tag} did not produce a complete final model (trainer status=${status})" >&2
        exit "${status:-1}"
    fi
    echo "Completed ${tag} (trainer status=${status})"
done

echo "Qwen3 scaling miniseries complete"
