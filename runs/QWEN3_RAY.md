# Qwen3 topologies on a 4x8 Ray cluster

This run uses Qwen3-0.6B's Transformer topology with nanochat's own 32K
tokenizer, ClimbMix loader, training loop, checkpoint format, and MuonAdamW.
The smaller vocabulary makes the model **474,021,888 unique parameters**;
the official checkpoint is about 0.6B largely because its tied embedding has
151,936 rows.

## Prerequisites

- 4 Ray nodes with 8 GPUs each and the same CUDA/PyTorch environment.
- The repository available to every worker (a Ray Job `--working-dir` is fine).
- Node-local storage at `/opt/tiger/nanochat_runtime` (the default).
- HDFS access through `/opt/tiger/nastk/bin/nastk`. Tokenizer and checkpoints
  are durably synchronized to
  `hdfs://harunavaali/home/byte_search_aisearch_strategy/wenjiedu/nanochat`.

Install the GPU environment and Ray Train on every node/image:

```bash
uv sync --extra gpu
uv pip install 'ray[train]>=2.40,<3'
```

The Ray launcher prepares disjoint node-local ClimbMix shards and trains the
nanochat tokenizer once. The 0.6B run uses 900 total shards. The subsequent 1B
run expands that cache to 1,500 total shards while reusing the exact tokenizer.

```bash
export NANOCHAT_BASE_DIR=/opt/tiger/nanochat_runtime
```

## Launch

From the Ray head node:

```bash
export NANOCHAT_BASE_DIR=/shared/nanochat
bash runs/qwen3_0_6b_30b_ray.sh
```

Or submit the same command through the Ray Jobs API:

```bash
ray job submit \
  --address http://127.0.0.1:8265 \
  --working-dir . \
  --runtime-env-json='{"env_vars":{"NANOCHAT_BASE_DIR":"/shared/nanochat"}}' \
  -- bash runs/qwen3_0_6b_30b_ray.sh
```

The default run uses sequence length 4096, per-GPU batch 2, global batch
2,097,152 tokens, and 8 gradient-accumulation steps. It runs 14,305 optimizer
steps, consuming 29,999,759,360 tokens (240,640 fewer than 30B, or 0.0008%).
Set `DEVICE_BATCH_SIZE=1` if memory is tight; the global batch and token horizon
remain unchanged and gradient accumulation adjusts automatically.

## Sequential 0.6B then 1B run

`scripts.train_1b_after_06b` waits for the complete 0.6B final checkpoint on
HDFS, compacts intermediate checkpoints, then launches `qwen3-1b-50b`:

```bash
python -m scripts.train_1b_after_06b
```

The 1B topology has hidden size 1,536, MLP size 5,376, 28 layers, 16 query
heads, 8 KV heads, and head dimension 128. With the 32K tied tokenizer it has
1,008,300,544 unique parameters. It consumes 50,000,297,984 tokens over 23,842
steps. Checkpoints are written at steps 5K, 10K, 15K, 20K, and the final step;
only the newest two are retained on node-local disks during training, and only
the complete final checkpoint remains after successful HDFS compaction.
