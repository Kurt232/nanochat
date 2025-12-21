#!/bin/bash

set -e

WANDB_RUN=d20 TAG=d20 bash speedrun.sh
WANDB_RUN=d20-hg TAG=d20-headwise_gated bash speedrun.sh
# WANDB_RUN=d20-eg TAG=d20-elementwise_gated bash speedrun.sh # OOM
