#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

mkdir -p logs

USE_WANDB=1 \
WANDB_ENTITY="ssdamle-carnegie-mellon-university" \
WANDB_PROJECT="Robot Learning 3DFA" \
WANDB_RUN_GROUP="place_cups" \
WANDB_MODE=online \
TASK=place_cups \
DATASET_DIR=zarr_datasets/peract_place_cups \
CUDA_VISIBLE_DEVICES=0 \
NGPUS=1 \
MODE=baseline \
EXP_NAME=Peract_place_cups_baseline \
WANDB_NAME=place_cups-baseline \
bash scripts/rlbench/train_peract_ndf.sh \
2>&1 | tee -a logs/place_cups_baseline.log

USE_WANDB=1 \
WANDB_ENTITY="ssdamle-carnegie-mellon-university" \
WANDB_PROJECT="Robot Learning 3DFA" \
WANDB_RUN_GROUP="place_cups" \
WANDB_MODE=online \
TASK=place_cups \
DATASET_DIR=zarr_datasets/peract_place_cups \
CUDA_VISIBLE_DEVICES=0 \
NGPUS=1 \
MODE=ndf \
NDF_CONFIG=configs/ndf.json \
EXP_NAME=Peract_place_cups_ndf \
WANDB_NAME=place_cups-ndf \
bash scripts/rlbench/train_peract_ndf.sh \
2>&1 | tee -a logs/place_cups_ndf.log
