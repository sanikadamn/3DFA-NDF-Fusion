#!/usr/bin/env bash
# Train separate baseline and NDF models for all 18 PerAct tasks (36 runs).
# Usage: bash scripts/rlbench/train_all_peract_ndf.sh
# Data: ${DATASET_ROOT:-zarr_datasets}/peract_<task>/{train,val}.zarr
# Optional: CUDA_VISIBLE_DEVICES, NGPUS, NDF_CONFIG, LOG_DIR, EXP_PREFIX,
#           and the usual WANDB_* / USE_WANDB environment variables.
# DRY_RUN=1 prints the run plan without requiring data or starting training.
# The delegated launcher uses PeractTwoCam (wrist/front), and resumes last.pth
# when present. Runs are sequential; any failure stops the launcher.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

tasks=(
    place_cups close_jar insert_onto_square_peg
    light_bulb_in meat_off_grill open_drawer
    place_shape_in_shape_sorter place_wine_at_rack_location
    push_buttons put_groceries_in_cupboard
    put_item_in_drawer put_money_in_safe reach_and_drag
    slide_block_to_color_target stack_blocks stack_cups
    sweep_to_dustpan_of_size turn_tap
)
dataset_root=${DATASET_ROOT:-zarr_datasets}
log_dir=${LOG_DIR:-logs/peract_all_two_cam}
exp_prefix=${EXP_PREFIX:-Peract}
ndf_config=${NDF_CONFIG:-configs/ndf.json}
dry_run=${DRY_RUN:-0}

# Check every dataset up front so missing tasks do not surface days later.
if [[ "$dry_run" != 1 ]]; then
    missing=0
    for task in "${tasks[@]}"; do
        for split in train val; do
            path="$dataset_root/peract_$task/$split.zarr"
            if [[ ! -d "$path" ]]; then
                printf 'Missing dataset: %s\n' "$path" >&2
                missing=1
            fi
        done
    done
    if [[ ! -f "$ndf_config" ]]; then
        printf 'Missing NDF config: %s\n' "$ndf_config" >&2
        missing=1
    fi
    if (( missing )); then exit 1; fi
    mkdir -p "$log_dir"
fi

for task in "${tasks[@]}"; do
    for mode in baseline ndf; do
        exp_name="${exp_prefix}_${task}_${mode}"
        printf 'Training task=%s mode=%s cameras=wrist,front iterations=300000 experiment=%s data=%s\n' \
            "$task" "$mode" "$exp_name" "$dataset_root/peract_$task"
        if [[ "$dry_run" == 1 ]]; then continue; fi
        TASK="$task" \
        MODE="$mode" \
        DATASET_DIR="$dataset_root/peract_$task" \
        TRAIN_ITERS=300000 \
        SMOKE_ONLY=0 \
        EXP_NAME="$exp_name" \
        NDF_CONFIG="$ndf_config" \
        WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-$task}" \
        WANDB_NAME="$exp_name" \
            bash scripts/rlbench/train_peract_ndf.sh \
            2>&1 | tee -a "$log_dir/${task}_${mode}.log"
    done
done
