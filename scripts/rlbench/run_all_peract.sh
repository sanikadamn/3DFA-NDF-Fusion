#!/usr/bin/env bash
# Activate the project's 3DFA/NDF + PerAct RLBench environment first.
# Usage: bash scripts/rlbench/run_all_peract.sh
# Requires wget, unzip, Xvfb, CUDA and an installed CoppeliaSim/PyRep/RLBench.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

: "${COPPELIASIM_ROOT:?Set COPPELIASIM_ROOT to your CoppeliaSim installation}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

tasks=(
    place_cups close_jar insert_onto_square_peg light_bulb_in meat_off_grill
    open_drawer place_shape_in_shape_sorter place_wine_at_rack_location
    push_buttons put_groceries_in_cupboard put_item_in_drawer put_money_in_safe
    reach_and_drag slide_block_to_color_target stack_blocks stack_cups
    sweep_to_dustpan_of_size turn_tap
)
dataset=zarr_datasets/peract_all
test_dir=${DATA_DIR:-peract_raw/peract_test}
run_name=denoise3d-PeractTwoCam-C120-B16-lr1e-4-constant-H3-rectified_flow
mkdir -p peract_raw "$dataset" checkpoints logs/peract_all

# Download official packaged demonstrations and held-out test states.
# Keep archives, and mark extraction complete only after unzip succeeds.
if [[ ! -f peract_raw/.all_train_extracted ]]; then
    wget -c -O peract_raw/Peract_packaged.zip \
        https://huggingface.co/katefgroup/3d_diffuser_actor/resolve/main/Peract_packaged.zip
    unzip -o peract_raw/Peract_packaged.zip -d peract_raw
    touch peract_raw/.all_train_extracted
fi
if [[ ! -d "$test_dir" ]]; then
    wget -c -O peract_raw/peract_test.zip \
        https://huggingface.co/katefgroup/3d_flowmatch_actor/resolve/main/peract_test.zip
    unzip -o peract_raw/peract_test.zip -d peract_raw
fi

# One combined dataset for BOTH models, not one dataset/model per task.
python data_processing/peract_to_zarr.py \
    --root peract_raw/Peract_packaged/ --tgt "$dataset/" --tasks "${tasks[@]}"

# Reject incomplete/subset data. Evaluation consumes every stored test episode,
# so exactly 25 per task must be present across all variations.
python - "$dataset" "$test_dir" <<'PY'
import sys
from pathlib import Path
import numpy as np
import zarr
from datasets.rlbench import PERACT_TASKS
for split in ('train', 'val'):
    store = zarr.open_group(str(Path(sys.argv[1]) / f'{split}.zarr'), mode='r')
    assert set(np.unique(store['task_id'][:])) == set(range(len(PERACT_TASKS))), split
    n = len(store['action'])
    assert n > 0 and all(len(store[k]) == n for k in ('rgb', 'pcd', 'proprioception', 'task_id', 'variation')), split
for task in PERACT_TASKS:
    episodes = list((Path(sys.argv[2]) / task).glob('variation*/episodes/episode*'))
    assert len(episodes) == 25, (task, len(episodes))
PY

# Pretrained NDF encoder used by configs/ndf.json.
if [[ ! -f checkpoints/multi_category_weights.pth ]]; then
    wget -O checkpoints/multi_category_weights.pth \
        'https://www.dropbox.com/s/hm4hty56ldu1wb5/multi_category_weights.pth?dl=1'
fi
python -m tools.check_ndf --config "${NDF_CONFIG:-configs/ndf.json}" --device cuda

# Exactly TWO training runs, each trained jointly on all 18 tasks.
for mode in baseline ndf; do
    TASK=all MODE="$mode" DATASET_DIR="$dataset" \
    TRAIN_ITERS=300000 SMOKE_ONLY=0 EXP_NAME="Peract_all_$mode" \
    WANDB_NAME="Peract_all_$mode" \
        bash scripts/rlbench/train_peract_ndf.sh \
        2>&1 | tee -a "logs/peract_all/train_$mode.log"
done

# Same two checkpoints evaluated on every task: 10 seeds x 25 episodes/task.
# Existing launcher uses wrist/front, max_steps=20 and max_tries=2.
for mode in baseline ndf; do
    checkpoint="$PWD/train_logs/Peract_all_$mode/$run_name/best.pth"
    [[ -f "$checkpoint" ]] || { echo "Missing $checkpoint" >&2; exit 1; }
    for seed in {0..9}; do
        CHECKPOINT="$checkpoint" CHECKPOINT_ALIAS=best \
        EXP_NAME="Peract_all_$mode" DATA_DIR="$test_dir" \
        TASKS="${tasks[*]}" SEED="$seed" \
            bash online_evaluation_rlbench/eval_peract.sh \
            2>&1 | tee "logs/peract_all/eval_${mode}_seed${seed}.log"
    done
done

echo 'Done. Results: eval_logs/Peract_all_{baseline,ndf}/best/seed*/<task>/eval.json'
