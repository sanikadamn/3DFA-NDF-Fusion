#!/usr/bin/env bash
# Evaluate teammate checkpoints with the local place_cups evaluation protocol.
# Run from the same activated environment used for baseline/NDF evaluation.
# Optional positional arguments: checkpoint paths (defaults to all bundle *.pth).
# DATASET=Peract reproduces the older four-camera runs; default matches *_twocam.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
shared_root=$(dirname "$(realpath "$repo_root/zarr_datasets")")
bundle=${BUNDLE_DIR:-$repo_root/../peract_official_2cam_local_eval}
test_dir=${DATA_DIR:-$shared_root/peract_raw/test}
dataset=${DATASET:-PeractTwoCam}
seed=${SEED:-0}
[[ "$seed" =~ ^[0-9]+$ ]] || { echo 'SEED must be a nonnegative integer' >&2; exit 2; }
case "$dataset" in Peract|PeractTwoCam) ;; *) echo 'DATASET must be Peract or PeractTwoCam' >&2; exit 2 ;; esac
if (( $# )); then
    checkpoints=("$@")
else
    shopt -s nullglob
    checkpoints=("$bundle"/checkpoints/*.pth)
fi
(( ${#checkpoints[@]} )) || { echo "No checkpoints in $bundle/checkpoints" >&2; exit 1; }
for checkpoint in "${checkpoints[@]}"; do
    [[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 1; }
done
for cmd in python xvfb-run; do command -v "$cmd" >/dev/null; done
for variation in 0 1; do
    episode="$test_dir/place_cups/variation$variation/episodes/episode0"
    [[ -d "$episode" ]] || { echo "Missing test episode: $episode" >&2; exit 1; }
done
export COPPELIASIM_ROOT=${COPPELIASIM_ROOT:-$shared_root/PyRep/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04}
[[ -d "$COPPELIASIM_ROOT" ]] || { echo "Missing simulator: $COPPELIASIM_ROOT" >&2; exit 1; }
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
cudnn_dir=$(python - <<'PY'
from pathlib import Path
import torch
print(Path(torch.__file__).resolve().parent.parent / 'nvidia/cudnn/lib')
PY
)
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:${LD_LIBRARY_PATH:-}"
if [[ -d "$cudnn_dir" ]]; then export LD_LIBRARY_PATH="$cudnn_dir:$LD_LIBRARY_PATH"; fi
# Ensure imports use this checkout, not the teammate's bundled source.
export PYTHONPATH="$repo_root:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
output_root=${OUTPUT_ROOT:-eval_logs/Peract_place_cups_teammate/$(date -u +%Y%m%dT%H%M%SZ)_$$}
mkdir -p "$output_root"
index=0
for checkpoint in "${checkpoints[@]}"; do
    index=$((index + 1))
    checkpoint=$(realpath "$checkpoint")
    alias="${index}_$(basename "${checkpoint%.pth}")_${dataset}"
    run_dir="$output_root/$alias"
    mkdir -p "$run_dir"
    args=(
        --checkpoint "$checkpoint" --task place_cups
        --max_tries 2 --max_steps 20 --headless true
        --collision_checking false --seed "$seed"
        --data_dir "$test_dir" --dataset "$dataset" --image_size 256,256
        --output_file "$run_dir/seed$seed/place_cups/eval.json"
        --model_type denoise3d --bimanual false --prediction_len 1
        --backbone clip --fps_subsampling_factor 4
        --embedding_dim 120 --num_attn_heads 8 --num_vis_instr_attn_layers 2
        --num_history 3 --num_shared_attn_layers 4 --relative_action false
        --rotation_format quat_xyzw --denoise_timesteps 5 --denoise_model rectified_flow
    )
    # Architecture defaults above also match the bundle's config/train_args.json.
    printf '%q ' xvfb-run -a python online_evaluation_rlbench/evaluate_policy.py "${args[@]}" > "$run_dir/command.txt"
    printf '\n' >> "$run_dir/command.txt"
    echo "Evaluating $checkpoint ($dataset); output: $run_dir"
    xvfb-run -a python online_evaluation_rlbench/evaluate_policy.py "${args[@]}" 2>&1 | tee "$run_dir/console.log"
    [[ -s "$run_dir/seed$seed/place_cups/eval.json" ]] || { echo 'Evaluation produced no result' >&2; exit 1; }
    python online_evaluation_rlbench/collect_results.py --folder "$run_dir/seed$seed/" 2>&1 | tee "$run_dir/summary.log"
done
echo "Results: $output_root"
