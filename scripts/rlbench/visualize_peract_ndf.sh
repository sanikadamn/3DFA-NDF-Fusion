#!/usr/bin/env bash
# Plot validation metrics and evaluate/record held-out RLBench rollouts after training.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

task=${TASK:-place_cups}
mode=${MODE:-ndf}
case "$task" in push_buttons|stack_cups|place_cups) ;; *) echo "Unsupported TASK: $task" >&2; exit 2 ;; esac
case "$mode" in baseline|ndf) ;; *) echo "MODE must be baseline or ndf" >&2; exit 2 ;; esac

exp_name=${EXP_NAME:-Peract_${task}_${mode}}
run_name=denoise3d-PeractTwoCam-C120-B16-lr1e-4-constant-H3-rectified_flow
run_dir=${RUN_DIR:-train_logs/$exp_name/$run_name}
vis_dir=${VIS_DIR:-visualizations/peract_${task}_${mode}}
test_dir=${TEST_DIR:-peract_raw/test}
checkpoint=$run_dir/best.pth

# This checkout may share its Zarr datasets with the original 3DFA checkout.
shared_root=$repo_root
if [[ -e "$repo_root/zarr_datasets" ]]; then
    shared_root=$(dirname "$(realpath "$repo_root/zarr_datasets")")
fi
if [[ ! -d "$test_dir" && -z "${TEST_DIR:-}" ]]; then
    test_dir=$shared_root/peract_raw/test
fi

[[ -f "$checkpoint" ]] || { echo "Missing best checkpoint: $checkpoint" >&2; exit 1; }
checkpoint=$(realpath "$checkpoint")
for command in python xvfb-run ffmpeg; do
    command -v "$command" >/dev/null || { echo "Missing command: $command" >&2; exit 1; }
done

mkdir -p "$vis_dir"
python scripts/rlbench/plot_peract_metrics.py \
    --run-dir "$run_dir" --task "$task" \
    --output "$vis_dir/validation_metrics.png"

for variation in 0 1; do
    episode_dir="$test_dir/$task/variation${variation}/episodes/episode0"
    [[ -d "$episode_dir" ]] || { echo "Missing held-out episode: $episode_dir" >&2; exit 1; }
done

if [[ -z "${COPPELIASIM_ROOT:-}" ]]; then
    COPPELIASIM_ROOT=$repo_root/PyRep/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
    if [[ ! -d "$COPPELIASIM_ROOT" ]]; then
        COPPELIASIM_ROOT=$shared_root/PyRep/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
    fi
fi
export COPPELIASIM_ROOT
[[ -d "$COPPELIASIM_ROOT" ]] || { echo "Missing CoppeliaSim: $COPPELIASIM_ROOT" >&2; exit 1; }
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
cudnn_dir=$(python - <<'PY'
from pathlib import Path
import torch
print(Path(torch.__file__).resolve().parent.parent / "nvidia/cudnn/lib")
PY
)
if [[ -d "$cudnn_dir" ]]; then
    export LD_LIBRARY_PATH="$cudnn_dir:$COPPELIASIM_ROOT:${LD_LIBRARY_PATH:-}"
else
    export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:${LD_LIBRARY_PATH:-}"
fi
export PYTHONUNBUFFERED=1

checkpoint_alias="${task}_${mode}_best_$(date -u +%Y%m%dT%H%M%SZ)"
CHECKPOINT="$checkpoint" CHECKPOINT_ALIAS="$checkpoint_alias" \
    DATA_DIR="$test_dir" TASKS="$task" EXP_NAME="$exp_name" \
    bash online_evaluation_rlbench/eval_peract.sh 2>&1 | tee "$run_dir/test_console.log"
test_result="eval_logs/$exp_name/$checkpoint_alias/seed0/$task/eval.json"
[[ -s "$test_result" ]] || { echo "Missing test result: $test_result" >&2; exit 1; }

for variation in 0 1; do
    xvfb-run -a python scripts/rlbench/record_peract_rollout.py \
        --task "$task" --checkpoint "$checkpoint" --data-dir "$test_dir" \
        --variation "$variation" --episode 0 \
        --output-dir "$vis_dir/test_variation${variation}_episode0"
done

echo "Test results: $test_result"
echo "Validation plot: $vis_dir/validation_metrics.png"
echo "Rollout MP4/GIF pairs: $vis_dir/test_variation{0,1}_episode0/"
