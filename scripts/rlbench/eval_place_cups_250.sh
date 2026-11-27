#!/usr/bin/env bash
# 250 rollouts per model: 25 held-out episodes at each of seeds 0..9.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
root=${OUTPUT_ROOT:-$repo_root/eval_logs/place_cups_250_$(date -u +%Y%m%dT%H%M%SZ)}
run_name=denoise3d-PeractTwoCam-C120-B16-lr1e-4-constant-H3-rectified_flow
for mode in baseline ndf; do
    checkpoint="$repo_root/train_logs/Peract_place_cups_$mode/$run_name/best.pth"
    [[ -f "$checkpoint" ]] || { echo "Missing $checkpoint" >&2; exit 1; }
done
mkdir -p "$root"
echo "Started $(date -Is); results: $root"
for mode in baseline ndf; do
    checkpoint="$repo_root/train_logs/Peract_place_cups_$mode/$run_name/best.pth"
    for seed in {0..9}; do
        echo "Starting $mode seed $seed at $(date -Is)"
        SEED="$seed" DATASET=PeractTwoCam OUTPUT_ROOT="$root/$mode/seed$seed" \
            bash scripts/rlbench/eval_teammate_place_cups.sh "$checkpoint"
    done
done
python - "$root" <<'PY'
import json, statistics, sys
from pathlib import Path
root = Path(sys.argv[1])
summary = {}
for mode in ('baseline', 'ndf'):
    files = sorted((root / mode).glob('seed*/*/seed*/place_cups/eval.json'))
    assert len(files) == 10, (mode, len(files))
    rates = [json.loads(p.read_text())['place_cups']['mean'] for p in files]
    summary[mode] = dict(episodes=250, seeds=10, mean_percent=100*statistics.mean(rates),
                         sample_sd_percent=100*statistics.stdev(rates), per_seed_rates=rates)
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
PY
echo "Finished $(date -Is); summary: $root/summary.json"
