# All-task baseline/NDF pipeline

From the repository root, in the installed 3DFA/NDF environment:

```bash
export COPPELIASIM_ROOT=/path/to/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
python scripts/rlbench/run_all_peract_pipeline.py --dry-run
python scripts/rlbench/run_all_peract_pipeline.py
```

Requires CUDA, curl, Xvfb, the PerAct-compatible RLBench/PyRep installation,
and the project's training/NDF dependencies. This does not install those dependencies.
Ensure adequate disk space for the full raw archives, extracted demonstrations,
one combined converted dataset, and two model runs. Downloaded archives are retained.

The pipeline:

1. Downloads the official packaged PerAct demonstrations and clean test states.
2. Verifies exactly 25 held-out episodes per task across variations.
3. Downloads the default NDF encoder weights and checks NDF functionality.
4. Converts all 18 tasks into one combined dataset at `zarr_datasets/peract_all`
   and verifies that both splits contain all 18 task IDs. A separate path avoids
   accidentally reusing the existing two-task `zarr_datasets/peract` dataset.
5. Trains **exactly two multi-task models: one baseline and one NDF**, each to
   300,000 iterations over the combined dataset, using `PeractTwoCam`: wrist and front.
   Checkpoints go to `train_logs/Peract_all_baseline/` and
   `train_logs/Peract_all_ndf/`. Stored Zarr data retains
   all four original cameras; the dataset selects the two used by the model.
6. Evaluates validation-selected `best.pth` checkpoints with seeds 0–9, each
   replaying the 25 held-out episodes: 250 rollouts/model/task, 9,000 total.
7. Writes per-task/model means and sample standard deviations to
   `eval_logs/peract_multitask_250/summary.json` (override with `--output`).

Evaluation matches upstream `online_evaluation_rlbench/eval_peract.sh` settings:
20 steps, 2 action attempts, no collision checking, 256x256 images. Upstream's
launcher uses four cameras and seed 0; this pipeline intentionally uses two
cameras and expands to ten seeds. These are seeded policy evaluations of the
same held-out initial states, not 250 newly sampled initial states.

`CUDA_VISIBLE_DEVICES`, `NGPUS`, `NDF_CONFIG`, and W&B environment settings are
inherited. Runs execute sequentially. Training restores `last.pth` through the
existing launcher. Successful conversion/training stages get completion markers;
existing evaluation JSON files are reused. Use a new `--output` directory after
changing checkpoints to avoid reusing old evaluations.

For safety, existing Zarr stores without `.pipeline_conversion_complete` stop
the pipeline. Validate that both splits are complete, contain all 18 tasks,
and retain the original camera layout before creating that marker in
`zarr_datasets/peract_all/`. Do not mark interrupted
conversions as complete. Invalid/partial downloads fail validation and need to be
re-downloaded explicitly. The pipeline never deletes or moves files.

Upstream reference:
https://github.com/nickgkan/3d_flowmatch_actor/blob/master/online_evaluation_rlbench/eval_peract.sh
