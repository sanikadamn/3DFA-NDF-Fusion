# Changes relative to the uploaded 3dfa_se3-master.zip

## Modified original files

| File | Change |
|---|---|
| `main.py` | Adds optional `--ndf_config`; all original argument defaults remain unchanged. |
| `modeling/policy/denoise_actor_3d.py` | Threads optional NDF settings into the encoder. |
| `modeling/encoder/multimodal/encoder3d.py` | Retains full observed cloud, queries frozen NDF at the existing token XYZ, and fuses concatenated descriptors/visual features back to embedding_dim. |
| `utils/trainers/base.py` | Constructs optional NDF, saves run/configuration metadata, restores checkpoints with compatibility checks, and guards absent checkpoint paths. The training loop, optimizer, loss, precision, augmentation, loaders and normalizer math are unchanged. |
| `online_evaluation_rlbench/evaluate_policy.py` | Restores architecture/camera/NDF settings from new checkpoints before environment selection; strips only an actual DDP prefix and strictly loads new checkpoints. |

## Added files

- `modeling/ndf/fusion.py`: actual official-model adapter; 2169-to-120 trainable feature fusion, identity visual initialization, frozen NDF eval/no_grad/optimizer exclusion, observation/query chunking.
- `modeling/ndf/config.py`: validated configuration, metadata and resume checks.
- `modeling/ndf/vendor/`: two supplied official NDF source files, package marker, original license and provenance. Only a relative import and four input-device allocations changed in vendored code.
- `modeling/ndf/__init__.py`: package marker.
- `configs/ndf.json`: weight path and NDF-only settings.
- `scripts/rlbench/train_peract_ndf.sh`: original launcher parameters with additional NDF argument and configurable task directory/mode/GPU count. The original launcher remains untouched.
- `scripts/rlbench/download_ndf_weights.sh`: downloads official pretrained weights to the configured location.
- `tools/check_ndf.py`: real pretrained weights/dimensions/gradient/invariance check.
- `tools/check_data.py`: task ID, array shape and instruction coverage check for task-specific Zarr stores.
- `tools/smoke_train.py`: actual real-batch DDP training step, frozen-state assertions and inference; no policy checkpoint mutation.
- `tools/__init__.py`: package marker.
- `tests/`: configuration/resume tests, actual launcher comparison, tensor-level frozen-fusion tests.
- `environment-ndf.yml`, `requirements-ndf.txt`: proposed versioned setup recipe (not a fully validated environment lock).
- `NDF_SETUP_GUIDE.md`: detailed AWS Linux, Miniconda, code edits, data, training and evaluation instructions.
- `docs/CHANGES.md`, `docs/VALIDATION.md`, `docs/source_manifest.json`, `docs/ndf_fusion.patch`: review/provenance artifacts.

No original files are deleted. Data conversion and dataset classes are unchanged:
this uploaded baseline already supports selecting arbitrary canonical PerAct task
subsets at conversion time. The task name controls paths in the new launcher, not
sample filtering. The new learned fusion can change density-sampler selections
through changed features; the sampler algorithm itself is unchanged.

This version uses per-token **feature concatenation** rather than appending extra
NDF tokens. It implements the user's current request for a larger encoder input
while preserving the transformer dimension and baseline training hyperparameters.
