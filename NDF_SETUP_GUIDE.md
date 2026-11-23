# Frozen NDF descriptors + 3DFA: AWS Linux setup and training

## NDF Implementation

For each spatial visual token, compute an NDF descriptor at that token's existing 3D location. Concatenate the two features and project back to the original transformer width.

Original sequence length maintained, new spatially invariant tokens are projected back to the original architecture.

```python
# visual_features: (B, N, 120)
# ndf_descriptors: (B, N, 2049), frozen, queried at corresponding token XYZ
combined = torch.cat([visual_features, ndf_descriptors], dim=-1)  # (B, N, 2169)
fused_features = fusion(combined)                              # (B, N, 120)
# fusion = nn.Linear(2169, 120)
```


The network flow is:

1. Original RGB/CLIP/FPN and vision-language attention produce visual features.
2. Original point-cloud resizing establishes the XYZ corresponding to each token.
3. The frozen NDF network encodes sampled observed geometry and describes those XYZ queries.
4. A trainable concatenation/projection layer produces the original-width features.
5. Original 3DFA predicts actions with its original loss, optimizer settings and flow-matching procedure.

NDF encoding and feature fusion occur once per observation forward pass, before the denoising loop. They are reused across all five denoising steps in that pass. No target action, task success label, or simulator ground-truth object pose is used to construct the descriptors.


The fusion layer is new trainable model capacity. Its learning rate and optimizer treatment use the original optimizer automatically.


Something to note: Frozen NDF trained on mug, bowl and bottle geometry. Should help for place cups (hopefully).



Updated files:

`main.py`: updated it to add the ndf argument \
`modeling/policy/denoise_actor_3d.py`: ndf confit, ndf pretrained arguments added \
`modeling/encoder/multimodal/encoder3d.py`: the ndf integration \
`utils/trainers/base.py`: ndf integration in the training loop \
`online_evaluation_rlbench/evaluate_policy.py`: ndf integration in the evaluation loop \
`modeling/ndf/fusion.py`: the fusion layer implementation, batching added \
`modeling/ndf/config.py`: the NDF configuration file



```shell
# Miniconda installation
mkdir -p "$HOME/installers"

curl -fL \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  -o "$HOME/installers/miniconda.sh"

bash "$HOME/installers/miniconda.sh" \
  -b -p "$HOME/miniconda3"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
```

```shell
# Required dependences
python -m pip install --upgrade pip

python -m pip install \
  torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install -r requirements-ndf.txt

python -m pip install \
  'git+https://github.com/openai/CLIP.git'

python -m pip install -e .
python -m pip check
```

```shell
# NDF weights
bash scripts/rlbench/download_ndf_weights.sh

python -m tools.check_ndf --device cuda

python -m unittest discover -s tests -v
```

```shell
# Push button task
bash scripts/rlbench/peract_datagen.sh push_buttons

mkdir -p zarr_datasets/peract_push_buttons

python -m data_processing.peract_to_zarr \
  --root peract_raw/Peract_packaged/ \
  --tgt zarr_datasets/peract_push_buttons/ \
  --tasks push_buttons
  
python -m tools.check_data \
  --dataset-dir zarr_datasets/peract_push_buttons \
  --tasks push_buttons

# Check data before training
CUDA_VISIBLE_DEVICES=0 \
TASK=push_buttons \
SMOKE_ONLY=1 \
bash scripts/rlbench/train_peract_ndf.sh
```

```shell
# Training
CUDA_VISIBLE_DEVICES=0 \
TASK=push_buttons \
bash scripts/rlbench/train_peract_ndf.sh
```

```shell
# Baseline
CUDA_VISIBLE_DEVICES=0 \
TASK=push_buttons \
MODE=baseline \
bash scripts/rlbench/train_peract_ndf.sh
```

For a different task, replace `push_buttons` with the desired task name in the above commands.
