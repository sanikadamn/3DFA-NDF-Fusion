# Source and license

Copied from the supplied `ndf_robot-master (1).zip` (SHA-256 in docs/source_manifest.json):

- `src/ndf_robot/model/vnn_occupancy_net_pointnet_dgcnn.py` -> `network.py`.
- `src/ndf_robot/model/layers_equi.py` -> `layers_equi.py`.
- `LICENSE.md` copied unchanged.

Only two compatibility edits were made: the network's layer import is package-relative,
and four `torch.device('cuda')` allocations use `x.device` for CPU/current-GPU support.
No parameter names or neural-network computations were changed. The official
PointNet-based Vector Neuron encoder and occupancy decoder are used with
`latent_dim=256, return_features=True, acts='all', sigmoid=True, scaling=10`.

Original repository: https://github.com/anthonysimeonov/ndf_robot
MIT copyright: Anthony Simeonov and Yilun Du, 2022. See LICENSE.md.
No upstream Git commit ID is asserted for the uploaded archive.
