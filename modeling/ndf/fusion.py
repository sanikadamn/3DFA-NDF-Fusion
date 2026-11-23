"""Concatenate official frozen NDF descriptors with aligned 3DFA visual tokens."""
from pathlib import Path

import torch
from torch import nn

from .config import NDFConfig
from .vendor.network import VNNOccNet


class FrozenNDFFusion(nn.Module):
    def __init__(self, visual_dim, config, load_pretrained=True):
        super().__init__()
        self.config = config if isinstance(config, NDFConfig) else NDFConfig(**config)
        self.visual_dim = visual_dim
        self.field = VNNOccNet(latent_dim=self.config.latent_dim, model_type="pointnet",
                               return_features=True, acts="all", sigmoid=True,
                               scaling=self.config.scaling)
        if load_pretrained:
            if not self.config.checkpoint or not Path(self.config.checkpoint).is_file():
                raise FileNotFoundError("Set the NDF JSON checkpoint to pretrained NDF weights")
            state = torch.load(self.config.checkpoint, map_location="cpu", weights_only=True)
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model_state_dict" in state:
                state = state["model_state_dict"]
            self.field.load_state_dict(state, strict=True)
        self.field.requires_grad_(False)
        self.field.eval()
        # Official acts='all': (2*L+1) invariant inputs + six L-wide activations.
        self.descriptor_dim = 8 * self.config.latent_dim + 1
        self.fusion = nn.Linear(visual_dim + self.descriptor_dim, visual_dim)
        # Starts as the original visual pathway; NDF weights can learn from step 1.
        with torch.no_grad():
            self.fusion.weight.zero_()
            self.fusion.weight[:, :visual_dim].copy_(torch.eye(visual_dim))
            self.fusion.bias.zero_()

    def train(self, mode=True):
        super().train(mode)
        self.field.eval()  # freeze running statistics as well as weights
        return self

    @torch.no_grad()
    def _prepare_clouds(self, cloud):
        """Sample (B,M,3) clouds, preserving the original finite-point ordering.

        The usual all-finite case uses a single batched gather. If observations
        contain invalid pixels, the fallback preserves the original per-cloud
        filtering and linspace sampling exactly. Each cloud gets its own center.
        """
        if cloud.ndim != 3 or cloud.shape[-1] != 3 or cloud.shape[0] < 1:
            raise ValueError('Expected a nonempty batch of (B,M,3) clouds')
        with torch.autocast(device_type=cloud.device.type, enabled=False):
            cloud = cloud.float()
            valid = torch.isfinite(cloud).all(-1)
            if bool(valid.all()):
                if cloud.shape[1] < 20:
                    raise ValueError('NDF requires at least 20 finite observed points per cloud')
                indices = torch.linspace(0, cloud.shape[1]-1, self.config.num_points,
                                         device=cloud.device).round().long()
                points = cloud[:, indices]
            else:
                sampled = []
                for b in range(cloud.shape[0]):
                    finite_cloud = cloud[b, valid[b]]
                    if len(finite_cloud) < 20:
                        raise ValueError('NDF requires at least 20 finite observed points per cloud')
                    indices = torch.linspace(0, len(finite_cloud)-1, self.config.num_points,
                                             device=cloud.device).round().long()
                    sampled.append(finite_cloud[indices])
                points = torch.stack(sampled, dim=0)
            center = points.mean(1, keepdim=True)
        return points-center, center

    @torch.no_grad()
    def _encode_points(self, centered_points):
        with torch.autocast(device_type=centered_points.device.type, enabled=False):
            return self.field.extract_latent({'point_cloud': centered_points.float()})

    @torch.no_grad()
    def encode_cloud_batch(self, cloud):
        """Encode a caller-sized (B,M,3) cloud batch. Returns latent, (B,1,3) centers."""
        points, center = self._prepare_clouds(cloud)
        return self._encode_points(points), center

    @torch.no_grad()
    def encode_cloud(self, cloud):
        """Keep the original single-cloud API for diagnostics and existing callers."""
        if cloud.ndim != 2 or cloud.shape[-1] != 3:
            raise ValueError('Expected one (M,3) cloud')
        latent, center = self.encode_cloud_batch(cloud.unsqueeze(0))
        return latent, center

    @torch.no_grad()
    def _describe_unchecked(self, latent, center, queries):
        with torch.autocast(device_type=queries.device.type, enabled=False):
            descriptors = self.field.forward_latent(latent, queries.float()-center)
        if descriptors.shape != (*queries.shape[:2], self.descriptor_dim):
            raise ValueError('Unexpected NDF descriptor shape')
        return descriptors

    @torch.no_grad()
    def describe(self, latent, center, queries):
        """Checked public API; forward consolidates finite checks across chunks."""
        if not bool(torch.isfinite(queries).all()):
            raise ValueError('Nonfinite query XYZ in 3DFA scene tokens')
        descriptors = self._describe_unchecked(latent, center, queries)
        if not bool(torch.isfinite(descriptors).all()):
            raise ValueError("Unexpected NDF descriptor shape or nonfinite features")
        return descriptors

    def forward(self, scene_pcd, query_xyz, visual_features):
        """(B,Cam,3,H,W), (B,N,3), (B,N,C) -> (B,N,C).

        Query exactly where the original encoder places each visual token; do not
        pool descriptors into one vector or change positional coordinates.
        cloud_batch_size controls observations per NDF encoder/decoder call.
        query_chunk_size controls queries per observation in each decoder call.
        Neither changes the policy batch, token count, or learned parameter shapes.
        """
        if scene_pcd.ndim != 5 or scene_pcd.shape[2] != 3:
            raise ValueError("Expected scene cloud (B,cameras,3,H,W)")
        if (query_xyz.ndim != 3 or visual_features.ndim != 3
                or query_xyz.shape[:2] != visual_features.shape[:2] or query_xyz.shape[-1] != 3):
            raise ValueError("Queries must align one-to-one with visual tokens")
        if scene_pcd.shape[0] != query_xyz.shape[0] or visual_features.shape[-1] != self.visual_dim:
            raise ValueError("Batch or feature dimension mismatch")
        if query_xyz.shape[0] == 0 or query_xyz.shape[1] == 0:
            raise ValueError('Expected nonempty observations and query tokens')
        if not bool(torch.isfinite(query_xyz).all()):
            raise ValueError('Nonfinite query XYZ in 3DFA scene tokens')
        cloud = scene_pcd.permute(0,1,3,4,2).reshape(scene_pcd.shape[0],-1,3)
        points, centers = self._prepare_clouds(cloud)
        batches = []
        all_finite = torch.ones((), device=query_xyz.device, dtype=torch.bool)
        # Batch observations, including a smaller final group when B is uneven.
        for b in range(0, len(cloud), self.config.cloud_batch_size):
            end = min(b+self.config.cloud_batch_size, len(cloud))
            latent = self._encode_points(points[b:end])
            descriptors_batch = visual_features.new_empty(
                (end-b, query_xyz.shape[1], self.descriptor_dim))
            for start in range(0, query_xyz.shape[1], self.config.query_chunk_size):
                stop = start+self.config.query_chunk_size
                descriptors = self._describe_unchecked(
                    latent, centers[b:end], query_xyz[b:end,start:stop])
                # Keep this reduction on the GPU; synchronize once after all chunks.
                all_finite = all_finite & torch.isfinite(descriptors).all()
                descriptors_batch[:, start:stop] = descriptors.to(visual_features.dtype)
            enlarged = torch.cat((visual_features[b:end], descriptors_batch), dim=-1)
            # OUTSIDE no_grad: train fusion and preserve gradients to visual features.
            batches.append(self.fusion(enlarged))
        if not bool(all_finite):
            raise ValueError('Nonfinite NDF descriptors')
        return torch.cat(batches, dim=0)
