"""Benchmark NDF + fusion only, not complete 3DFA training or data loading.

Uses pretrained weights from --config and synthetic observation tensors.
Run from the project root: python -m tools.benchmark_ndf_batching --help
"""
import argparse
from contextlib import nullcontext
from dataclasses import replace
import json
import time

import torch

from modeling.ndf.config import NDFConfig
from modeling.ndf.fusion import FrozenNDFFusion


def legacy_forward(model, scene_pcd, query_xyz, visual_features):
    """Original sequential loop, including per-chunk validation and fusion."""
    cloud = scene_pcd.permute(0, 1, 3, 4, 2).reshape(scene_pcd.shape[0], -1, 3)
    batches = []
    for b in range(len(cloud)):
        # Reproduce the old preprocessing too, rather than using the new fast path.
        with torch.no_grad(), torch.autocast(device_type=cloud.device.type, enabled=False):
            finite_cloud = cloud[b].float()
            finite_cloud = finite_cloud[torch.isfinite(finite_cloud).all(-1)]
            if len(finite_cloud) < 20:
                raise ValueError('NDF requires at least 20 finite observed points')
            indices = torch.linspace(0, len(finite_cloud)-1, model.config.num_points,
                                     device=cloud.device).round().long()
            points = finite_cloud[indices].unsqueeze(0)
            center = points.mean(1, keepdim=True)
            latent = model.field.extract_latent({'point_cloud': points-center})
        chunks = []
        for start in range(0, query_xyz.shape[1], model.config.query_chunk_size):
            stop = start+model.config.query_chunk_size
            descriptors = model.describe(latent, center, query_xyz[b:b+1, start:stop])
            visual = visual_features[b:b+1, start:stop]
            chunks.append(model.fusion(torch.cat((visual, descriptors.to(visual.dtype)), -1)))
        batches.append(torch.cat(chunks, dim=1))
    return torch.cat(batches, dim=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/ndf.json')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--cameras', type=int, default=2)
    parser.add_argument('--height', type=int, default=128)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--tokens', type=int, default=512,
                        help='Synthetic visual-token count; set to your actual count for closer timing')
    parser.add_argument('--cloud-batches', type=int, nargs='+', default=[1, 4, 8])
    parser.add_argument('--query-chunk-size', type=int, default=256)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--amp', choices=['bf16', 'none'], default='bf16')
    args = parser.parse_args()
    for name in ('batch_size', 'cameras', 'height', 'width', 'tokens', 'steps',
                 'query_chunk_size'):
        if getattr(args, name) < 1:
            parser.error(f'--{name.replace("_", "-")} must be positive')
    if args.warmup < 0 or any(b < 1 for b in args.cloud_batches):
        parser.error('warmup must be nonnegative; cloud batches must be positive')
    device = torch.device(args.device)
    if device.type not in ('cuda', 'cpu'):
        parser.error('Use a CUDA or CPU device')
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; use --device cpu --amp none for a CPU check')
    torch.manual_seed(23)
    base_config = NDFConfig.read(args.config)
    model = FrozenNDFFusion(120, base_config).to(device).train()
    # Activate descriptor columns for the equivalence check; no weights are saved.
    with torch.no_grad():
        model.fusion.weight[:, 120:].normal_(0, .01)
    pcd = torch.randn(args.batch_size, args.cameras, 3, args.height,
                       args.width, device=device) * .08
    queries = torch.randn(args.batch_size, args.tokens, 3, device=device) * .08
    visual_dtype = torch.bfloat16 if args.amp == 'bf16' else torch.float32
    visual = torch.randn(args.batch_size, args.tokens, 120, device=device,
                         dtype=visual_dtype, requires_grad=True)

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def amp_context():
        return (torch.autocast(device_type=device.type, dtype=torch.bfloat16)
                if args.amp == 'bf16' else nullcontext())

    # Compare float32 outputs to avoid BF16 reduction-rounding obscuring errors.
    with torch.no_grad():
        reference = legacy_forward(model, pcd, queries, visual.float())
        for b in args.cloud_batches:
            model.config = replace(base_config, cloud_batch_size=b,
                                   query_chunk_size=args.query_chunk_size)
            actual = model(pcd, queries, visual.float())
            torch.testing.assert_close(actual, reference, atol=1e-4, rtol=1e-3)
    del actual, reference
    print('Float32 output equivalence: passed', flush=True)
    print('Synthetic NDF + fusion forward/backward timings (not full training):', flush=True)

    runs = [('legacy_sequential', None)] + [('batched', b) for b in args.cloud_batches]
    for name, b in runs:
        model.config = (base_config if b is None else
                        replace(base_config, cloud_batch_size=b,
                                query_chunk_size=args.query_chunk_size))

        def step():
            model.zero_grad(set_to_none=True)
            visual.grad = None
            with amp_context():
                out = (legacy_forward(model, pcd, queries, visual) if b is None
                       else model(pcd, queries, visual))
                loss = out.float().square().mean()
            loss.backward()

        for _ in range(args.warmup):
            step()
        sync()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(args.steps):
            step()
        sync()
        elapsed = time.perf_counter()-start
        print(json.dumps({
            'implementation': name, 'cloud_batch_size': 1 if b is None else b,
            'query_chunk_size': model.config.query_chunk_size,
            'seconds_per_step': elapsed/args.steps,
            'steps_per_second': args.steps/elapsed,
            'peak_allocated_MiB': (torch.cuda.max_memory_allocated(device)/2**20
                                   if device.type == 'cuda' else None),
        }), flush=True)


if __name__ == '__main__':
    main()
