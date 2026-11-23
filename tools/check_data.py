"""Fail early on wrong task IDs, empty stores, camera layout or instructions."""
import argparse
import json
from pathlib import Path
import numpy as np
import zarr
from datasets.rlbench import PERACT_TASKS


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--dataset-dir',required=True)
    p.add_argument('--tasks',nargs='+',default=['push_buttons'],choices=PERACT_TASKS)
    p.add_argument('--instructions',default='instructions/peract/instructions.json')
    args=p.parse_args()
    instructions=json.loads(Path(args.instructions).read_text())
    expected={PERACT_TASKS.index(t) for t in args.tasks}
    for split in ('train','val'):
        group=zarr.open_group(str(Path(args.dataset_dir)/(split+'.zarr')),mode='r')
        count=len(group['action'])
        assert count>0, f'{split} has no samples'
        assert all(len(group[k])==count for k in group.array_keys()), 'array lengths differ'
        actual=set(map(int,np.unique(group['task_id'][:])))
        assert actual==expected, f'{split} tasks {actual}; expected {expected}; use a new conversion directory'
        assert group['rgb'].shape[1:]==(4,3,256,256)
        assert group['pcd'].shape[1:]==(4,3,256,256)
        assert group['action'].shape[1:]==(1,1,8)
        assert group['proprioception'].shape[1:]==(3,1,8)
        for tid,var in set(zip(group['task_id'][:].tolist(),group['variation'][:].tolist())):
            assert instructions[PERACT_TASKS[tid]][str(var)], f'Missing instruction: {tid}, {var}'
        # Sample integrity checks, not an expensive full geometry scan.
        for i in np.linspace(0,count-1,min(16,count)).astype(int):
            assert np.isfinite(group['pcd'][i]).all(), f'nonfinite pcd sample {i}'
            assert np.isfinite(group['action'][i]).all(), f'nonfinite action sample {i}'
        print(split, count, 'keypose samples;', args.tasks, '; canonical IDs', sorted(actual))


if __name__=='__main__':
    main()
