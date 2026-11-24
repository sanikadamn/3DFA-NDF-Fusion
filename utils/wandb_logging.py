"""Optional rank-zero W&B mirror of existing TensorBoard scalar metrics."""
import json
import os
from pathlib import Path
import uuid


class WandbLogger:
    def __init__(self, args, rank=0):
        self.run = None
        enabled = os.environ.get('USE_WANDB', '0').lower() in ('1', 'true', 'yes')
        if not enabled or rank != 0 or args.eval_only:
            return
        import wandb  # Optional dependency when logging is disabled.

        project = os.environ.get('WANDB_PROJECT', '3dfa-ndf-comparison')
        entity = os.environ.get('WANDB_ENTITY') or None
        task = os.environ.get('TASK', 'unspecified')
        variant = 'ndf' if args.ndf_settings is not None else 'baseline'
        group = os.environ.get('WANDB_RUN_GROUP', task)
        name = os.environ.get('WANDB_NAME', f'{task}-{variant}-{args.exp_log_dir}')
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        identity_path = log_dir / 'wandb_run.json'
        checkpoint_exists = bool(args.checkpoint and Path(args.checkpoint).is_file())
        identity = None
        if checkpoint_exists and identity_path.exists():
            identity = json.loads(identity_path.read_text())
            if (identity['project'] != project or identity['variant'] != variant
                    or (entity and identity['entity'] != entity)):
                raise ValueError('W&B project/entity/variant differs from saved wandb_run.json. '
                                 'Restore the original settings or use a new experiment directory.')
        run_id = identity['id'] if identity else uuid.uuid4().hex[:12]
        config = json.loads(json.dumps(vars(args), default=str))
        config.update(task=task, variant=variant,
                      global_batch_size=args.batch_size * int(os.environ.get('WORLD_SIZE', '1')))
        self.run = wandb.init(
            project=project, entity=entity, name=name, group=group,
            tags=[variant, task], job_type='train', config=config,
            dir=str(log_dir), id=run_id,
            resume='allow' if identity else 'never',
            mode=os.environ.get('WANDB_MODE', 'online'),
            # Direct mirroring avoids logging each TensorBoard scalar twice.
            sync_tensorboard=False,
        )
        self.run.define_metric('global_step')
        self.run.define_metric('*', step_metric='global_step')
        identity = {'id': self.run.id, 'project': self.run.project,
                    'entity': self.run.entity, 'variant': variant}
        temporary = identity_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(identity, indent=2) + '\n')
        temporary.replace(identity_path)

    def log(self, values, step_id):
        if self.run is not None and step_id >= 0:
            # W&B's internal step increments per call. Plot against global_step
            # so separate train/val calls share the exact TensorBoard x coordinate.
            self.run.log({**values, 'global_step': int(step_id)})

    def finish(self, exit_code=0):
        if self.run is not None:
            self.run.finish(exit_code=exit_code)
            self.run = None
