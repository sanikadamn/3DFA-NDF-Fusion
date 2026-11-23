from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class NDFConfig:
    checkpoint: str = ""
    latent_dim: int = 256
    num_points: int = 512
    query_chunk_size: int = 256
    scaling: float = 10.0
    # Internal NDF batch only; does not change the policy/optimizer batch.
    cloud_batch_size: int = 8

    def __post_init__(self):
        for name in ('latent_dim', 'num_points', 'query_chunk_size', 'cloud_batch_size'):
            if type(getattr(self, name)) is not int:
                raise ValueError(f'{name} must be an integer')
        if self.cloud_batch_size < 1:
            raise ValueError('cloud_batch_size must be >= 1')
        if self.latent_dim < 1 or self.num_points < 20 or self.query_chunk_size < 1:
            raise ValueError("Require latent_dim >= 1, num_points >= 20, query_chunk_size >= 1")
        if not math.isfinite(self.scaling) or self.scaling <= 0:
            raise ValueError("NDF scaling must be finite and positive")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def read(cls, filename):
        path = Path(filename).resolve()
        data = json.loads(path.read_text())
        if data.get("checkpoint"):
            weights = Path(data["checkpoint"]).expanduser()
            data["checkpoint"] = str((path.parent / weights).resolve())
        return cls(**data)


MODEL_KEYS = (
    "model_type", "backbone", "num_vis_instr_attn_layers", "fps_subsampling_factor",
    "embedding_dim", "num_attn_heads", "num_history", "bimanual",
    "num_shared_attn_layers", "relative_action", "rotation_format",
    "denoise_timesteps", "denoise_model", "dataset",
)


def metadata(args):
    return {"ndf_config": args.ndf_settings,
            "model_config": {key: getattr(args, key) for key in MODEL_KEYS}}


def check_resume(saved, args):
    current = metadata(args)
    # Prevent silent loss/addition of NDF and mismatched architecture on resume.
    def numerical(config):
        if config is None:
            return None
        # These affect file location or execution memory, not the descriptor.
        execution_keys = {'checkpoint', 'query_chunk_size', 'cloud_batch_size'}
        return {k: v for k, v in NDFConfig(**config).to_dict().items()
                if k not in execution_keys}
    if numerical(saved.get("ndf_config")) != numerical(current["ndf_config"]):
        raise ValueError("Resume NDF configuration mismatch; use a new experiment directory")
    if "model_config" in saved and saved["model_config"] != current["model_config"]:
        raise ValueError("Resume model/camera configuration mismatch")
