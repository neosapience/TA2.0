from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ConfigNode(dict[str, Any]):
    """Dictionary with attribute access for the vendored Wan pipeline."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _node(value: Any) -> Any:
    if isinstance(value, dict):
        return ConfigNode({key: _node(item) for key, item in value.items()})
    return value


@dataclass(frozen=True)
class GenerationConfig:
    width: int = 480
    height: int = 832
    fps: int = 25
    frames: int = 45
    steps: int = 30
    guidance_scale: float = 5.0
    audio_guidance_scale: float = 3.0
    shift: float = 5.0
    seed: int = 42
    base_num_frames: int = 45
    overlap_history: int = 9
    addnoise_condition: int = 20
    ref_copy_motion_idx: int = 1
    prefix_frame: int = 5

    def validate(self) -> None:
        if self.fps != 25:
            raise ValueError("The released TalkingAvatar model supports 25 fps only.")
        if self.width % 16 or self.height % 16:
            raise ValueError("Width and height must be divisible by 16.")
        if self.frames < 1 or (self.frames - 1) % 4:
            raise ValueError("Frames must be of the form 4n+1.")
        if self.base_num_frames < 1 or (self.base_num_frames - 1) % 4:
            raise ValueError("base_num_frames must be of the form 4n+1.")
        if not 0 <= self.overlap_history < self.base_num_frames:
            raise ValueError("overlap_history must be in [0, base_num_frames).")
        if self.steps < 1:
            raise ValueError("steps must be positive.")

    def runtime_config(self) -> ConfigNode:
        return _node(
            {
                "full_face_mask": True,
                "mask_border_ratio": 0.1,
                "network": {
                    "audio_last_layer": False,
                    "audio_temp_interpolate": True,
                    "audio_token_per_latent_frame": 4,
                    "audio_out_attn_mask": True,
                    "backbone": {"cond_pure_norm": False},
                },
            }
        )
