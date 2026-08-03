from __future__ import annotations

import tempfile
from pathlib import Path

import imageio.v2 as imageio
import torch

from .checkpoint import load_embedding
from .config import GenerationConfig
from .generator import TalkingAvatarGenerator
from .hub import (
    DEFAULT_MODEL_ID,
    download_pretrained_artifacts,
    load_model_config,
    resolve_model_dir,
)
from .models import TextEncoder, load_components
from .output import mux_audio
from .preprocessing import (
    frames_for_audio,
    load_audio,
    load_audio_mask,
    load_reference_image,
)


class TalkingAvatarPipeline:
    def __init__(
        self,
        video_model_dir: Path,
        audio_model_dir: Path,
        release_checkpoint: Path,
        device: str = "cuda",
        negative_prompt_embeds: Path | None = None,
    ) -> None:
        if not torch.cuda.is_available() and device.startswith("cuda"):
            raise RuntimeError("A CUDA GPU is required for TalkingAvatar inference.")
        self.video_model_dir = video_model_dir
        self.audio_model_dir = audio_model_dir
        self.default_negative_prompt_embeds = negative_prompt_embeds
        self.device = device
        self.dtype = torch.bfloat16
        self.components = load_components(
            video_model_dir,
            audio_model_dir,
            release_checkpoint,
            device=device,
            dtype=self.dtype,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path = DEFAULT_MODEL_ID,
        *,
        video_model_name_or_path: str | Path | None = None,
        audio_model_name_or_path: str | Path | None = None,
        revision: str | None = None,
        video_model_revision: str | None = None,
        audio_model_revision: str | None = None,
        cache_dir: str | Path | None = None,
        token: bool | str | None = None,
        local_files_only: bool = False,
        device: str = "cuda",
    ) -> "TalkingAvatarPipeline":
        artifacts = download_pretrained_artifacts(
            pretrained_model_name_or_path,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        model_config = load_model_config(artifacts.model_config)
        if video_model_name_or_path is None:
            video_model_name_or_path = model_config["video_model_name_or_path"]
        if audio_model_name_or_path is None:
            audio_model_name_or_path = model_config["audio_model_name_or_path"]

        video_model_dir = resolve_model_dir(
            video_model_name_or_path,
            revision=video_model_revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        audio_model_dir = resolve_model_dir(
            audio_model_name_or_path,
            revision=audio_model_revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        return cls(
            video_model_dir=video_model_dir,
            audio_model_dir=audio_model_dir,
            release_checkpoint=artifacts.release_checkpoint,
            device=device,
            negative_prompt_embeds=artifacts.negative_prompt_embeds,
        )

    @torch.inference_mode()
    def generate(
        self,
        image_path: Path,
        audio_path: Path,
        prompt: str,
        negative_prompt_embeds: Path | None = None,
        output_path: Path | None = None,
        config: GenerationConfig | None = None,
        mask_path: Path | None = None,
    ) -> Path:
        if output_path is None:
            raise TypeError("output_path is required.")
        if config is None:
            config = GenerationConfig(frames=frames_for_audio(audio_path))
        if negative_prompt_embeds is None:
            negative_prompt_embeds = self.default_negative_prompt_embeds
        if negative_prompt_embeds is None:
            raise ValueError(
                "negative_prompt_embeds is required when the pipeline was not "
                "created with TalkingAvatarPipeline.from_pretrained()."
            )
        config.validate()
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)

        text_encoder = TextEncoder(
            self.video_model_dir, device=self.device, dtype=self.dtype
        )
        prompt_embeds = text_encoder.encode(prompt)
        text_encoder.close()
        negative_embeds = load_embedding(negative_prompt_embeds, self.device).to(
            dtype=self.dtype
        )

        image = load_reference_image(
            image_path, width=config.width, height=config.height
        ).unsqueeze(0).to(device=self.device, dtype=self.dtype)
        audio = load_audio(audio_path, self.audio_model_dir, frames=config.frames)
        audio_mask = (
            load_audio_mask(mask_path, width=config.width, height=config.height)
            if mask_path is not None
            else None
        )

        generator = TalkingAvatarGenerator(
            transformer=self.components.transformer,
            vae=self.components.vae,
            device=self.device,
            weight_dtype=self.dtype,
            config=config.runtime_config(),
            audio_encoder=self.components.audio_encoder,
            audio_projection=self.components.audio_projection,
        )
        frames = generator.generate(
            img=image,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            max_area=config.width * config.height,
            frame_num=config.frames,
            shift=config.shift,
            sample_solver="unipc",
            sampling_steps=config.steps,
            guide_scale=config.guidance_scale,
            seed=config.seed,
            base_num_frames=config.base_num_frames,
            overlap_history=config.overlap_history,
            addnoise_condition=config.addnoise_condition,
            ref_copy_motion_idx=config.ref_copy_motion_idx,
            ref_latent_attn=False,
            audio=audio,
            ref_lmk=None,
            audio_guidance_scale=config.audio_guidance_scale,
            audio_mask=audio_mask,
            prefix_frame=config.prefix_frame,
        )[0]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="talking-avatar-") as temp_dir:
            video_only = Path(temp_dir) / "video.mp4"
            imageio.mimwrite(
                video_only,
                frames,
                fps=config.fps,
                quality=8,
                output_params=["-loglevel", "error"],
            )
            mux_audio(
                video_only,
                audio_path,
                output_path,
                frames=config.frames,
                fps=config.fps,
            )
        return output_path
