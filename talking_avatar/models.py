from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .checkpoint import load_release_state_dict, split_component_state
from .custom_wav2vec2 import CustomWav2Vec2
from .vendor.wan.modules.model_v2 import WanModel
from .vendor.wan.modules.t5 import umt5_xxl
from .vendor.wan.modules.tokenizers import HuggingfaceTokenizer
from .vendor.wan.modules.vae2_2 import Wan2_2_VAE


class AudioProjection(nn.Module):
    def __init__(self, audio_in_dim: int = 768, cross_attention_dim: int = 3072):
        super().__init__()
        self.proj = nn.Linear(audio_in_dim, cross_attention_dim, bias=False)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, audio_embeds: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(audio_embeds))


@dataclass
class TalkingAvatarComponents:
    transformer: WanModel
    vae: Wan2_2_VAE
    audio_encoder: CustomWav2Vec2
    audio_projection: AudioProjection


def _assert_exact_keys(
    module: nn.Module, state_dict: dict[str, torch.Tensor], component: str
) -> None:
    expected = set(module.state_dict())
    actual = set(state_dict)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise RuntimeError(
            f"{component} checkpoint mismatch: {len(missing)} missing and "
            f"{len(unexpected)} unexpected keys. "
            f"Examples: missing={missing[:3]}, unexpected={unexpected[:3]}"
        )


def load_components(
    base_model_dir: Path,
    audio_model_dir: Path,
    release_checkpoint: Path,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> TalkingAvatarComponents:
    vae_path = base_model_dir / "Wan2.2_VAE.pth"
    if not vae_path.is_file():
        raise FileNotFoundError(f"Wan VAE not found: {vae_path}")

    vae = Wan2_2_VAE(vae_pth=str(vae_path), dtype=dtype, device=device)
    vae.model.eval().requires_grad_(False)

    transformer = WanModel.from_pretrained(
        base_model_dir,
        time_scale=1.0,
        ref_rope_idx=-1,
        ref_end_mode="replace",
        torch_dtype=dtype,
        device_map=device,
    )
    for block in transformer.blocks:
        block.init_audio_cross_attn_multi_talk(
            audio_dim=3072,
            hidden_dim=3072,
            audio_scale=1.0,
        )
    transformer.to(device=device, dtype=dtype)

    audio_projection = AudioProjection().to(device=device, dtype=dtype)
    release_state = load_release_state_dict(release_checkpoint, device=device)
    transformer_state = split_component_state(release_state, "transformer")
    projection_state = split_component_state(release_state, "audio_projection")
    _assert_exact_keys(transformer, transformer_state, "transformer")
    _assert_exact_keys(audio_projection, projection_state, "audio_projection")
    transformer.load_state_dict(transformer_state, strict=True)
    audio_projection.load_state_dict(projection_state, strict=True)
    del release_state, transformer_state, projection_state

    transformer.eval().requires_grad_(False)
    audio_projection.eval().requires_grad_(False)
    audio_encoder = CustomWav2Vec2.from_pretrained(
        audio_model_dir, output_hidden_states=True, mask_time_prob=0.0
    ).to(device=device, dtype=dtype)
    audio_encoder.eval().requires_grad_(False)
    return TalkingAvatarComponents(transformer, vae, audio_encoder, audio_projection)


class TextEncoder:
    def __init__(
        self,
        base_model_dir: Path,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        text_len: int = 512,
    ) -> None:
        checkpoint = base_model_dir / "models_t5_umt5-xxl-enc-bf16.pth"
        tokenizer = base_model_dir / "google" / "umt5-xxl"
        if not checkpoint.is_file() or not tokenizer.is_dir():
            raise FileNotFoundError("Wan UMT5 weights or tokenizer are missing.")

        self.device = device
        self.model = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=dtype,
            device=device,
        ).eval().requires_grad_(False)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(device=device, dtype=dtype)
        self.tokenizer = HuggingfaceTokenizer(
            name=str(tokenizer), seq_len=text_len, clean="whitespace"
        )

    @torch.inference_mode()
    def encode(self, text: str) -> torch.Tensor:
        ids, mask = self.tokenizer(
            text, return_mask=True, add_special_tokens=True
        )
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        context = self.model(ids, mask=mask)
        return context * mask.unsqueeze(-1)

    def close(self) -> None:
        self.model.to("cpu")
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
