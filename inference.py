#!/usr/bin/env python3
"""Generate a talking-avatar video with TalkingAvatar."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Sequence

from talking_avatar.config import GenerationConfig
from talking_avatar.hub import DEFAULT_MODEL_ID
from talking_avatar.preprocessing import frames_for_audio


DEFAULT_PROMPT = "The person is speaking moderately."


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _model_name_or_path(value: str) -> str | Path:
    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.exists() else value


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a talking-avatar video with TalkingAvatar."
    )
    parser.add_argument(
        "--model_id",
        type=_model_name_or_path,
        default=DEFAULT_MODEL_ID,
        help="TalkingAvatar Hugging Face repo ID or local snapshot directory.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Disable Hub downloads and use cached or local model files only.",
    )
    parser.add_argument("--image", type=_path, required=True, help="Reference image.")
    parser.add_argument("--audio", type=_path, required=True, help="Speech audio.")
    parser.add_argument(
        "--mask",
        type=_path,
        default=None,
        help="Optional image indicating where audio conditioning is focused.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Text prompt.")
    parser.add_argument(
        "--save_file",
        type=_path,
        help="Output MP4 path. Defaults to outputs/talking_avatar_<timestamp>.mp4.",
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help=(
            "Override the generated frame count; must be 4n+1. "
            "Defaults to the shortest valid frame count that covers the audio."
        ),
    )
    parser.add_argument(
        "--sample_steps", type=int, default=30, help="Denoising steps."
    )
    parser.add_argument(
        "--sample_shift", type=float, default=5.0, help="Flow-matching shift."
    )
    parser.add_argument(
        "--text_guide_scale", type=float, default=5.0, help="Text CFG scale."
    )
    parser.add_argument(
        "--audio_guide_scale", type=float, default=3.0, help="Audio CFG scale."
    )
    parser.add_argument("--base_seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default="cuda", help="Inference device.")
    return parser.parse_args(argv)


def _default_save_file() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / f"talking_avatar_{timestamp}.mp4"


def _validate_args(args: argparse.Namespace) -> None:
    if not args.image.is_file():
        raise FileNotFoundError(f"Reference image not found: {args.image}")
    if not args.audio.is_file():
        raise FileNotFoundError(f"Speech audio not found: {args.audio}")
    if args.mask is not None and not args.mask.is_file():
        raise FileNotFoundError(f"Mask image not found: {args.mask}")


def generate(args: argparse.Namespace) -> Path:
    from talking_avatar.pipeline import TalkingAvatarPipeline

    _validate_args(args)
    save_file = args.save_file or _default_save_file().resolve()
    frame_num = args.frame_num
    if frame_num is None:
        frame_num = frames_for_audio(args.audio, fps=GenerationConfig().fps)
        logging.info(
            "Selected %d frames from the input audio duration", frame_num
        )
    config = GenerationConfig(
        frames=frame_num,
        steps=args.sample_steps,
        shift=args.sample_shift,
        seed=args.base_seed,
        guidance_scale=args.text_guide_scale,
        audio_guidance_scale=args.audio_guide_scale,
    )

    logging.info("Loading TalkingAvatar model from %s", args.model_id)
    model = TalkingAvatarPipeline.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
        device=args.device,
    )
    logging.info("Generating %d frames", config.frames)
    return model.generate(
        image_path=args.image,
        audio_path=args.audio,
        prompt=args.prompt,
        output_path=save_file,
        mask_path=args.mask,
        config=config,
    )


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )
    output = generate(_parse_args(argv))
    logging.info("Saved video to %s", output)


if __name__ == "__main__":
    main()
