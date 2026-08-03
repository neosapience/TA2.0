from __future__ import annotations

from pathlib import Path
from math import gcd

import soundfile
import torch
from PIL import Image
from scipy.signal import resample_poly
from transformers import Wav2Vec2FeatureExtractor


def audio_duration_seconds(path: Path) -> float:
    info = soundfile.info(path)
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError(f"Audio must contain at least one sample: {path}")
    return info.frames / info.samplerate


def frames_for_audio(path: Path, fps: int = 25) -> int:
    """Return the smallest 4n+1 frame count that covers the full audio."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    info = soundfile.info(path)
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError(f"Audio must contain at least one sample: {path}")

    minimum_frames = max(
        1, (info.frames * fps + info.samplerate - 1) // info.samplerate
    )
    remainder = (minimum_frames - 1) % 4
    if remainder:
        minimum_frames += 4 - remainder
    return minimum_frames


def load_reference_image(path: Path, width: int, height: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
        pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        pixels = pixels.reshape(height, width, 3).permute(2, 0, 1).float()
    return pixels.div(127.5).sub(1.0).clamp(-1.0, 1.0)


def load_audio_mask(path: Path, width: int, height: int) -> torch.Tensor:
    """Load a mask image as a normalized single-channel tensor."""
    with Image.open(path) as image:
        image = image.convert("L").resize(
            (width, height), Image.Resampling.NEAREST
        )
        pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        pixels = pixels.reshape(1, 1, height, width).float()
    return pixels.div(255.0).clamp(0.0, 1.0)


def load_audio(
    path: Path,
    processor_dir: Path,
    frames: int,
    samples_per_frame: int = 640,
) -> torch.Tensor:
    waveform, sample_rate = soundfile.read(path, dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    if sample_rate != 16_000:
        divisor = gcd(sample_rate, 16_000)
        waveform = resample_poly(
            waveform, 16_000 // divisor, sample_rate // divisor
        ).astype("float32", copy=False)
    waveform = torch.from_numpy(waveform)

    target_samples = frames * samples_per_frame
    waveform = waveform[:target_samples]
    if waveform.numel() < target_samples:
        waveform = torch.nn.functional.pad(waveform, (0, target_samples - waveform.numel()))

    processor = Wav2Vec2FeatureExtractor.from_pretrained(processor_dir)
    values = processor(
        [waveform.numpy()], sampling_rate=16_000, return_tensors="pt"
    ).input_values
    return values
