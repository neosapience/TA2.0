from __future__ import annotations

import subprocess
from pathlib import Path

from imageio_ffmpeg import get_ffmpeg_exe

from .preprocessing import audio_duration_seconds


def mux_audio(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    *,
    frames: int,
    fps: int,
) -> Path:
    """Encode a frame-accurate video stream and mux it with the input audio.

    Stream-copy trimming is not frame-accurate for H.264 with reordered frames.
    Re-encoding gives the video, audio, and MP4 container the same requested
    duration instead of allowing ``-shortest`` to truncate one stream early.
    """
    if frames <= 0 or fps <= 0:
        raise ValueError("frames and fps must be positive")

    target_duration = min(audio_duration_seconds(audio_path), frames / fps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-t",
            f"{target_duration:.9f}",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        check=True,
    )
    return output_path
