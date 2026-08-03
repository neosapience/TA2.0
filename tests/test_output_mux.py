import json
import shutil
import subprocess

import imageio.v2 as imageio
import numpy as np
import pytest
import soundfile

from talking_avatar.output import mux_audio


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe is required")
def test_mux_matches_audio_duration_without_shortening_video(tmp_path):
    video = tmp_path / "video.mp4"
    audio = tmp_path / "audio.wav"
    output = tmp_path / "output.mp4"
    imageio.mimwrite(
        video,
        np.zeros((229, 16, 16, 3), dtype=np.uint8),
        fps=25,
        quality=8,
        output_params=["-loglevel", "error"],
    )
    soundfile.write(audio, np.zeros(round(9.04 * 44_100)), 44_100)

    mux_audio(video, audio, output, frames=229, fps=25)

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,duration,nb_frames",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(probe.stdout)
    streams = {stream["codec_type"]: stream for stream in metadata["streams"]}
    assert int(streams["video"]["nb_frames"]) == 226
    assert float(streams["video"]["duration"]) == pytest.approx(9.04, abs=0.001)
    assert float(streams["audio"]["duration"]) == pytest.approx(9.04, abs=0.001)
    assert float(metadata["format"]["duration"]) == pytest.approx(9.04, abs=0.001)
