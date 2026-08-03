import numpy as np
import pytest
import soundfile

from talking_avatar.preprocessing import audio_duration_seconds, frames_for_audio


def _write_audio(path, duration, sample_rate=44_100):
    samples = round(duration * sample_rate)
    soundfile.write(path, np.zeros(samples, dtype=np.float32), sample_rate)


def test_frames_for_audio_covers_full_duration_and_matches_model_shape(tmp_path):
    audio = tmp_path / "speech.wav"
    _write_audio(audio, 9.04)

    frames = frames_for_audio(audio, fps=25)

    assert frames == 229
    assert (frames - 1) % 4 == 0
    assert frames / 25 >= audio_duration_seconds(audio)


def test_frames_for_audio_keeps_exact_valid_duration(tmp_path):
    audio = tmp_path / "speech.wav"
    _write_audio(audio, 1.8)

    assert frames_for_audio(audio, fps=25) == 45


def test_frames_for_audio_rejects_invalid_fps(tmp_path):
    audio = tmp_path / "speech.wav"
    _write_audio(audio, 1.0)

    with pytest.raises(ValueError, match="fps"):
        frames_for_audio(audio, fps=0)
