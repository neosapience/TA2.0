import pytest

from talking_avatar.config import GenerationConfig


def test_release_profile_is_valid():
    GenerationConfig().validate()


@pytest.mark.parametrize("frames", [0, 2, 44, 46])
def test_invalid_frame_count_is_rejected(frames):
    with pytest.raises(ValueError):
        GenerationConfig(frames=frames).validate()


def test_only_25_fps_is_supported():
    with pytest.raises(ValueError):
        GenerationConfig(fps=12).validate()
