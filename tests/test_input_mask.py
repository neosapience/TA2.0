import pytest
from PIL import Image

from inference import _parse_args, _validate_args
from talking_avatar.preprocessing import load_audio_mask


def test_load_audio_mask_resizes_and_normalizes_image(tmp_path):
    path = tmp_path / "mask.png"
    image = Image.new("L", (2, 1))
    image.putdata([0, 255])
    image.save(path)

    mask = load_audio_mask(path, width=4, height=2)

    assert mask.shape == (1, 1, 2, 4)
    assert mask.min().item() == 0.0
    assert mask.max().item() == 1.0


def test_inference_accepts_optional_mask_image(tmp_path):
    mask_path = tmp_path / "mask.png"
    Image.new("L", (16, 16), 255).save(mask_path)

    args = _parse_args(
        [
            "--image",
            "reference.png",
            "--audio",
            "speech.wav",
            "--mask",
            str(mask_path),
        ]
    )

    assert args.mask == mask_path.resolve()


def test_inference_rejects_missing_mask_image(tmp_path):
    args = _parse_args(
        [
            "--image",
            __file__,
            "--audio",
            __file__,
            "--mask",
            str(tmp_path / "missing.png"),
        ]
    )
    with pytest.raises(FileNotFoundError, match="Mask image"):
        _validate_args(args)
