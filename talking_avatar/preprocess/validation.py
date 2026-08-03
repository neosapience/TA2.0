from __future__ import annotations

from pathlib import Path

from .image_ops import open_rgb_image
from .manifest import PreprocessItem, validate_sample_id


MIN_IMAGE_SIZE = 256


def validate_item(item: PreprocessItem) -> tuple[int, int]:
    validate_sample_id(item.sample_id)
    if not item.image.is_file():
        raise FileNotFoundError(f"image not found: {item.image}")
    image = open_rgb_image(item.image)
    width, height = image.size
    if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
        raise ValueError(
            f"image must be at least {MIN_IMAGE_SIZE}x{MIN_IMAGE_SIZE}: {item.image}"
        )
    if item.audio is not None and not item.audio.is_file():
        raise FileNotFoundError(f"audio not found: {item.audio}")
    return width, height


def validate_output_dir(path: Path, overwrite: bool = False) -> None:
    if not path.exists():
        return
    if not path.is_dir():
        raise NotADirectoryError(path)
    if any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory must be empty or use --overwrite: {path}")


def validate_model_path_or_id(value: str | None, require: bool = True) -> None:
    if not value:
        if require:
            raise ValueError("--qwen-model or QWEN3VL_MODEL is required")
        return
    path = Path(value).expanduser()
    if path.exists():
        return
    if value.count("/") == 1 and all(part for part in value.split("/", 1)):
        return
    raise FileNotFoundError(f"model path not found and not a valid model id: {value}")
