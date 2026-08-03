from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps


TA2_WIDTH = 480
TA2_HEIGHT = 832


def open_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def center_crop_resize(
    image: Image.Image,
    width: int = TA2_WIDTH,
    height: int = TA2_HEIGHT,
) -> Image.Image:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    image = image.convert("RGB")
    src_width, src_height = image.size
    if src_width <= 0 or src_height <= 0:
        raise ValueError("image dimensions must be positive")

    target_aspect = width / height
    source_aspect = src_width / src_height

    if source_aspect > target_aspect:
        crop_width = round(src_height * target_aspect)
        left = (src_width - crop_width) // 2
        box = (left, 0, left + crop_width, src_height)
    else:
        crop_height = round(src_width / target_aspect)
        top = (src_height - crop_height) // 2
        box = (0, top, src_width, top + crop_height)

    return image.crop(box).resize((width, height), Image.Resampling.BICUBIC)


def save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
