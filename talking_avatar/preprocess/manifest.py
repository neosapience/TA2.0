from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SAFE_ID_RE = re.compile(r"^(?!\.+$)[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class PreprocessItem:
    sample_id: str
    image: Path
    audio: Path | None = None
    domain: str | None = None

    @property
    def source_label(self) -> str:
        return self.image.name


def make_safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return safe or "sample"


def validate_sample_id(sample_id: str) -> None:
    if not sample_id:
        raise ValueError("sample id must not be empty")
    if not SAFE_ID_RE.match(sample_id):
        raise ValueError(
            "sample id must contain only letters, numbers, underscore, dash, or dot"
        )


def item_from_image(
    path: Path,
    sample_id: str | None = None,
    *,
    audio: Path | None = None,
) -> PreprocessItem:
    image = path.expanduser().resolve()
    resolved_audio = audio.expanduser().resolve() if audio is not None else None
    return PreprocessItem(
        sample_id or make_safe_id(image.stem),
        image=image,
        audio=resolved_audio,
    )


def _path_from_manifest(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def load_manifest(path: Path) -> list[PreprocessItem]:
    manifest_path = path.expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as file:
        payload: dict[str, Any] = json.load(file)

    raw_items = payload.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("manifest must contain a non-empty items list")

    items: list[PreprocessItem] = []
    seen: set[str] = set()
    base_dir = manifest_path.parent
    for index, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            raise ValueError(f"manifest item {index} must be an object")
        image_value = raw.get("image")
        if not isinstance(image_value, str) or not image_value:
            raise ValueError(f"manifest item {index} requires image")

        image = _path_from_manifest(image_value, base_dir)
        sample_id = str(raw.get("id") or make_safe_id(image.stem))
        validate_sample_id(sample_id)
        if sample_id in seen:
            raise ValueError(f"duplicate sample id: {sample_id}")
        seen.add(sample_id)

        audio_value = raw.get("audio")
        audio = None
        if audio_value:
            if not isinstance(audio_value, str):
                raise ValueError(f"manifest item {sample_id} audio must be a string")
            audio = _path_from_manifest(audio_value, base_dir)

        domain_value = raw.get("domain")
        domain = str(domain_value) if domain_value is not None else None
        items.append(
            PreprocessItem(
                sample_id=sample_id,
                image=image,
                audio=audio,
                domain=domain,
            )
        )
    return items
