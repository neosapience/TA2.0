from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .manifest import PreprocessItem


def default_run_id(prefix: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def artifact(path: Path, sample_dir: Path) -> str:
    return path.relative_to(sample_dir).as_posix()


def normalized_manifest(items: list[PreprocessItem]) -> dict[str, Any]:
    return {
        "items": [
            {
                "id": item.sample_id,
                "source_label": item.source_label,
                "audio_label": item.audio.name if item.audio else None,
                "domain": item.domain,
            }
            for item in items
        ]
    }


def write_failure(path: Path, sample_id: str, error: Exception | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"id": sample_id, "status": "failed", "error": str(error)}
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, sort_keys=True) + "\n")
