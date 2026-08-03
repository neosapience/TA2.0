from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_MODEL_ID = "neosapience/TA2.0"
RELEASE_CHECKPOINT_DIR = "checkpoint"
NEGATIVE_PROMPT_FILE = "negative_prompt.safetensors"
RELEASE_MANIFEST_FILE = "release-manifest.json"
MODEL_CONFIG_FILE = "config.json"


@dataclass(frozen=True)
class HubArtifacts:
    snapshot_dir: Path
    release_checkpoint: Path
    negative_prompt_embeds: Path
    release_manifest: Path
    model_config: Path


def _snapshot(
    repo_id: str,
    *,
    revision: str | None,
    cache_dir: str | Path | None,
    token: bool | str | None,
    local_files_only: bool,
    allow_patterns: list[str] | None = None,
) -> Path:
    return Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            token=token,
            local_files_only=local_files_only,
            allow_patterns=allow_patterns,
        )
    )


def resolve_model_dir(
    model_name_or_path: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: bool | str | None = None,
    local_files_only: bool = False,
) -> Path:
    candidate = Path(model_name_or_path).expanduser()
    if candidate.exists():
        if not candidate.is_dir():
            raise NotADirectoryError(candidate)
        return candidate.resolve()
    if isinstance(model_name_or_path, Path):
        raise FileNotFoundError(candidate)
    return _snapshot(
        model_name_or_path,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )


def download_pretrained_artifacts(
    model_name_or_path: str | Path = DEFAULT_MODEL_ID,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: bool | str | None = None,
    local_files_only: bool = False,
) -> HubArtifacts:
    candidate = Path(model_name_or_path).expanduser()
    if candidate.exists():
        if not candidate.is_dir():
            raise NotADirectoryError(candidate)
        snapshot_dir = candidate.resolve()
    elif isinstance(model_name_or_path, Path):
        raise FileNotFoundError(candidate)
    else:
        snapshot_dir = _snapshot(
            model_name_or_path,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            allow_patterns=[
                f"{RELEASE_CHECKPOINT_DIR}/*",
                NEGATIVE_PROMPT_FILE,
                RELEASE_MANIFEST_FILE,
                MODEL_CONFIG_FILE,
            ],
        )

    artifacts = HubArtifacts(
        snapshot_dir=snapshot_dir,
        release_checkpoint=snapshot_dir / RELEASE_CHECKPOINT_DIR,
        negative_prompt_embeds=snapshot_dir / NEGATIVE_PROMPT_FILE,
        release_manifest=snapshot_dir / RELEASE_MANIFEST_FILE,
        model_config=snapshot_dir / MODEL_CONFIG_FILE,
    )
    index_path = artifacts.release_checkpoint / "model.safetensors.index.json"
    required = [
        index_path,
        artifacts.negative_prompt_embeds,
        artifacts.release_manifest,
        artifacts.model_config,
    ]
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid checkpoint index: {index_path}")
        required.extend(
            artifacts.release_checkpoint / filename
            for filename in sorted(set(weight_map.values()))
        )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Incomplete TA2.0 pretrained snapshot; missing: "
            + ", ".join(str(path) for path in missing)
        )
    return artifacts


def load_model_config(config_path: Path) -> dict[str, str]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = ("video_model_name_or_path", "audio_model_name_or_path")
    missing = [
        key
        for key in required
        if not isinstance(config.get(key), str) or not config[key].strip()
    ]
    if missing:
        raise ValueError(
            f"Invalid TalkingAvatar model config; missing: {', '.join(missing)}"
        )
    return config
