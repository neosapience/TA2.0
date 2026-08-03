from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


TRANSFORMER_PREFIXES = ("transformer._orig_mod.", "transformer.")
AUDIO_PREFIX = "audio_projection."


def _release_key(key: str) -> str | None:
    for prefix in TRANSFORMER_PREFIXES:
        if key.startswith(prefix):
            return "transformer." + key.removeprefix(prefix)
    if key.startswith(AUDIO_PREFIX):
        return key
    return None


def extract_release_state_dict(
    checkpoint: Mapping[str, object],
) -> dict[str, torch.Tensor]:
    raw_state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(raw_state, Mapping):
        raise TypeError("Checkpoint does not contain a state dictionary.")

    release_state: dict[str, torch.Tensor] = {}
    for key, value in raw_state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            continue
        release_key = _release_key(key)
        if release_key is not None:
            release_state[release_key] = value.detach().cpu().contiguous()

    groups = {key.split(".", 1)[0] for key in release_state}
    if groups != {"transformer", "audio_projection"}:
        raise ValueError(
            "Expected both transformer and audio_projection weights; "
            f"found {sorted(groups)}."
        )
    return release_state


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _shards(
    state_dict: Mapping[str, torch.Tensor], max_shard_bytes: int
) -> Iterator[dict[str, torch.Tensor]]:
    shard: dict[str, torch.Tensor] = {}
    shard_bytes = 0
    for key in sorted(state_dict):
        tensor = state_dict[key]
        size = _tensor_bytes(tensor)
        if shard and shard_bytes + size > max_shard_bytes:
            yield shard
            shard = {}
            shard_bytes = 0
        shard[key] = tensor
        shard_bytes += size
    if shard:
        yield shard


def sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def convert_lightning_checkpoint(
    source: Path,
    output_dir: Path,
    max_shard_gib: float = 5.0,
    trust_pickle: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")

    load_kwargs = {"map_location": "cpu", "mmap": True}
    try:
        checkpoint = torch.load(source, weights_only=True, **load_kwargs)
    except Exception:
        if not trust_pickle:
            raise RuntimeError(
                "Safe checkpoint loading failed. Re-run with --trust-pickle only for a "
                "checkpoint from a trusted source."
            ) from None
        checkpoint = torch.load(source, weights_only=False, **load_kwargs)

    state_dict = extract_release_state_dict(checkpoint)
    del checkpoint
    max_shard_bytes = int(max_shard_gib * 1024**3)
    shard_list = list(_shards(state_dict, max_shard_bytes))
    weight_map: dict[str, str] = {}
    total = len(shard_list)

    for index, shard in enumerate(shard_list, start=1):
        filename = f"model-{index:05d}-of-{total:05d}.safetensors"
        save_file(shard, output_dir / filename, metadata={"format": "pt"})
        weight_map.update({key: filename for key in shard})

    index_path = output_dir / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "format": "talking-avatar-v1",
                    "total_size": sum(_tensor_bytes(t) for t in state_dict.values()),
                },
                "weight_map": weight_map,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return index_path


def load_release_state_dict(path: Path, device: str) -> dict[str, torch.Tensor]:
    if path.is_file() and path.suffix == ".safetensors":
        return load_file(path, device=device)

    index_path = path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Release checkpoint index not found: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid checkpoint index: {index_path}")

    state_dict: dict[str, torch.Tensor] = {}
    for filename in sorted(set(weight_map.values())):
        shard = load_file(path / filename, device=device)
        overlap = state_dict.keys() & shard.keys()
        if overlap:
            raise ValueError(f"Duplicate checkpoint keys: {sorted(overlap)[:3]}")
        state_dict.update(shard)
    if state_dict.keys() != weight_map.keys():
        raise ValueError("Checkpoint index and shard keys do not match.")
    return state_dict


def split_component_state(
    state_dict: Mapping[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    marker = prefix + "."
    return {
        key.removeprefix(marker): value
        for key, value in state_dict.items()
        if key.startswith(marker)
    }


def convert_embedding(source: Path, output: Path, trust_pickle: bool = False) -> None:
    try:
        embedding = torch.load(source, map_location="cpu", weights_only=True)
    except Exception:
        if not trust_pickle:
            raise RuntimeError(
                "Safe embedding loading failed. Use --trust-pickle only for a trusted file."
            ) from None
        embedding = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(embedding, torch.Tensor):
        raise TypeError("Embedding file must contain one tensor.")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    save_file({"embedding": embedding.detach().cpu().contiguous()}, output)


def load_embedding(path: Path, device: str) -> torch.Tensor:
    if path.suffix != ".safetensors":
        raise ValueError("Runtime embeddings must use the safetensors format.")
    state = load_file(path, device=device)
    if set(state) != {"embedding"}:
        raise ValueError("Embedding safetensors must contain only the 'embedding' key.")
    embedding = state["embedding"]
    if embedding.ndim == 2:
        embedding = embedding.unsqueeze(0)
    if embedding.ndim != 3 or embedding.shape[0] != 1:
        raise ValueError(f"Expected embedding shape [1, L, D], got {embedding.shape}.")
    return embedding
