from __future__ import annotations

import json
from pathlib import Path

from .checkpoint import sha256


def verify_release_manifest(manifest_path: Path, artifact_root: Path) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Release manifest has no files.")

    verified = 0
    root = artifact_root.resolve()
    allowed_roots = [root]
    model_cache = root.parent.parent
    if root.parent.name == "snapshots" and model_cache.name.startswith("models--"):
        allowed_roots.append((model_cache / "blobs").resolve())
    for entry in files:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe artifact path: {relative}")
        artifact = root / relative
        resolved_artifact = artifact.resolve()
        if not any(
            resolved_artifact.is_relative_to(allowed_root)
            for allowed_root in allowed_roots
        ) or not artifact.is_file():
            raise FileNotFoundError(artifact)
        actual_size = artifact.stat().st_size
        if actual_size != entry["size"]:
            raise ValueError(
                f"Size mismatch for {relative}: {actual_size} != {entry['size']}"
            )
        actual_hash = sha256(artifact)
        if actual_hash != entry["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {relative}")
        verified += 1
    return verified
