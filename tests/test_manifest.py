import hashlib
import json

import pytest

from talking_avatar.manifest import verify_release_manifest


def test_verify_release_manifest(tmp_path):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    artifact = artifact_root / "weights.bin"
    artifact.write_bytes(b"weights")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "weights.bin",
                        "size": artifact.stat().st_size,
                        "sha256": hashlib.sha256(b"weights").hexdigest(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert verify_release_manifest(manifest, artifact_root) == 1


def test_verify_release_manifest_supports_hugging_face_cache_symlinks(tmp_path):
    model_cache = tmp_path / "models--org--repo"
    blob_dir = model_cache / "blobs"
    blob_dir.mkdir(parents=True)
    blob = blob_dir / "abc123"
    blob.write_bytes(b"checkpoint")
    snapshot = model_cache / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    (snapshot / "weights.bin").symlink_to("../../blobs/abc123")
    manifest = snapshot / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "weights.bin",
                        "size": blob.stat().st_size,
                        "sha256": hashlib.sha256(b"checkpoint").hexdigest(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert verify_release_manifest(manifest, snapshot) == 1


def test_verify_release_manifest_rejects_symlink_outside_hub_cache(tmp_path):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"checkpoint")
    (artifact_root / "weights.bin").symlink_to(outside)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "weights.bin",
                        "size": outside.stat().st_size,
                        "sha256": hashlib.sha256(b"checkpoint").hexdigest(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError):
        verify_release_manifest(manifest, artifact_root)
