import json

import pytest
from PIL import Image

from talking_avatar.preprocess.manifest import load_manifest


def test_load_manifest_resolves_relative_images(tmp_path):
    image = tmp_path / "source.png"
    Image.new("RGB", (300, 300), "white").save(image)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"items": [{"id": "sample_001", "image": "source.png"}]}),
        encoding="utf-8",
    )

    items = load_manifest(manifest)

    assert items[0].sample_id == "sample_001"
    assert items[0].image == image.resolve()


def test_load_manifest_rejects_duplicate_ids(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "items": [
                    {"id": "same", "image": "a.png"},
                    {"id": "same", "image": "b.png"},
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate sample id"):
        load_manifest(manifest)


def test_load_manifest_rejects_unsafe_ids(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"items": [{"id": "../bad", "image": "source.png"}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sample id"):
        load_manifest(manifest)


@pytest.mark.parametrize("sample_id", [".", "..", "..."])
def test_load_manifest_rejects_dot_only_ids(tmp_path, sample_id):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"items": [{"id": sample_id, "image": "source.png"}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sample id"):
        load_manifest(manifest)
