import json

import torch

from talking_avatar.checkpoint import (
    convert_lightning_checkpoint,
    extract_release_state_dict,
)


def test_extract_release_state_dict_strips_lightning_prefixes():
    checkpoint = {
        "state_dict": {
            "transformer.layer.weight": torch.ones(1),
            "audio_projection.proj.weight": torch.ones(1),
            "optimizer.unused": torch.ones(1),
        }
    }
    result = extract_release_state_dict(checkpoint)
    assert set(result) == {
        "transformer.layer.weight",
        "audio_projection.proj.weight",
    }


def test_compiled_transformer_prefix_is_normalized():
    checkpoint = {
        "transformer._orig_mod.layer.weight": torch.ones(1),
        "audio_projection.norm.weight": torch.ones(1),
    }
    result = extract_release_state_dict(checkpoint)
    assert "transformer.layer.weight" in result


def test_conversion_omits_source_metadata(tmp_path):
    source = tmp_path / "source.ckpt"
    torch.save(
        {
            "state_dict": {
                "transformer.layer.weight": torch.ones(1),
                "audio_projection.proj.weight": torch.ones(1),
            }
        },
        source,
    )
    output_dir = tmp_path / "release"

    index_path = convert_lightning_checkpoint(source, output_dir)
    index = json.loads(index_path.read_text(encoding="utf-8"))

    assert index["metadata"] == {
        "format": "talking-avatar-v1",
        "total_size": 8,
    }
