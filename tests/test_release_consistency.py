from __future__ import annotations

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
HUB_README = (ROOT / "hub/README.md").read_text(encoding="utf-8")
EXAMPLES_README = (ROOT / "examples/README.md").read_text(encoding="utf-8")


def _cli_options() -> set[str]:
    tree = ast.parse((ROOT / "inference.py").read_text(encoding="utf-8"))
    options: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if not node.args:
            continue
        value = node.args[0]
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            options.add(value.value)
    return options


def test_bundled_example_manifest_references_bundled_files():
    for relative in ("reference.png", "audio.wav", "preprocess_manifest.json"):
        assert (ROOT / "examples" / relative).is_file()

    payload = json.loads((ROOT / "examples/preprocess_manifest.json").read_text())
    for item in payload["items"]:
        assert (ROOT / "examples" / item["image"]).is_file()
        assert (ROOT / "examples" / item["audio"]).is_file()


def test_release_docs_use_consistent_example_paths_and_wording():
    for text in (README, HUB_README):
        assert "examples/speech.wav" not in text
        assert "--audio examples/audio.wav" in text
        assert "TA2.0_animation_dataset" in text
        for phrase in (
            "hf auth",
            "HF_TOKEN",
            "access-controlled",
            "private repository",
            "public repository",
        ):
            assert phrase not in text

    assert "examples/reference.png" in EXAMPLES_README
    assert "examples/audio.wav" in EXAMPLES_README
    assert "examples/preprocess_manifest.json" in EXAMPLES_README


def test_documented_generation_options_exist_in_cli():
    section = README.split("## Generation Options", 1)[1].split("\n## ", 1)[0]
    documented = set(re.findall(r"`(--[a-z_]+)`", section))
    cli_options = _cli_options()

    assert documented <= cli_options

    removed_options = {
        "--video_model_id",
        "--audio_model_id",
        "--revision",
        "--video_model_revision",
        "--audio_model_revision",
        "--cache_dir",
        "--sample_guide_scale",
    }
    assert removed_options.isdisjoint(cli_options)


def test_hub_metadata_uses_canonical_talking_avatar_tag():
    assert "  - talking_avatar\n" in HUB_README
    assert "  - talking-avatar\n" not in HUB_README


def test_pipeline_uses_talking_avatar_generator_interface():
    pipeline_source = (ROOT / "talking_avatar/pipeline.py").read_text()
    config_source = (ROOT / "talking_avatar/config.py").read_text()

    assert "TalkingAvatarGenerator" in pipeline_source
    assert "WanTI2V" not in pipeline_source
    assert ".vendor" not in pipeline_source
    assert "wan_config" not in config_source


def test_acknowledgments_match_across_release_readmes():
    acknowledgment = (
        "TalkingAvatar was developed as part of the **첨단 GPU 활용 지원 사업**\n"
        "(project period 2026-04-01 – 2026-07-15), supported by\n"
        "**정보통신산업진흥원** and **한국정보통신진흥협회**."
    )

    assert README.endswith(acknowledgment + "\n")
    assert HUB_README.endswith(acknowledgment + "\n")
