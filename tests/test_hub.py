import json
import sys
import types
from pathlib import Path

import pytest

from talking_avatar import hub


def _make_release(root: Path) -> None:
    checkpoint = root / hub.RELEASE_CHECKPOINT_DIR
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors.index.json").write_text(
        '{"weight_map":{"transformer.key":"model-00001-of-00001.safetensors"}}'
    )
    (checkpoint / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    (root / hub.NEGATIVE_PROMPT_FILE).write_bytes(b"embedding")
    (root / hub.RELEASE_MANIFEST_FILE).write_text("{}")
    (root / hub.MODEL_CONFIG_FILE).write_text(
        json.dumps(
            {
                "video_model_name_or_path": "test/video-model",
                "audio_model_name_or_path": "test/audio-model",
            }
        ),
        encoding="utf-8",
    )


def test_download_pretrained_artifacts_accepts_local_snapshot(tmp_path: Path):
    _make_release(tmp_path)

    result = hub.download_pretrained_artifacts(tmp_path)

    assert result.snapshot_dir == tmp_path.resolve()
    assert result.release_checkpoint.name == hub.RELEASE_CHECKPOINT_DIR
    assert result.negative_prompt_embeds.name == hub.NEGATIVE_PROMPT_FILE
    assert result.model_config.name == hub.MODEL_CONFIG_FILE


def test_download_pretrained_artifacts_filters_remote_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _make_release(tmp_path)
    call = {}

    def fake_snapshot_download(**kwargs):
        call.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(hub, "snapshot_download", fake_snapshot_download)

    result = hub.download_pretrained_artifacts("neosapience/TA2.0", revision="release")

    assert result.snapshot_dir == tmp_path
    assert call["repo_id"] == "neosapience/TA2.0"
    assert call["revision"] == "release"
    assert call["token"] is None
    assert f"{hub.RELEASE_CHECKPOINT_DIR}/*" in call["allow_patterns"]


def test_load_model_config(tmp_path: Path):
    _make_release(tmp_path)

    config = hub.load_model_config(tmp_path / hub.MODEL_CONFIG_FILE)

    assert config["video_model_name_or_path"] == "test/video-model"
    assert config["audio_model_name_or_path"] == "test/audio-model"


def test_download_pretrained_artifacts_rejects_incomplete_snapshot(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Incomplete TA2.0"):
        hub.download_pretrained_artifacts(tmp_path)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_pipeline_resolves_dependencies_from_model_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    lightweight_models = types.ModuleType("talking_avatar.models")
    lightweight_models.TextEncoder = object
    lightweight_models.load_components = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "talking_avatar.models", lightweight_models)

    from talking_avatar import pipeline

    artifacts = type(
        "Artifacts",
        (),
        {
            "model_config": tmp_path / "config.json",
            "release_checkpoint": tmp_path / "checkpoint",
            "negative_prompt_embeds": tmp_path / "negative.safetensors",
        },
    )()
    resolved = []

    monkeypatch.setattr(
        pipeline, "download_pretrained_artifacts", lambda *args, **kwargs: artifacts
    )
    monkeypatch.setattr(
        pipeline,
        "load_model_config",
        lambda path: {
            "video_model_name_or_path": "test/video-model",
            "audio_model_name_or_path": "test/audio-model",
        },
    )

    def fake_resolve(name, **kwargs):
        resolved.append(name)
        return tmp_path / name.replace("/", "--")

    monkeypatch.setattr(pipeline, "resolve_model_dir", fake_resolve)

    class LightweightPipeline:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    result = pipeline.TalkingAvatarPipeline.from_pretrained.__func__(
        LightweightPipeline, "neosapience/TA2.0"
    )

    assert resolved == ["test/video-model", "test/audio-model"]
    assert result.kwargs["video_model_dir"].name == "test--video-model"
    assert result.kwargs["audio_model_dir"].name == "test--audio-model"


def test_inference_defaults_to_model_hub():
    from inference import _parse_args

    args = _parse_args(
        [
            "--image",
            "reference.png",
            "--audio",
            "speech.wav",
        ]
    )

    assert args.model_id == "neosapience/TA2.0"
    assert args.mask is None
    assert args.frame_num is None
    assert args.text_guide_scale == 5.0
