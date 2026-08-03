import json
from types import SimpleNamespace

from PIL import Image

from scripts.preprocess import run


def test_cli_writes_redacted_summary_and_reference(tmp_path):
    image = tmp_path / "input.png"
    Image.new("RGB", (640, 640), "white").save(image)
    output_dir = tmp_path / "out"

    args = SimpleNamespace(
        image=image,
        audio=None,
        manifest=None,
        output_dir=output_dir,
        qwen_model=None,
        caption_prompt=None,
        max_new_tokens=256,
        temperature=0.1,
        overwrite=False,
        skip_caption=True,
    )

    run(args)

    sample_dir = output_dir / "input"
    assert Image.open(sample_dir / "01_reference_480x832.png").size == (480, 832)
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    summary_text = json.dumps(summary)
    assert str(image) not in summary_text
    assert summary["succeeded"] == 1


def test_cli_redacts_failure_paths(tmp_path):
    image = tmp_path / "missing.png"
    output_dir = tmp_path / "out"
    args = SimpleNamespace(
        image=image,
        audio=None,
        manifest=None,
        output_dir=output_dir,
        qwen_model=None,
        caption_prompt=None,
        max_new_tokens=256,
        temperature=0.1,
        overwrite=False,
        skip_caption=True,
    )

    run(args)

    summary_text = (output_dir / "summary.json").read_text(encoding="utf-8")
    failures_text = (output_dir / "failures.jsonl").read_text(encoding="utf-8")
    assert str(image) not in summary_text
    assert str(image) not in failures_text
    assert image.name in summary_text
    assert image.name in failures_text


def test_cli_writes_single_image_audio_to_inference_args(tmp_path):
    image = tmp_path / "input.png"
    audio = tmp_path / "speech.wav"
    Image.new("RGB", (640, 640), "white").save(image)
    audio.write_bytes(b"placeholder")
    output_dir = tmp_path / "out"

    args = SimpleNamespace(
        image=image,
        audio=audio,
        manifest=None,
        output_dir=output_dir,
        qwen_model=None,
        caption_prompt=None,
        max_new_tokens=256,
        temperature=0.1,
        overwrite=False,
        skip_caption=True,
    )

    run(args)

    sample_dir = output_dir / "input"
    inference_args = json.loads(
        (sample_dir / "inference_args.json").read_text(encoding="utf-8")
    )
    normalized = json.loads(
        (output_dir / "manifest.normalized.json").read_text(encoding="utf-8")
    )
    assert inference_args["audio"] == str(audio.resolve())
    assert normalized["items"][0]["audio_label"] == audio.name
