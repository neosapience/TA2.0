#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from talking_avatar.preprocess.image_ops import (
    TA2_HEIGHT,
    TA2_WIDTH,
    center_crop_resize,
    open_rgb_image,
    save_png,
)
from talking_avatar.preprocess.manifest import PreprocessItem, item_from_image, load_manifest
from talking_avatar.preprocess.outputs import (
    artifact,
    default_run_id,
    normalized_manifest,
    write_failure,
    write_json,
)
from talking_avatar.preprocess.qwen_caption import CaptionConfig, QwenImageCaptioner
from talking_avatar.preprocess.validation import (
    validate_item,
    validate_model_path_or_id,
    validate_output_dir,
)


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess TalkingAvatar inputs.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=_path, help="Single reference image.")
    source.add_argument("--manifest", type=_path, help="JSON image manifest.")
    parser.add_argument(
        "--audio",
        type=_path,
        help="Speech audio paired with --image and written to inference_args.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=_path,
        help="Output directory. Defaults to outputs/preprocess/<run_id>.",
    )
    parser.add_argument("--qwen-model", default=os.environ.get("QWEN3VL_MODEL"))
    parser.add_argument("--caption-prompt", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-caption",
        action="store_true",
        help="Write a placeholder caption for local structural tests only.",
    )
    return parser.parse_args(argv)


def _load_items(args: argparse.Namespace) -> list[PreprocessItem]:
    audio = getattr(args, "audio", None)
    if args.image:
        return [item_from_image(args.image, audio=audio)]
    if audio is not None:
        raise ValueError("--audio can only be used with --image")
    return load_manifest(args.manifest)


def _captioner(args: argparse.Namespace) -> QwenImageCaptioner | None:
    validate_model_path_or_id(args.qwen_model, require=not args.skip_caption)
    if args.skip_caption:
        return None
    config = CaptionConfig(
        model_path=args.qwen_model,
        prompt=args.caption_prompt or CaptionConfig.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    return QwenImageCaptioner(config)


def _redacted_error(error: Exception, item: PreprocessItem) -> str:
    message = str(error)
    for path in (item.image, item.audio):
        if path is not None:
            message = message.replace(str(path), path.name)
    return message


def preprocess_item(
    item: PreprocessItem,
    output_root: Path,
    captioner: QwenImageCaptioner | None,
) -> dict:
    source_size = validate_item(item)
    sample_dir = output_root / item.sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    original_path = sample_dir / "00_original.png"
    reference_path = sample_dir / "01_reference_480x832.png"
    caption_path = sample_dir / "02_caption.txt"

    image = open_rgb_image(item.image)
    save_png(image, original_path)
    reference = center_crop_resize(image, width=TA2_WIDTH, height=TA2_HEIGHT)
    save_png(reference, reference_path)

    if captioner is None:
        caption = "Caption skipped by --skip-caption."
    else:
        caption = captioner.caption(reference_path)
    caption_path.write_text(caption + "\n", encoding="utf-8")

    artifacts = {
        "original": artifact(original_path, sample_dir),
        "reference": artifact(reference_path, sample_dir),
        "caption": artifact(caption_path, sample_dir),
    }
    preprocess_payload = {
        "artifacts": artifacts,
        "branch": "preprocess",
        "id": item.sample_id,
        "output_size": [TA2_WIDTH, TA2_HEIGHT],
        "source_label": item.source_label,
        "source_size": list(source_size),
        "status": "done",
    }
    write_json(sample_dir / "preprocess.json", preprocess_payload)
    write_json(
        sample_dir / "inference_args.json",
        {
            "audio": str(item.audio) if item.audio else None,
            "height": TA2_HEIGHT,
            "image": artifact(reference_path, sample_dir),
            "prompt": caption,
            "width": TA2_WIDTH,
        },
    )
    return {
        "artifacts": artifacts,
        "id": item.sample_id,
        "source_label": item.source_label,
        "status": "done",
    }


def run(args: argparse.Namespace) -> Path:
    items = _load_items(args)
    captioner = _captioner(args)
    output_root = args.output_dir or Path("outputs/preprocess") / default_run_id("run")
    output_root = output_root.expanduser().resolve()
    validate_output_dir(output_root, overwrite=args.overwrite)
    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "manifest.normalized.json", normalized_manifest(items))

    failures_path = output_root / "failures.jsonl"
    results = []
    for item in items:
        try:
            results.append(preprocess_item(item, output_root, captioner))
        except Exception as exc:
            error = _redacted_error(exc, item)
            write_failure(failures_path, item.sample_id, error)
            results.append(
                {
                    "error": error,
                    "id": item.sample_id,
                    "source_label": item.source_label,
                    "status": "failed",
                }
            )

    write_json(
        output_root / "summary.json",
        {
            "branch": "preprocess",
            "failed": sum(item["status"] == "failed" for item in results),
            "items": results,
            "succeeded": sum(item["status"] == "done" for item in results),
            "total": len(results),
        },
    )
    return output_root


def main(argv: Sequence[str] | None = None) -> None:
    output_root = run(_parse_args(argv))
    print(output_root)


if __name__ == "__main__":
    main()
