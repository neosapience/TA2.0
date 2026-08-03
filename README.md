<div align="center">

# TalkingAvatar (TA2.0): Audio-Driven Talking Avatar Generation

### Reference Image + Speech Audio → Expressive Avatar Video by [Neosapience](https://www.neosapience.com/)

[![Neosapience](https://img.shields.io/badge/Neosapience-Voice%20%26%20Avatar%20AI-6C5CE7)](https://www.neosapience.com/)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-2EA44F)](LICENSE)

[Neosapience](https://www.neosapience.com/) |
[Model Card](MODEL_CARD.md) |
[Model Weights](https://huggingface.co/neosapience/TA2.0) |
[Training Dataset](https://huggingface.co/datasets/neosapience/TA2.0_animation_dataset) |
[Verification](docs/VERIFICATION.md)

</div>

## Demo

<table align="center">
  <tr>
    <th>Reference Image</th>
    <th>Generated Talking Avatar</th>
  </tr>
  <tr>
    <td><img src="examples/reference.png" width="240" alt="TalkingAvatar reference image"></td>
    <td><img src="assets/talking-avatar-demo.webp" width="240" alt="Audio-driven TalkingAvatar generation demo"></td>
  </tr>
</table>

<p align="center">
  <sub>Generated from the bundled reference image and speech audio. The animated
  preview omits audio; inference writes an MP4 with the original speech.</sub>
</p>

## Overview

**TalkingAvatar (TA2.0)** is Neosapience's audio-driven talking avatar generation
model. It creates expressive, lip-synchronized avatar videos from a reference
image, speech audio, and an optional text prompt.

This repository provides the inference runtime and release configuration for
TalkingAvatar.

## Training Dataset

TalkingAvatar was trained using the
[TA2.0 Animation Dataset](https://huggingface.co/datasets/neosapience/TA2.0_animation_dataset).
See the dataset repository for its documentation, structure, and access
details.

## Highlights

- **Script-first inference** - run the top-level `inference.py` directly.
- **Automatic model composition** - resolves the video, VAE, text, and audio
  assets declared by the model configuration.
- **Focused runtime** - keeps inference, model loading, preprocessing, and
  output encoding in a concise, auditable package.
- **Artifact integrity** - release weights use sharded safetensors with an index
  and a SHA-256 manifest.
- **Reproducible profile** - defines release defaults for resolution, frame
  rate, sampling, and guidance.

## Quick Start

### 1. Clone

```bash
git clone https://github.com/neosapience/TA2.0.git
cd TA2.0
```

### 2. Create the uv environment

The verified NVIDIA container already provides CUDA-enabled PyTorch, so the
environment inherits system packages instead of replacing PyTorch with a PyPI
build.

```bash
uv venv --system-site-packages .venv
source .venv/bin/activate
uv pip install --no-deps -r requirements.lock
```

### 3. Run inference

Place a consented reference image and speech recording under `examples/`, then
run:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python inference.py \
  --image examples/reference.png \
  --audio examples/audio.wav \
  --prompt "The person is speaking moderately." \
  --save_file outputs/result.mp4
```

Run `python inference.py --help` for the complete option list.

### 4. Optional: preprocess a raw reference image

TA2.0 works best when the reference is already composed like an upper-body
portrait: shoulders or chest visible, the head slightly above center, and no
tiny full-body framing. The examples below show the intended 480x832 reference
composition.

![Reference alignment examples](assets/reference_alignment_examples.jpg)

The preprocessing workflow center-crops a raw image to the released 480x832
inference frame and asks Qwen3-VL for a caption.
The script writes the normalized reference image, caption, metadata, and
ready-to-copy inference arguments under one output folder.

For a single image, optional `--audio` records the paired speech path in
`inference_args.json` so the result can be passed directly to inference. It
does not modify or copy the audio; omit it for image-only preprocessing.

```bash
uv pip install --no-deps -r requirements.preprocess-caption.txt

CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/preprocess.py \
  --image examples/reference.png \
  --audio examples/audio.wav \
  --qwen-model Qwen/Qwen3-VL-8B-Instruct \
  --output-dir outputs/preprocess/demo \
  --overwrite
```

For a single image, the sample folder is named from the image stem:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python inference.py \
  --image outputs/preprocess/demo/reference/01_reference_480x832.png \
  --audio examples/audio.wav \
  --prompt "$(cat outputs/preprocess/demo/reference/02_caption.txt)" \
  --save_file outputs/result.mp4
```

For manifest-based preprocessing, use the included example:

```json
{
  "items": [
    {
      "id": "example",
      "image": "reference.png",
      "audio": "audio.wav"
    }
  ]
}
```

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/preprocess.py \
  --manifest examples/preprocess_manifest.json \
  --qwen-model Qwen/Qwen3-VL-8B-Instruct \
  --output-dir outputs/preprocess/batch \
  --overwrite
```

## Model Assets

The first inference run automatically downloads and caches the following
components:

| Component | Source | Purpose |
| --- | --- | --- |
| TalkingAvatar | `neosapience/TA2.0` | Audio-conditioned Transformer and audio projection |
| Video components | Resolved from model config | Video generation, VAE, and text encoding |
| Audio components | Resolved from model config | Speech feature extraction |


## Generation Options

| Option | Default | Description |
| --- | ---: | --- |
| `--frame_num` | Audio length | Optional override; must be `4n+1` |
| `--mask` | Full image | Optional image indicating where audio conditioning is focused |
| `--sample_steps` | 30 | Denoising steps |
| `--sample_shift` | 5.0 | Flow-matching shift |
| `--text_guide_scale` | 5.0 | Text guidance scale |
| `--audio_guide_scale` | 3.0 | Audio guidance scale |
| `--base_seed` | 42 | Random seed |
| `--save_file` | Timestamped MP4 | Output path |

## Inference Profile

| Property | Value |
| --- | --- |
| Resolution | 480 x 832 |
| Frame rate | 25 fps |
| Duration | Derived from the input audio |
| Denoising steps | 30 |
| Text guidance | 5.0 |
| Audio guidance | 3.0 |
| Seed | 42 |

By default, TalkingAvatar selects the shortest valid `4n+1` frame count that
covers the full input audio at 25 fps. The generated stream and AAC audio are
muxed to the original audio duration. Passing `--frame_num` explicitly keeps
the manual fixed-length workflow; audio is then trimmed or zero-padded to that
requested duration.

Artifact sizes and SHA-256 hashes are recorded in
[release-manifest.json](release-manifest.json).

## Repository Structure

```text
TA2.0/
|-- assets/                     # README images and examples
|-- configs/preprocess/         # example Qwen captioning config
|-- inference.py                # inference entry point
|-- scripts/                    # standalone preprocessing utilities
|-- talking_avatar/             # model loading and generation implementation
|   |-- config.py               # released inference profile
|   |-- generator.py            # TalkingAvatar generation interface
|   |-- hub.py                  # Hugging Face snapshot resolution
|   |-- models.py               # video, VAE, text, and audio model loading
|   |-- pipeline.py             # TalkingAvatar generation flow
|   |-- preprocess/             # image normalization and caption helpers
|   |-- preprocessing.py        # image and audio preprocessing
|   `-- vendor/                 # attributed generation runtime modules
|-- examples/                   # reference image and speech audio
|-- tests/
|-- MODEL_CARD.md
|-- release-manifest.json
|-- requirements.lock
`-- pyproject.toml
```

## Verification

The release was verified in an NVIDIA PyTorch 26.03 container on an NVIDIA H100.

- Unit, runtime-boundary, and output-timing tests passed.
- All checkpoint shards passed manifest size and SHA-256 verification.
- Transformer and audio-projection weights loaded strictly.
- Both one-step smoke inference and the default 30-step generation completed.
- Generated outputs were H.264, 480 x 832, 25 fps, with AAC mono audio.

See [docs/VERIFICATION.md](docs/VERIFICATION.md) for the verification record.

```bash
uv pip install "pytest>=8.3"
uv run --no-sync pytest
```

## Responsible Use

TalkingAvatar is intended for research, evaluation, and authorized creative
work with audio-driven avatars. Use only images and voices for which you have
the necessary consent and rights. Generated media should be disclosed where
appropriate. See the [Model Card](MODEL_CARD.md) for intended use and
limitations.

## About Neosapience

TalkingAvatar is developed by **[Neosapience, Inc.](https://www.neosapience.com/)**,
a generative AI company focused on expressive voice and avatar technologies.
Neosapience also develops [Typecast](https://typecast.ai/), an AI-powered content
creation platform for voices and avatars.

## License

Copyright 2026 Neosapience, Inc. Licensed under the
[Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for third-party attribution.

## Acknowledgments

TalkingAvatar was developed as part of the **첨단 GPU 활용 지원 사업**
(project period 2026-04-01 – 2026-07-15), supported by
**정보통신산업진흥원** and **한국정보통신진흥협회**.
