---
license: apache-2.0
pipeline_tag: image-to-video
tags:
  - audio-driven
  - talking_avatar
  - safetensors
---

# TalkingAvatar

### Audio-Driven Talking Avatar Generation by [Neosapience](https://www.neosapience.com/)

TalkingAvatar generates an expressive portrait video from a reference image,
speech audio, and an optional text prompt. Model weights are distributed as
sharded safetensors with an integrity manifest and runtime configuration.

[Source Code](https://github.com/neosapience/TA2.0) |
[Training Dataset](https://huggingface.co/datasets/neosapience/TA2.0_animation_dataset) |
[Neosapience](https://www.neosapience.com/) |
[Typecast](https://typecast.ai/)

## Training Dataset

TalkingAvatar was trained using the
[TA2.0 Animation Dataset](https://huggingface.co/datasets/neosapience/TA2.0_animation_dataset).
See the dataset repository for its documentation, structure, and access
details.

## Quick Start

Clone the source repository and create its uv environment:

```bash
git clone https://github.com/neosapience/TA2.0.git
cd TA2.0

uv venv --system-site-packages .venv
source .venv/bin/activate
uv pip install --no-deps -r requirements.lock
```

Generate a video with the top-level inference script:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python inference.py \
  --image examples/reference.png \
  --audio examples/audio.wav \
  --prompt "The person is speaking moderately." \
  --save_file outputs/result.mp4
```

The script downloads the TalkingAvatar checkpoint and resolves its required
video and audio dependencies through the Hugging Face cache.
Run `python inference.py --help` for all options.

## Inference Profile

| Property | Value |
| --- | --- |
| Resolution | 480 x 832 |
| Frame rate | 25 fps |
| Duration | Derived from the input audio |
| Denoising steps | 30 |
| Text guidance | 5.0 |
| Audio guidance | 3.0 |

The runtime automatically chooses the shortest model-valid frame count that
covers the complete input audio. `--frame_num` remains available as an override.

## Repository Layout

```text
checkpoint/
  model-00001-of-00003.safetensors
  model-00002-of-00003.safetensors
  model-00003-of-00003.safetensors
  model.safetensors.index.json
negative_prompt.safetensors
release-manifest.json
config.json
```

## Intended Use

TalkingAvatar is intended for research, evaluation, and authorized creative
work with audio-driven avatars. Use only images and voices for which you have
the necessary consent and rights, and disclose generated media where
appropriate.

## Limitations

Output quality and lip synchronization vary with pose, occlusion, audio
quality, language, and image domain. The runtime does not provide identity
verification, consent management, watermarking, or content moderation.

## License

TalkingAvatar is released under the Apache License 2.0. See the source
repository for complete license terms, third-party notices, and the full model
card.

## Acknowledgments

TalkingAvatar was developed as part of the **첨단 GPU 활용 지원 사업**
(project period 2026-04-01 – 2026-07-15), supported by
**정보통신산업진흥원** and **한국정보통신진흥협회**.
