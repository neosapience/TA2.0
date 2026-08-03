# TalkingAvatar Model Card

## Model Description

TalkingAvatar is an audio-driven talking avatar model developed by Neosapience.
Given a reference image, speech audio, and an optional text prompt, it generates
a portrait video synchronized to the input speech.

## Model Details

| Field | Value |
| --- | --- |
| Developer | Neosapience, Inc. |
| Model type | Audio-driven image-to-video generation |
| Inputs | Reference image, speech audio, optional text prompt |
| Output | Portrait video with synchronized audio |
| License | Apache-2.0 |

## Inference Profile

| Field | Value |
| --- | --- |
| Resolution | 480 x 832 |
| Frame rate | 25 fps |
| Duration | Derived from the input audio |
| Text guidance | 5.0 |
| Audio guidance | 3.0 |
| Default steps | 30 |
| Default seed | 42 |

## Model Files

The Hugging Face model repository
[`neosapience/TA2.0`](https://huggingface.co/neosapience/TA2.0) distributes:

1. `checkpoint/model.safetensors.index.json` and its referenced shards.
2. `negative_prompt.safetensors`, containing the default negative-prompt
   embedding.
3. `release-manifest.json`, containing artifact sizes and SHA-256 hashes.
4. `config.json`, declaring the runtime model dependencies.

The top-level `inference.py` script resolves these files automatically.

## Intended Use

TalkingAvatar is intended for research, evaluation, and authorized creative
work involving audio-driven avatar generation. Users should have the necessary
rights and consent for every likeness and voice supplied as input, and should
disclose generated media where appropriate.

## Out-of-Scope Use

Do not use TalkingAvatar for deceptive impersonation, non-consensual identity
or voice replication, harassment, fraud, or other unlawful activity.

## Limitations

- The released inference profile supports 25 fps.
- The audio encoder is used for speech features; its CTC language-model head is
  not used.
- Full-frame audio attention may influence motion outside the face region.
- Output quality and lip synchronization vary with pose, occlusion, audio
  quality, language, and image domain.
- Long-form generation may accumulate temporal inconsistencies across chunks.

## Safety Considerations

The runtime does not perform identity verification, consent management,
watermarking, or content moderation. Applications built with TalkingAvatar
should add safeguards appropriate to their users and deployment context.

## License

TalkingAvatar is released under the [Apache License 2.0](LICENSE). Third-party
attribution is provided in [NOTICE](NOTICE).
