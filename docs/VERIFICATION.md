# GPU Verification

Verification date: 2026-07-24

## Environment

Validation was performed in a clean NVIDIA PyTorch 26.03 container on an NVIDIA
H100. The environment used the versioned source checkout, release artifacts,
declared model dependencies, one reference image, one audio input, and an
isolated output directory.
The uv environment inherited the CUDA-enabled PyTorch stack from the container
and installed the pinned runtime packages without resolving a second PyTorch
stack.


## Results

- Unit, runtime-boundary, preprocessing, and consistency tests: 47 passed.
- Release checkpoint: 1,188 keys, 14,541,834,624 tensor bytes.
- Strict load: 1,185 Transformer keys and 3 audio-projection keys matched.
- CUDA allocation after component load: 13,031,007,744 bytes.
- Audio-length-driven one-step end-to-end smoke generation: passed.
- Default 30-step end-to-end generation: passed.
- Duration probe: H.264 video and AAC mono audio both reported 9.04 seconds;
  the MP4 container also reported 9.04 seconds.
- Automatic frame selection: 229 model-valid generation frames for the 9.04
  second input, encoded as 226 output frames at 25 fps.

The default run covered safetensors loading, strict model assembly, image and
audio preprocessing, text encoding, audio-conditioned diffusion, VAE decoding,
video encoding, and audio muxing.
