from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_IMAGE_CAPTION_PROMPT = (
    "Describe the visible avatar reference image for a talking-avatar video prompt. "
    "Focus on appearance, framing, clothing, art style, expression, and background. "
    "Do not infer identity. Use one concise paragraph."
)


@dataclass(frozen=True)
class CaptionConfig:
    model_path: str
    prompt: str = DEFAULT_IMAGE_CAPTION_PROMPT
    max_new_tokens: int = 256
    temperature: float = 0.1
    device_map: str = "auto"


class QwenImageCaptioner:
    def __init__(self, config: CaptionConfig):
        self.config = config
        self.processor = None
        self.model = None

    def _load(self) -> None:
        if self.model is not None:
            return

        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor

        try:
            from transformers import Qwen3VLForConditionalGeneration as model_cls
        except ImportError:
            from transformers import AutoModelForImageTextToText as model_cls

        self._process_vision_info = process_vision_info
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
        )
        self.model = model_cls.from_pretrained(
            self.config.model_path,
            dtype=torch.bfloat16,
            device_map=self.config.device_map,
            trust_remote_code=True,
        )
        self.model.eval()

    @staticmethod
    def _messages(image_path: Path, prompt: str) -> list[dict]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def caption(self, image_path: Path) -> str:
        self._load()
        import torch

        messages = self._messages(image_path.resolve(), self.config.prompt)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs, video_kwargs = self._process_vision_info(
            messages,
            return_video_kwargs=True,
        )
        video_kwargs = normalize_video_kwargs(video_kwargs)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        device = getattr(self.model, "device", None)
        if device is None:
            device = next(self.model.parameters()).device
        inputs = inputs.to(device)

        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.temperature > 0,
        }
        if self.config.temperature > 0:
            generation_kwargs["temperature"] = self.config.temperature

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation_kwargs)
        input_len = inputs["input_ids"].shape[1]
        return self.processor.decode(
            output_ids[0][input_len:],
            skip_special_tokens=True,
        ).strip()


def normalize_video_kwargs(video_kwargs: dict | None) -> dict:
    normalized = {}
    for key, value in (video_kwargs or {}).items():
        if isinstance(value, list):
            if not value:
                continue
            if len(value) == 1:
                value = value[0]
        normalized[key] = value
    return normalized
