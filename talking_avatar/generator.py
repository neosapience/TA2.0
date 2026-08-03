from __future__ import annotations

from typing import Any

from .vendor.runtime import create_generation_runtime


class TalkingAvatarGenerator:
    """Internal generation interface used by TalkingAvatarPipeline."""

    def __init__(self, **components: Any) -> None:
        self._runtime = create_generation_runtime(**components)

    def generate(self, **kwargs: Any) -> Any:
        return self._runtime.generate(**kwargs)
