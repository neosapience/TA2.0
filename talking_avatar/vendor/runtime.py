from __future__ import annotations

from typing import Any

from .wan.image2video_df_audio import WanTI2V


def create_generation_runtime(**kwargs: Any) -> Any:
    """Create the attributed vendor generation runtime."""
    return WanTI2V(**kwargs)
