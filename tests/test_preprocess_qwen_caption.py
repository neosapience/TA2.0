from talking_avatar.preprocess.qwen_caption import normalize_video_kwargs


def test_normalize_video_kwargs_drops_empty_image_only_lists():
    assert normalize_video_kwargs({"fps": [], "other": [3]}) == {"other": 3}
