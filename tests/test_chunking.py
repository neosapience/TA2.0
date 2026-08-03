from talking_avatar.vendor.wan.image2video_df_audio import calculate_chunk_count


def test_one_context_uses_one_chunk():
    assert calculate_chunk_count(45, 45, 9) == 1


def test_long_video_uses_overlapping_chunks():
    assert calculate_chunk_count(46, 45, 9) == 2
    assert calculate_chunk_count(81, 45, 9) == 2
    assert calculate_chunk_count(82, 45, 9) == 3
