from PIL import Image

from talking_avatar.preprocess.image_ops import TA2_HEIGHT, TA2_WIDTH, center_crop_resize


def test_center_crop_resize_outputs_ta2_size():
    image = Image.new("RGB", (1200, 800), "white")

    result = center_crop_resize(image)

    assert result.size == (TA2_WIDTH, TA2_HEIGHT)


def test_center_crop_resize_keeps_center_content():
    image = Image.new("RGB", (1000, 1000), "black")
    for x in range(490, 510):
        for y in range(490, 510):
            image.putpixel((x, y), (255, 0, 0))

    result = center_crop_resize(image)

    center = result.getpixel((TA2_WIDTH // 2, TA2_HEIGHT // 2))
    assert center[0] > 200
