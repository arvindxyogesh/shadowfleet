import io

import pytest
from PIL import Image


@pytest.fixture
def sample_image_bytes() -> bytes:
    image = Image.new("RGB", (320, 240), color=(120, 130, 140))
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return buf.getvalue()
