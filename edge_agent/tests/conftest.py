import io
import os

# Must run before `edge_agent.app.config` is imported anywhere (its
# module-level `settings = Settings()` reads the environment at import
# time), so /admin/model (NFR-8) is reachable with the right token.
os.environ.setdefault("SHADOWFLEET_SERVICE_TOKEN", "test-service-token")

import pytest  # noqa: E402
from PIL import Image


@pytest.fixture
def sample_image_bytes() -> bytes:
    image = Image.new("RGB", (320, 240), color=(120, 130, 140))
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return buf.getvalue()
