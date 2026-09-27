from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image

from app.main import VISION_IMAGE_BATCH_MAX_B64_CHARS, _encode_vision_images


def test_vision_image_batch_stays_under_proxy_budget(tmp_path):
    paths = []
    for index in range(12):
        # Noisy source images make this exercise the batch-level compression path
        # instead of passing only because the test PNGs are trivially compressible.
        image = Image.effect_noise((1200, 900), 120).convert("RGB")
        path = tmp_path / f"vision-{index}.png"
        image.save(path, format="PNG")
        paths.append(path)

    encoded = _encode_vision_images(paths)

    assert len(encoded) == len(paths)
    assert sum(len(item) for item in encoded) <= VISION_IMAGE_BATCH_MAX_B64_CHARS

    for item in encoded:
        with Image.open(BytesIO(base64.b64decode(item))) as decoded:
            assert decoded.format == "JPEG"
            assert max(decoded.size) <= 640
