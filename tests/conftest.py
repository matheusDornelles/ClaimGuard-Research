from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def png_bytes() -> bytes:
    array = np.zeros((64, 64, 3), dtype=np.uint8)
    array[:, :, 0] = np.arange(64, dtype=np.uint8)[None, :] * 4
    array[:, :, 1] = np.arange(64, dtype=np.uint8)[:, None] * 4
    array[16:48, 16:48] = (180, 90, 40)
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return stream.getvalue()


@pytest.fixture
def jpeg_bytes(png_bytes: bytes) -> bytes:
    with Image.open(io.BytesIO(png_bytes)) as image:
        stream = io.BytesIO()
        image.save(stream, format="JPEG", quality=83)
    return stream.getvalue()
