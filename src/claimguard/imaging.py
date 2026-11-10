"""Image decoding and identity helpers shared across analysis agents."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, UnidentifiedImageError

MAX_IMAGE_PIXELS = 40_000_000


@dataclass(frozen=True)
class DecodedImage:
    image: Image.Image
    width: int
    height: int
    sha256: str


def decode_image(image_bytes: bytes) -> DecodedImage:
    """Decode and validate image bytes, returning a detached RGB Pillow image."""
    if not image_bytes:
        raise ValueError("The uploaded image is empty.")
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            source.verify()
        with Image.open(io.BytesIO(image_bytes)) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise ValueError(
                    f"Image dimensions exceed the {MAX_IMAGE_PIXELS:,}-pixel safety limit."
                )
            rgb_image = source.convert("RGB")
            return DecodedImage(
                image=rgb_image,
                width=rgb_image.width,
                height=rgb_image.height,
                sha256=hashlib.sha256(image_bytes).hexdigest(),
            )
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("The uploaded bytes are not a supported, valid image.") from error


def image_array(image: Image.Image) -> np.ndarray:
    """Return a contiguous RGB uint8 array."""
    return np.asarray(image.convert("RGB"), dtype=np.uint8)
