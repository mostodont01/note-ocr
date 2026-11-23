"""Loading pages and turning them into clean grayscale / binary images."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_gray(path: str | Path) -> np.ndarray:
    """Load an image as uint8 grayscale, white paper / dark ink.

    Transparent pixels (e.g. screenshots with alpha) are put on white.
    """
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return np.asarray(im.convert("L"), dtype=np.uint8).copy()


def binarize(gray: np.ndarray) -> np.ndarray:
    """Ink mask (bool, True = ink) using Otsu's threshold."""
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return ink > 0
