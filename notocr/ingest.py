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


def darkness(gray: np.ndarray) -> np.ndarray:
    """Ink darkness 0..1, normalised: paper -> 0, solid ink -> 1 (handles tinted paper,
    low contrast and grey scans)."""
    g = gray.astype(np.float32)
    paper = float(np.percentile(g, 60))
    ink_px = g[g < paper - 0.3 * max(paper - float(g.min()), 1.0)]
    ink = float(np.percentile(ink_px, 10)) if ink_px.size > 50 else float(g.min())
    if paper - ink < 10:
        ink = paper - 10
    return np.clip((paper - g) / (paper - ink), 0.0, 1.0)
