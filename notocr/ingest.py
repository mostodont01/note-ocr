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


def flatten_light(gray: np.ndarray) -> np.ndarray:
    """Divide out uneven lighting (phone photos, shadows): local paper brightness is
    estimated with a max filter (removes thin ink) and blurred."""
    h, w = gray.shape
    k = max(15, (min(h, w) // 40) | 1)
    bg = cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 2).astype(np.float32)
    out = gray.astype(np.float32) / np.maximum(bg, 1.0) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def binarize(gray: np.ndarray) -> np.ndarray:
    """Ink mask (bool, True = ink): lighting flattened, then Otsu's threshold."""
    _, ink = cv2.threshold(flatten_light(gray), 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return ink > 0


def darkness(gray: np.ndarray) -> np.ndarray:
    """Ink darkness 0..1, normalised: paper -> 0, solid ink -> 1 (handles tinted paper,
    low contrast and grey scans)."""
    g = flatten_light(gray).astype(np.float32)
    paper = float(np.percentile(g, 60))
    ink_px = g[g < paper - 0.3 * max(paper - float(g.min()), 1.0)]
    ink = float(np.percentile(ink_px, 10)) if ink_px.size > 50 else float(g.min())
    if paper - ink < 10:
        ink = paper - 10
    return np.clip((paper - g) / (paper - ink), 0.0, 1.0)
