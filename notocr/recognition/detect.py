"""Run the symbol detector on a page and return symbols in page coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as Fn

from .net import CLASSES, NORM_SPACING, Detector, decode

# classes that cannot share a spot: keep only the best one where they overlap
EXCLUSIVE_GROUPS = [
    {"notehead_black", "notehead_half", "notehead_whole", "notehead_x"},
    {c for c in CLASSES if c.startswith("rest_")},
    {"sharp", "flat", "natural", "double_sharp", "double_flat"},
    {c for c in CLASSES if c.startswith("flag_")},
    {"clef_g", "clef_f", "clef_c", "clef_perc"},
    {c for c in CLASSES if c.startswith("time_")},
]


@dataclass
class Symbol:
    cls: str
    score: float
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0


def _iou(a: Symbol, b: Symbol) -> float:
    ix = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    iy = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    inter = ix * iy
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def suppress(symbols: list[Symbol], iou: float = 0.45) -> list[Symbol]:
    group_of = {}
    for gi, g in enumerate(EXCLUSIVE_GROUPS):
        for c in g:
            group_of[c] = gi
    keep: list[Symbol] = []
    for s in sorted(symbols, key=lambda s: -s.score):
        g = group_of.get(s.cls, s.cls)
        if any(group_of.get(k.cls, k.cls) == g and _iou(s, k) > iou for k in keep):
            continue
        keep.append(s)
    return keep


class SymbolDetector:
    def __init__(self, weights: str | Path, threads: int | None = None):
        if threads:
            torch.set_num_threads(threads)
        ckpt = torch.load(weights, map_location="cpu")
        self.classes = list(ckpt.get("classes", CLASSES))
        self.model = Detector(len(self.classes))
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    def normalise(self, gray: np.ndarray, spacing: float) -> tuple[np.ndarray, float]:
        scale = NORM_SPACING / spacing
        img = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        return img, scale

    @torch.no_grad()
    def detect(self, gray: np.ndarray, spacing: float, threshold: float = 0.3) -> list[Symbol]:
        img, scale = self.normalise(gray, spacing)
        h, w = img.shape
        x = torch.from_numpy(1.0 - img.astype(np.float32) / 255.0)[None, None]
        ph, pw = (-h) % 16, (-w) % 16
        x = Fn.pad(x, (0, pw, 0, ph))
        heat, size, off = self.model(x)
        dets = decode(heat, size, off, threshold=threshold)[0]
        out = []
        for cls, score, cx, cy, bw, bh in dets:
            out.append(
                Symbol(
                    self.classes[cls],
                    score,
                    (cx - bw / 2) / scale,
                    (cy - bh / 2) / scale,
                    (cx + bw / 2) / scale,
                    (cy + bh / 2) / scale,
                )
            )
        return suppress(out)


def draw_symbols(gray: np.ndarray, symbols: list[Symbol], scale: float = 1.0) -> np.ndarray:
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    for s in symbols:
        i = CLASSES.index(s.cls)
        color = [(0, 0, 230), (0, 150, 0), (230, 0, 0), (0, 140, 255), (200, 0, 200), (140, 140, 0)][i % 6]
        p0 = (int(s.x0 * scale), int(s.y0 * scale))
        p1 = (int(s.x1 * scale), int(s.y1 * scale))
        cv2.rectangle(img, p0, p1, color, 1)
    return img
