"""Run the symbol detector on a page and return symbols in page coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .classes import CLASSES, NORM_SPACING, STRIDE

# classes that cannot share a spot: keep only the best one where they overlap
EXCLUSIVE_GROUPS = [
    {"notehead_black", "notehead_half", "notehead_whole", "notehead_x", "notehead_x_half", "notehead_x_whole"},
    {c for c in CLASSES if c.startswith("rest_")},
    {"sharp", "flat", "natural", "double_sharp", "double_flat"},
    {c for c in CLASSES if c.startswith("flag_")},
    {"clef_g", "clef_f", "clef_c", "clef_perc"},
    {c for c in CLASSES if c.startswith("time_")},
    {c for c in CLASSES if c.startswith("dyn_")},
    {"staccato", "accent", "tenuto", "marcato"},
]


@dataclass
class Symbol:
    cls: str
    score: float
    x0: float
    y0: float
    x1: float
    y1: float
    uid: int = -1  # stable id within a page (set by the app session)

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


def decode_np(heat_logits: np.ndarray, size: np.ndarray, offset: np.ndarray, threshold=0.3, max_det=3000):
    """Heatmap peaks -> (class, score, cx, cy, w, h) in input pixels (numpy; one image)."""
    heat = 1.0 / (1.0 + np.exp(-heat_logits))
    c, h, w = heat.shape
    pad = np.pad(heat, ((0, 0), (1, 1), (1, 1)), constant_values=-1)
    neigh = np.max(np.stack([pad[:, dy : dy + h, dx : dx + w] for dy in range(3) for dx in range(3)]), axis=0)
    peaks = (heat >= neigh) & (heat >= threshold)
    cls, ys, xs = np.nonzero(peaks)
    scores = heat[cls, ys, xs]
    order = np.argsort(-scores)[:max_det]
    out = []
    for i in order:
        y, x = ys[i], xs[i]
        ox, oy = offset[:, y, x]
        sw, sh = np.exp(size[:, y, x])
        out.append((int(cls[i]), float(scores[i]), (x + ox) * STRIDE, (y + oy) * STRIDE,
                    sw * NORM_SPACING, sh * NORM_SPACING))
    return out


class SymbolDetector:
    """Runs a .pt checkpoint (PyTorch) or an exported .onnx model (ONNX Runtime, used
    by the packaged app so it doesn't need PyTorch)."""

    def __init__(self, weights: str | Path, threads: int | None = None):
        weights = Path(weights)
        if weights.suffix == ".onnx" or not weights.exists() and weights.with_suffix(".onnx").exists():
            import onnxruntime as ort

            path = weights if weights.suffix == ".onnx" else weights.with_suffix(".onnx")
            opts = ort.SessionOptions()
            if threads:
                opts.intra_op_num_threads = threads
            self._ort = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
            meta = self._ort.get_modelmeta().custom_metadata_map
            self.classes = meta["classes"].split(",") if "classes" in meta else list(CLASSES)
            self.model = None
            return
        import torch

        from .net import Detector

        if threads:
            torch.set_num_threads(threads)
        ckpt = torch.load(weights, map_location="cpu")
        self.classes = list(ckpt.get("classes", CLASSES))
        self.model = Detector(len(self.classes))
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self._ort = None

    def normalise(self, gray: np.ndarray, spacing: float) -> tuple[np.ndarray, float]:
        scale = NORM_SPACING / spacing
        img = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        return img, scale

    def raw(self, x: np.ndarray):
        """(1,1,H,W) float32 -> heat, size, offset as numpy (first image)."""
        if self._ort is not None:
            heat, size, off = self._ort.run(None, {"image": x})
            return heat[0], size[0], off[0]
        import torch

        with torch.no_grad():
            heat, size, off = self.model(torch.from_numpy(x))
        return heat[0].numpy(), size[0].numpy(), off[0].numpy()

    def detect(self, gray: np.ndarray, spacing: float, threshold: float = 0.3) -> list[Symbol]:
        img, scale = self.normalise(gray, spacing)
        h, w = img.shape
        x = (1.0 - img.astype(np.float32) / 255.0)[None, None]
        x = np.pad(x, ((0, 0), (0, 0), (0, (-h) % 16), (0, (-w) % 16)))
        dets = decode_np(*self.raw(x), threshold=threshold)
        out = []
        for cls, score, cx, cy, bw, bh in dets:
            out.append(Symbol(self.classes[cls], score, (cx - bw / 2) / scale, (cy - bh / 2) / scale,
                              (cx + bw / 2) / scale, (cy + bh / 2) / scale))
        return suppress(out)


def export_onnx(weights: str | Path, out: str | Path | None = None) -> Path:
    """.pt checkpoint -> .onnx (dynamic image size), class list stored as metadata."""
    import onnx
    import torch

    from .net import Detector

    ckpt = torch.load(weights, map_location="cpu")
    classes = list(ckpt.get("classes", CLASSES))
    model = Detector(len(classes))
    model.load_state_dict(ckpt["model"])
    model.eval()
    out = Path(out) if out else Path(weights).with_suffix(".onnx")
    torch.onnx.export(
        model, torch.zeros(1, 1, 256, 256), str(out), input_names=["image"], output_names=["heat", "size", "offset"],
        dynamic_axes={"image": {2: "h", 3: "w"}, "heat": {2: "h4", 3: "w4"}, "size": {2: "h4", 3: "w4"},
                      "offset": {2: "h4", 3: "w4"}},
        opset_version=17, dynamo=False,
    )
    m = onnx.load(str(out))
    m.metadata_props.add(key="classes", value=",".join(classes))
    onnx.save(m, str(out))
    return out


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
