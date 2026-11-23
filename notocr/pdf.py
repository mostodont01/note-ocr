"""PDF input.

Every page is rendered at a resolution picked from its staff spacing (so the
detector sees what it was trained on), then recognised like an image.

Fast path for PDFs exported from notation software (MuseScore, Sibelius, Dorico,
Finale, LilyPond...): their music fonts follow SMuFL, so each glyph's code point says
exactly what it is. Those glyphs correct the detector's classes and fill in symbols it
missed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pymupdf

from .ingest import binarize
from .recognition.detect import EXCLUSIVE_GROUPS, Symbol
from .recognition.staff import detect_staves

TARGET_SPACING = 12.0  # px between staff lines in the rendered page

# SMuFL code point -> detector class
SMUFL = {
    0xE0A4: "notehead_black", 0xE0A3: "notehead_half", 0xE0A2: "notehead_whole", 0xE0A9: "notehead_x",
    0xE0A8: "notehead_x", 0xE0A7: "notehead_x",
    0xE4E3: "rest_whole", 0xE4E4: "rest_half", 0xE4E5: "rest_quarter", 0xE4E6: "rest_8th",
    0xE4E7: "rest_16th", 0xE4E8: "rest_32nd",
    0xE050: "clef_g", 0xE07A: "clef_g", 0xE062: "clef_f", 0xE07C: "clef_f", 0xE05C: "clef_c", 0xE07B: "clef_c",
    0xE069: "clef_perc", 0xE06A: "clef_perc",
    0xE262: "sharp", 0xE260: "flat", 0xE261: "natural", 0xE263: "double_sharp", 0xE264: "double_flat",
    0xE1E7: "dot",
    0xE240: "flag_8th_up", 0xE241: "flag_8th_down", 0xE242: "flag_16th_up", 0xE243: "flag_16th_down",
    0xE244: "flag_32nd_up", 0xE245: "flag_32nd_down",
    **{0xE080 + i: f"time_{i}" for i in range(1, 10)},
    0xE08A: "time_common", 0xE08B: "time_cut",
    0xE883: "tuplet_3",
}
# classes whose glyph we may add when the detector missed it (box from same-class detections)
ADDABLE = {"notehead_black", "notehead_half", "notehead_whole", "notehead_x", "sharp", "flat", "natural",
           "dot", "rest_quarter", "rest_8th", "rest_16th", "rest_32nd"}


@dataclass
class Glyph:
    cls: str
    x: float  # origin (left edge, baseline) in rendered pixels
    y: float
    box: tuple[float, float, float, float]  # font bbox in rendered pixels (loose)


@dataclass
class PdfPage:
    gray: np.ndarray
    dpi: float
    glyphs: list[Glyph]


def _render(page, dpi: float) -> np.ndarray:
    pix = page.get_pixmap(dpi=int(round(dpi)), colorspace=pymupdf.csGRAY, alpha=False)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)[:, : pix.width].copy()


def _glyphs(page, scale: float) -> list[Glyph]:
    out = []
    for b in page.get_text("rawdict")["blocks"]:
        for line in b.get("lines", []):
            for span in line["spans"]:
                for ch in span["chars"]:
                    cls = SMUFL.get(ord(ch["c"]))
                    if cls:
                        ox, oy = ch["origin"]
                        x0, y0, x1, y1 = ch["bbox"]
                        out.append(Glyph(cls, ox * scale, oy * scale, (x0 * scale, y0 * scale, x1 * scale, y1 * scale)))
    return out


def load_pdf(path: str | Path, pages: list[int] | None = None) -> list[PdfPage]:
    doc = pymupdf.open(path)
    out = []
    for i, page in enumerate(doc):
        if pages is not None and i not in pages:
            continue
        probe = _render(page, 100)
        layout = detect_staves(binarize(probe))
        if layout.staves:
            spacing = float(np.median([s.spacing() for s in layout.staves]))
            dpi = float(np.clip(100 * TARGET_SPACING / spacing, 72, 400))
        else:
            dpi = 200.0
        out.append(PdfPage(_render(page, dpi), dpi, _glyphs(page, dpi / 72)))
    return out


def refine(symbols: list[Symbol], glyphs: list[Glyph], spacing: float) -> list[Symbol]:
    """Correct detector classes with exact vector glyphs; add missed glyphs."""
    if not glyphs:
        return symbols
    group_of = {}
    for gi, g in enumerate(EXCLUSIVE_GROUPS):
        for c in g:
            group_of[c] = gi
    sizes: dict[str, list[tuple[float, float]]] = {}
    for s in symbols:
        sizes.setdefault(s.cls, []).append((s.w, s.h))
    used = set()
    out = list(symbols)
    for g in glyphs:
        grp = group_of.get(g.cls, g.cls)
        # SMuFL glyphs start at their origin; vertical anchor varies, so match by the
        # horizontal start and the loose font box
        best, best_d = None, 1e9
        for i, s in enumerate(out):
            if i in used or group_of.get(s.cls, s.cls) != grp:
                continue
            if not (g.box[1] - 0.5 * spacing <= s.cy <= g.box[3] + 0.5 * spacing):
                continue
            dx = abs(s.x0 - g.x)
            if dx < 0.6 * spacing and dx < best_d:
                best, best_d = i, dx
        if best is not None:
            used.add(best)
            if out[best].cls != g.cls:
                s = out[best]
                out[best] = Symbol(g.cls, max(s.score, 0.9), s.x0, s.y0, s.x1, s.y1)
            continue
        if g.cls in ADDABLE and g.cls in sizes:
            w, h = np.median(np.array(sizes[g.cls]), axis=0)
            # noteheads, accidentals, dots, rests: centred on the baseline in SMuFL
            out.append(Symbol(g.cls, 0.9, g.x, g.y - h / 2, g.x + w, g.y + h / 2))
            used.add(len(out) - 1)
    return out
