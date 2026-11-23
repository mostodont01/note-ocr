"""Image -> Score: the whole recognition pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ingest import binarize, darkness, load_gray
from .model import Score
from .recognition.assemble import assemble, check_measures
from .recognition.detect import Symbol, SymbolDetector
from .recognition.staff import PageLayout, detect_staves, estimate_skew, rotate

DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "models" / "detector.pt"

# symbols erased before counting beams, so they aren't mistaken for beams
_BEAM_MASK_SKIP = {"stem", "barline"}


@dataclass
class PageResult:
    gray: np.ndarray
    layout: PageLayout
    symbols: list[Symbol]
    score: Score
    bad_measures: list[int]


class Recognizer:
    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS, threshold: float = 0.3):
        self.detector = SymbolDetector(weights)
        self.threshold = threshold

    def page(self, gray: np.ndarray, deskew: bool = True, glyphs=None, state=None) -> PageResult:
        """Recognise one page. `glyphs`: SMuFL glyphs from a vector PDF (fast path);
        `state`: clef / key / time carried over from the previous page."""
        ink = binarize(gray)
        if deskew and not glyphs:
            deg = estimate_skew(ink)
            if abs(deg) >= 0.1:
                gray = rotate(gray, deg)
                ink = binarize(gray)
        layout = detect_staves(ink)
        if not layout.staves:
            return PageResult(gray, layout, [], Score(parts=[]), [])
        spacing = float(np.median([s.spacing() for s in layout.staves]))
        symbols = self.detector.detect(gray, spacing, self.threshold)
        if glyphs:
            from .pdf import refine

            symbols = refine(symbols, glyphs, spacing)
        # ink darkness for stems / beams; other symbols blanked so they aren't taken for beams
        dark = darkness(gray)
        for s in symbols:
            if s.cls not in _BEAM_MASK_SKIP:
                dark[max(int(s.y0), 0) : int(np.ceil(s.y1)), max(int(s.x0), 0) : int(np.ceil(s.x1))] = 0.0
        part = assemble(symbols, layout, dark, state)
        bad = check_measures(part)
        return PageResult(gray, layout, symbols, Score(parts=[part]), bad)

    def file(self, path: str | Path) -> PageResult:
        """First page of an image or PDF."""
        return self.document(path)[1][0]

    def document(self, path: str | Path) -> tuple[Score, list[PageResult]]:
        """All pages of an image or PDF -> one score (pages continue the same part)."""
        from .recognition.assemble import _State

        state = _State()
        if str(path).lower().endswith(".pdf"):
            from .pdf import load_pdf

            pages = [self.page(p.gray, glyphs=p.glyphs, state=state) for p in load_pdf(path)]
        else:
            pages = [self.page(load_gray(path), state=state)]
        return merge_pages(pages), pages


def merge_pages(pages: list[PageResult]) -> Score:
    parts = [p.score.parts[0] for p in pages if p.score.parts and p.score.parts[0].measures]
    if not parts:
        return Score(parts=[])
    merged = parts[0]
    for p in parts[1:]:
        merged.measures.extend(p.measures)
    return Score(parts=[merged])
