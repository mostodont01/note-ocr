"""App logic without Qt: open, recognise, edit, re-assemble, save, export.

Two kinds of edits (see UI.md):
- symbol edits (class change, delete, add) -> exact training labels, music re-assembled;
- note overrides (pitch / duration / accidental of one note) -> applied after assembly.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from fractions import Fraction as F
from pathlib import Path

import cv2
import numpy as np

from ..ingest import binarize, darkness, load_gray
from ..model import Event, Pitch, Score
from ..paths import CORRECTIONS, active_model
from ..pipeline import _BEAM_MASK_SKIP, merge_pages
from ..recognition.assemble import _State, assemble, check_measures
from ..recognition.detect import Symbol, SymbolDetector
from ..recognition.classes import NORM_SPACING
from ..recognition.staff import PageLayout, detect_staves

# default symbol sizes in staff spaces (w, h), used when adding a symbol by hand
DEFAULT_SIZE = {
    "notehead": (1.2, 1.0), "rest_whole": (1.2, 0.6), "rest_half": (1.2, 0.6), "rest": (1.0, 2.6),
    "clef": (2.6, 7.0), "sharp": (1.0, 2.8), "flat": (0.9, 2.4), "natural": (0.7, 2.8),
    "double": (1.0, 1.0), "dot": (0.4, 0.4), "flag": (1.1, 3.0), "time": (1.6, 2.0),
    "tuplet": (0.8, 1.0), "tie": (2.5, 0.8), "barline": (0.2, 4.0), "stem": (0.12, 3.5),
}


def default_size(cls: str) -> tuple[float, float]:
    if cls in DEFAULT_SIZE:
        return DEFAULT_SIZE[cls]
    for k, v in DEFAULT_SIZE.items():
        if cls.startswith(k):
            return v
    return (1.0, 1.0)


@dataclass
class Page:
    gray: np.ndarray
    layout: PageLayout
    symbols: list[Symbol]
    spacing: float
    glyphs: list = field(default_factory=list)
    edited: bool = False
    next_uid: int = 0

    def symbol(self, uid: int) -> Symbol | None:
        return next((s for s in self.symbols if s.uid == uid), None)


@dataclass
class MeasureInfo:
    index: int
    page: int
    x: float
    y: float
    w: float
    h: float
    ok: bool
    total: str
    expected: str


class Session:
    def __init__(self, weights: str | Path | None = None, detector: SymbolDetector | None = None,
                 corrections: str | Path | None = CORRECTIONS):
        self.weights = Path(weights) if weights else active_model()
        self.corrections_dir = Path(corrections) if corrections else None
        self._detector = detector
        self.source: Path | None = None
        self.pages: list[Page] = []
        self.overrides: dict[str, dict] = {}  # "page:uid" -> changes
        self.corrections: list[dict] = []
        self.score: Score = Score(parts=[])
        self.measures: list[MeasureInfo] = []
        self.event_at: dict[tuple[int, int], tuple[int, int]] = {}  # (page, uid) -> (measure, event)

    # ------------------------------------------------------------------ loading

    @property
    def detector(self) -> SymbolDetector:
        if self._detector is None:
            self._detector = SymbolDetector(self.weights)
        return self._detector

    def project_path(self, source: Path | None = None) -> Path:
        source = source or self.source
        return source.with_name(source.name + ".notocr.json")

    def open(self, path: str | Path, use_saved: bool = True):
        path = Path(path)
        self.source = path
        self.pages, self.overrides, self.corrections = [], {}, []
        if path.suffix.lower() == ".pdf":
            from ..pdf import load_pdf

            raw = [(p.gray, p.glyphs) for p in load_pdf(path)]
        else:
            raw = [(load_gray(path), [])]
        saved = self._load_project() if use_saved else None
        for i, (gray, glyphs) in enumerate(raw):
            if not glyphs:
                from ..photo import prepare

                gray = prepare(gray)  # page detection, deskew, dewarp (each only if needed)
            ink = binarize(gray)
            layout = detect_staves(ink)
            spacing = float(np.median([s.spacing() for s in layout.staves])) if layout.staves else 10.0
            page = Page(gray, layout, [], spacing, glyphs)
            if saved and i < len(saved["pages"]):
                sp = saved["pages"][i]
                page.symbols = [Symbol(**s) for s in sp["symbols"]]
                page.edited = sp.get("edited", False)
                page.next_uid = max((s.uid for s in page.symbols), default=-1) + 1
            else:
                self._detect(page)
            self.pages.append(page)
        if saved:
            self.overrides = saved.get("overrides", {})
            self.corrections = saved.get("corrections", [])
        self.reassemble()

    def _detect(self, page: Page):
        if not page.layout.staves:
            page.symbols = []
            return
        symbols = self.detector.detect(page.gray, page.spacing)
        if page.glyphs:
            from ..pdf import refine

            symbols = refine(symbols, page.glyphs, page.spacing)
        for i, s in enumerate(symbols):
            s.uid = i
        page.symbols = symbols
        page.next_uid = len(symbols)

    # --------------------------------------------------------------- assembling

    def reassemble(self):
        state = _State()
        results = []
        for pi, page in enumerate(self.pages):
            if not page.layout.staves:
                continue
            dark = darkness(page.gray)
            for s in page.symbols:
                if s.cls not in _BEAM_MASK_SKIP:
                    dark[max(int(s.y0), 0) : int(np.ceil(s.y1)), max(int(s.x0), 0) : int(np.ceil(s.x1))] = 0.0
            part = assemble(page.symbols, page.layout, dark, state)
            for m in part.measures:
                if m.region:
                    m.region.page = pi
                for e in m.events:
                    if e.region:
                        e.region.page = pi
                    for n in e.notes:
                        if n.region:
                            n.region.page = pi
            results.append(part)

        class _R:  # minimal PageResult stand-in for merge_pages
            def __init__(self, part):
                self.score = Score(parts=[part])

        self.score = merge_pages([_R(p) for p in results])
        self._apply_overrides()
        self._index()

    def _apply_overrides(self):
        for part in self.score.parts:
            for m in part.measures:
                for e in m.events:
                    if not e.region:
                        continue
                    ch = self.overrides.get(f"{e.region.page}:{e.region.uid}")
                    if ch:
                        apply_override(e, ch)

    def _index(self):
        self.measures, self.event_at = [], {}
        if not self.score.parts:
            return
        part = self.score.parts[0]
        bad = set(check_measures(part))
        time_sig = (4, 4)
        for mi, m in enumerate(part.measures):
            time_sig = m.time or time_sig
            total = sum((e.duration for e in m.events if e.staff == 1 and e.voice == 1), F(0))
            r = m.region
            self.measures.append(MeasureInfo(mi, r.page if r else 0, r.x if r else 0, r.y if r else 0,
                                             r.w if r else 0, r.h if r else 0, mi not in bad,
                                             str(total), f"{time_sig[0]}/{time_sig[1]}"))
            for ei, e in enumerate(m.events):
                if e.region:
                    self.event_at[(e.region.page, e.region.uid)] = (mi, ei)
                for n in e.notes:
                    if n.region:
                        self.event_at[(n.region.page, n.region.uid)] = (mi, ei)

    def event_for(self, page: int, uid: int) -> Event | None:
        at = self.event_at.get((page, uid))
        if at is None:
            return None
        return self.score.parts[0].measures[at[0]].events[at[1]]

    # -------------------------------------------------------------------- edits

    def _log(self, page: int, kind: str, **data):
        self.corrections.append({"time": time.time(), "page": page, "kind": kind, **data})

    def set_class(self, page: int, uid: int, cls: str):
        p = self.pages[page]
        s = p.symbol(uid)
        if s is None or s.cls == cls:
            return
        self._log(page, "class", uid=uid, old=s.cls, new=cls, box=[s.x0, s.y0, s.x1, s.y1])
        s.cls = cls
        s.score = 1.0
        p.edited = True
        self.reassemble()

    def delete(self, page: int, uid: int):
        p = self.pages[page]
        s = p.symbol(uid)
        if s is None:
            return
        self._log(page, "delete", uid=uid, old=s.cls, box=[s.x0, s.y0, s.x1, s.y1])
        p.symbols.remove(s)
        p.edited = True
        self.reassemble()

    def add(self, page: int, cls: str, x: float, y: float, w: float | None = None, h: float | None = None) -> int:
        """Add a symbol centred at (x, y); size from same-class symbols on the page."""
        p = self.pages[page]
        same = [(s.w, s.h) for s in p.symbols if s.cls == cls]
        if w is None or h is None:
            if same:
                w, h = np.median(np.array(same), axis=0)
            else:
                fw, fh = default_size(cls)
                w, h = fw * p.spacing, fh * p.spacing
        s = Symbol(cls, 1.0, x - w / 2, y - h / 2, x + w / 2, y + h / 2, uid=p.next_uid)
        p.next_uid += 1
        p.symbols.append(s)
        p.edited = True
        self._log(page, "add", uid=s.uid, new=cls, box=[s.x0, s.y0, s.x1, s.y1])
        self.reassemble()
        return s.uid

    def override(self, page: int, uid: int, **changes):
        """Note override for the event containing symbol `uid` (pitch steps, alter,
        base, dots, rest)."""
        e = self.event_for(page, uid)
        if e is None or not e.region:
            return
        key = f"{e.region.page}:{e.region.uid}"
        cur = self.overrides.setdefault(key, {})
        if "steps" in changes:
            cur["steps"] = cur.get("steps", 0) + changes.pop("steps")
        cur.update(changes)
        self._log(page, "override", key=key, changes=dict(cur))
        self.reassemble()

    # --------------------------------------------------------------- persistence

    def save(self):
        if not self.source:
            return None
        data = {
            "source": str(self.source),
            "pages": [{"edited": p.edited, "symbols": [asdict(s) for s in p.symbols]} for p in self.pages],
            "overrides": self.overrides,
            "corrections": self.corrections,
        }
        path = self.project_path()
        path.write_text(json.dumps(data))
        if self.corrections_dir and any(p.edited for p in self.pages):
            self.export_training(self.corrections_dir)
        return path

    def _load_project(self):
        path = self.project_path()
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (OSError, ValueError):
                return None
        return None

    def export_training(self, root: str | Path, only_edited: bool = True) -> int:
        """Edited pages -> training data (normalised image + boxes), same format as
        the synthetic sets. Returns the number of pages written."""
        from ..recognition.classes import CLASS_INDEX

        root = Path(root)
        (root / "images").mkdir(parents=True, exist_ok=True)
        index_path = root / "index.json"
        index = json.loads(index_path.read_text()) if index_path.exists() else []
        stem = self.source.stem.replace(" ", "_") if self.source else "page"
        n = 0
        for pi, p in enumerate(self.pages):
            if only_edited and not p.edited:
                continue
            k = NORM_SPACING / p.spacing
            img = cv2.resize(p.gray, None, fx=k, fy=k, interpolation=cv2.INTER_LINEAR)
            name = f"images/{stem}_p{pi + 1}.png"
            cv2.imwrite(str(root / name), img)
            boxes = [[CLASS_INDEX[s.cls], s.x0 * k, s.y0 * k, s.x1 * k, s.y1 * k] for s in p.symbols if s.cls in CLASS_INDEX]
            index = [it for it in index if it["image"] != name]
            index.append({"image": name, "font": "corrections", "boxes": boxes})
            n += 1
        index_path.write_text(json.dumps(index))
        return n


def apply_override(e: Event, ch: dict):
    if ch.get("rest"):
        e.notes = []
    if "steps" in ch and e.notes:
        for n in e.notes:
            n.pitch = Pitch.from_diatonic(n.pitch.diatonic + ch["steps"], n.pitch.alter)
    if "alter" in ch and e.notes:
        n = e.notes[0]
        n.pitch = Pitch(n.pitch.step, n.pitch.octave, ch["alter"])
    if "base" in ch:
        e.base = F(ch["base"])
        e.measure_rest = False
    if "dots" in ch:
        e.dots = ch["dots"]
