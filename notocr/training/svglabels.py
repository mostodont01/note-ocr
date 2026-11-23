"""Training labels from MuseScore SVG exports.

MuseScore writes every element with a class (Note, Rest, Hook, Accidental, ...) and
absolute coordinates. The glyph *subtype* (filled / half / whole notehead, 8th / 16th
flag, sharp / flat ...) is not tagged, so each glyph outline gets a shape signature,
and a calibration table maps signatures to classes. The table is built automatically
by rendering tiny scores that contain only one known glyph type (per music font).
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from fractions import Fraction as F
from pathlib import Path

from ..export import musescore
from ..export.musicxml import write_musicxml
from ..model import ALTO, BASS, PERCUSSION, TENOR, TREBLE, Event, Measure, NoteHead, Part, Pitch, Score

SVG_NS = "{http://www.w3.org/2000/svg}"
NUM = re.compile(r"-?\d+(?:\.\d+)?(?:e-?\d+)?")

# SVG classes whose glyph subtype is looked up in the calibration table
GLYPH_GROUPS = {"Note": "Note", "Rest": "Rest", "Hook": "Hook", "Accidental": "Accidental",
                "KeySig": "Accidental", "Clef": "Clef", "TimeSig": "TimeSig"}
# SVG classes with a fixed label
FIXED = {"NoteDot": "dot", "TieSegment": "tie", "BarLine": "barline", "Stem": "stem"}


@dataclass
class Element:
    cls: str
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 (SVG units)
    sig: str = ""
    polyline: bool = False


def _apply_transform(nums: list[float], transform: str | None) -> list[float]:
    """Apply an SVG matrix(a,b,c,d,e,f) transform to a flat list of x, y pairs."""
    if not transform:
        return nums
    m = re.search(r"matrix\(([^)]*)\)", transform)
    if not m:
        return nums
    a, b, c, d, e, f = (float(v) for v in NUM.findall(m.group(1)))
    out = []
    for x, y in zip(nums[0::2], nums[1::2]):
        out += [a * x + c * y + e, b * x + d * y + f]
    return out


def _bbox(nums: list[float], pad: float = 0.0):
    xs, ys = nums[0::2], nums[1::2]
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)


def parse_svg(path: str | Path) -> tuple[list[Element], tuple[float, float], float]:
    """Elements, (width, height) of the viewBox, staff space (line distance).

    Staff lines come back as elements of class "StaffLines" / "LedgerLine"."""
    root = ET.parse(path).getroot()
    vb = [float(v) for v in root.get("viewBox").split()]
    elements: list[Element] = []
    staff_ys: list[float] = []
    for el in root.iter():
        cls = el.get("class")
        if not cls:
            continue
        tag = el.tag.replace(SVG_NS, "")
        if tag == "polyline":
            nums = _apply_transform([float(v) for v in NUM.findall(el.get("points", ""))], el.get("transform"))
            if cls == "StaffLines":
                staff_ys.append(nums[1])
            pad = float(el.get("stroke-width", 0)) / 2
            elements.append(Element(cls, _bbox(nums, pad), polyline=True))
        elif tag == "path":
            d = el.get("d", "")
            nums = _apply_transform([float(v) for v in NUM.findall(d)], el.get("transform"))
            if len(nums) < 2:
                continue
            box = _bbox(nums, float(el.get("stroke-width", 0)) / 2)
            letters = re.sub(r"[^A-Za-z]", "", d)
            w, h = box[2] - box[0], box[3] - box[1]
            # start point relative to the box separates rotated twins (6 / 9)
            sx = (nums[0] - box[0]) / max(w, 1e-6)
            sy = (nums[1] - box[1]) / max(h, 1e-6)
            sig = f"{letters}|{w / max(h, 1e-6):.1f}|{sx:.1f},{sy:.1f}"
            elements.append(Element(cls, box, sig))
    ys = sorted(set(round(y, 1) for y in staff_ys))
    gaps = [b - a for a, b in zip(ys, ys[1:]) if b - a > 0]
    space = sorted(gaps)[len(gaps) // 4] if gaps else 0.0  # small quantile = in-staff gap
    return elements, (vb[2], vb[3]), space


# --------------------------------------------------------------------- calibration


def _score(measures, clef=TREBLE, key=0, time=(4, 4), time_symbol=None, percussion=False) -> Score:
    m0 = measures[0]
    m0.clefs, m0.key, m0.time, m0.time_symbol = {1: clef}, key, time, time_symbol
    return Score(parts=[Part(measures=measures, percussion=percussion)])


def _drum_notes(drum_ids, base, n_measures=4):
    from ..percussion import BY_ID

    per = int(F(4) / base)
    ms = []
    for _ in range(n_measures):
        evs = []
        for i in range(per):
            d = BY_ID[drum_ids[i % len(drum_ids)]]
            evs.append(Event([NoteHead(d.display, notehead=d.notehead, instrument=d.id)], base))
        ms.append(Measure(evs))
    return ms


def _notes(pitches, base, n_measures=4, rest_between=False, alter=0):
    per = int(F(4) / base)
    ms = []
    for _ in range(n_measures):
        evs = []
        i = 0
        while len(evs) < per:
            p = pitches[i % len(pitches)]
            evs.append(Event([NoteHead(Pitch(p[0], p[1], alter))], base))
            if rest_between and len(evs) < per:
                evs.append(Event([], base))
            i += 1
        ms.append(Measure(evs))
    return ms


def _rests(base, n_measures=4, note_between=True):
    per = int(F(4) / base)
    ms = []
    for _ in range(n_measures):
        evs = []
        while len(evs) < per:
            evs.append(Event([], base))
            if note_between and len(evs) < per:
                evs.append(Event([NoteHead(Pitch("B", 4))], base))
        ms.append(Measure(evs))
    return ms


LOW = [("E", 4), ("F", 4), ("G", 4)]  # stems up in treble
HIGH = [("A", 5), ("G", 5), ("B", 5)]  # stems down


def calibration_scores() -> list[tuple[str, str, Score, str | None]]:
    """(label, SVG group, score, which) — `which` picks top/bottom TimeSig numbers."""
    out: list[tuple[str, str, Score, str | None]] = []
    out.append(("notehead_black", "Note", _score(_notes(LOW + HIGH, F(1))), None))
    out.append(("notehead_half", "Note", _score(_notes(LOW + HIGH, F(2))), None))
    out.append(("notehead_whole", "Note", _score(_notes(LOW + HIGH, F(4))), None))
    out.append(("rest_whole", "Rest", _score([Measure([Event([], F(4), measure_rest=True)]) for _ in range(4)]), None))
    out.append(("rest_whole", "Rest", _score([Measure([Event([], F(4))]) for _ in range(4)]), None))
    for label, base in [("rest_half", F(2)), ("rest_quarter", F(1)), ("rest_8th", F(1, 2)),
                        ("rest_16th", F(1, 4)), ("rest_32nd", F(1, 8))]:
        out.append((label, "Rest", _score(_rests(base)), None))
    for name, base in [("8th", F(1, 2)), ("16th", F(1, 4)), ("32nd", F(1, 8))]:
        out.append((f"flag_{name}_up", "Hook", _score(_notes(LOW, base, rest_between=True)), None))
        out.append((f"flag_{name}_down", "Hook", _score(_notes(HIGH, base, rest_between=True)), None))
    for base in (F(1), F(2), F(4)):
        out.append(("notehead_x", "Note", _score(_drum_notes(["hh", "ride", "crash", "ph"], base), clef=PERCUSSION,
                                                 percussion=True), None))
    out.append(("clef_perc", "Clef", _score(_drum_notes(["sn", "bd"], F(1)), clef=PERCUSSION, percussion=True), None))
    out.append(("clef_g", "Clef", _score(_notes(LOW, F(4)), clef=TREBLE), None))
    out.append(("clef_f", "Clef", _score(_notes([("C", 3)], F(4)), clef=BASS), None))
    out.append(("clef_c", "Clef", _score(_notes([("C", 4)], F(4)), clef=ALTO), None))
    out.append(("clef_c", "Clef", _score(_notes([("C", 4)], F(4)), clef=TENOR), None))
    for label, alter in [("sharp", 1), ("flat", -1), ("double_sharp", 2), ("double_flat", -2)]:
        out.append((label, "Accidental", _score(_notes([("C", 5), ("D", 5), ("E", 5)], F(1), alter=alter)), None))
    # naturals: F natural in G major
    out.append(("natural", "Accidental", _score(_notes([("F", 5), ("F", 4)], F(1)), key=1), None))
    # key signatures share glyphs with accidentals
    out.append(("sharp", "KeySig", _score(_notes(LOW, F(4)), key=7), None))
    out.append(("flat", "KeySig", _score(_notes(LOW, F(4)), key=-7), None))
    times = [(2, 4), (3, 4), (4, 4), (5, 4), (6, 8), (7, 8), (9, 8), (12, 8), (3, 8), (2, 2), (3, 2), (6, 4)]
    for t in times:
        length = F(4 * t[0], t[1])
        ms = [Measure([Event([], length, measure_rest=True)]) for _ in range(2)]
        out.append((f"time:{t[0]}", "TimeSig", _score(ms, time=t), "top"))
        ms = [Measure([Event([], length, measure_rest=True)]) for _ in range(2)]
        out.append((f"time:{t[1]}", "TimeSig", _score(ms, time=t), "bottom"))
    for sym, t in [("common", (4, 4)), ("cut", (2, 2))]:
        ms = [Measure([Event([], F(4), measure_rest=True)]) for _ in range(2)]
        out.append((f"time_{sym}", "TimeSig", _score(ms, time=t, time_symbol=sym), None))
    return out


def calibrate(workdir: str | Path, style: str | Path | None = None) -> dict[str, str]:
    """Render the calibration scores and return {group|signature: label}."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    items = calibration_scores()
    jobs = []
    for i, (label, group, score, which) in enumerate(items):
        xml = write_musicxml(score, workdir / f"cal{i:03d}.musicxml")
        jobs.append((xml, workdir / f"cal{i:03d}.svg"))
    musescore.convert_batch(jobs, style=style)
    table: dict[str, str] = {}
    conflicts: dict[str, set[str]] = {}
    for i, (label, group, score, which) in enumerate(items):
        for svg in sorted(workdir.glob(f"cal{i:03d}-*.svg")):
            elements, _, _ = parse_svg(svg)
            els = [e for e in elements if e.cls == group and not e.polyline]
            if which and els:
                ys = sorted({round(e.box[1]) for e in els})
                mid = (ys[0] + ys[-1]) / 2
                els = [e for e in els if (e.box[1] < mid) == (which == "top")] if len(ys) > 1 else els
            pairs = [(e, label) for e in els]
            if label.startswith("time:"):
                # one glyph per digit, left to right
                digits = label.split(":")[1]
                els = sorted(els, key=lambda e: e.box[0])
                pairs = [(e, f"time_{d}") for e, d in zip(els, digits)] if len(els) == len(digits) else []
            elif label in ("rest_whole", "rest_half"):
                pairs = [(e, "rest_block") for e in els]  # same glyph, told apart by position
            for e, lab in pairs:
                key = f"{GLYPH_GROUPS[e.cls]}|{e.sig}"
                if key in table and table[key] != lab:
                    conflicts.setdefault(key, {table[key]}).add(lab)
                table[key] = lab
    for key, labels in conflicts.items():
        # a glyph seen under several labels is ambiguous: drop it
        table.pop(key, None)
    return table


def _rest_block(box, line_ys: list[float], space: float) -> str:
    """Whole rest hangs from a line, half rest sits on one."""
    if not line_ys:
        return "rest_half"
    tol = 0.2 * space
    top = min(abs(y - box[1]) for y in line_ys)
    bottom = min(abs(y - box[3]) for y in line_ys)
    return "rest_whole" if top < bottom and top < tol else "rest_half"


def label_elements(elements: list[Element], table: dict[str, str], space: float = 0.0):
    """-> (labelled [(label, box)], unknown [(svg class, box)])."""
    labelled, unknown = [], []
    line_ys = [
        (e.box[1] + e.box[3]) / 2
        for e in elements
        if e.cls in ("StaffLines", "LedgerLine") and e.polyline
    ]
    for e in elements:
        if e.cls in FIXED:
            labelled.append((FIXED[e.cls], e.box))
        elif e.cls == "Tuplet" and not e.polyline:
            labelled.append(("tuplet_3", e.box))
        elif e.cls in GLYPH_GROUPS and not e.polyline:
            label = table.get(f"{GLYPH_GROUPS[e.cls]}|{e.sig}")
            if label == "rest_block":
                label = _rest_block(e.box, line_ys, space)
            if label:
                labelled.append((label, e.box))
            else:
                unknown.append((e.cls, e.box))
    return labelled, unknown


def save_table(table: dict[str, str], path: str | Path):
    Path(path).write_text(json.dumps(table, indent=1, sort_keys=True))


def load_table(path: str | Path) -> dict[str, str]:
    return json.loads(Path(path).read_text())
