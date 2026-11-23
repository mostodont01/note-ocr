"""Symbols + staff geometry -> music.

Per staff (single voice for now):
  barlines            -> measures
  clef / key / time   -> running state (key signature = accidentals not attached to a note)
  noteheads + stems   -> chords; flags / beams / dots -> durations
  position on staff   -> pitch (clef + key signature + accidentals in the measure)
  ties, triplets      -> attached to neighbouring notes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction as F

import numpy as np

from ..model import (
    ALTO,
    BASS,
    PERCUSSION,
    TREBLE,
    Clef,
    Event,
    Measure,
    NoteHead,
    Part,
    Pitch,
    Region,
    Score,
    Tuplet,
    key_alters,
    measure_length,
)
from ..percussion import drum_at
from .detect import Symbol
from .staff import PageLayout, Staff

HEAD_BASE = {"notehead_black": F(1), "notehead_half": F(2), "notehead_whole": F(4), "notehead_x": F(1),
             "notehead_x_half": F(2), "notehead_x_whole": F(4)}
WHOLE = ("notehead_whole", "notehead_x_whole")  # no stem
BLACKISH = ("notehead_black", "notehead_x")
REST_BASE = {
    "rest_whole": F(4),
    "rest_half": F(2),
    "rest_quarter": F(1),
    "rest_8th": F(1, 2),
    "rest_16th": F(1, 4),
    "rest_32nd": F(1, 8),
}
FLAG_LEVEL = {"8th": 1, "16th": 2, "32nd": 3}
ACC_ALTER = {"sharp": 1, "flat": -1, "natural": 0, "double_sharp": 2, "double_flat": -2}
CLEFS = {"clef_g": TREBLE, "clef_f": BASS, "clef_c": ALTO, "clef_perc": PERCUSSION}


@dataclass
class Head:
    sym: Symbol
    pos: int  # staff position, half spaces above the bottom line
    stem: Symbol | None = None
    accidental: Symbol | None = None
    dots: int = 0


@dataclass
class Group:
    """Note heads sharing a stem (a chord), or a lone stemless head."""

    heads: list[Head]
    stem: Symbol | None
    up: bool = True
    levels: int = 0  # beams / flags
    dots: int = 0
    tuplet: bool = False

    @property
    def x(self) -> float:
        return min(h.sym.cx for h in self.heads)


@dataclass
class StaffSymbols:
    staff: Staff
    syms: list[Symbol] = field(default_factory=list)

    def of(self, *prefixes: str) -> list[Symbol]:
        return [s for s in self.syms if s.cls.startswith(prefixes)]


def _assign_to_staves(symbols: list[Symbol], layout: PageLayout) -> list[StaffSymbols]:
    out = [StaffSymbols(s) for s in layout.staves]
    for sym in sorted(symbols, key=lambda s: s.cls.startswith("tuplet")):
        if sym.cls == "barline":
            spans = [ss for ss in out if sym.y0 <= ss.staff.top(sym.cx) + 0.5 * ss.staff.spacing()
                     and sym.y1 >= ss.staff.bottom(sym.cx) - 0.5 * ss.staff.spacing()
                     and ss.staff.x0 - ss.staff.spacing() <= sym.cx <= ss.staff.x1 + ss.staff.spacing()]
            if spans:
                for ss in spans:
                    ss.syms.append(sym)
                continue
        best, best_d = None, 1e9
        for ss in out:
            st = ss.staff
            if sym.cx < st.x0 - 3 * st.spacing() or sym.cx > st.x1 + st.spacing():
                continue
            top, bot = st.top(sym.cx), st.bottom(sym.cx)
            d = 0.0 if top <= sym.cy <= bot else min(abs(sym.cy - top), abs(sym.cy - bot))
            if d < best_d:
                best, best_d = ss, d
        if sym.cls.startswith("tuplet") and best is not None:
            # a tuplet number belongs to the notes it labels, which may be on the other
            # staff (e.g. under the upper staff's beam, between the staves)
            near = [
                (abs(o.cy - sym.cy) + abs(o.cx - sym.cx), ss)
                for ss in out
                for o in ss.syms
                if o.cls == "stem" or o.cls.startswith("notehead")
            ]
            if near:
                best = min(near, key=lambda t: t[0])[1]
                best_d = 0.0
        if best is not None and best_d < 7 * best.staff.spacing():
            best.syms.append(sym)
    return out


def _region(s: Symbol) -> Region:
    return Region(s.x0, s.y0, s.w, s.h, confidence=s.score, uid=s.uid)


# ------------------------------------------------------------------------ stems, beams


def _stem_for(head: Symbol, stems: list[Symbol], d: float) -> Symbol | None:
    best, best_dx = None, 1e9
    for st in stems:
        dx = min(abs(st.cx - head.x0), abs(st.cx - head.x1))
        if dx > 0.45 * d:
            continue
        if not (st.y0 - 0.6 * d <= head.cy <= st.y1 + 0.6 * d):
            continue
        if dx < best_dx:
            best, best_dx = st, dx
    return best


def _find_stem_classic(ink: np.ndarray, head: Symbol, d: float) -> Symbol | None:
    """Fallback when the detector missed a stem: a long vertical run at a head edge."""
    h, w = ink.shape
    best = None
    for x_edge, up in ((head.x1, True), (head.x0, False)):
        for dx in (-1, 0, 1, -2, 2):
            x = int(round(x_edge + dx - (1 if up else -1) * 0.5))
            if not 0 <= x < w:
                continue
            y = int(round(head.cy))
            if up:
                y_end = y
                while y_end > 0 and ink[y_end - 1, x]:
                    y_end -= 1
                length = y - y_end
                box = (x - 0.5, y_end, x + 0.5, y)
            else:
                y_end = y
                while y_end < h - 1 and ink[y_end + 1, x]:
                    y_end += 1
                length = y_end - y
                box = (x - 0.5, y, x + 0.5, y_end)
            if length >= 2.2 * d and (best is None or length > best[0]):
                best = (length, box)
    if best is None:
        return None
    return Symbol("stem", 0.3, *best[1])


def _stem_tip(ink: np.ndarray, stem: Symbol, up: bool, d: float) -> float:
    """Follow the stem in the image to its real free end (detector boxes can be short)."""
    h, w = ink.shape
    box_tip = stem.y0 if up else stem.y1
    y_mid = int(np.clip(round(stem.cy), 0, h - 1))
    y_lo, y_hi = max(int(stem.y0), 0), min(int(np.ceil(stem.y1)) + 1, h)
    cols = [x for x in range(int(round(stem.cx)) - 1, int(round(stem.cx)) + 2) if 0 <= x < w]
    if not cols or y_hi <= y_lo:
        return box_tip
    x = max(cols, key=lambda c: int(ink[y_lo:y_hi, c].sum()))
    step = -1 if up else 1
    limit = int(1.5 * d)  # how far past the box end we may follow
    y, gap = y_mid, 0
    end = int(box_tip) + step * limit
    while 0 <= y + step < h and (y + step - end) * step <= 0:
        if ink[y + step, x] or (0 < x < w - 1 and (ink[y + step, x - 1] or ink[y + step, x + 1])):
            gap = 0
        else:
            gap += 1
            if gap > 1:
                break
        y += step
    tip = y - step * gap
    # never shorter than the detector's box
    return min(tip, box_tip) if up else max(tip, box_tip)


def _line_darkness(dark: np.ndarray, staff: Staff, d: float) -> float:
    """Ink one bare staff line contributes to a column (median over free columns)."""
    h, w = dark.shape
    vals = []
    for x in np.arange(staff.x0 + d, staff.x1 - d, max(d / 2, 1)).astype(int):
        for ln in staff.lines:
            y = float(ln.y_at(x))
            a, b = int(round(y - 0.4 * d)), int(round(y + 0.4 * d)) + 1
            if a < 0 or b > h:
                continue
            core = dark[a:b, x].sum()
            above = dark[max(a - int(0.4 * d), 0) : a, x].sum()
            below = dark[b : b + int(0.4 * d), x].sum()
            if above < 0.3 and below < 0.3:
                vals.append(core)
    return float(np.median(vals)) if vals else 1.0


def _count_beams(dark: np.ndarray, stem: Symbol, up: bool, d: float, staff: Staff | None = None,
                 line_dark: float = 0.0) -> int:
    """Count beams at the stem's free end by sampling columns beside the stem.

    `dark` is the ink darkness (0..1) with staff lines kept: at low resolution a thin
    beam lying on a staff line would vanish with line removal. Each dark run's total
    ink minus what the staff lines it covers contribute = beam ink (about 0.5 space
    per beam), which also splits beams that merged into one run."""
    h, w = dark.shape
    tip = _stem_tip(dark > 0.5, stem, up, d)
    reach = 2.1 * d
    best = 0
    for side in (-1, 1):
        x = int(round(stem.cx + side * (0.45 * d + stem.w / 2)))
        if not 0 <= x < w:
            continue
        if up:
            y0, y1 = int(tip - 0.5 * d), int(tip + reach)
        else:
            y0, y1 = int(tip - reach), int(np.ceil(tip + 0.5 * d))
        y0, y1 = max(y0, 0), min(y1, h)
        col = dark[y0:y1, x]
        if not len(col):
            continue
        on = (col > 0.35).astype(np.int8)
        edges = np.diff(np.concatenate([[0], on, [0]]))
        starts, ends = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]
        line_ys = [float(ln.y_at(x)) - y0 for ln in staff.lines] if staff is not None else []
        n = 0
        for a, b in zip(starts, ends):
            covered = sum(1 for ly in line_ys if a - 1 <= ly <= b)
            excess = float(col[a:b].sum()) - covered * line_dark
            if excess < 0.28 * d:
                continue
            n += max(1, int(round(excess / (0.5 * d))))
        best = max(best, n)
    return min(best, 4)


# ---------------------------------------------------------------------- header parts


def _key_signature(accs: list[Symbol]) -> int:
    sharps = sum(1 for a in accs if a.cls == "sharp")
    flats = sum(1 for a in accs if a.cls == "flat")
    if sharps > flats:
        return min(sharps, 7)
    if flats > sharps:
        return -min(flats, 7)
    return 0


def _time_signature(digits: list[Symbol], staff: Staff) -> tuple[tuple[int, int] | None, str | None]:
    if not digits:
        return None, None
    for s in digits:
        if s.cls in ("time_common", "time_cut"):
            return ((4, 4), "common") if s.cls == "time_common" else ((2, 2), "cut")
    mid = (staff.top(digits[0].cx) + staff.bottom(digits[0].cx)) / 2
    top = sorted((s for s in digits if s.cy < mid), key=lambda s: s.cx)
    bot = sorted((s for s in digits if s.cy >= mid), key=lambda s: s.cx)
    if not top or not bot:
        return None, None
    num = int("".join(s.cls.split("_")[1] for s in top))
    den = int("".join(s.cls.split("_")[1] for s in bot))
    if den not in (1, 2, 4, 8, 16, 32):
        return None, None
    return (num, den), None


def _cluster_x(symbols: list[Symbol], gap: float) -> list[list[Symbol]]:
    out: list[list[Symbol]] = []
    for s in sorted(symbols, key=lambda s: s.cx):
        if out and s.x0 - max(t.x1 for t in out[-1]) < gap:
            out[-1].append(s)
        else:
            out.append([s])
    return out


# ------------------------------------------------------------------------- assembly


@dataclass
class _State:
    clef: Clef = TREBLE
    key: int = 0
    time: tuple[int, int] = (4, 4)


def _assemble_staff(ss: StaffSymbols, dark: np.ndarray, state: _State, first_staff: bool) -> list[Measure]:
    st = ss.staff
    d = st.spacing()
    ink = dark > 0.5
    line_dark = _line_darkness(dark, st, d)

    heads_syms = ss.of("notehead_")
    rests = ss.of("rest_")
    stems = ss.of("stem")
    flags = ss.of("flag_")
    dots = ss.of("dot")
    accs = ss.of("sharp", "flat", "natural", "double_")
    # a clef sits on the staff and is tall (rejects e.g. an "ff" dynamic below it)
    clefs = [
        c for c in ss.of("clef_")
        if st.top(c.cx) - 0.8 * d <= c.cy <= st.bottom(c.cx) + 0.8 * d and c.h >= (1.0 if c.cls == "clef_perc" else 1.8) * d
    ]
    times = ss.of("time_")
    ties = ss.of("tie")
    tuplets = ss.of("tuplet_")

    # ---- barlines -> measure boundaries
    bars = sorted(
        (b for b in ss.of("barline") if b.h > 2.5 * d), key=lambda b: b.cx
    )
    bounds = []
    for b in bars:
        if bounds and b.cx - bounds[-1] < 1.2 * d:  # double / final barline
            bounds[-1] = b.cx
        else:
            bounds.append(b.cx)
    xs = [st.x0 - 2 * d] + [b for b in bounds if b > st.x0 + 2 * d] + [st.x1 + 2 * d]
    intervals = list(zip(xs[:-1], xs[1:]))

    # ---- accidentals: attached to a note head, or part of a key signature
    attached: dict[int, Symbol] = {}
    free_accs: list[Symbol] = []
    for a in sorted(accs, key=lambda a: -a.cx):
        cands = [
            (h.x0 - a.x1, i)
            for i, h in enumerate(heads_syms)
            if -0.4 * d <= h.x0 - a.x1 <= 1.2 * d and a.y0 - 0.2 * d <= h.cy <= a.y1 + 0.2 * d and i not in attached
        ]
        if cands:
            attached[min(cands)[1]] = a
        else:
            free_accs.append(a)

    # ---- note heads -> groups (chords)
    heads = []
    for i, hs in enumerate(heads_syms):
        pos = int(round(st.position(hs.cx, hs.cy)))
        heads.append(Head(hs, pos, accidental=attached.get(i)))
    groups: list[Group] = []
    by_stem: dict[int, Group] = {}
    for h in heads:
        stem = None
        if h.sym.cls not in WHOLE:
            stem = _stem_for(h.sym, stems, d) or _find_stem_classic(ink, h.sym, d)
            if stem is not None and stem.score < 0.31:
                stems.append(stem)  # classic stem, shared by the rest of its chord
        if stem is not None:
            key = id(stem)
            if key in by_stem:
                by_stem[key].heads.append(h)
                continue
            g = Group([h], stem, up=h.sym.cy > stem.cy)
            by_stem[key] = g
            groups.append(g)
        else:
            # stemless: chord if another stemless head is right above / below
            for g in groups:
                if g.stem is None and abs(g.heads[0].sym.cx - h.sym.cx) < 0.6 * d:
                    g.heads.append(h)
                    break
            else:
                groups.append(Group([h], None))

    # ---- stem direction from the whole chord: the stem sticks out above (up) or below
    # (down) all of its note heads
    for g in groups:
        if g.stem is not None:
            top = min(h.sym.cy for h in g.heads)
            bottom = max(h.sym.cy for h in g.heads)
            g.up = (top - g.stem.y0) > (g.stem.y1 - bottom)

    # ---- durations: flags, beams
    for g in groups:
        if g.stem is None:
            continue
        lv = 0
        tip = g.stem.y0 if g.up else g.stem.y1
        for f in flags:
            if abs(f.x0 - g.stem.cx) < 0.6 * d and f.y0 - 0.5 * d <= tip <= f.y1 + 0.5 * d:
                lv = max(lv, FLAG_LEVEL[f.cls.split("_")[1]])
        if lv == 0 and g.heads[0].sym.cls in BLACKISH:
            lv = _count_beams(dark, g.stem, g.up, d, st, line_dark)
        g.levels = lv

    # ---- dots: right of a head (or rest), about the same height
    rest_dots: dict[int, int] = {}
    for dot in dots:
        best, best_dx = None, 1e9
        for g in groups:
            for h in g.heads:
                dx = dot.cx - h.sym.x1
                if 0 <= dx <= 1.6 * d and abs(dot.cy - h.sym.cy) < 0.8 * d and dx < best_dx:
                    best, best_dx = ("g", id(g)), dx
        for ri, r in enumerate(rests):
            dx = dot.cx - r.x1
            if 0 <= dx <= 1.4 * d and r.y0 - 0.3 * d <= dot.cy <= r.y1 + 0.3 * d and dx < best_dx:
                best, best_dx = ("r", ri), dx
        if best is None:
            continue
        if best[0] == "g":
            for g in groups:
                if id(g) == best[1]:
                    g.dots = 1
        else:
            rest_dots[best[1]] = 1

    # ---- ties: pairs of heads (same position) at both ends
    tie_pairs: list[tuple[Head, Head]] = []
    for t in ties:
        left = [h for h in heads if abs(h.sym.x1 - t.x0) < 1.2 * d and abs(h.sym.cy - t.cy) < 1.6 * d]
        right = [h for h in heads if abs(h.sym.x0 - t.x1) < 1.2 * d and abs(h.sym.cy - t.cy) < 1.6 * d]
        best = None
        for a in left:
            for b in right:
                if a.pos == b.pos and b.sym.cx > a.sym.cx:
                    cost = abs(a.sym.x1 - t.x0) + abs(b.sym.x0 - t.x1)
                    if best is None or cost < best[0]:
                        best = (cost, a, b)
        if best:
            tie_pairs.append((best[1], best[2]))

    # ---- walk the measures
    measures: list[Measure] = []
    clefs_sorted = sorted(clefs, key=lambda c: c.cx)
    pending_clef: Clef | None = None
    for mi, (x0, x1) in enumerate(intervals):
        m = Measure(region=Region(x0, st.top() - 2 * d, x1 - x0, st.bottom() - st.top() + 4 * d))
        if pending_clef is not None:
            m.clefs = {1: pending_clef}
            pending_clef = None
        in_m = lambda s: x0 <= s.cx < x1  # noqa: E731
        m_groups = sorted((g for g in groups if in_m(g.heads[0].sym)), key=lambda g: g.x)
        m_rests = [(ri, r) for ri, r in enumerate(rests) if in_m(r)]
        first_x = min([g.x for g in m_groups] + [r.cx for _, r in m_rests] + [x1])

        # clef / key / time before the first note of the measure
        for c in clefs_sorted:
            if in_m(c) and c.cx < first_x:
                clef = CLEFS[c.cls]
                if clef != state.clef or (mi == 0 and first_staff):
                    m.clefs = {1: clef}
                state.clef = clef
        head_accs = [a for a in free_accs if in_m(a) and a.cx < first_x]
        if head_accs:
            k = _key_signature(head_accs)
            if k != state.key or (mi == 0 and first_staff):
                m.key = k
            state.key = k
        elif mi == 0 and first_staff:
            m.key = state.key
        t_syms = [s for s in times if in_m(s) and s.cx < first_x]
        time, symbol = _time_signature(t_syms, st)
        if time and (time != state.time or (mi == 0 and first_staff)):
            m.time, m.time_symbol = time, symbol
            state.time = time
        elif mi == 0 and first_staff:
            m.time = state.time

        # events in x order
        items: list[tuple[float, object]] = [(g.x, g) for g in m_groups] + [(r.cx, (ri, r)) for ri, r in m_rests]
        # clef changes inside the measure (MuseScore draws a change at the start of a
        # measure just before the previous barline)
        items += [(c.cx, c) for c in clefs_sorted if in_m(c) and c.cx >= first_x]
        items.sort(key=lambda t: t[0])
        n_notes = sum(1 for _, it in items if not isinstance(it, Symbol))
        alters_in_measure: dict[tuple[str, int], int] = {}
        kalters = key_alters(state.key)
        seen = 0
        for _, it in items:
            if isinstance(it, Symbol):  # clef change
                state.clef = CLEFS[it.cls]
                if seen < n_notes:
                    m.clefs = {1: state.clef}
                else:
                    pending_clef = state.clef
                continue
            seen += 1
            if isinstance(it, Group):
                g = it
                head_cls = g.heads[0].sym.cls
                base = HEAD_BASE[head_cls]
                if head_cls in BLACKISH:
                    base = F(1, 2**g.levels)
                notes = []
                for h in sorted(g.heads, key=lambda h: h.pos):
                    if state.clef.sign == "percussion":
                        drum = drum_at(h.pos, "x" if h.sym.cls.startswith("notehead_x") else "normal")
                        nh = NoteHead(drum.display, notehead=drum.notehead, instrument=drum.id, region=_region(h.sym))
                        nh._head = h  # type: ignore[attr-defined]
                        notes.append(nh)
                        continue
                    natural = state.clef.pitch_at(h.pos)
                    kk = (natural.step, natural.octave)
                    if h.accidental is not None:
                        alter = ACC_ALTER[h.accidental.cls]
                        alters_in_measure[kk] = alter
                    else:
                        alter = alters_in_measure.get(kk, kalters.get(natural.step, 0))
                    notes.append(NoteHead(Pitch(natural.step, natural.octave, alter), region=_region(h.sym)))
                    h_note = notes[-1]
                    h_note._head = h  # type: ignore[attr-defined]
                score = min(h.sym.score for h in g.heads)
                x_lo = min(h.sym.x0 for h in g.heads)
                y_lo = min([h.sym.y0 for h in g.heads] + ([g.stem.y0] if g.stem else []))
                x_hi = max(h.sym.x1 for h in g.heads)
                y_hi = max([h.sym.y1 for h in g.heads] + ([g.stem.y1] if g.stem else []))
                ev = Event(notes, base, g.dots, region=Region(x_lo, y_lo, x_hi - x_lo, y_hi - y_lo, confidence=score,
                                                             uid=min(h.sym.uid for h in g.heads)))
                ev._up = g.up if g.stem is not None else None  # type: ignore[attr-defined]
            else:
                ri, r = it
                ev = Event([], REST_BASE[r.cls], rest_dots.get(ri, 0), region=_region(r))
                if r.cls == "rest_whole" and n_notes == 1:
                    ev.measure_rest = True
                    ev.base = measure_length(state.time)
            m.events.append(ev)

        # triplets: the three events closest to each tuplet number in this measure
        for tsym in tuplets:
            if not in_m(tsym) or len(m.events) < 3:
                continue
            centres = [(e.region.x + e.region.w / 2) if e.region else 0 for e in m.events]
            best = min(range(len(m.events) - 2), key=lambda i: abs(sum(centres[i : i + 3]) / 3 - tsym.cx))
            for k in range(3):
                m.events[best + k].tuplet = Tuplet(3, 2, start=k == 0, stop=k == 2)

        _split_voices(m, st, d, percussion=state.clef.sign == "percussion")

        # measure rests written as a whole rest in other time signatures
        if m.events or mi < len(intervals) - 1:
            if m.events or any(in_m(s) for s in ss.syms):
                measures.append(m)

    _attach_marks(ss, measures, d)

    # ties (after all events exist)
    for a, b in tie_pairs:
        for m in measures:
            for e in m.events:
                for n in e.notes:
                    h = getattr(n, "_head", None)
                    if h is a:
                        n.tie_start = True
                    if h is b:
                        n.tie_stop = True
    # a tied note keeps the pitch of the note it is tied from (accidental carries over)
    pitch_of = {}
    for m in measures:
        for e in m.events:
            for n in e.notes:
                h = getattr(n, "_head", None)
                pitch_of[id(h)] = n
    for a, b in tie_pairs:
        na, nb = pitch_of.get(id(a)), pitch_of.get(id(b))
        if na and nb and na.pitch.step == nb.pitch.step and na.pitch.octave == nb.pitch.octave:
            nb.pitch = na.pitch
    for m in measures:
        for e in m.events:
            if hasattr(e, "_up"):
                del e._up
            for n in e.notes:
                if hasattr(n, "_head"):
                    del n._head
    # drop empty measures (e.g. the space after the final barline)
    return [m for m in measures if m.events]


def _split_voices(m: Measure, st: Staff, d: float, percussion: bool):
    """Two voices on one staff: stems up = voice 1, stems down = voice 2.

    Used when up- and down-stem notes sound together (same horizontal position), e.g.
    drums with hands up / feet down. Rests go to the voice whose side they are on."""
    def cx(e):
        return e.region.x + e.region.w / 2 if e.region else 0.0

    ups = [e for e in m.events if getattr(e, "_up", None) is True]
    downs = [e for e in m.events if getattr(e, "_up", None) is False]
    if not ups or not downs:
        return
    # only when up- and down-stem notes sound together; a single voice may mix stem
    # directions (also on drum staves)
    if not any(abs(cx(a) - cx(b)) < 0.6 * d for a in ups for b in downs):
        return
    mid = (st.top() + st.bottom()) / 2
    for e in m.events:
        up = getattr(e, "_up", None)
        if up is None:
            below = e.region is not None and e.region.y + e.region.h / 2 > mid + 0.5 * d
            e.voice = 2 if (e.is_rest and below) else 1
        else:
            e.voice = 1 if up else 2
    m.events.sort(key=lambda e: e.voice)  # stable: x order kept within each voice
    _align_voice2(m, d)


def _rest_values(length: F) -> list[tuple[F, int]]:
    """Split a gap into standard rest values (base, dots), longest first."""
    out = []
    for base in (F(4), F(2), F(1), F(1, 2), F(1, 4), F(1, 8), F(1, 16)):
        while length >= base:
            if length >= base * F(3, 2) and base >= F(1, 4):
                out.append((base, 1))
                length -= base * F(3, 2)
            else:
                out.append((base, 0))
                length -= base
    return out


def _align_voice2(m: Measure, d: float):
    """Voice 2 timing from the x position of voice 1 notes above it (its rests are often
    hidden); gaps become invisible rests."""
    v1 = [e for e in m.events if e.voice == 1]
    v2 = [e for e in m.events if e.voice == 2]
    if not v1 or not v2:
        return
    onsets, pos = [], F(0)
    for e in v1:
        if e.region:
            onsets.append((e.region.x + e.region.w / 2, pos))
        pos += e.duration
    new_v2, pos = [], F(0)
    for e in v2:
        if e.region:
            x = e.region.x + e.region.w / 2
            near = min(onsets, key=lambda o: abs(o[0] - x))
            if abs(near[0] - x) < 0.8 * d and near[1] > pos:
                for base, dots in _rest_values(near[1] - pos):
                    new_v2.append(Event([], base, dots, voice=2, staff=e.staff))
                pos = near[1]
        new_v2.append(e)
        pos += e.duration
    m.events = v1 + new_v2


ARTICULATION_NAMES = {"staccato": "staccato", "accent": "accent", "tenuto": "tenuto", "marcato": "strong-accent"}


def _attach_marks(ss: StaffSymbols, measures: list[Measure], d: float):
    """Dynamics, articulations, fermatas and slurs -> nearest notes; repeat dots -> measures."""
    events = [e for m in measures for e in m.events if e.region]
    if not events:
        return

    def centre(e):
        return e.region.x + e.region.w / 2

    def nearest(x, max_dx, notes_only=True, key=centre):
        cands = [e for e in events if (e.notes or not notes_only) and abs(key(e) - x) <= max_dx]
        return min(cands, key=lambda e: abs(key(e) - x)) if cands else None

    for s in ss.of("dyn_"):
        e = nearest(s.x0, 3 * d, notes_only=False, key=lambda e: e.region.x)
        if e is not None:
            e.dynamic = s.cls[4:]
    for s in ss.of("staccato", "accent", "tenuto", "marcato"):
        e = nearest(s.cx, 1.2 * d)
        if e is not None and ARTICULATION_NAMES[s.cls] not in e.articulations:
            e.articulations.append(ARTICULATION_NAMES[s.cls])
    for s in ss.of("fermata"):
        e = nearest(s.cx, 2 * d, notes_only=False)
        if e is not None:
            e.fermata = True
    for s in ss.of("slur"):
        a = nearest(s.x0, 1.5 * d)
        b = nearest(s.x1, 1.5 * d)
        if a is not None and b is not None and a is not b and centre(a) < centre(b):
            a.slur_start = True
            b.slur_stop = True
    dots = ss.of("repeat_dot")
    for m in measures:
        if not m.region or not dots:
            continue
        x0, x1 = m.region.x, m.region.x + m.region.w
        if sum(1 for s in dots if 0 < s.cx - x0 < 1.5 * d) >= 1 and x0 > ss.staff.x0:
            m.repeat_start = True
        if sum(1 for s in dots if 0 < x1 - s.cx < 1.5 * d) >= 1:
            m.repeat_end = True


def check_measures(part: Part) -> list[int]:
    """Indexes of measures whose durations don't add up to the time signature
    (checked per staff and voice)."""
    bad = []
    time = (4, 4)
    for i, m in enumerate(part.measures):
        time = m.time or time
        want = measure_length(time)
        totals: dict[tuple[int, int], F] = {}
        for e in m.events:
            totals[(e.staff, e.voice)] = totals.get((e.staff, e.voice), F(0)) + e.duration
        ok = all(t == want or (i == 0 and t < want) for t in totals.values())
        if not ok:
            bad.append(i)
            if m.region:
                m.region.confidence = 0.3
    return bad


def assemble(symbols: list[Symbol], layout: PageLayout, dark: np.ndarray, state: _State | None = None) -> Part:
    """Symbols -> one part.

    Systems of one staff continue a single-staff part; systems of two staves form a
    grand staff (piano: staff 1 + 2). A percussion clef makes it a drum part.
    `dark`: ink darkness 0..1 (staff lines kept, detected symbols other than stems and
    barlines blanked out); used for stems and beams."""
    per_staff = _assign_to_staves(symbols, layout)
    by_staff = {id(ss.staff): ss for ss in per_staff}
    sizes = [len(sy.staves) for sy in layout.systems] or [1]
    n = max(set(sizes), key=sizes.count)
    if n != 2:
        state = state or _State()
        part = Part()
        for si, ss in enumerate(per_staff):
            part.measures.extend(_assemble_staff(ss, dark, state, first_staff=(si == 0)))
        part.percussion = state.clef.sign == "percussion"
        if part.percussion:
            part.name, part.abbreviation = "Drumset", "D. Set"
        return part

    # grand staff
    upper = state or _State()
    lower = getattr(upper, "lower", None) or _State(clef=BASS)
    upper.lower = lower  # type: ignore[attr-defined]  (carried to the next page)
    part = Part(name="Piano", abbreviation="Pno.", staves=2)
    # pairs of staves: detected two-staff systems, plus neighbouring single staves whose
    # joining barline wasn't found (e.g. bent photos)
    pairs, singles = [], []
    for sy in layout.systems:
        if len(sy.staves) == 2:
            pairs.append((sy.staves[0], sy.staves[1]))
        elif len(sy.staves) == 1:
            singles.append(sy.staves[0])
            if len(singles) == 2:
                pairs.append((singles[0], singles[1]))
                singles = []
        if len(sy.staves) == 2 and singles:
            singles = []  # an odd single staff before a real pair: can't place it
    pairs.sort(key=lambda p: p[0].top())
    first = True
    for st_top, st_bot in pairs:
        top = _assemble_staff(by_staff[id(st_top)], dark, upper, first_staff=first)
        bot = _assemble_staff(by_staff[id(st_bot)], dark, lower, first_staff=first)
        lower.key, lower.time = upper.key, upper.time
        for i in range(max(len(top), len(bot))):
            mt = top[i] if i < len(top) else Measure()
            mb = bot[i] if i < len(bot) else None
            for e in mt.events:
                e.staff = 1
            if mb is None or not mb.events:
                mb = mb or Measure()
                mb.events = [Event([], measure_length(upper.time), measure_rest=True)]
            for e in mb.events:
                e.staff = 2
            m = Measure(events=mt.events + mb.events, time=mt.time, time_symbol=mt.time_symbol, key=mt.key,
                        region=mt.region)
            m.clefs = {1: c for c in mt.clefs.values()} | {2: c for c in mb.clefs.values()}
            if first and i == 0:
                m.clefs.setdefault(1, upper.clef)
                m.clefs.setdefault(2, BASS)
            part.measures.append(m)
        first = False
    return part
