"""MusicXML (partwise 4.0) writer, and appending measures to an existing file."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

from ..model import TYPE_NAMES, Clef, Event, Measure, Part, Score, key_alters

_PART_ID = "P1"  # set while writing a part (instrument ids are per part)
ACCIDENTAL_NAMES = {-2: "flat-flat", -1: "flat", 0: "natural", 1: "sharp", 2: "double-sharp"}
DOCTYPE = (
    '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
    '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
    '"http://www.musicxml.org/dtds/partwise.dtd">\n'
)


def _sub(parent, tag, text=None, **attrib):
    el = ET.SubElement(parent, tag, {k.replace("_", "-"): str(v) for k, v in attrib.items()})
    if text is not None:
        el.text = str(text)
    return el


def _divisions(parts: list[Part]) -> int:
    dens = {1}
    for part in parts:
        for m in part.measures:
            for e in m.events:
                dens.add(e.duration.denominator)
                dens.add(e.written.denominator)
    return math.lcm(*dens)


def _write_clef(attrs, clef: Clef, number: int, staves: int):
    c = _sub(attrs, "clef", number=number) if staves > 1 else _sub(attrs, "clef")
    _sub(c, "sign", clef.sign)
    if clef.sign != "percussion":
        _sub(c, "line", clef.line)
    if clef.octave_change:
        _sub(c, "clef-octave-change", clef.octave_change)


def _write_note(parent, e: Event, idx: int, divisions: int, shown: list[str | None], percussion: bool):
    dur = int(e.duration * divisions)
    if e.dynamic:
        d = _sub(parent, "direction", placement="below")
        _sub(_sub(_sub(d, "direction-type"), "dynamics"), e.dynamic)
        _sub(d, "staff", e.staff)
    heads = e.notes or [None]
    for i, head in enumerate(heads):
        n = _sub(parent, "note")
        if i > 0:
            _sub(n, "chord")
        if head is None:
            r = _sub(n, "rest", measure="yes") if e.measure_rest else _sub(n, "rest")
        elif percussion:
            u = _sub(n, "unpitched")
            _sub(u, "display-step", head.pitch.step)
            _sub(u, "display-octave", head.pitch.octave)
        else:
            p = _sub(n, "pitch")
            _sub(p, "step", head.pitch.step)
            if head.pitch.alter:
                _sub(p, "alter", head.pitch.alter)
            _sub(p, "octave", head.pitch.octave)
        _sub(n, "duration", dur)
        if head is not None:
            if head.tie_stop:
                _sub(n, "tie", type="stop")
            if head.tie_start:
                _sub(n, "tie", type="start")
            if percussion and head.instrument:
                _sub(n, "instrument", id=f"{_PART_ID}-{head.instrument}")
        _sub(n, "voice", e.voice)
        if not (head is None and e.measure_rest):
            name = TYPE_NAMES.get(e.base)
            if name:
                _sub(n, "type", name)
            for _ in range(e.dots):
                _sub(n, "dot")
        if head is not None and shown[i]:
            _sub(n, "accidental", shown[i])
        if e.tuplet:
            tm = _sub(n, "time-modification")
            _sub(tm, "actual-notes", e.tuplet.actual)
            _sub(tm, "normal-notes", e.tuplet.normal)
        if head is not None and head.notehead != "normal":
            _sub(n, "notehead", head.notehead)
        _sub(n, "staff", e.staff)
        if e.beams and i == 0:
            for level, value in enumerate(e.beams, start=1):
                _sub(n, "beam", value, number=level)
        notations = []
        if head is not None and (head.tie_start or head.tie_stop):
            notations.append(("tied", head))
        if e.tuplet and i == 0 and (e.tuplet.start or e.tuplet.stop):
            notations.append(("tuplet", None))
        first = i == 0 and head is not None
        if first and (e.articulations or e.fermata or e.slur_start or e.slur_stop):
            notations.append(("extra", None))
        if notations:
            nt = _sub(n, "notations")
            if head is not None and head.tie_stop:
                _sub(nt, "tied", type="stop")
            if head is not None and head.tie_start:
                _sub(nt, "tied", type="start")
            if e.tuplet and i == 0:
                if e.tuplet.start:
                    _sub(nt, "tuplet", type="start", bracket="no" if e.beams else "yes")
                if e.tuplet.stop:
                    _sub(nt, "tuplet", type="stop")
            if first:
                if e.slur_stop:
                    _sub(nt, "slur", type="stop", number=1)
                if e.slur_start:
                    _sub(nt, "slur", type="start", number=1)
                if e.articulations:
                    art = _sub(nt, "articulations")
                    for a in e.articulations:
                        _sub(art, a)
                if e.fermata:
                    _sub(nt, "fermata", type="upright")


class _State:
    """Running key / time / clef state while writing a part."""

    def __init__(self):
        self.key = 0
        self.time = (4, 4)
        self.clefs: dict[int, Clef] = {}


def _write_measure(
    pel, m: Measure, number: int, part: Part, divisions: int, state: _State, first: bool
):
    mel = _sub(pel, "measure", number=number)
    if m.repeat_start:
        bl = _sub(mel, "barline", location="left")
        _sub(bl, "bar-style", "heavy-light")
        _sub(bl, "repeat", direction="forward")
    if first or m.time or m.key is not None or m.clefs:
        attrs = _sub(mel, "attributes")
        if first:
            _sub(attrs, "divisions", divisions)
        if m.key is not None or first:
            state.key = m.key if m.key is not None else state.key
            k = _sub(attrs, "key")
            _sub(k, "fifths", state.key)
        if m.time or first:
            state.time = m.time or state.time
            t = _sub(attrs, "time", symbol=m.time_symbol) if m.time_symbol else _sub(attrs, "time")
            _sub(t, "beats", state.time[0])
            _sub(t, "beat-type", state.time[1])
        if first and part.staves > 1:
            _sub(attrs, "staves", part.staves)
        for s in range(1, part.staves + 1):
            if s in m.clefs or (first and s not in state.clefs):
                state.clefs[s] = m.clefs.get(s, Clef())
                _write_clef(attrs, state.clefs[s], s, part.staves)

    # group events by (staff, voice), keeping order; write each group then back up
    groups: dict[tuple[int, int], list[Event]] = {}
    for e in m.events:
        groups.setdefault((e.staff, e.voice), []).append(e)
    keys = sorted(groups)
    base_alters = key_alters(state.key)
    for gi, key in enumerate(keys):
        alters: dict[tuple[str, int], int] = {}
        total = Fraction(0)
        for idx, e in enumerate(groups[key]):
            shown = []
            for head in e.notes:
                p = head.pitch
                expected = alters.get((p.step, p.octave), base_alters.get(p.step, 0))
                acc = head.accidental
                if acc is None and not part.percussion and p.alter != expected:
                    acc = ACCIDENTAL_NAMES[p.alter]
                alters[(p.step, p.octave)] = p.alter
                shown.append(acc)
            _write_note(mel, e, idx, divisions, shown, part.percussion)
            total += e.duration
        if gi < len(keys) - 1 and total > 0:
            b = _sub(mel, "backup")
            _sub(b, "duration", int(total * divisions))
    if m.repeat_end:
        bl = _sub(mel, "barline", location="right")
        _sub(bl, "bar-style", "light-heavy")
        _sub(bl, "repeat", direction="backward")
    return mel


def to_musicxml(score: Score) -> ET.ElementTree:
    root = ET.Element("score-partwise", version="4.0")
    if score.title:
        _sub(_sub(root, "work"), "work-title", score.title)
    ident = _sub(root, "identification")
    if score.composer:
        _sub(ident, "creator", score.composer, type="composer")
    enc = _sub(ident, "encoding")
    _sub(enc, "software", "note-OCR")

    plist = _sub(root, "part-list")
    for i, part in enumerate(score.parts, start=1):
        sp = _sub(plist, "score-part", id=f"P{i}")
        _sub(sp, "part-name", part.name)
        if part.abbreviation:
            _sub(sp, "part-abbreviation", part.abbreviation)
        if part.percussion:
            from ..percussion import KIT

            for d in KIT:
                si = _sub(sp, "score-instrument", id=f"P{i}-{d.id}")
                _sub(si, "instrument-name", d.name)
            for d in KIT:
                mi = _sub(sp, "midi-instrument", id=f"P{i}-{d.id}")
                _sub(mi, "midi-channel", 10)
                _sub(mi, "midi-unpitched", d.midi)
        else:
            si = _sub(sp, "score-instrument", id=f"P{i}-I1")
            _sub(si, "instrument-name", part.name)
            mi = _sub(sp, "midi-instrument", id=f"P{i}-I1")
            _sub(mi, "midi-channel", i if i < 10 else i + 1)
            _sub(mi, "midi-program", part.midi_program)

    divisions = _divisions(score.parts)
    for i, part in enumerate(score.parts, start=1):
        pel = _sub(root, "part", id=f"P{i}")
        global _PART_ID
        _PART_ID = f"P{i}"
        state = _State()
        for n, m in enumerate(part.measures, start=1):
            _write_measure(pel, m, n, part, divisions, state, first=(n == 1))
    ET.indent(root, space="  ")
    return ET.ElementTree(root)


def write_musicxml(score: Score, path: str | Path) -> Path:
    path = Path(path)
    tree = to_musicxml(score)
    with open(path, "w", encoding="utf-8") as f:
        f.write(DOCTYPE)
        f.write(ET.tostring(tree.getroot(), encoding="unicode"))
        f.write("\n")
    return path


# ---------------------------------------------------------------------- appending


def _last_state(part_el: ET.Element) -> tuple[int | None, tuple[int, int] | None, dict[int, Clef]]:
    key = time = None
    clefs: dict[int, Clef] = {}
    for attrs in part_el.iter("attributes"):
        k = attrs.find("key/fifths")
        if k is not None:
            key = int(k.text)
        t = attrs.find("time")
        if t is not None and t.find("beats") is not None:
            time = (int(t.findtext("beats")), int(t.findtext("beat-type")))
        for c in attrs.findall("clef"):
            num = int(c.get("number", 1))
            clefs[num] = Clef(c.findtext("sign", "G"), int(c.findtext("line", "2") or 2))
    return key, time, clefs


def append_measures(existing: str | Path, new: Score, out: str | Path, part_index: int = 0) -> Path:
    """Append the measures of `new` (its first part) to part `part_index` of an existing
    MusicXML file. Everything already in the file is kept untouched at the XML level;
    key / time / clef are only written where they differ from the file's last state."""
    tree = ET.parse(existing)
    root = tree.getroot()
    parts = root.findall("part")
    target = parts[part_index]
    key, time, clefs = _last_state(target)
    last_number = len(target.findall("measure"))

    src = new.parts[0]
    measures = [Measure(**{**m.__dict__}) for m in src.measures]
    if measures:
        m0 = measures[0]
        if m0.key is not None and m0.key == key:
            m0.key = None
        if m0.time and m0.time == time:
            m0.time = None
        m0.clefs = {s: c for s, c in m0.clefs.items() if clefs.get(s) != c}

    divisions = _divisions([src])
    state = _State()
    state.key = key or 0
    state.time = time or (4, 4)
    state.clefs = dict(clefs)
    for i, m in enumerate(measures):
        mel = _write_measure(target, m, last_number + i + 1, src, divisions, state, first=False)
        if i == 0:
            # our durations use our own divisions: declare them
            attrs = mel.find("attributes")
            if attrs is None:
                attrs = ET.Element("attributes")
                mel.insert(0, attrs)
            div = ET.Element("divisions")
            div.text = str(divisions)
            attrs.insert(0, div)

    # keep the other parts the same length: fill them with full-measure rests
    for pi, part_el in enumerate(parts):
        if pi == part_index:
            continue
        pkey, ptime, _ = _last_state(part_el)
        ptime = ptime or (4, 4)
        n0 = len(part_el.findall("measure"))
        for i, m in enumerate(measures):
            ptime = m.time or ptime
            mel = ET.SubElement(part_el, "measure", number=str(n0 + i + 1))
            if i == 0:
                attrs = ET.SubElement(mel, "attributes")
                ET.SubElement(attrs, "divisions").text = str(divisions)
            note = ET.SubElement(mel, "note")
            ET.SubElement(note, "rest", measure="yes")
            ET.SubElement(note, "duration").text = str(int(Fraction(4 * ptime[0], ptime[1]) * divisions))
            ET.SubElement(note, "voice").text = "1"
    ET.indent(root, space="  ")
    out = Path(out)
    with open(out, "w", encoding="utf-8") as f:
        f.write(DOCTYPE)
        f.write(ET.tostring(root, encoding="unicode"))
        f.write("\n")
    return out
