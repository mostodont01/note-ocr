"""Random music for synthetic training data.

Not meant to sound good: it covers the symbols the recogniser must learn
(clefs, keys, times, all note/rest values, dots, chords, accidentals, ties,
triplets, ledger lines) in plausible proportions.
"""

from __future__ import annotations

import random
from fractions import Fraction as F

from ..model import (
    ALTO,
    BASS,
    TREBLE,
    Event,
    Measure,
    NoteHead,
    Part,
    Pitch,
    Score,
    Tuplet,
    key_alters,
    measure_length,
)

TIMES = [(4, 4)] * 4 + [(3, 4)] * 3 + [(2, 4)] * 2 + [(6, 8)] * 2 + [(2, 2), (3, 8), (9, 8), (12, 8), (5, 4)]
VALUES = [F(4), F(2), F(1), F(1, 2), F(1, 4), F(1, 8)]
VALUE_WEIGHTS = [1, 4, 10, 12, 7, 2]


def _clef(rng: random.Random):
    return rng.choices([TREBLE, BASS, ALTO], weights=[6, 3, 1])[0]


def _pitch(rng: random.Random, clef, key: int, prev: Pitch | None) -> Pitch:
    lo = clef.bottom_line.diatonic - 6  # 3 ledger lines below
    hi = clef.bottom_line.diatonic + 8 + 6  # 3 ledger lines above
    if prev is None or rng.random() < 0.25:
        center = clef.bottom_line.diatonic + 4
        d = int(rng.gauss(center, 4))
    else:
        d = prev.diatonic + rng.choice([-3, -2, -1, -1, 1, 1, 2, 3, 4, -4, 0])
    d = max(lo, min(hi, d))
    alter = key_alters(key).get("CDEFGAB"[d % 7], 0)
    r = rng.random()
    if r < 0.08:
        alter += rng.choice([-1, 1])
    elif r < 0.12:
        alter = 0
    alter = max(-2, min(2, alter))
    return Pitch.from_diatonic(d, alter)


def _fill(rng: random.Random, length: F) -> list[tuple[F, int, bool]]:
    """Split a measure into (base value, dots, is_triplet_group) chunks."""
    out = []
    left = length
    while left > 0:
        if left >= 1 and rng.random() < 0.07:
            out.append((F(1, 2), 0, True))  # triplet of eighths = 1 quarter
            left -= 1
            continue
        choices = [(v, w) for v, w in zip(VALUES, VALUE_WEIGHTS) if v <= left]
        v = rng.choices([c[0] for c in choices], weights=[c[1] for c in choices])[0]
        dots = 0
        if v * F(3, 2) <= left and v >= F(1, 4) and rng.random() < 0.15:
            dots = 1
        out.append((v, dots, False))
        left -= v * (2 - F(1, 2**dots))
    return out


def _beam(events: list[Event], time: tuple[int, int]):
    """Beam groups per beat (per dotted quarter in compound time)."""
    beat = F(3, 2) if time[1] == 8 and time[0] % 3 == 0 else F(4, time[1]) if time[1] <= 4 else F(1, 2)
    if time == (2, 2):
        beat = F(1)
    pos = F(0)
    groups: list[list[Event]] = []
    cur: list[Event] = []
    cur_beat = None
    for e in events:
        b = pos // beat
        beamable = not e.is_rest and e.base <= F(1, 2)
        if not beamable or b != cur_beat:
            if len(cur) > 1:
                groups.append(cur)
            cur = []
        if beamable:
            if not cur:
                cur_beat = b
            cur.append(e)
        pos += e.duration
    if len(cur) > 1:
        groups.append(cur)
    levels = {F(1, 2): 1, F(1, 4): 2, F(1, 8): 3}
    for g in groups:
        n = [levels[e.base] for e in g]
        for i, e in enumerate(g):
            beams = []
            for lv in range(1, n[i] + 1):
                left = i > 0 and n[i - 1] >= lv
                right = i < len(g) - 1 and n[i + 1] >= lv
                if left and right:
                    beams.append("continue")
                elif right:
                    beams.append("begin")
                elif left:
                    beams.append("end")
                else:
                    beams.append("forward hook" if i == 0 else "backward hook")
            e.beams = beams


def random_part(rng: random.Random, n_measures: int, name: str = "Part") -> Part:
    clef = _clef(rng)
    key = rng.randint(-6, 6) if rng.random() < 0.8 else 0
    time = rng.choice(TIMES)
    measures = []
    prev: Pitch | None = None
    for mi in range(n_measures):
        m = Measure()
        if mi == 0:
            m.time, m.key, m.clefs = time, key, {1: clef}
        elif rng.random() < 0.03:
            key = rng.randint(-6, 6)
            m.key = key
        elif rng.random() < 0.02:
            clef = _clef(rng)
            m.clefs = {1: clef}
        length = measure_length(time)
        if rng.random() < 0.05:
            m.events.append(Event([], length, measure_rest=True))
            measures.append(m)
            continue
        for base, dots, triplet in _fill(rng, length):
            count = 3 if triplet else 1
            for k in range(count):
                if rng.random() < 0.15:
                    ev = Event([], base, dots)
                else:
                    p = _pitch(rng, clef, key, prev)
                    heads = [NoteHead(p)]
                    if rng.random() < 0.12:
                        for _ in range(rng.choice([1, 1, 2])):
                            step = rng.choice([2, 3, 4, 5])
                            q = Pitch.from_diatonic(heads[-1].pitch.diatonic + step, key_alters(key).get("CDEFGAB"[(heads[-1].pitch.diatonic + step) % 7], 0))
                            heads.append(NoteHead(q))
                    prev = p
                    ev = Event(heads, base, dots)
                if triplet:
                    ev.tuplet = Tuplet(3, 2, start=k == 0, stop=k == 2)
                m.events.append(ev)
        # ties between equal neighbouring single notes
        for a, b in zip(m.events, m.events[1:]):
            if a.notes and b.notes and len(a.notes) == len(b.notes) == 1 and rng.random() < 0.08:
                b.notes[0] = NoteHead(a.notes[0].pitch, tie_stop=True)
                a.notes[0].tie_start = True
        _beam(m.events, time)
        measures.append(m)
    return Part(name=name, measures=measures)


WORDS = ["Canon", "Sonata", "Etude", "Minuet", "Waltz", "March", "Prelude", "Air", "Rondo", "in", "D", "G",
         "No.", "2", "Allegro", "Andante", "Song", "Dance", "Theme", "Variations", "Little", "Study"]
NAMES = ["Violin", "Flute", "Trumpet in Bb", "Clarinet", "Cello", "Piano", "Horn", "Oboe", "Part", "Voice"]
DYNAMICS = ["pp", "p", "mp", "mf", "f", "ff", "sf", "fp"]


def random_score(seed: int, n_measures: int | None = None) -> Score:
    rng = random.Random(seed)
    n = n_measures or rng.randint(16, 40)
    part = random_part(rng, n, name=rng.choice(NAMES))
    for m in part.measures:
        for e in m.events:
            if rng.random() < 0.03:
                e.dynamic = rng.choice(DYNAMICS)
    title = " ".join(rng.choice(WORDS) for _ in range(rng.randint(1, 4))) if rng.random() < 0.7 else ""
    composer = rng.choice(["J. S. Bach", "Anon.", "W. A. Mozart", "Johann Pachelbel", ""])
    return Score(title=title, composer=composer, parts=[part])
