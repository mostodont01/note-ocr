"""Drum kit notation: staff position + notehead shape <-> instrument.

Common 5-line drum notation (also MuseScore's default drumset). Positions are half
staff spaces above the bottom line (0 = bottom line, 8 = top line).
"""

from __future__ import annotations

from dataclasses import dataclass

from .model import Pitch


@dataclass(frozen=True)
class Drum:
    id: str
    name: str
    midi: int  # General MIDI percussion key
    pos: int  # staff position
    notehead: str = "normal"  # normal / x

    @property
    def display(self) -> Pitch:
        # percussion clef: positions like treble clef (bottom line = E4)
        return Pitch.from_diatonic(Pitch("E", 4).diatonic + self.pos)


DRUMS = [
    Drum("bd", "Acoustic Bass Drum", 35, 1),  # F4
    Drum("bd2", "Bass Drum 1", 36, 1, "normal"),
    Drum("ft", "High Floor Tom", 43, 3),  # A4
    Drum("sn", "Acoustic Snare", 38, 5),  # C5
    Drum("lt", "Low-Mid Tom", 47, 6),  # D5
    Drum("ht", "High Tom", 50, 7),  # E5
    Drum("ride", "Ride Cymbal 1", 51, 8, "x"),  # F5
    Drum("hh", "Closed Hi-Hat", 42, 9, "x"),  # G5
    Drum("crash", "Crash Cymbal 1", 49, 10, "x"),  # A5
    Drum("ph", "Pedal Hi-Hat", 44, -1, "x"),  # D4
    Drum("ss", "Side Stick", 37, 5, "x"),  # C5 x
]
KIT = [d for d in DRUMS if d.id != "bd2"]
BY_ID = {d.id: d for d in DRUMS}


def drum_at(pos: int, notehead: str) -> Drum:
    """Best matching drum for a recognised head (exact match, else nearest same shape)."""
    for d in KIT:
        if d.pos == pos and d.notehead == notehead:
            return d
    same = [d for d in KIT if d.notehead == notehead] or KIT
    return min(same, key=lambda d: abs(d.pos - pos))
