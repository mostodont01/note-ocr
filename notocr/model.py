"""Score model.

Score > Part > Measure > Event (chord or rest) > NoteHead.

Durations are in quarter notes as Fractions (1 = quarter, 1/2 = eighth...).
Every recognised element can carry a Region: where on which page it was read from,
and how confident the recogniser was. That is what lets the app overlay results on
the image and turn corrections into training data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

STEPS = "CDEFGAB"
STEP_SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

TYPE_NAMES = {
    Fraction(8): "breve",
    Fraction(4): "whole",
    Fraction(2): "half",
    Fraction(1): "quarter",
    Fraction(1, 2): "eighth",
    Fraction(1, 4): "16th",
    Fraction(1, 8): "32nd",
    Fraction(1, 16): "64th",
}

# order in which sharps / flats appear in key signatures
SHARP_ORDER = "FCGDAEB"
FLAT_ORDER = "BEADGCF"


def key_alters(fifths: int) -> dict[str, int]:
    """Alteration per step for a key signature (fifths > 0 sharps, < 0 flats)."""
    if fifths >= 0:
        return {s: 1 for s in SHARP_ORDER[:fifths]}
    return {s: -1 for s in FLAT_ORDER[: -fifths]}


@dataclass
class Region:
    x: float
    y: float
    w: float
    h: float
    page: int = 0
    confidence: float = 1.0


@dataclass(frozen=True)
class Pitch:
    step: str  # C D E F G A B
    octave: int  # scientific: C4 = middle C
    alter: int = 0  # -2..2

    @property
    def diatonic(self) -> int:
        """Diatonic step number (C0 = 0)."""
        return self.octave * 7 + STEPS.index(self.step)

    @staticmethod
    def from_diatonic(n: int, alter: int = 0) -> Pitch:
        return Pitch(STEPS[n % 7], n // 7, alter)

    @property
    def midi(self) -> int:
        return 12 * (self.octave + 1) + STEP_SEMITONES[self.step] + self.alter

    def __str__(self) -> str:
        acc = {-2: "bb", -1: "b", 0: "", 1: "#", 2: "##"}[self.alter]
        return f"{self.step}{acc}{self.octave}"


@dataclass(frozen=True)
class Clef:
    sign: str = "G"  # G, F, C, percussion
    line: int = 2
    octave_change: int = 0

    @property
    def bottom_line(self) -> Pitch:
        """Pitch of the bottom staff line."""
        # diatonic number of the clef's reference pitch, placed on its line
        ref = {"G": Pitch("G", 4), "F": Pitch("F", 3), "C": Pitch("C", 4)}.get(self.sign, Pitch("E", 4))
        line_pos = (self.line - 1) * 2  # half spaces above bottom line
        return Pitch.from_diatonic(ref.diatonic - line_pos + 7 * self.octave_change)

    def pitch_at(self, position: int) -> Pitch:
        """Natural pitch at a staff position (half spaces above the bottom line)."""
        return Pitch.from_diatonic(self.bottom_line.diatonic + position)


TREBLE = Clef("G", 2)
BASS = Clef("F", 4)
ALTO = Clef("C", 3)
TENOR = Clef("C", 4)
PERCUSSION = Clef("percussion", 3)


@dataclass
class NoteHead:
    pitch: Pitch
    tie_start: bool = False
    tie_stop: bool = False
    accidental: str | None = None  # displayed accidental: sharp, flat, natural, ...
    notehead: str = "normal"  # normal, x, diamond, ... (percussion)
    instrument: str | None = None  # percussion instrument id
    region: Region | None = None


@dataclass
class Tuplet:
    actual: int = 3
    normal: int = 2
    start: bool = False
    stop: bool = False


@dataclass
class Event:
    """A chord (one or more note heads) or, with no note heads, a rest."""

    notes: list[NoteHead]
    base: Fraction  # written value without dots, in quarters
    dots: int = 0
    tuplet: Tuplet | None = None
    voice: int = 1
    staff: int = 1
    beams: list[str] | None = None  # per beam level: begin / continue / end / forward hook / backward hook
    measure_rest: bool = False
    dynamic: str | None = None  # p, mf, ff ... placed below this event
    region: Region | None = None

    @property
    def is_rest(self) -> bool:
        return not self.notes

    @property
    def written(self) -> Fraction:
        """Dotted value, without tuplet scaling."""
        return self.base * (2 - Fraction(1, 2**self.dots))

    @property
    def duration(self) -> Fraction:
        d = self.written
        if self.tuplet:
            d = d * self.tuplet.normal / self.tuplet.actual
        return d


@dataclass
class Measure:
    events: list[Event] = field(default_factory=list)
    time: tuple[int, int] | None = None  # set where the time signature appears / changes
    time_symbol: str | None = None  # "common" / "cut" when shown as C / ¢
    key: int | None = None  # fifths, set where it appears / changes
    clefs: dict[int, Clef] = field(default_factory=dict)  # staff number -> clef change
    region: Region | None = None


@dataclass
class Part:
    name: str = "Part"
    abbreviation: str = ""
    staves: int = 1
    measures: list[Measure] = field(default_factory=list)
    midi_program: int = 1  # 1-based GM program; 41 = violin
    percussion: bool = False


@dataclass
class Score:
    title: str = ""
    composer: str = ""
    parts: list[Part] = field(default_factory=list)


def measure_length(time: tuple[int, int]) -> Fraction:
    return Fraction(4 * time[0], time[1])
