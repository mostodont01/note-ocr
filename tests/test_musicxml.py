from fractions import Fraction as F
import xml.etree.ElementTree as ET

from notocr.model import Event, Measure, NoteHead, Part, Pitch, Score, TREBLE, Tuplet, Clef
from notocr.export.musicxml import to_musicxml, write_musicxml, append_measures


def q(step, octave, alter=0, base=F(1), **kw):
    return Event([NoteHead(Pitch(step, octave, alter))], base, **kw)


def canon_start():
    m1 = Measure([Event([], F(4), measure_rest=True)], time=(4, 4), key=2, clefs={1: TREBLE})
    m2 = Measure([q("F", 5, 1), q("E", 5), q("D", 5), q("C", 5, 1)])
    eighths = [q(s, o, a, F(1, 2)) for s, o, a in [("D", 4, 0), ("F", 4, 1), ("A", 4, 0), ("G", 4, 0)] * 2]
    for i, e in enumerate(eighths):
        e.beams = ["begin" if i % 4 == 0 else "end" if i % 4 == 3 else "continue"]
    m3 = Measure(eighths)
    trip = [q("C", 5, 1, F(1, 2), tuplet=Tuplet(3, 2, start=i == 0, stop=i == 2)) for i in range(3)]
    m4 = Measure(trip + [q("D", 5, 0, F(1), dots=1), q("B", 4, 0, F(1, 2)), Event([], F(1))])
    m4.events[3].notes[0].tie_start = True
    return Score("Canon test", "Pachelbel", [Part("Violin", "Vln.", 1, [m1, m2, m3, m4], midi_program=41)])


def test_durations_and_accidentals():
    root = to_musicxml(canon_start()).getroot()
    assert root.find("part/measure/attributes/divisions").text == "6"  # lcm for triplet 8ths
    measures = root.findall("part/measure")
    for m in measures[1:]:
        total = sum(int(n.findtext("duration")) for n in m.findall("note") if n.find("chord") is None)
        assert total == 24  # 4 quarters * 6 divisions
    # F#5 in D major: no accidental shown; C natural would be
    assert measures[1].find("note/accidental") is None
    assert measures[3].find("note/time-modification/actual-notes").text == "3"


def test_append(tmp_path):
    a = write_musicxml(canon_start(), tmp_path / "a.musicxml")
    extra = Score(parts=[Part(measures=[Measure([q("D", 5, base=F(4))], key=2, time=(4, 4), clefs={1: TREBLE})])])
    out = append_measures(a, extra, tmp_path / "b.musicxml")
    root = ET.parse(out).getroot()
    ms = root.findall("part/measure")
    assert len(ms) == 5 and ms[-1].get("number") == "5"
    # same key / time / clef as before -> not repeated
    assert ms[-1].find("attributes/key") is None and ms[-1].find("attributes/clef") is None
