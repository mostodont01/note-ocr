"""Session + playback logic, with a fake detector (no model weights needed)."""

from fractions import Fraction as F

import cv2
import numpy as np
import pytest

from notocr.app.session import Session
from notocr.playback import render, timeline, wav_bytes
from notocr.recognition.detect import Symbol


def page_image(tmp_path, spacing=12):
    h, w = 200, 600
    img = np.full((h, w), 255, np.uint8)
    top = 60
    for i in range(5):
        cv2.line(img, (40, top + i * spacing), (560, top + i * spacing), 0, 1)
    for x in (300, 560):  # barlines
        cv2.line(img, (x, top), (x, top + 4 * spacing), 0, 2)
    path = tmp_path / "page.png"
    cv2.imwrite(str(path), img)
    return path, top, spacing


class FakeDetector:
    """Four whole notes on the middle line, one per half of each measure... in 4/4:
    two whole notes = two measures."""

    def __init__(self, top, spacing):
        self.top, self.sp = top, spacing

    def detect(self, gray, spacing, threshold=0.3):
        y = self.top + 2 * self.sp  # middle line (B4 in treble)
        sp = self.sp
        out = [
            Symbol("clef_g", 0.9, 45, self.top - sp, 70, self.top + 5 * sp),
            Symbol("time_4", 0.9, 80, self.top, 95, self.top + 2 * sp),
            Symbol("time_4", 0.9, 80, self.top + 2 * sp, 95, self.top + 4 * sp),
            Symbol("barline", 0.9, 299, self.top, 301, self.top + 4 * sp),
            Symbol("barline", 0.9, 559, self.top, 561, self.top + 4 * sp),
        ]
        for x in (180, 420):
            out.append(Symbol("notehead_whole", 0.9, x - 8, y - sp / 2, x + 8, y + sp / 2))
        return out


@pytest.fixture
def session(tmp_path):
    path, top, sp = page_image(tmp_path)
    s = Session(detector=FakeDetector(top, sp), corrections=tmp_path / "corrections")
    s.open(path, use_saved=False)
    return s


def heads(s):
    return [x for x in s.pages[0].symbols if x.cls.startswith("notehead")]


def test_open_and_assemble(session):
    part = session.score.parts[0]
    assert len(part.measures) == 2
    assert [str(e.notes[0].pitch) for m in part.measures for e in m.events] == ["B4", "B4"]
    assert all(m.ok for m in session.measures)


def test_symbol_edit_and_override(session):
    h = heads(session)[1]
    session.set_class(0, h.uid, "notehead_half")
    assert session.event_for(0, h.uid).duration == F(2)
    assert not session.measures[1].ok  # half note in 4/4 doesn't fill the bar (bar 1 may be a pickup)
    session.override(0, h.uid, steps=2)
    assert str(session.event_for(0, h.uid).notes[0].pitch) == "D5"
    session.override(0, h.uid, base="4")
    assert session.event_for(0, h.uid).duration == F(4)
    kinds = [c["kind"] for c in session.corrections]
    assert kinds == ["class", "override", "override"]


def test_add_delete_save_reload(session, tmp_path):
    uid = session.add(0, "notehead_whole", 240, 60 + 24)
    assert len(session.score.parts[0].measures[0].events) == 2
    session.delete(0, uid)
    assert len(session.score.parts[0].measures[0].events) == 1
    h = heads(session)[0]
    session.override(0, h.uid, steps=-1)
    session.save()
    assert (tmp_path / "corrections" / "index.json").exists()  # edited page exported on save
    s2 = Session(detector=None, corrections=None)
    s2.open(session.source)  # loads saved symbols: no detector needed
    assert str(s2.score.parts[0].measures[0].events[0].notes[0].pitch) == "A4"
    n = s2.export_training(tmp_path / "corr")
    assert n == 1 and (tmp_path / "corr" / "index.json").exists()


def test_playback(session):
    snd = timeline(session.score, bpm=120)
    assert [round(s.start, 2) for s in snd] == [0.0, 2.0]
    audio = render(snd)
    assert audio.shape[0] > 22050 * 3 and np.abs(audio).max() <= 1.0
    assert wav_bytes(audio)[:4] == b"RIFF"


def test_playback_repeats_and_dynamics():
    from notocr.model import Event, Measure, NoteHead, Part, Pitch, Score
    from notocr.playback import play_order

    ms = [Measure([Event([NoteHead(Pitch("C", 4))], F(4))]) for _ in range(4)]
    ms[1].repeat_start = True
    ms[2].repeat_end = True
    assert play_order(ms) == [0, 1, 2, 1, 2, 3]
    ms[0].events[0].dynamic = "pp"
    ms[3].events[0].dynamic = "ff"
    snd = timeline(Score(parts=[Part(measures=ms)]), bpm=60)
    assert len(snd) == 6 and snd[0].gain < snd[-1].gain
