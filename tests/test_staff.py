import numpy as np

from notocr.model import ALTO, BASS, TREBLE, Pitch
from notocr.ingest import binarize
from notocr.recognition.staff import detect_staves, estimate_skew, remove_staff_lines, rotate


def synthetic_page(spacing=12, staves=3, skew=0.01):
    h, w = 120 + staves * 10 * spacing, 800
    ink = np.zeros((h, w), bool)
    for s in range(staves):
        top = 60 + s * 10 * spacing
        for i in range(5):
            for x in range(50, 750):
                ink[int(round(top + i * spacing + skew * x)), x] = True
    # a "note head" crossing a line
    ink[60 + spacing * 2 - 4 : 60 + spacing * 2 + 5, 300:312] = True
    return ink


def test_deskew_then_detect():
    ink = synthetic_page()
    deg = estimate_skew(ink)
    assert abs(deg - np.degrees(np.arctan(0.01))) < 0.1
    gray = np.where(ink, 0, 255).astype(np.uint8)
    layout = detect_staves(binarize(rotate(gray, deg)))
    assert len(layout.staves) == 3
    for s in layout.staves:
        assert abs(s.spacing() - 12) < 0.3
        assert s.x0 <= 52 and s.x1 >= 748


def test_position_and_line_removal():
    ink = synthetic_page(skew=0.0)
    layout = detect_staves(ink)
    st = layout.staves[0]
    assert round(st.position(400, st.lines[-1].y_at(400))) == 0
    assert round(st.position(400, st.lines[0].y_at(400))) == 8
    clean = remove_staff_lines(ink, layout)
    assert not clean[60, 100]  # bare line pixel removed
    assert clean[60 + 24, 305]  # note head kept


def test_clef_pitches():
    assert TREBLE.pitch_at(0) == Pitch("E", 4)
    assert TREBLE.pitch_at(-2) == Pitch("C", 4)
    assert BASS.pitch_at(8) == Pitch("A", 3)
    assert ALTO.pitch_at(4) == Pitch("C", 4)
    assert Pitch("C", 4).midi == 60
