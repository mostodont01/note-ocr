from notocr.pdf import Glyph, refine
from notocr.recognition.detect import Symbol


def test_refine_fixes_class_and_adds_missing():
    sp = 10.0
    dets = [Symbol("notehead_half", 0.6, 100, 95, 112, 105), Symbol("notehead_black", 0.9, 200, 95, 212, 105)]
    glyphs = [
        Glyph("notehead_black", 100.5, 100, (100, 80, 112, 120)),  # detector said half: fix
        Glyph("notehead_black", 200.2, 100, (200, 80, 212, 120)),  # agrees
        Glyph("notehead_black", 300.0, 100, (300, 80, 312, 120)),  # missed: add
    ]
    out = refine(dets, glyphs, sp)
    classes = sorted((round(s.x0), s.cls) for s in out)
    assert classes == [(100, "notehead_black"), (200, "notehead_black"), (300, "notehead_black")]
    added = [s for s in out if round(s.x0) == 300][0]
    assert abs(added.cy - 100) < 0.01 and abs(added.w - 12) < 0.01
