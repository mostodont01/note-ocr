"""Command line: recognise a page and write MusicXML / MuseScore.

    notocr recognize page.png -o out/page.musicxml [--mscz] [--debug out/]
    notocr recognize page2.png --append out/page.musicxml -o out/both.musicxml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def cmd_recognize(a):
    from .export import musescore
    from .export.musicxml import append_measures, write_musicxml
    from .pipeline import Recognizer
    from .recognition.detect import draw_symbols
    from .recognition.staff import draw_layout

    rec = Recognizer(a.weights) if a.weights else Recognizer()
    res = rec.file(a.image)
    out = Path(a.output or Path(a.image).with_suffix(".musicxml"))
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.append:
        append_measures(a.append, res.score, out)
    else:
        write_musicxml(res.score, out)
    n_measures = len(res.score.parts[0].measures) if res.score.parts else 0
    print(f"{len(res.layout.staves)} staves, {len(res.symbols)} symbols, {n_measures} measures -> {out}")
    if res.bad_measures:
        print("measures that don't add up (check these):", ", ".join(str(i + 1) for i in res.bad_measures))
    if a.mscz:
        mscz = out.with_suffix(".mscz")
        musescore.convert(out, mscz)
        print("->", mscz)
    if a.debug:
        dbg = Path(a.debug)
        dbg.mkdir(parents=True, exist_ok=True)
        stem = Path(a.image).stem.replace(" ", "_")
        cv2.imwrite(str(dbg / f"{stem}_staves.png"), draw_layout(res.gray, res.layout))
        cv2.imwrite(str(dbg / f"{stem}_symbols.png"), draw_symbols(res.gray, res.symbols, scale=2.0))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="notocr")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("recognize", help="image -> MusicXML")
    r.add_argument("image")
    r.add_argument("-o", "--output")
    r.add_argument("--append", help="existing MusicXML to append the measures to")
    r.add_argument("--mscz", action="store_true", help="also write .mscz via MuseScore")
    r.add_argument("--debug", help="folder for debug images")
    r.add_argument("--weights")
    r.set_defaults(func=cmd_recognize)
    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
