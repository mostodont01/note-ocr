"""Command line: recognise a page and write MusicXML / MuseScore.

    notocr recognize page.png -o out/page.musicxml [--mscz] [--debug out/]
    notocr recognize page2.png --append out/page.musicxml -o out/both.musicxml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


def cmd_recognize(a):
    from .export import musescore
    from .export.musicxml import append_measures, write_musicxml
    from .pipeline import Recognizer
    from .recognition.assemble import check_measures
    from .recognition.detect import draw_symbols
    from .recognition.staff import draw_layout

    rec = Recognizer(a.weights) if a.weights else Recognizer()
    score, pages = rec.document(a.image)
    res = pages[0]
    res.score = score
    res.bad_measures = check_measures(score.parts[0]) if score.parts else []
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
    r = sub.add_parser("recognize", help="image or PDF -> MusicXML")
    r.add_argument("image")
    r.add_argument("-o", "--output")
    r.add_argument("--append", help="existing MusicXML to append the measures to")
    r.add_argument("--mscz", action="store_true", help="also write .mscz via MuseScore")
    r.add_argument("--debug", help="folder for debug images")
    r.add_argument("--weights")
    r.set_defaults(func=cmd_recognize)
    app = sub.add_parser("app", help="open the desktop app")
    app.add_argument("file", nargs="?")
    app.set_defaults(func=lambda a: sys.exit(__import__("notocr.app.backend", fromlist=["main"]).main(
        [sys.argv[0]] + ([a.file] if a.file else []))))
    rt = sub.add_parser("retrain", help="fine-tune the detector with your corrections")
    rt.add_argument("--steps", type=int, default=400)
    rt.set_defaults(func=lambda a: __import__("notocr.training.finetune", fromlist=["finetune"]).finetune(steps=a.steps))
    ff = sub.add_parser("install-firefox", help="register the Firefox extension's helper")
    ff.set_defaults(func=lambda a: [print("installed:", p) for p in __import__(
        "notocr.native_host", fromlist=["install"]).install()])
    inst = sub.add_parser("install-desktop", help="add note-OCR to the app menu")
    inst.set_defaults(func=cmd_install_desktop)
    a = ap.parse_args(argv)
    a.func(a)


def cmd_install_desktop(a):
    import shutil

    apps = Path.home() / ".local" / "share" / "applications"
    icons = Path.home() / ".local" / "share" / "icons" / "hicolor" / "scalable" / "apps"
    apps.mkdir(parents=True, exist_ok=True)
    icons.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).parent / "app" / "assets" / "notocr.svg", icons / "notocr.svg")
    exe = shutil.which("notocr") or f"{sys.executable} -m notocr.cli"
    (apps / "notocr.desktop").write_text(
        "[Desktop Entry]\nType=Application\nName=note-OCR\nComment=Sheet music to MuseScore\n"
        f"Exec={exe} app %f\nIcon=notocr\nTerminal=false\nCategories=AudioVideo;Music;\n"
        "MimeType=application/pdf;image/png;image/jpeg;\nStartupWMClass=notocr\n"
    )
    print("installed:", apps / "notocr.desktop")


if __name__ == "__main__":
    main()
