"""Firefox native-messaging helper for the note-OCR extension.

Firefox starts this program and talks to it over stdin/stdout: each message is a
4-byte little-endian length followed by UTF-8 JSON.

Request:  {"action": "recognise", "image": "data:image/png;base64,...",
           "rect": {"x", "y", "w", "h"} in CSS px or null, "dpr": float, "mode": "open"|"quick"}
Reply:    {"ok": true, "path": snippet image, "summary": "...", "musicxml"?: ..., "mscz"?: ...}

Snippets are saved to ~/Music/note-OCR/. "open" opens them in the app for review;
"quick" recognises right away and saves MusicXML (+ .mscz when MuseScore is available).
"""

from __future__ import annotations

import base64
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

SNIPPETS = Path.home() / "Music" / "note-OCR"


def read_message(stream) -> dict | None:
    raw = stream.read(4)
    if len(raw) < 4:
        return None
    (n,) = struct.unpack("<I", raw)
    return json.loads(stream.read(n).decode("utf-8"))


def write_message(stream, msg: dict):
    data = json.dumps(msg).encode("utf-8")
    stream.write(struct.pack("<I", len(data)))
    stream.write(data)
    stream.flush()


def decode_capture(image: str, rect: dict | None, dpr: float) -> np.ndarray:
    """Data URL -> BGR image, cropped to `rect` (CSS px, scaled by device pixel ratio)."""
    b64 = image.split(",", 1)[1] if "," in image else image
    img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode the captured image")
    if rect:
        h, w = img.shape[:2]
        x0 = int(max(0, round(rect["x"] * dpr)))
        y0 = int(max(0, round(rect["y"] * dpr)))
        x1 = int(min(w, round((rect["x"] + rect["w"]) * dpr)))
        y1 = int(min(h, round((rect["y"] + rect["h"]) * dpr)))
        if x1 - x0 < 8 or y1 - y0 < 8:
            raise ValueError("selection is too small")
        img = img[y0:y1, x0:x1]
    return img


def handle(msg: dict, open_app=None, out_dir: Path = SNIPPETS) -> dict:
    if msg.get("action") != "recognise":
        return {"ok": False, "error": f"unknown action {msg.get('action')!r}"}
    img = decode_capture(msg["image"], msg.get("rect"), float(msg.get("dpr") or 1.0))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"snippet-{time.strftime('%Y%m%d-%H%M%S')}.png"
    cv2.imwrite(str(path), img)
    if msg.get("mode") == "quick":
        from .export.musicxml import write_musicxml
        from .pipeline import Recognizer

        score, pages = Recognizer().document(path)
        if not score.parts:
            return {"ok": False, "path": str(path), "error": "no staves found in the selection"}
        xml = path.with_suffix(".musicxml")
        write_musicxml(score, xml)
        reply = {"ok": True, "path": str(path), "musicxml": str(xml),
                 "summary": f"{len(score.parts[0].measures)} measures saved to {xml.name}"}
        try:
            from .export import musescore

            mscz = path.with_suffix(".mscz")
            musescore.convert(xml, mscz)
            reply["mscz"] = str(mscz)
        except Exception:  # noqa: BLE001 - MusicXML is still there
            pass
        return reply
    (open_app or _open_app)(path)
    return {"ok": True, "path": str(path), "summary": "opened in note-OCR"}


def _open_app(path: Path):
    subprocess.Popen([sys.executable, "-m", "notocr.cli", "app", str(path)], start_new_session=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    while True:
        msg = read_message(stdin)
        if msg is None:
            return
        try:
            reply = handle(msg)
        except Exception as e:  # noqa: BLE001 - reported to the extension
            reply = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        write_message(stdout, reply)


def install(extension_id: str = "notocr@mostodont01") -> list[Path]:
    """Register the helper with Firefox (normal and snap installs)."""
    launcher = Path.home() / ".local" / "share" / "notocr" / "notocr-native-host"
    launcher.parent.mkdir(parents=True, exist_ok=True)
    launcher.write_text(f"#!/bin/sh\nexec {sys.executable} -m notocr.native_host \"$@\"\n")
    launcher.chmod(0o755)
    manifest = {
        "name": "notocr",
        "description": "note-OCR helper",
        "path": str(launcher),
        "type": "stdio",
        "allowed_extensions": [extension_id],
    }
    dirs = [Path.home() / ".mozilla" / "native-messaging-hosts",
            Path.home() / "snap" / "firefox" / "common" / ".mozilla" / "native-messaging-hosts"]
    written = []
    for d in dirs:
        if d == dirs[1] and not (Path.home() / "snap" / "firefox").exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        (d / "notocr.json").write_text(json.dumps(manifest, indent=2))
        written.append(d / "notocr.json")
    return written


if __name__ == "__main__":
    main()
