"""MuseScore command-line wrapper: format conversion and rendering.

MuseScore reads MusicXML and writes .mscz, .svg, .png, .pdf, .musicxml ...
Set NOTOCR_MSCORE to override the command (e.g. "mscore" or an AppImage path).
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

FLATPAK_ID = "org.musescore.MuseScore"


def mscore_command() -> list[str]:
    env = os.environ.get("NOTOCR_MSCORE")
    if env:
        return shlex.split(env)
    for name in ("mscore4portable", "mscore", "musescore4", "musescore", "mscore3"):
        path = shutil.which(name)
        if path:
            return [path]
    if shutil.which("flatpak"):
        r = subprocess.run(["flatpak", "info", FLATPAK_ID], capture_output=True)
        if r.returncode == 0:
            return ["flatpak", "run", "--command=mscore", FLATPAK_ID]
    raise FileNotFoundError(
        "MuseScore not found. Install it (e.g. `flatpak install flathub org.musescore.MuseScore`) "
        "or set NOTOCR_MSCORE."
    )


def _run(args: list[str], timeout: float = 600) -> subprocess.CompletedProcess:
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    r = subprocess.run(mscore_command() + args, capture_output=True, text=True, timeout=timeout, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"MuseScore failed ({r.returncode}):\n{r.stderr[-2000:]}")
    return r


def convert(src: str | Path, dst: str | Path, style: str | Path | None = None) -> Path:
    """Convert one file (format from the extension), e.g. .musicxml -> .mscz."""
    args = ["-o", str(Path(dst).resolve()), str(Path(src).resolve())]
    if style:
        args = ["-S", str(Path(style).resolve())] + args
    _run(args)
    return Path(dst)


def convert_batch(jobs: list[tuple[str | Path, str | Path]], style: str | Path | None = None) -> None:
    """Many conversions in one MuseScore start (startup takes seconds)."""
    spec = [{"in": str(Path(a).resolve()), "out": str(Path(b).resolve())} for a, b in jobs]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, dir=Path(jobs[0][1]).parent) as f:
        json.dump(spec, f)
        job_file = f.name
    try:
        args = ["-j", job_file]
        if style:
            args = ["-S", str(Path(style).resolve())] + args
        _run(args, timeout=60 + 20 * len(jobs))
    finally:
        os.unlink(job_file)
