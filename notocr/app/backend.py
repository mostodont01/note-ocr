"""Qt bridge between QML and the Session (all logic lives in session.py)."""

from __future__ import annotations

import sys
import tempfile
import threading
import traceback
from pathlib import Path

import cv2
from PySide6.QtCore import Property, QObject, QTimer, QUrl, Signal, Slot

from ..recognition.classes import CLASSES
from .session import Session

THEME = {
    "background": "#17191d", "panel": "#202329", "panelRaised": "#282c33", "border": "#383d46",
    "text": "#e3e7ed", "muted": "#929ba8", "faint": "#687280", "accent": "#72a7ff",
    "ok": "#6fcf97", "warn": "#e5b567", "error": "#f28b82",
}

KIND_COLORS = {
    "notehead": "#72a7ff", "rest": "#6fcf97", "sharp": "#e5b567", "flat": "#e5b567", "natural": "#e5b567",
    "double": "#e5b567", "clef": "#c084fc", "time": "#c084fc", "flag": "#94a3b8", "stem": "#687280",
    "barline": "#687280", "dot": "#f472b6", "tie": "#f472b6", "tuplet": "#f472b6",
}

DURATIONS = {"4": "whole", "2": "half", "1": "quarter", "1/2": "8th", "1/4": "16th", "1/8": "32nd", "1/16": "64th"}


def kind_color(cls: str) -> str:
    for k, v in KIND_COLORS.items():
        if cls.startswith(k):
            return v
    return "#929ba8"


def url_path(url: str) -> Path:
    u = QUrl(url)
    return Path(u.toLocalFile() if u.isLocalFile() else url)


class Backend(QObject):
    changed = Signal()
    pageChanged = Signal()
    selectionChanged = Signal()
    busyChanged = Signal()
    playChanged = Signal()
    resultChanged = Signal()
    message = Signal(str)
    _done = Signal(str, str)  # (job, error) from worker threads

    def __init__(self, weights: str | None = None):
        super().__init__()
        self.session = Session(weights) if weights else Session()
        # not /tmp: MuseScore runs as a flatpak with a private /tmp and could not see it
        cache = Path.home() / ".cache" / "notocr"
        cache.mkdir(parents=True, exist_ok=True)
        _clean_cache(cache)
        self._tmp = Path(tempfile.mkdtemp(prefix="session-", dir=cache))
        self._page = 0
        self._sel = -1
        self._busy = ""
        self._tool = "select"
        self._add_class = "notehead_black"
        self._page_urls: list[str] = []
        self._result: list[str] = []
        self._result_busy = False
        self._play_uid = -1
        self._play_page = -1
        self._sounds = []
        self._player = None
        self._timer = QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._tick)
        self._done.connect(self._on_done)

    # ------------------------------------------------------------- properties

    @Property("QVariantMap", constant=True)
    def theme(self):
        return THEME

    @Property(list, constant=True)
    def classes(self):
        return CLASSES

    @Property(str, notify=busyChanged)
    def busy(self):
        return self._busy

    @Property(str, notify=changed)
    def title(self):
        return self.session.source.name if self.session.source else ""

    @Property(int, notify=changed)
    def pageCount(self):
        return len(self.session.pages)

    @Property(int, notify=pageChanged)
    def currentPage(self):
        return self._page

    @Property(str, notify=pageChanged)
    def pageUrl(self):
        return self._page_urls[self._page] if self._page < len(self._page_urls) else ""

    @Property(int, notify=pageChanged)
    def pageWidth(self):
        return self.session.pages[self._page].gray.shape[1] if self.session.pages else 0

    @Property(int, notify=pageChanged)
    def pageHeight(self):
        return self.session.pages[self._page].gray.shape[0] if self.session.pages else 0

    @Property(list, notify=changed)
    def symbols(self):
        if not self.session.pages:
            return []
        p = self.session.pages[self._page]
        return [
            {"uid": s.uid, "cls": s.cls, "x": s.x0, "y": s.y0, "w": s.w, "h": s.h, "score": s.score,
             "color": kind_color(s.cls)}
            for s in p.symbols
        ]

    @Property(list, notify=changed)
    def measures(self):
        return [
            {"index": m.index, "page": m.page, "x": m.x, "y": m.y, "w": m.w, "h": m.h, "ok": m.ok,
             "label": f"{m.index + 1}" + ("" if m.ok else f"   {m.total} ≠ {m.expected}")}
            for m in self.session.measures
        ]

    @Property(str, notify=changed)
    def summary(self):
        if not self.session.pages:
            return "Open an image or PDF of sheet music (Ctrl+O)"
        bad = sum(not m.ok for m in self.session.measures)
        edits = len(self.session.corrections)
        return (f"{len(self.session.measures)} measures · {bad} to check · {edits} corrections · "
                f"model {self.session.weights.name}")

    @Property(int, notify=selectionChanged)
    def selected(self):
        return self._sel

    @Property("QVariantMap", notify=selectionChanged)
    def selInfo(self):
        if self._sel < 0 or not self.session.pages:
            return {}
        s = self.session.pages[self._page].symbol(self._sel)
        if s is None:
            return {}
        info = {"cls": s.cls, "score": round(s.score, 2), "note": "", "duration": "", "hasEvent": False}
        e = self.session.event_for(self._page, self._sel)
        if e is not None:
            info["hasEvent"] = True
            info["note"] = "rest" if e.is_rest else " + ".join(str(n.pitch) for n in e.notes)
            if e.notes and e.notes[0].instrument:
                info["note"] = " + ".join(n.instrument for n in e.notes)
            name = DURATIONS.get(str(e.base), str(e.base))
            info["duration"] = name + ("." * e.dots) + (" (triplet)" if e.tuplet else "")
        return info

    @Property(str, notify=selectionChanged)
    def tool(self):
        return self._tool

    @Property(str, notify=selectionChanged)
    def addClass(self):
        return self._add_class

    @Property(list, notify=resultChanged)
    def resultImages(self):
        return self._result

    @Property(bool, notify=resultChanged)
    def resultBusy(self):
        return self._result_busy

    @Property(bool, notify=playChanged)
    def playing(self):
        return self._timer.isActive()

    @Property(int, notify=playChanged)
    def playUid(self):
        return self._play_uid if self._play_page == self._page else -1

    # --------------------------------------------------------------- workers

    def _run(self, job: str, fn):
        self._busy = job
        self.busyChanged.emit()

        def work():
            try:
                fn()
                self._done.emit(job, "")
            except Exception as e:  # noqa: BLE001 - shown to the user
                traceback.print_exc()
                self._done.emit(job, f"{type(e).__name__}: {e}")

        threading.Thread(target=work, daemon=True).start()

    @Slot(str, str)
    def _on_done(self, job, error):
        self._busy = ""
        self.busyChanged.emit()
        if error:
            self.message.emit(f"{job} failed: {error}")
        if job == "Recognising":
            self._write_pages()
            self._page = 0
            self._sel = -1
            self.pageChanged.emit()
            self.changed.emit()
            self.selectionChanged.emit()
            self.renderResult()
        elif job.startswith("Exporting"):
            if not error:
                self.message.emit("Saved")
        elif job == "Rendering":
            self._result_busy = False
            self.resultChanged.emit()
        elif job == "Retraining" and not error:
            self.message.emit("Retrained: new model in use")
            self.changed.emit()

    def _write_pages(self):
        self._page_urls = []
        for i, p in enumerate(self.session.pages):
            path = self._tmp / f"page{i + 1}_{id(p.gray)}.png"
            cv2.imwrite(str(path), p.gray)
            self._page_urls.append(QUrl.fromLocalFile(str(path)).toString())

    # ----------------------------------------------------------------- slots

    @Slot(str)
    def openFile(self, url):
        path = url_path(url)
        self.stop()
        self._run("Recognising", lambda: self.session.open(path))

    @Slot(int)
    def setPage(self, i):
        if 0 <= i < len(self.session.pages) and i != self._page:
            self._page = i
            self._sel = -1
            self.pageChanged.emit()
            self.changed.emit()
            self.selectionChanged.emit()

    @Slot(int)
    def select(self, uid):
        self._sel = uid
        self.selectionChanged.emit()

    @Slot(str)
    def setTool(self, tool):
        self._tool = tool
        self.selectionChanged.emit()

    @Slot(str)
    def setAddClass(self, cls):
        self._add_class = cls
        self.selectionChanged.emit()

    @Slot(float, float)
    def clickAt(self, x, y):
        """Click on the page (image pixels): add a symbol with the add tool, else select
        the smallest symbol under the cursor."""
        if not self.session.pages:
            return
        if self._tool == "add":
            uid = self.session.add(self._page, self._add_class, x, y)
            self._after_edit(uid)
            return
        p = self.session.pages[self._page]
        pad = 0.25 * p.spacing
        hits = [s for s in p.symbols if s.x0 - pad <= x <= s.x1 + pad and s.y0 - pad <= y <= s.y1 + pad]
        hits.sort(key=lambda s: (not s.cls.startswith("notehead"), s.w * s.h))
        self.select(hits[0].uid if hits else -1)

    def _after_edit(self, uid=None):
        if uid is not None:
            self._sel = uid
        self.session.save()
        self.changed.emit()
        self.selectionChanged.emit()
        self._schedule_render()

    def _schedule_render(self):
        if not hasattr(self, "_render_timer"):
            self._render_timer = QTimer(self)
            self._render_timer.setSingleShot(True)
            self._render_timer.setInterval(1500)
            self._render_timer.timeout.connect(self.renderResult)
        self._render_timer.start()

    @Slot(str)
    def setClass(self, cls):
        if self._sel >= 0:
            self.session.set_class(self._page, self._sel, cls)
            self._after_edit()

    @Slot()
    def deleteSelected(self):
        if self._sel >= 0:
            self.session.delete(self._page, self._sel)
            self._sel = -1
            self._after_edit()

    def _override(self, **ch):
        if self._sel >= 0 and self.session.event_for(self._page, self._sel) is not None:
            self.session.override(self._page, self._sel, **ch)
            self._after_edit()

    @Slot(int)
    def step(self, n):
        self._override(steps=n)

    @Slot(int)
    def accidental(self, alter):
        self._override(alter=alter)

    @Slot(str)
    def duration(self, base):
        self._override(base=base)

    @Slot()
    def toggleDot(self):
        e = self.session.event_for(self._page, self._sel) if self._sel >= 0 else None
        if e is not None:
            self._override(dots=0 if e.dots else 1)

    @Slot()
    def toggleRest(self):
        self._override(rest=True)

    @Slot(int)
    def nextBad(self, direction):
        bad = [m for m in self.session.measures if not m.ok]
        if not bad:
            self.message.emit("All measures add up")
            return
        cur = self._cur_measure if hasattr(self, "_cur_measure") else -1
        if direction > 0:
            nxt = next((m for m in bad if m.index > cur), bad[0])
        else:
            nxt = next((m for m in reversed(bad) if m.index < cur), bad[-1])
        self._cur_measure = nxt.index
        self.setPage(nxt.page)
        self.focusMeasure.emit(nxt.x, nxt.y, nxt.w, nxt.h)

    focusMeasure = Signal(float, float, float, float)

    @Slot(int)
    def gotoMeasure(self, index):
        m = self.session.measures[index]
        self._cur_measure = index
        self.setPage(m.page)
        self.focusMeasure.emit(m.x, m.y, m.w, m.h)

    # ---------------------------------------------------------------- export

    @Slot(str)
    def exportXml(self, url):
        from ..export.musicxml import write_musicxml

        path = url_path(url)
        self._run("Exporting MusicXML", lambda: write_musicxml(self.session.score, path))

    @Slot(str)
    def exportMscz(self, url):
        from ..export import musescore
        from ..export.musicxml import write_musicxml

        path = url_path(url)

        def job():
            xml = self._tmp / "export.musicxml"
            write_musicxml(self.session.score, xml)
            musescore.convert(xml, path)

        self._run("Exporting MuseScore", job)

    @Slot(str, str)
    def appendTo(self, existing_url, out_url):
        """Append this result's measures to an existing .musicxml or .mscz file."""
        from ..export import musescore
        from ..export.musicxml import append_measures

        existing, out = url_path(existing_url), url_path(out_url)

        def job():
            src = existing
            if existing.suffix.lower() == ".mscz":
                src = self._tmp / "existing.musicxml"
                musescore.convert(existing, src)
            if out.suffix.lower() == ".mscz":
                xml = self._tmp / "appended.musicxml"
                append_measures(src, self.session.score, xml)
                musescore.convert(xml, out)
            else:
                append_measures(src, self.session.score, out)

        self._run("Exporting (append)", job)

    @Slot()
    def renderResult(self):
        """Render the recognised music with MuseScore, to compare with the original."""
        if not self.session.score.parts or self._result_busy:
            return
        from ..export import musescore
        from ..export.musicxml import write_musicxml

        self._result_busy = True
        self.resultChanged.emit()
        out_dir = self._tmp / f"render{len(self._result)}_{id(self.session.score)}"
        out_dir.mkdir(exist_ok=True)

        def job():
            xml = out_dir / "result.musicxml"
            write_musicxml(self.session.score, xml)
            musescore.convert(xml, out_dir / "result.png", dpi=110)
            self._result = [QUrl.fromLocalFile(str(p)).toString() for p in sorted(out_dir.glob("result-*.png"))]

        self._run("Rendering", job)

    # -------------------------------------------------------------- playback

    @Slot()
    def play(self):
        from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

        from ..playback import render, timeline, wav_bytes

        if not self.session.score.parts:
            return
        self.stop()
        self._sounds = timeline(self.session.score)
        wav = self._tmp / f"play{id(self._sounds)}.wav"
        wav.write_bytes(wav_bytes(render(self._sounds)))
        if self._player is None:
            self._player = QMediaPlayer(self)
            self._audio = QAudioOutput(self)
            self._player.setAudioOutput(self._audio)
        self._player.setSource(QUrl.fromLocalFile(str(wav)))
        self._player.play()
        self._timer.start()
        self.playChanged.emit()

    @Slot()
    def stop(self):
        if self._player is not None:
            self._player.stop()
        self._timer.stop()
        self._play_uid = -1
        self.playChanged.emit()

    @Slot()
    def togglePlay(self):
        if self._timer.isActive():
            self.stop()
        else:
            self.play()

    def _tick(self):
        if self._player is None:
            return
        from PySide6.QtMultimedia import QMediaPlayer

        if self._player.playbackState() == QMediaPlayer.StoppedState and self._player.position() > 0:
            self.stop()
            return
        t = self._player.position() / 1000.0
        cur = None
        for s in self._sounds:
            if s.start <= t < s.start + s.dur:
                cur = s
                break
        if cur is not None and (cur.uid != self._play_uid or cur.page != self._play_page):
            self._play_uid, self._play_page = cur.uid, cur.page
            if cur.page >= 0 and cur.page != self._page:
                self.setPage(cur.page)
            self.playChanged.emit()

    # -------------------------------------------------------------- retraining

    @Slot()
    def retrain(self):
        from ..training.finetune import finetune

        def job():
            self.session.save()
            res = finetune([self.session], log=lambda line: print(line, flush=True))
            if res.get("switched"):
                from ..recognition.detect import SymbolDetector

                self.session._detector = SymbolDetector(res["weights"])
                self.session.weights = Path(res["weights"])

        self._run("Retraining", job)


def _clean_cache(cache: Path, keep: int = 5):
    """Remove old session folders (page images, renders) but the newest few."""
    import shutil

    sessions = sorted(cache.glob("session-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in sessions[keep:]:
        shutil.rmtree(old, ignore_errors=True)


def main(argv=None) -> int:
    from PySide6.QtGui import QGuiApplication, QIcon
    from PySide6.QtQml import QQmlApplicationEngine

    argv = argv if argv is not None else sys.argv
    app = QGuiApplication(argv)
    app.setApplicationName("note-OCR")
    app.setDesktopFileName("notocr")
    icon = Path(__file__).parent / "assets" / "notocr.svg"
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))
    backend = Backend()
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("app", backend)
    engine.rootContext().setContextProperty("theme", THEME)
    engine.load(QUrl.fromLocalFile(str(Path(__file__).parent / "qml" / "Main.qml")))
    if not engine.rootObjects():
        return 1
    files = [a for a in argv[1:] if not a.startswith("-")]
    if files:
        backend.openFile(QUrl.fromLocalFile(str(Path(files[0]).resolve())).toString())
    return app.exec()
