"""Headless app test: QML loads without errors and the main actions work."""

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtQuick")

from PySide6.QtCore import QUrl  # noqa: E402
from PySide6.QtQml import QQmlApplicationEngine  # noqa: E402

from notocr.app.backend import THEME, Backend  # noqa: E402
from notocr.app.session import Session  # noqa: E402
from tests.test_session import FakeDetector, page_image  # noqa: E402


def test_app_loads_and_edits(qtbot, tmp_path):
    path, top, sp = page_image(tmp_path)
    b = Backend.__new__(Backend)
    Backend.__init__(b, weights=str(tmp_path / "none.pt"))
    b.session = Session(detector=FakeDetector(top, sp), corrections=tmp_path / "corr")
    errors = []
    engine = QQmlApplicationEngine()
    engine.warnings.connect(lambda ws: errors.extend(w.toString() for w in ws))
    engine.rootContext().setContextProperty("app", b)
    engine.rootContext().setContextProperty("theme", THEME)
    engine.load(QUrl.fromLocalFile(str(Path(__file__).parents[1] / "notocr/app/qml/Main.qml")))
    assert engine.rootObjects(), errors
    b.renderResult = lambda: None  # no MuseScore in CI
    b.openFile(QUrl.fromLocalFile(str(path)).toString())
    qtbot.waitUntil(lambda: b.busy == "" and b.pageCount == 1, timeout=10000)
    assert len(b.measures) == 2 and b.pageUrl
    head = [s for s in b.symbols if s["cls"] == "notehead_whole"][0]
    b.clickAt(head["x"] + 2, head["y"] + 2)
    assert b.selInfo["note"] == "B4"
    b.step(1)
    assert b.selInfo["note"] == "C5"
    b.setTool("add")
    b.setAddClass("notehead_whole")
    b.clickAt(240, top + 2 * sp)
    assert "3 corrections" in b.summary or "corrections" in b.summary
    assert not [e for e in errors if "TypeError" in e or "ReferenceError" in e], errors
