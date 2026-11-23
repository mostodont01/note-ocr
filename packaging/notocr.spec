# PyInstaller spec: `uv run pyinstaller packaging/notocr.spec` -> dist/notocr/
# Inference only (ONNX Runtime); PyTorch and the training tools are left out.
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

root = Path(SPECPATH).parent
hidden = [m for m in collect_submodules("notocr") if not m.startswith(("notocr.training", "notocr.recognition.net"))]
hidden += ["notocr.training.finetune"]  # imported lazily; reports that retraining needs the dev install

a = Analysis(
    [str(root / "packaging" / "launch.py")],
    pathex=[str(root)],
    datas=[
        (str(root / "notocr" / "app" / "qml"), "notocr/app/qml"),
        (str(root / "notocr" / "app" / "assets"), "notocr/app/assets"),
        (str(root / "models" / "detector.onnx"), "models"),
    ],
    hiddenimports=hidden,
    excludes=["torch", "torchvision", "cairosvg", "onnx", "pytest", "tkinter", "IPython", "matplotlib",
              "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
              "PySide6.Qt3DCore", "PySide6.QtCharts", "PySide6.QtDataVisualization", "PySide6.QtGraphs"],
    noarchive=False,
)
# Qt modules the app doesn't use but QML plugin discovery drags in
UNUSED = ("WebEngine", "WebView", "Quick3D", "3DCore", "3DRender", "3DInput", "3DLogic", "3DAnimation",
          "3DExtras", "3DQuick", "Charts", "DataVisualization", "Graphs", "Pdf", "Scxml", "VirtualKeyboard",
          "Location", "Positioning", "Sensors", "RemoteObjects", "SerialBus", "Bluetooth", "Nfc", "Designer",
          "Help", "Sql", "Test", "Lottie", "Spatial")


def keep(entry):
    name = entry[0].replace("\\", "/")
    return not any(u in name for u in UNUSED)


a.binaries = [b for b in a.binaries if keep(b)]
a.datas = [d for d in a.datas if keep(d)]
pyz = PYZ(a.pure)
icon = root / "packaging" / "notocr.ico"
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="notocr", console=False,
          icon=str(icon) if icon.exists() else None)
coll = COLLECT(exe, a.binaries, a.datas, name="notocr")
