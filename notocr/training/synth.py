"""Synthetic training data: random scores -> MuseScore SVG -> labelled page images.

Pages are rendered at a random *source* resolution (staff line distance 5..16 px,
like screenshots and scans), degraded, then rescaled to the network's normalised
resolution (NORM_SPACING px per line distance) exactly like real input is.

    python -m notocr.training.synth data/synth --scores 40
"""

from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path

import cairosvg
import cv2
import numpy as np
from PIL import Image

from ..export import musescore
from ..export.musicxml import write_musicxml
from ..recognition.classes import CLASS_INDEX, NORM_SPACING
from .generate import random_score
from .svglabels import calibrate, label_elements, load_table, parse_svg, save_table

FONTS = {
    "Leland": "Leland Text",
    "Bravura": "Bravura Text",
    "Emmentaler": "MScore Text",
    "Gonville": "Gonville Text",
    "MuseJazz": "MuseJazz Text",
    "Petaluma": "Petaluma Text",
    "Finale Maestro": "Finale Maestro Text",
    "Finale Broadway": "Finale Broadway Text",
}


def write_style(path: Path, font: str) -> Path:
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<museScore version="4.70">\n  <Style>\n'
        f"    <musicalSymbolFont>{font}</musicalSymbolFont>\n"
        f"    <musicalTextFont>{FONTS[font]}</musicalTextFont>\n"
        "  </Style>\n</museScore>\n"
    )
    return path


def render_svg(svg: Path, scale: float) -> np.ndarray:
    """Rasterise an SVG page at `scale` px per SVG unit -> gray uint8 on white."""
    root_w = float(svg.read_text()[:2000].split('viewBox="')[1].split()[2])
    png = cairosvg.svg2png(url=str(svg), output_width=max(int(round(root_w * scale)), 16), background_color="white")
    return np.asarray(Image.open(io.BytesIO(png)).convert("L")).copy()


def degrade(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Scanner / screenshot / JPEG style damage, applied at source resolution."""
    out = img.astype(np.float32)
    if rng.random() < 0.3:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.3, 0.9))
    if rng.random() < 0.4:  # paper tint / contrast
        lo, hi = rng.uniform(0, 60), rng.uniform(200, 255)
        out = lo + out * (hi - lo) / 255.0
    if rng.random() < 0.4:
        out += np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(2, 12), out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(35, 90)])
        out = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return out


def warp(img: np.ndarray, boxes: list, rng: random.Random, spacing: float):
    """Photo-like geometry: small rotation, perspective, page curvature. Boxes follow.

    The warp is defined as a dst -> src map g; boxes use the first-order inverse
    p_dst ~ p_src - (g(p_src) - p_src), fine for these small deformations."""
    h, w = img.shape
    ang = np.radians(rng.uniform(-1.2, 1.2))
    amp = rng.uniform(0, 0.6) * spacing * rng.choice([-1, 1])  # page bend
    phase = rng.uniform(0, np.pi)
    px, py = rng.uniform(-0.012, 0.012), rng.uniform(-0.012, 0.012)  # perspective-ish

    def g(x, y):
        cx, cy = x - w / 2, y - h / 2
        sx = cx * np.cos(ang) - cy * np.sin(ang)
        sy = cx * np.sin(ang) + cy * np.cos(ang)
        sx = sx * (1 + px * cy / h)
        sy = sy * (1 + py * cx / w) + amp * np.sin(np.pi * x / w + phase)
        return sx + w / 2, sy + h / 2

    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    mx, my = g(xs, ys)
    out = cv2.remap(img, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR, borderValue=255)
    new_boxes = []
    for c, x0, y0, x1, y1 in boxes:
        pts = np.array([(x0, y0), (x1, y0), (x0, y1), (x1, y1), ((x0 + x1) / 2, y0), ((x0 + x1) / 2, y1)])
        gx, gy = g(pts[:, 0], pts[:, 1])
        fx, fy = 2 * pts[:, 0] - gx, 2 * pts[:, 1] - gy
        # keep the box size, move its centre (rotation would inflate thin boxes)
        dx, dy = fx.mean() - pts[:, 0].mean(), fy.mean() - pts[:, 1].mean()
        new_boxes.append([c, x0 + dx, y0 + dy, x1 + dx, y1 + dy])
    return out, new_boxes


def lighting(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Uneven light / shadow like a phone photo."""
    h, w = img.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    a = rng.uniform(0, 2 * np.pi)
    grad = (np.cos(a) * xs / w + np.sin(a) * ys / h)
    field = 1.0 - rng.uniform(0.1, 0.45) * (grad - grad.min()) / max(np.ptp(grad), 1e-6)
    if rng.random() < 0.4:  # a soft shadow blob
        cx, cy, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(0.2, 0.6) * w
        field *= 1.0 - 0.35 * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * r**2))
    return np.clip(img.astype(np.float32) * field, 0, 255).astype(np.uint8)


def build(out_dir: str | Path, n_scores: int = 40, variants: int = 2, seed: int = 0, fonts=None,
          kind: str | None = None):
    """kind: single / piano / drums, or None for a mix."""
    out_dir = Path(out_dir)
    rng = random.Random(seed)
    fonts = fonts or list(FONTS)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    index = []
    for fi, font in enumerate(fonts):
        work = out_dir / "work" / font.replace(" ", "_")
        work.mkdir(parents=True, exist_ok=True)
        style = write_style(work / "style.mss", font)
        table_path = work / "table.json"
        if table_path.exists():
            table = load_table(table_path)
        else:
            table = calibrate(work / "cal", style)
            save_table(table, table_path)
        jobs = []
        for i in range(n_scores):
            s = seed * 100000 + fi * 1000 + i
            xml = write_musicxml(random_score(s, kind=kind), work / f"s{s}.musicxml")
            jobs.append((xml, work / f"s{s}.svg"))
        musescore.convert_batch(jobs, style=style)
        unknown_total = 0
        for svg in sorted(work.glob("s*-*.svg")):
            elements, (vw, vh), space = parse_svg(svg)
            labelled, unknown = label_elements(elements, table, space)
            unknown_total += len(unknown)
            if not labelled or space <= 0:
                continue
            for v in range(variants):
                src_d = rng.choice([rng.uniform(5.0, 9.0), rng.uniform(7.0, 16.0)])
                img = degrade(render_svg(svg, src_d / space), rng)
                # back to the normalised resolution, with a small scale error
                norm = NORM_SPACING * rng.uniform(0.93, 1.07) / src_d
                img = cv2.resize(img, None, fx=norm, fy=norm, interpolation=cv2.INTER_LINEAR)
                k = src_d / space * norm  # SVG units -> output px
                boxes = [
                    [CLASS_INDEX[lab], b[0] * k, b[1] * k, b[2] * k, b[3] * k]
                    for lab, b in labelled
                    if lab in CLASS_INDEX
                ]
                if rng.random() < 0.35:  # photo-like
                    img, boxes = warp(img, boxes, rng, NORM_SPACING)
                    img = lighting(img, rng)
                name = f"{svg.stem}_v{v}"
                cv2.imwrite(str(out_dir / "images" / f"{name}.png"), img)
                index.append({"image": f"images/{name}.png", "font": font, "boxes": boxes})
        print(f"{font}: {len(list(work.glob('s*-*.svg')))} pages, {unknown_total} unlabelled glyphs", flush=True)
    (out_dir / "index.json").write_text(json.dumps(index))
    return index


def relabel(out_dir: str | Path):
    """Recompute the labels of an existing dataset from its SVGs (after label fixes).

    The render scale is recovered from the image width (image width = viewBox width * k).
    """
    out_dir = Path(out_dir)
    index = json.loads((out_dir / "index.json").read_text())
    tables = {}
    for it in index:
        font = it["font"]
        work = out_dir / "work" / font.replace(" ", "_")
        if font not in tables:
            if not (work / "table.json").exists():  # labels changed: calibrate again
                save_table(calibrate(work / "cal", work / "style.mss"), work / "table.json")
            tables[font] = load_table(work / "table.json")
        stem = Path(it["image"]).stem.rsplit("_v", 1)[0]
        elements, (vw, vh), space = parse_svg(work / f"{stem}.svg")
        labelled, _ = label_elements(elements, tables[font], space)
        w = cv2.imread(str(out_dir / it["image"]), cv2.IMREAD_GRAYSCALE).shape[1]
        k = w / vw
        it["boxes"] = [
            [CLASS_INDEX[lab], b[0] * k, b[1] * k, b[2] * k, b[3] * k] for lab, b in labelled if lab in CLASS_INDEX
        ]
    (out_dir / "index.json").write_text(json.dumps(index))
    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--scores", type=int, default=40, help="random scores per font")
    ap.add_argument("--variants", type=int, default=2, help="renders per page")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--relabel", action="store_true", help="only recompute labels of an existing dataset")
    a = ap.parse_args()
    if a.relabel:
        relabel(a.out)
    else:
        build(a.out, a.scores, a.variants, a.seed)


if __name__ == "__main__":
    main()
