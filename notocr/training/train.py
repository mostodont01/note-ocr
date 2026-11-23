"""Train the symbol detector on labelled page images (synthetic and/or corrections).

    python -m notocr.training.train data/synth --steps 3000 --out models/detector.pt

Runs on the CPU. Checkpoints are written every --save-every steps, so it can be
stopped at any time and resumed with --resume.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as Fn
from torch.utils.data import DataLoader, IterableDataset

from ..recognition.net import CLASSES, NORM_SPACING, STRIDE, Detector, focal_loss

CROP = 256


class CropStream(IterableDataset):
    def __init__(self, roots: list[str], crop: int = CROP, cache: int = 48):
        self.items = []
        for root in roots:
            root, _, rep = str(root).partition("*")  # "dir*K" repeats a set K times
            root = Path(root)
            for it in json.loads((root / "index.json").read_text()) * int(rep or 1):
                self.items.append((str(root / it["image"]), np.array(it["boxes"], dtype=np.float32).reshape(-1, 5)))
        self.crop = crop
        self.cache_size = cache
        # instances per class, for class-balanced sampling of crop centres
        per_class: dict[int, list[tuple[int, int]]] = {}
        for ii, (_, boxes) in enumerate(self.items):
            for bi, b in enumerate(boxes):
                per_class.setdefault(int(b[0]), []).append((ii, bi))
        self.per_class = per_class
        counts = np.array([len(per_class.get(c, [])) for c in range(len(CLASSES))], dtype=float)
        w = np.where(counts > 0, counts**0.3, 0)
        self.class_p = w / w.sum()

    def _image(self, i, cache):
        path = self.items[i][0]
        if path in cache:
            cache.move_to_end(path)
            return cache[path]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        cache[path] = img
        if len(cache) > self.cache_size:
            cache.popitem(last=False)
        return img

    def sample(self, rng: random.Random, cache):
        if rng.random() < 0.8:
            c = int(np.searchsorted(np.cumsum(self.class_p), rng.random()))
            c = min(c, len(CLASSES) - 1)
            ii, bi = rng.choice(self.per_class[c])
            b = self.items[ii][1][bi]
            cx = (b[1] + b[3]) / 2 + rng.uniform(-0.4, 0.4) * self.crop
            cy = (b[2] + b[4]) / 2 + rng.uniform(-0.4, 0.4) * self.crop
        else:
            ii = rng.randrange(len(self.items))
            cx = cy = None
        img = self._image(ii, cache)
        h, w = img.shape
        if cx is None:
            cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        x0, y0 = int(cx - self.crop / 2), int(cy - self.crop / 2)
        crop = np.full((self.crop, self.crop), 255, np.uint8)
        sx0, sy0 = max(x0, 0), max(y0, 0)
        sx1, sy1 = min(x0 + self.crop, w), min(y0 + self.crop, h)
        if sx1 > sx0 and sy1 > sy0:
            crop[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = img[sy0:sy1, sx0:sx1]
        if rng.random() < 0.3:  # extra brightness / contrast jitter
            crop = np.clip(crop.astype(np.float32) * rng.uniform(0.8, 1.1) + rng.uniform(-20, 20), 0, 255).astype(np.uint8)
        boxes = self.items[ii][1].copy()
        boxes[:, [1, 3]] -= x0
        boxes[:, [2, 4]] -= y0
        return crop, boxes

    def targets(self, boxes):
        n = self.crop // STRIDE
        heat = np.zeros((len(CLASSES), n, n), np.float32)
        size = np.zeros((2, n, n), np.float32)
        off = np.zeros((2, n, n), np.float32)
        mask = np.zeros((n, n), np.float32)
        ys, xs = np.mgrid[0:n, 0:n]
        for c, x0, y0, x1, y1 in boxes:
            cx, cy = (x0 + x1) / 2 / STRIDE, (y0 + y1) / 2 / STRIDE
            if not (0 <= cx < n and 0 <= cy < n):
                continue
            bw, bh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
            sigma = max(0.6, 0.12 * min(bw, bh) / STRIDE + 0.4)
            ix, iy = int(cx), int(cy)
            g = np.exp(-((xs - ix) ** 2 + (ys - iy) ** 2) / (2 * sigma**2))
            ci = int(c)
            heat[ci] = np.maximum(heat[ci], g)
            heat[ci, iy, ix] = 1.0
            size[:, iy, ix] = (math.log(bw / NORM_SPACING), math.log(bh / NORM_SPACING))
            off[:, iy, ix] = (cx - ix, cy - iy)
            mask[iy, ix] = 1.0
        return heat, size, off, mask

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = random.Random((info.id if info else 0) * 7919 + int(time.time() * 1000) % 100000)
        cache: OrderedDict = OrderedDict()
        while True:
            crop, boxes = self.sample(rng, cache)
            heat, size, off, mask = self.targets(boxes)
            x = torch.from_numpy(1.0 - crop.astype(np.float32) / 255.0)[None]
            yield x, torch.from_numpy(heat), torch.from_numpy(size), torch.from_numpy(off), torch.from_numpy(mask)


def load_partial(model, ckpt):
    """Load a checkpoint even if the class list grew: matching tensors are copied, the
    heatmap head keeps the rows of classes the checkpoint knows."""
    old_classes = list(ckpt.get("classes", CLASSES))
    state = model.state_dict()
    for k, v in ckpt["model"].items():
        if k not in state:
            continue
        if state[k].shape == v.shape:
            state[k] = v
        elif k.startswith("heat."):
            for oi, c in enumerate(old_classes):
                if c in CLASSES:
                    state[k][CLASSES.index(c)] = v[oi]
    model.load_state_dict(state)


def compute_loss(model, batch):
    x, heat, size, off, mask = batch
    ph, ps, po = model(x)
    l_heat = focal_loss(ph, heat)
    m = mask[:, None]
    n = mask.sum().clamp(min=1)
    l_size = (Fn.l1_loss(ps, size, reduction="none") * m).sum() / n
    l_off = (Fn.l1_loss(po, off, reduction="none") * m).sum() / n
    return l_heat + 0.5 * l_size + l_off, (l_heat.item(), l_size.item(), l_off.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default="models/detector.pt")
    ap.add_argument("--resume", default=None, help="start from this checkpoint (fine-tuning)")
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    model = Detector()
    if a.resume:
        load_partial(model, torch.load(a.resume, map_location="cpu"))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    loader = DataLoader(CropStream(a.data), batch_size=a.batch, num_workers=2, persistent_workers=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    model.train()
    t0 = time.time()
    run = None
    for step, batch in enumerate(loader, start=1):
        loss, parts = compute_loss(model, batch)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        run = parts if run is None else tuple(0.98 * r + 0.02 * p for r, p in zip(run, parts))
        if step % 25 == 0:
            el = time.time() - t0
            print(
                f"step {step}/{a.steps}  heat {run[0]:.3f}  size {run[1]:.3f}  off {run[2]:.3f}  "
                f"{el / step:.2f}s/step  eta {el / step * (a.steps - step) / 60:.0f} min",
                flush=True,
            )
        if step % a.save_every == 0 or step == a.steps:
            torch.save({"model": model.state_dict(), "classes": CLASSES, "step": step}, out)
        if step >= a.steps:
            break
    print("saved", out)


if __name__ == "__main__":
    main()
