"""Evaluate on a synthetic validation set (built by synth.py).

    python -m notocr.training.evaluate data/synth_val --weights models/detector.pt

1. Detector: per-class precision / recall (a hit = right class, centre inside the box).
2. End to end: recognised notes vs the generated ground truth, aligned like a diff.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2

from ..model import Score
from ..pipeline import Recognizer
from ..recognition.net import CLASSES
from .generate import random_score


def tokens(score: Score) -> list[tuple]:
    out = []
    for part in score.parts:
        for m in part.measures:
            for e in m.events:
                pitches = tuple(sorted((n.pitch.midi, n.notehead) for n in e.notes)) or ("rest",)
                out.append((pitches, e.duration))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--kind", default=None, help="score kind of the dataset (single for the old v1 sets)")
    a = ap.parse_args()
    root = Path(a.data)
    items = json.loads((root / "index.json").read_text())
    if a.limit:
        items = items[: a.limit]
    rec = Recognizer(a.weights) if a.weights else Recognizer()

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    by_seed: dict[int, list[tuple[int, Score]]] = defaultdict(list)
    for it in items:
        gray = cv2.imread(str(root / it["image"]), cv2.IMREAD_GRAYSCALE)
        res = rec.page(gray)
        gts = [(CLASSES[int(b[0])], b[1:]) for b in it["boxes"]]
        used = set()
        for s in res.symbols:
            hit = None
            for gi, (c, b) in enumerate(gts):
                if gi not in used and c == s.cls and b[0] - 1 <= s.cx <= b[2] + 1 and b[1] - 1 <= s.cy <= b[3] + 1:
                    hit = gi
                    break
            if hit is None:
                fp[s.cls] += 1
            else:
                used.add(hit)
                tp[s.cls] += 1
        for gi, (c, _) in enumerate(gts):
            if gi not in used:
                fn[c] += 1
        m = re.match(r"s(\d+)-(\d+)_v\d+", Path(it["image"]).stem)
        if m:
            by_seed[int(m.group(1))].append((int(m.group(2)), res.score))

    print(f"{'class':16s} {'prec':>6s} {'recall':>6s} {'n':>6s}")
    for c in CLASSES:
        n = tp[c] + fn[c]
        if n == 0 and fp[c] == 0:
            continue
        p = tp[c] / max(tp[c] + fp[c], 1)
        r = tp[c] / max(n, 1)
        print(f"{c:16s} {p:6.3f} {r:6.3f} {n:6d}")

    tot = match = pitch_ok = dur_ok = 0
    for seed, pages in sorted(by_seed.items()):
        gt = tokens(random_score(seed, kind=a.kind))
        rec_score = Score(parts=[p for _, s in sorted(pages, key=lambda t: t[0]) for p in s.parts])
        got = tokens(rec_score)
        sm = difflib.SequenceMatcher(a=gt, b=got, autojunk=False)
        match += sum(b.size for b in sm.get_matching_blocks())
        tot += max(len(gt), len(got))
        # pitch / duration separately
        sm_p = difflib.SequenceMatcher(a=[t[0] for t in gt], b=[t[0] for t in got], autojunk=False)  # noqa
        sm_d = difflib.SequenceMatcher(a=[t[1] for t in gt], b=[t[1] for t in got], autojunk=False)
        pitch_ok += sum(b.size for b in sm_p.get_matching_blocks())
        dur_ok += sum(b.size for b in sm_d.get_matching_blocks())
    if tot:
        print(f"\nend to end over {len(by_seed)} scores, {tot} events:")
        print(f"  events fully right (pitch + duration): {match / tot:.1%}")
        print(f"  pitch right: {pitch_ok / tot:.1%}   duration right: {dur_ok / tot:.1%}")


if __name__ == "__main__":
    main()
