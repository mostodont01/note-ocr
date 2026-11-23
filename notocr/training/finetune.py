"""M6: retrain the detector with your corrections.

    notocr retrain [--steps 400]

1. Corrected pages (exported on every save of the app) are the new labels.
2. Fine-tune the active model on them (oversampled) mixed with synthetic data, so it
   learns your sheets without forgetting the rest.
3. Compare old vs new: detection on your corrected pages, end-to-end on the synthetic
   validation set. The new model is only used if it is better on your pages and not
   worse on the synthetic set; the previous one is kept as detector_prev.pt.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ..paths import CORRECTIONS, USER_MODELS, active_model, synthetic_sets, synthetic_val


def finetune(sessions=None, steps: int = 400, log=print, repeat: int = 30, val_limit: int = 8) -> dict:
    for s in sessions or []:
        s.export_training(CORRECTIONS)
    index = CORRECTIONS / "index.json"
    if not index.exists() or not json.loads(index.read_text()):
        raise RuntimeError("No corrected pages yet: fix some symbols in the app first")
    n_pages = len(json.loads(index.read_text()))

    old = active_model()
    USER_MODELS.mkdir(parents=True, exist_ok=True)
    new = USER_MODELS / f"detector_{time.strftime('%Y%m%d-%H%M%S')}.pt"
    roots = [f"{CORRECTIONS}*{repeat}"] + [str(p) for p in synthetic_sets()]
    cmd = [sys.executable, "-u", "-m", "notocr.training.train", *roots, "--steps", str(steps), "--lr", "5e-4",
           "--resume", str(old), "--out", str(new), "--save-every", str(steps)]
    log(f"fine-tuning on {n_pages} corrected page(s) + {len(roots) - 1} synthetic set(s), {steps} steps")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            cwd=Path(__file__).resolve().parents[2])
    for line in proc.stdout:
        log(line.rstrip())
    if proc.wait() != 0:
        raise RuntimeError("training failed")

    from .evaluate import evaluate

    res = {"weights": str(new), "old_weights": str(old)}
    res["old_mine"] = evaluate(old, CORRECTIONS)["det_f1"]
    res["new_mine"] = evaluate(new, CORRECTIONS)["det_f1"]
    val = synthetic_val()
    if val is not None:
        res["old_val"] = evaluate(old, val, limit=val_limit)["acc"]
        res["new_val"] = evaluate(new, val, limit=val_limit)["acc"]
    better_mine = res["new_mine"] >= res["old_mine"]
    not_worse = val is None or res["new_val"] is None or res["new_val"] >= res["old_val"] - 0.01
    res["switched"] = bool(better_mine and not_worse)
    log(f"your pages: detection F1 {res['old_mine']:.3f} -> {res['new_mine']:.3f}")
    if val is not None and res.get("new_val") is not None:
        log(f"synthetic val: notes right {res['old_val']:.1%} -> {res['new_val']:.1%}")
    if res["switched"]:
        active = USER_MODELS / "detector.pt"
        if active.exists():
            shutil.copy2(active, USER_MODELS / "detector_prev.pt")
        shutil.copy2(new, active)
        res["weights"] = str(active)
        log("new model in use")
    else:
        log("kept the previous model (the new one was not better)")
    return res


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(prog="notocr retrain")
    ap.add_argument("--steps", type=int, default=400)
    a = ap.parse_args(argv)
    finetune(steps=a.steps)


if __name__ == "__main__":
    main()
