"""Where models and user data live.

The active model is the first that exists of: $NOTOCR_MODEL, the user's own (retrained)
model in ~/.local/share/notocr/models/detector.pt, the repository's models/detector.pt.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
USER_DATA = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "notocr"
USER_MODELS = USER_DATA / "models"
CORRECTIONS = USER_DATA / "corrections"
REPO_MODEL = REPO / "models" / "detector.pt"


def active_model() -> Path:
    env = os.environ.get("NOTOCR_MODEL")
    if env:
        return Path(env)
    user = USER_MODELS / "detector.pt"
    if user.exists():
        return user
    return REPO_MODEL


def synthetic_sets() -> list[Path]:
    """Synthetic training sets in the repository's data folder (if built)."""
    return [p for p in (REPO / "data" / "synth", REPO / "data" / "synth_v2") if (p / "index.json").exists()]


def synthetic_val() -> Path | None:
    for p in (REPO / "data" / "synth_v2_val", REPO / "data" / "synth_val"):
        if (p / "index.json").exists():
            return p
    return None
