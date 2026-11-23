"""Built-in playback: score -> timeline -> audio (no external synth or SoundFont).

Pitched notes: a few decaying harmonics (plucked / piano-ish). Drums: noise bursts,
a pitch-swept sine for bass drum. Good enough to hear wrong notes and rhythms.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from fractions import Fraction as F

import numpy as np

from .model import Score

SR = 22050


@dataclass
class Sound:
    start: float  # seconds
    dur: float
    midi: list[int]
    drums: list[str]
    measure: int
    event: int
    page: int = -1
    uid: int = -1


def timeline(score: Score, bpm: float = 100.0) -> list[Sound]:
    """One Sound per event; staves / voices start together at each measure."""
    q = 60.0 / bpm  # seconds per quarter
    out: list[Sound] = []
    t0 = 0.0
    if not score.parts:
        return out
    part = score.parts[0]
    for mi, m in enumerate(part.measures):
        pos: dict[tuple[int, int], F] = {}
        longest = F(0)
        for ei, e in enumerate(m.events):
            key = (e.staff, e.voice)
            p = pos.get(key, F(0))
            if e.notes and not all(n.tie_stop for n in e.notes):
                out.append(Sound(
                    t0 + float(p) * q, float(e.duration) * q,
                    [n.pitch.midi for n in e.notes if not part.percussion],
                    [n.instrument or "sn" for n in e.notes] if part.percussion else [],
                    mi, ei,
                    e.region.page if e.region else -1, e.region.uid if e.region else -1,
                ))
            pos[key] = p + e.duration
            longest = max(longest, pos[key])
        t0 += float(longest) * q
    return out


def _tone(midi: int, dur: float) -> np.ndarray:
    f = 440.0 * 2 ** ((midi - 69) / 12)
    n = int(SR * (dur + 0.25))
    t = np.arange(n) / SR
    env = np.exp(-t * 3.0) * np.minimum(1.0, t / 0.005)
    env[int(SR * dur):] *= np.exp(-(t[int(SR * dur):] - dur) * 25)
    y = np.sin(2 * np.pi * f * t) + 0.4 * np.sin(4 * np.pi * f * t) + 0.15 * np.sin(6 * np.pi * f * t)
    return (y * env).astype(np.float32)


def _drum(kind: str, rng: np.random.Generator) -> np.ndarray:
    n = int(SR * 0.3)
    t = np.arange(n) / SR
    if kind in ("bd", "bd2"):
        f = 110 * np.exp(-t * 18) + 45
        return (np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9)).astype(np.float32)
    noise = rng.standard_normal(n).astype(np.float32)
    if kind in ("hh", "ph"):
        return np.diff(noise, prepend=0) * np.exp(-t * 60) * 0.4
    if kind in ("ride", "crash"):
        return np.diff(noise, prepend=0) * np.exp(-t * (6 if kind == "crash" else 14)) * 0.3
    if kind in ("ht", "lt", "ft"):
        f = {"ht": 200, "lt": 150, "ft": 100}[kind]
        return (np.sin(2 * np.pi * f * t) * np.exp(-t * 12)).astype(np.float32)
    return (noise * np.exp(-t * 20) * 0.6 + np.sin(2 * np.pi * 190 * t) * np.exp(-t * 25) * 0.5).astype(np.float32)


def render(sounds: list[Sound]) -> np.ndarray:
    if not sounds:
        return np.zeros(SR // 10, np.float32)
    end = max(s.start + s.dur for s in sounds) + 0.5
    buf = np.zeros(int(SR * end) + SR, np.float32)
    rng = np.random.default_rng(0)
    for s in sounds:
        i = int(SR * s.start)
        for m in s.midi:
            y = _tone(m, s.dur) * 0.25
            buf[i : i + len(y)] += y[: len(buf) - i]
        for d in s.drums:
            y = _drum(d, rng) * 0.6
            buf[i : i + len(y)] += y[: len(buf) - i]
    peak = float(np.abs(buf).max()) or 1.0
    return buf / max(peak, 1.0) * 0.9


def wav_bytes(audio: np.ndarray) -> bytes:
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    bio = io.BytesIO()
    with wave.open(bio, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return bio.getvalue()
