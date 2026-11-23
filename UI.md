# note-OCR app — UI design (2026-11-23)

PySide6 + QML, same look as VL / School Notes / Photoroom (dark theme, accent `#72a7ff`).
Logic lives in `notocr/app/session.py` (no Qt, tested headless); `backend.py` only exposes
it to QML. Everything the app does is also available from the `notocr` CLI.

```
┌ toolbar ───────────────────────────────────────────────────────────────────────┐
│ Open…  Recognise  │  ▶ Play ■  │ Export MusicXML  Export .mscz  Append to…  │ zoom │
├ sidebar ─────┬ original page (zoomable) ───────────────┬ inspector ────────────┤
│ Pages        │  page image                              │ [Note] [Result]       │
│  ▸ page 1    │  + symbol boxes (colour = kind)          │ Note: C#5, 16th       │
│  ▸ page 2    │  + measures that don't add up tinted red │  pitch ↑ ↓  # b ♮     │
│ Measures     │  + selected symbol / playing note glow   │  duration 1 2 4 8 …   │
│  1 ✓         │                                          │  dot, rest, delete    │
│  2 ✓         │  click: select symbol                    │ Symbol: notehead_half │
│  3 ⚠ 5/4     │  tool "Add": click to add a symbol       │  change kind ▾        │
│  …           │                                          │ Result tab: MuseScore │
│              │                                          │  render of the output │
├──────────────┴──────────────────────────────────────────┴───────────────────────┤
│ 25 measures · 3 to check · model v2 · 14 corrections  [Retrain with corrections] │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Two kinds of edits

1. **Symbol edits** (on the image): change a symbol's kind, delete a false one, add a
   missed one by clicking where it is. The music is re-assembled instantly (no detector
   run). These are exact training labels, so the page becomes training data (M6).
2. **Note overrides** (inspector): fix pitch / duration / accidental of one note when the
   symbols are right but the assembly is wrong (e.g. a beam counted wrong). Applied
   after assembly; logged too, but not used as detector labels.

## Keys

| Key | Action |
|---|---|
| Ctrl+O / Ctrl+E / Ctrl+Shift+E | open / export MusicXML / export .mscz |
| Space | play / stop |
| ↑ ↓ | selected note one step up / down |
| # b n | sharp / flat / natural |
| 1–6 | whole, half, quarter, 8th, 16th, 32nd |
| . | toggle dot |
| Del | delete selected symbol |
| A | add-symbol tool (kind from the inspector) |
| N / P | next / previous measure that doesn't add up |
| + − 0 | zoom in / out / fit |

## Playback

Built-in: the result is synthesised to audio (simple additive tones, noise for drums)
and played with QtMultimedia; the note being played is highlighted on the original page.
