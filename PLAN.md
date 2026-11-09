# note-OCR — plan

Optical music recognition (OMR) for my own use: turn sheet music (PDF, scans, phone
photos, web pages) into MuseScore files, check and fix the result, and let every
correction improve the recogniser.

## Decisions (2026-11-09)

| Topic | Decision |
|---|---|
| First front end | Desktop app. Firefox extension later, as a thin front end on the same engine. |
| Stack | Python + PySide6/QML (same as VL). OpenCV, PyTorch, ONNX Runtime for inference. |
| Output | MusicXML; `.mscz` produced by MuseScore's CLI (`mscore -o out.mscz in.musicxml`). No hand-written `.mscz` writer (format is version-specific and undocumented). |
| Sources | All: digital PDFs, scans, phone photos, web pages/viewers. |
| Notation v1 | Single-staff parts, piano grand staff, drums/percussion. (Full scores later.) |
| Symbols v1 | Notes, rests, clefs, key + time signatures, accidentals, dots, ties, tuplets. Not v1: repeats/voltas, dynamics, articulations, text, lyrics. |
| Recognition | Hybrid: classic image processing for staff lines / geometry, trainable ML detector for symbols. Pitch from geometry. |
| Editing | Full editor in the app. Every fix is logged as training data. |
| Training data | Synthetic (rendered with MuseScore) + my own corrections. |
| Training hardware | Laptop CPU (small models, background/overnight). Fallback: Colab. Desktop Quadro K4000 (if that's it) is too old for current PyTorch. |
| Add to existing file | Append measures to the end of the same part. |
| Playback | Built-in (FluidSynth + SoundFont), highlighting the current note on the image. |

## Core principle: everything keeps its source region

Every recognised symbol stores *where it came from*: page/snippet id + bounding box +
model confidence. That makes three things possible:
- overlay boxes on the original image and highlight low-confidence symbols,
- click a note in the result → jump to it in the image (and back),
- a correction becomes a labelled training example (crop + right answer).

## Architecture

```
notocr/
  model/        score model: Score > Part > Staff > Measure > Voice > Event (Note/Rest/Chord)
                + source region on every element; percussion instrument map
  ingest/       PDF (PyMuPDF), images; preprocessing: deskew, binarise,
                phone photos: perspective correction, dewarp, lighting
  recognition/
    staff.py    run-length stats (line thickness, staff space), line detection + tracking
                (curved/skewed lines), group 5 lines → staff, staves → systems
                (brace → grand staff, single-line percussion staff)
    detect.py   ML symbol detector (noteheads, stems, beams, flags, rests, clefs,
                accidentals, dots, ties, barlines, time-sig digits, tuplet numbers)
    assemble.py symbols → music: notehead+stem+beams/flags → duration; position +
                clef + key + accidentals → pitch; barlines → measures; voices;
                sanity check (measure durations vs time signature)
    percussion.py staff position + notehead shape → drum instrument
  export/       MusicXML writer; .mscz via MuseScore CLI; append to existing file
  app/          PySide6/QML: import, side-by-side view, overlay, editor, playback
  training/
    synth.py    MusicXML corpus → MuseScore render (SVG/PNG) → images + boxes,
                augmentation (blur, skew, curvature, shadows, JPEG noise)
    corrections store (SQLite + crops), retrain / fine-tune, export to ONNX
```

Detector candidates: small YOLO-style detector (watch licences: Ultralytics is AGPL,
fine for personal use) or torchvision detection models. Run inference via ONNX Runtime.

## Milestones

1. **Staff detection prototype** — clean scanned page → detected staves, debug images.
2. **Score model + export** — hand-built score → MusicXML → `.mscz`, opens in MuseScore.
3. **Synthetic data generator** — render MusicXML with MuseScore, extract symbol boxes.
4. **Detector v1 + assembly** — first end-to-end result on printed single-staff parts.
5. **Desktop app** — import, overlay, editor, correction logging, playback.
6. **Retraining loop** — corrections + synthetic → fine-tune → new model version, compare.
7. **Piano grand staff** — braces, two staves, multiple voices, cross-staff notes.
8. **Drums** — percussion staff, notehead shapes, instrument mapping.
9. **Phone photos** — dewarping + heavier augmentation.
10. **Digital PDF fast path** — read music-font glyphs straight from vector PDFs.
11. **Firefox extension** — box overlay → capture → send to the app (native messaging).
12. **Packaging** — Linux AppImage, Windows `.exe`.

## Later (after v1)

Repeats, voltas, D.C./D.S./coda, dynamics (p/f/cresc.), articulations (staccato,
accents), tempo text, lyrics, chord symbols, full multi-instrument scores,
handwritten music.

## Open questions / to verify

- MuseScore SVG export: do elements carry class names (Note, Rest, Clef, …) usable as
  bounding-box labels? If not, get boxes another way (layout data / per-element render).
- Firefox: `tabs.captureVisibleTab` only captures the visible tab; check whether it works
  on Firefox's built-in PDF viewer.
- Which music corpus to render for synthetic data (OpenScore, own MuseScore files, …).
- Drum notation conventions to support (which instrument on which line/space).
- Exact desktop GPU model (`nvidia-smi`).
