# note-OCR

Open-source optical music recognition: turn sheet music (PDFs, scans, phone photos,
web pages) into MuseScore files, check and fix the result, and let every correction
improve the recogniser.

**Status:** working prototype (2026-11-23). All planned milestones have a first
version; see [PLAN.md](PLAN.md) for what's next.

## Features

- **Desktop app** (PySide6/QML): open a PNG/JPEG/PDF, see what was recognised on top of
  the original, measures that don't add up highlighted, fix symbols or notes, play it
  back (with the playing note highlighted), compare with MuseScore's rendering.
- **Input:** screenshots, scans, phone photos (page detection, perspective, bent
  pages), PDFs (rendered at the right resolution; notation-software PDFs read their
  music-font glyphs directly), web pages via the **Firefox extension**.
- **Notation:** single-staff parts, piano grand staff, drum kit; clefs, keys, time
  signatures, notes/rests down to 32nds, dots, accidentals, ties, triplets, chords,
  repeats, dynamics, articulations, fermatas, slurs.
- **Output:** MusicXML and MuseScore `.mscz` (via MuseScore's CLI); new file, or append
  the measures to an existing score.
- **Learns from you:** every symbol fix is a training label; *Retrain* fine-tunes the
  detector on your pages and only switches to the new model if it's better.
- **Packaged** as a Linux AppImage and a Windows app (GitHub releases), running the
  detector with ONNX Runtime (no PyTorch needed).

Not yet: text (lyrics, chord symbols, tempo text, D.C./D.S.), several voices on one
staff, full multi-instrument scores, handwritten music.

## Usage

Needs Python ≥ 3.12, [uv](https://docs.astral.sh/uv/) and MuseScore 4
(`flatpak install flathub org.musescore.MuseScore`, or set `NOTOCR_MSCORE`).

```bash
uv sync

# recognise a page -> MusicXML (+ .mscz), with debug images
uv run notocr recognize page.png -o out/page.musicxml --mscz --debug out/

# next page of the same part: append its measures
uv run notocr recognize page2.png --append out/page.musicxml -o out/both.musicxml

# the desktop app (and an app-menu entry)
uv run notocr app
uv run notocr install-desktop

# Firefox extension helper (then load firefox/manifest.json, see firefox/README.md)
uv run notocr install-firefox

# fine-tune the detector with the pages you corrected in the app
uv run notocr retrain
```

### Training the detector

```bash
# random scores rendered by MuseScore in 8 music fonts -> labelled images
uv run python -m notocr.training.synth data/synth --scores 40 --variants 2 --seed 1
uv run python -m notocr.training.synth data/synth_val --scores 4 --variants 1 --seed 2

uv run python -m notocr.training.train data/synth --steps 2500 --out models/detector.pt
uv run python -m notocr.training.evaluate data/synth_val --weights models/detector.pt
```

## How it works

1. **Photos**: the page is found and flattened (perspective, bend), then deskewed.
2. **Staves** (classic image processing): run-length statistics give line thickness and
   spacing, long thin horizontal runs give staff lines, five evenly spaced lines a staff.
   The page is deskewed first.
3. **Symbols** (small CNN, CenterNet-style, ~0.9 M parameters): the page is scaled so
   staff lines are 10 px apart, then the network finds 57 kinds of symbols (note heads,
   rests, clefs, accidentals, flags, dots, time digits, ties, slurs, triplet numbers,
   barlines, stems, dynamics, articulations, fermatas, repeat dots).
4. **Assembly** (rules): barlines → measures, stems → chords, beams counted next to the
   stem tips, staff position + clef + key signature + accidentals → pitch.
5. **Export**: MusicXML; MuseScore's command line turns it into `.mscz`.

Training data is synthetic: random music is rendered by MuseScore to SVG, and every
SVG element tells what it is and where, so labels come for free.

## Licence

MIT, see [LICENSE](LICENSE).
