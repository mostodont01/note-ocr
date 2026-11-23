# note-OCR

Open-source optical music recognition: turn sheet music (PDFs, scans, phone photos,
web pages) into MuseScore files, check and fix the result, and let every correction
improve the recogniser.

**Status:** early prototype (milestones 1–4: staff detection, score model + export,
synthetic training data, symbol detector + assembly). See [PLAN.md](PLAN.md).

## Planned features

- Desktop app (Python + PySide6/QML): import PDF/PNG/JPEG, see the original next to
  the recognised music, fix mistakes, play it back.
- Export to MusicXML and MuseScore (`.mscz`); start a new file or append to an existing one.
- Single-staff parts, piano (grand staff) and drums.
- Learns from your corrections: every fix becomes training data for retraining.
- Later: a Firefox extension to grab music straight from a web page.

## Usage

Needs Python ≥ 3.12, [uv](https://docs.astral.sh/uv/) and MuseScore 4
(`flatpak install flathub org.musescore.MuseScore`, or set `NOTOCR_MSCORE`).

```bash
uv sync

# recognise a page -> MusicXML (+ .mscz), with debug images
uv run notocr recognize page.png -o out/page.musicxml --mscz --debug out/

# next page of the same part: append its measures
uv run notocr recognize page2.png --append out/page.musicxml -o out/both.musicxml
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

1. **Staves** (classic image processing): run-length statistics give line thickness and
   spacing, long thin horizontal runs give staff lines, five evenly spaced lines a staff.
   The page is deskewed first.
2. **Symbols** (small CNN, CenterNet-style): the page is scaled so staff lines are 10 px
   apart, then the network finds note heads, rests, clefs, accidentals, flags, dots,
   time-signature digits, ties, triplet numbers, barlines and stems.
3. **Assembly** (rules): barlines → measures, stems → chords, beams counted next to the
   stem tips, staff position + clef + key signature + accidentals → pitch.
4. **Export**: MusicXML; MuseScore's command line turns it into `.mscz`.

Training data is synthetic: random music is rendered by MuseScore to SVG, and every
SVG element tells what it is and where, so labels come for free.

## Licence

MIT, see [LICENSE](LICENSE).
