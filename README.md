# note-OCR

Open-source optical music recognition: turn sheet music (PDFs, scans, phone photos,
web pages) into MuseScore files, check and fix the result, and let every correction
improve the recogniser.

**Status:** planning. See [PLAN.md](PLAN.md).

## Planned features

- Desktop app (Python + PySide6/QML): import PDF/PNG/JPEG, see the original next to
  the recognised music, fix mistakes, play it back.
- Export to MusicXML and MuseScore (`.mscz`); start a new file or append to an existing one.
- Single-staff parts, piano (grand staff) and drums.
- Learns from your corrections: every fix becomes training data for retraining.
- Later: a Firefox extension to grab music straight from a web page.

## Licence

MIT, see [LICENSE](LICENSE).
