# note-OCR for Firefox

Select sheet music on any web page and turn it into MuseScore / MusicXML.

1. Register the helper once: `notocr install-firefox` (works for normal and snap Firefox).
2. Load the extension:
   - quick test: `about:debugging#/runtime/this-firefox` → *Load Temporary Add-on…* →
     pick `firefox/manifest.json` (stays until Firefox restarts);
   - permanent: sign it as an unlisted add-on on addons.mozilla.org
     (`npx web-ext sign --channel unlisted` with your AMO API keys), then install the `.xpi`.
3. Click the toolbar button (or Alt+Shift+M): a box appears. Drag it over the music,
   resize it with the corner handle, then:
   - **Open in note-OCR**: the app opens with the snippet, to check / fix it and export
     as a new file or append it to an existing one;
   - **Quick save**: recognised right away, saved to `~/Music/note-OCR/` as `.musicxml`
     and `.mscz`.

On pages where extensions can't run scripts (Firefox's own PDF viewer), the button
sends the whole visible tab to the app instead.
