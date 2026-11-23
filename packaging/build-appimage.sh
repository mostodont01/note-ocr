#!/bin/sh
# PyInstaller output (dist/notocr) -> note-OCR-x86_64.AppImage. Used by the release
# workflow; APPIMAGETOOL must point to an appimagetool binary.
set -e
DIST=${1:-build/dist/notocr}
OUT=${2:-build/note-OCR-x86_64.AppImage}
APPDIR=build/AppDir
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/lib"
cp -r "$DIST" "$APPDIR/usr/lib/notocr"
cp packaging/notocr.png "$APPDIR/notocr.png"
cat > "$APPDIR/notocr.desktop" <<DESK
[Desktop Entry]
Type=Application
Name=note-OCR
Comment=Sheet music to MuseScore
Exec=notocr %f
Icon=notocr
Categories=AudioVideo;Audio;
MimeType=application/pdf;image/png;image/jpeg;
DESK
cat > "$APPDIR/AppRun" <<'RUN'
#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
exec "$HERE/usr/lib/notocr/notocr" "$@"
RUN
chmod +x "$APPDIR/AppRun"
ARCH=x86_64 "${APPIMAGETOOL:?set APPIMAGETOOL to the appimagetool binary}" --appimage-extract-and-run "$APPDIR" "$OUT"
echo "$OUT"
