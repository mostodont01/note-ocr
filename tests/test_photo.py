import cv2
import numpy as np

from notocr.ingest import binarize
from notocr.photo import find_page, prepare
from notocr.recognition.staff import detect_staves


def fake_photo(n_staves=6, spacing=14):
    h, w = 120 + n_staves * 9 * spacing, 1100
    page = np.full((h, w), 245, np.uint8)
    for s in range(n_staves):
        top = 80 + s * 9 * spacing
        for i in range(5):
            cv2.line(page, (60, top + i * spacing), (w - 60, top + i * spacing), 20, 2)
        for x in range(150, w - 100, 90):  # some "notes"
            cv2.ellipse(page, (x, top + (x // 90 % 9) * spacing // 2), (8, 6), -20, 0, 360, 20, -1)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    page = cv2.remap(page, xs, ys + 1.2 * spacing * np.sin(np.pi * xs / w), cv2.INTER_LINEAR, borderValue=245)
    H, W = int(h * 1.4), int(w * 1.4)
    dst = np.float32([[W * 0.15, H * 0.1], [W * 0.88, H * 0.14], [W * 0.9, H * 0.92], [W * 0.1, H * 0.87]])
    m = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), dst)
    canvas = np.full((H, W), 50, np.uint8)
    photo = cv2.warpPerspective(page, m, (W, H), dst=canvas, borderMode=cv2.BORDER_TRANSPARENT)
    shade = np.linspace(1.0, 0.65, W)[None, :]
    return np.clip(photo * shade, 0, 255).astype(np.uint8), n_staves


def test_photo_page_found_and_all_staves_detected():
    photo, n = fake_photo()
    assert find_page(photo) is not None
    flat = prepare(photo)
    assert len(detect_staves(binarize(flat)).staves) == n


def test_clean_page_untouched():
    page = np.full((400, 600), 255, np.uint8)
    for i in range(5):
        cv2.line(page, (40, 100 + 12 * i), (560, 100 + 12 * i), 0, 1)
    assert find_page(page) is None
    assert np.array_equal(prepare(page), page)
