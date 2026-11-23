"""Phone photos: find the page, undo perspective, flatten a curved page.

1. Page: the largest bright four-sided region (paper on a darker background) is warped
   to a rectangle.
2. Curvature: the page is cut into vertical strips; each strip's row profile (staff
   lines are dark rows) is aligned to its neighbour's, which gives a vertical shift per
   strip; a smooth shift field then straightens the lines.
"""

from __future__ import annotations

import cv2
import numpy as np


def _order(pts: np.ndarray) -> np.ndarray:
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def find_page(gray: np.ndarray) -> np.ndarray | None:
    """Corners (tl, tr, br, bl) of the paper, or None if the image is already just the page."""
    h, w = gray.shape
    k = 800 / max(h, w)
    small = cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else gray.copy()
    k = min(k, 1.0)
    blur = cv2.GaussianBlur(small, (0, 0), 3)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c) / (small.shape[0] * small.shape[1])
    if not 0.2 < area < 0.97:
        return None
    hull = cv2.convexHull(c)
    for eps in (0.02, 0.03, 0.05, 0.08):
        quad = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
        if len(quad) == 4:
            pts = quad.reshape(4, 2).astype(np.float32) / k
            # touching the image border on all sides = it's the page itself
            m = 0.01 * max(h, w)
            if (pts[:, 0].min() < m and pts[:, 1].min() < m and pts[:, 0].max() > w - m and pts[:, 1].max() > h - m):
                return None
            return _order(pts)
    return None


def unwarp_page(gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = corners
    width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    m = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(gray, m, (width, height), flags=cv2.INTER_LINEAR, borderValue=255)


def dewarp(gray: np.ndarray, ink: np.ndarray, spacing: float, strips: int = 12) -> tuple[np.ndarray, float]:
    """Straighten bent staff lines. Returns (image, max shift in px)."""
    h, w = gray.shape
    klen = max(int(2 * spacing), 5)
    horiz = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1)))
    edges = np.linspace(0, w, strips + 1).astype(int)
    profiles = [horiz[:, a:b].sum(1).astype(np.float32) for a, b in zip(edges[:-1], edges[1:])]
    max_shift = int(2.5 * spacing)
    shifts = [0.0]
    for p0, p1 in zip(profiles, profiles[1:]):
        if p0.sum() < 10 or p1.sum() < 10:
            shifts.append(shifts[-1])
            continue
        best, best_s = -1.0, 0
        for s in range(-max_shift, max_shift + 1):
            a = p0[max(0, s) : h + min(0, s)]
            b = p1[max(0, -s) : h + min(0, -s)]
            score = float(np.dot(a, b))
            if score > best:
                best, best_s = score, s
        shifts.append(shifts[-1] - best_s)
    centers = (edges[:-1] + edges[1:]) / 2
    shifts = np.array(shifts) - np.median(shifts)
    if np.abs(shifts).max() < 0.25 * spacing:
        return gray, float(np.abs(shifts).max())
    # smooth curve through the strip shifts, one value per column
    coeffs = np.polyfit(centers, shifts, min(3, strips - 1))
    field = np.polyval(coeffs, np.arange(w)).astype(np.float32)
    map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    map_y = np.arange(h, dtype=np.float32)[:, None] + field[None, :]
    out = cv2.remap(gray, map_x, map_y, cv2.INTER_LINEAR, borderValue=255)
    return out, float(np.abs(field).max())


def prepare(gray: np.ndarray, log=None) -> np.ndarray:
    """Photo -> flat, straight page: page detection, deskew, dewarp (each only if needed)."""
    from .ingest import binarize
    from .recognition.staff import estimate_skew, rotate, run_length_stats

    corners = find_page(gray)
    if corners is not None:
        gray = unwarp_page(gray, corners)
        if log:
            log("page found: perspective corrected")
    ink = binarize(gray)
    deg = estimate_skew(ink, max_deg=8.0)
    if abs(deg) >= 0.2:
        gray = rotate(gray, deg)
        ink = binarize(gray)
        if log:
            log(f"rotated {deg:.2f} degrees")
    _, spacing = run_length_stats(ink)
    if spacing >= 4:
        out, shift = dewarp(gray, ink, spacing)
        if out is not gray:
            gray = out
            if log:
                log(f"page bend flattened (max {shift:.1f} px)")
    return gray
