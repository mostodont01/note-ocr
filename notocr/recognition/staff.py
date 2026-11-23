"""Staff detection: the classic first step of OMR.

1. Run-length statistics give the staff line thickness and the distance between lines.
   Every later step is scaled by these two numbers.
2. Long thin horizontal structures are staff line candidates; five evenly spaced
   candidates make a staff.
3. Each line is refined to a curve y(x) (handles slight skew / curvature).
4. Staves joined by a vertical line at their left edge form one system.
5. Staff lines can be erased where no symbol crosses them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class StaffLine:
    coeffs: np.ndarray  # polynomial y(x), highest power first (np.polyval)
    thickness: float
    x0: int
    x1: int

    def y_at(self, x):
        return np.polyval(self.coeffs, x)


@dataclass
class Staff:
    lines: list[StaffLine]  # top to bottom
    index: int = 0

    @property
    def x0(self) -> int:
        return min(line.x0 for line in self.lines)

    @property
    def x1(self) -> int:
        return max(line.x1 for line in self.lines)

    def spacing(self, x=None) -> float:
        """Distance between neighbouring lines (one 'staff space' incl. line)."""
        if x is None:
            x = (self.x0 + self.x1) / 2
        n = len(self.lines)
        if n < 2:
            return 0.0
        return float((self.lines[-1].y_at(x) - self.lines[0].y_at(x)) / (n - 1))

    def top(self, x=None) -> float:
        x = (self.x0 + self.x1) / 2 if x is None else x
        return float(self.lines[0].y_at(x))

    def bottom(self, x=None) -> float:
        x = (self.x0 + self.x1) / 2 if x is None else x
        return float(self.lines[-1].y_at(x))

    def position(self, x: float, y: float) -> float:
        """Vertical position in half staff spaces above the bottom line.

        0 = bottom line, 1 = first space, 2 = second line, ... 8 = top line.
        Negative / > 8 = ledger-line territory.
        """
        bottom = self.lines[-1].y_at(x)
        return float((bottom - y) / (self.spacing(x) / 2))

    def y_of_position(self, x: float, pos: float) -> float:
        return float(self.lines[-1].y_at(x) - pos * self.spacing(x) / 2)


@dataclass
class System:
    staves: list[Staff]


@dataclass
class PageLayout:
    shape: tuple[int, int]
    line_thickness: float
    line_distance: float  # global estimate of the staff line period
    staves: list[Staff] = field(default_factory=list)
    systems: list[System] = field(default_factory=list)


# --------------------------------------------------------------------------- stats


def _vertical_runs(ink: np.ndarray):
    """Start rows, end rows (exclusive) and column of every vertical ink run."""
    cols = np.pad(ink.T.astype(np.int8), ((0, 0), (1, 1)))
    d = np.diff(cols, axis=1)
    sc, sr = np.nonzero(d == 1)
    ec, er = np.nonzero(d == -1)
    return sc, sr, er


def run_length_stats(ink: np.ndarray) -> tuple[float, float]:
    """(line thickness, line distance) from the most common vertical runs."""
    col, start, end = _vertical_runs(ink)
    black = end - start
    h = ink.shape[0]
    hist = np.bincount(black, minlength=h + 1)
    thickness = int(np.argmax(hist[1 : h // 4] if h > 8 else hist[1:])) + 1

    # period: distance from one run start to the next run start in the same column,
    # counted only between runs that look like staff lines (thin)
    thin = black <= thickness + 1
    col, start = col[thin], start[thin]
    same = col[1:] == col[:-1]
    period = (start[1:] - start[:-1])[same]
    period = period[period > thickness + 1]
    phist = np.bincount(period, minlength=h + 1)[: h // 4]
    distance = int(np.argmax(phist))
    # sub-pixel refinement: weighted mean around the mode
    lo, hi = max(distance - 1, 0), min(distance + 2, len(phist))
    w = phist[lo:hi].astype(float)
    distance_f = float(np.dot(np.arange(lo, hi), w) / w.sum()) if w.sum() else float(distance)
    return float(thickness), distance_f


# ------------------------------------------------------------------------- deskew


def estimate_skew(ink: np.ndarray, max_deg: float = 3.0, step: float = 0.05) -> float:
    """Page rotation in degrees (positive = lines go down to the right).

    Staff lines give the sharpest horizontal projection when the angle is right.
    """
    _, distance = run_length_stats(ink)
    klen = max(int(round(1.5 * distance)), 5)
    horiz = cv2.morphologyEx(
        ink.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1))
    )
    ys, xs = np.nonzero(horiz)
    if len(xs) < 100:
        return 0.0
    if len(xs) > 200_000:
        sel = np.random.default_rng(0).choice(len(xs), 200_000, replace=False)
        ys, xs = ys[sel], xs[sel]
    xs = xs - ink.shape[1] / 2

    def sharpness(deg):
        yy = np.rint(ys - np.tan(np.radians(deg)) * xs).astype(int)
        yy -= yy.min()
        return float((np.bincount(yy).astype(float) ** 2).sum())

    angles = np.arange(-max_deg, max_deg + step / 2, step)
    coarse = angles[:: 4]
    best = coarse[int(np.argmax([sharpness(a) for a in coarse]))]
    fine = angles[np.abs(angles - best) <= 0.2]
    return float(fine[int(np.argmax([sharpness(a) for a in fine]))])


def rotate(gray: np.ndarray, deg: float) -> np.ndarray:
    """Rotate so lines at `deg` become horizontal (white fill)."""
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)


# ------------------------------------------------------------------------ detection


def _line_candidates(ink: np.ndarray, thickness: float, distance: float):
    """Row clusters of long thin horizontal ink: (y_center, thickness)."""
    klen = max(int(round(3 * distance)), 5)
    horiz = cv2.morphologyEx(
        ink.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1))
    )
    profile = horiz.sum(axis=1).astype(float)
    if profile.max() <= 0:
        return [], horiz
    on = profile > 0.3 * np.percentile(profile[profile > 0], 99)
    cands = []
    y = 0
    h = len(on)
    while y < h:
        if not on[y]:
            y += 1
            continue
        y2 = y
        while y2 + 1 < h and on[y2 + 1]:
            y2 += 1
        rows = np.arange(y, y2 + 1)
        w = profile[rows]
        cands.append((float(np.dot(rows, w) / w.sum()), float(y2 - y + 1)))
        y = y2 + 1
    max_thick = max(2 * thickness + 1, 0.35 * distance)
    cands = [c for c in cands if c[1] <= max_thick]
    return cands, horiz


def _group_staves(cands, distance: float, n_lines: int = 5):
    """Find runs of n evenly spaced candidates."""
    ys = [c[0] for c in cands]
    groups = []
    i = 0
    while i + n_lines <= len(ys):
        gaps = np.diff(ys[i : i + n_lines])
        if np.all(np.abs(gaps - distance) < 0.25 * distance):
            groups.append(list(range(i, i + n_lines)))
            i += n_lines
        else:
            i += 1
    return groups


def _fit_line(ink: np.ndarray, horiz: np.ndarray, y_est: float, thickness: float, distance: float):
    """Refine one staff line to a polynomial y(x) and find its horizontal extent."""
    h, w = ink.shape
    band = max(int(round(distance / 3)), 2)
    y0, y1 = max(int(y_est) - band, 0), min(int(y_est) + band + 1, h)
    strip = horiz[y0:y1]
    present = strip.any(axis=0)
    xs = np.nonzero(present)[0]
    if len(xs) == 0:
        return None
    # longest stretch of presence allowing small gaps (= the actual line)
    gap = max(int(distance), 3)
    breaks = np.nonzero(np.diff(xs) > gap)[0]
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [len(xs) - 1]])
    k = int(np.argmax(xs[ends] - xs[starts]))
    x0, x1 = int(xs[starts[k]]), int(xs[ends[k]])

    # sample the centre of the line in chunks
    step = max(int(2 * distance), 4)
    sx, sy = [], []
    rows = np.arange(y0, y1)[:, None]
    for cx in range(x0, x1 + 1, step):
        sub = strip[:, cx : min(cx + step, x1 + 1)].astype(float)
        m = sub.sum()
        if m < 0.5 * sub.shape[1]:
            continue
        sx.append(cx + sub.shape[1] / 2)
        sy.append(float((rows * sub).sum() / m))
    sx, sy = np.array(sx), np.array(sy)
    if len(sx) < 3:
        coeffs = np.array([0.0, 0.0, y_est])
    else:
        deg = 2 if len(sx) >= 8 else 1
        coeffs = np.polyfit(sx, sy, deg)
        resid = np.abs(np.polyval(coeffs, sx) - sy)
        keep = resid < max(0.15 * distance, 1.0)
        if keep.sum() >= deg + 2:
            coeffs = np.polyfit(sx[keep], sy[keep], deg)
        if deg == 1:
            coeffs = np.concatenate([[0.0], coeffs])
    return StaffLine(coeffs=coeffs, thickness=thickness, x0=x0, x1=x1)


def _connected(ink: np.ndarray, a: Staff, b: Staff, distance: float) -> bool:
    """True if a vertical line joins staff a (above) and staff b at the left edge."""
    x0 = max(a.x0, b.x0)
    ya, yb = int(a.bottom(x0)) + 1, int(b.top(x0))
    if yb <= ya:
        return True
    xs = range(max(x0 - int(distance), 0), min(x0 + int(distance) + 1, ink.shape[1]))
    for x in xs:
        if ink[ya:yb, x].mean() > 0.95:
            return True
    return False


def detect_staves(ink: np.ndarray) -> PageLayout:
    thickness, distance = run_length_stats(ink)
    cands, horiz = _line_candidates(ink, thickness, distance)
    groups = _group_staves(cands, distance)
    layout = PageLayout(shape=ink.shape, line_thickness=thickness, line_distance=distance)
    for g in groups:
        lines = []
        for idx in g:
            line = _fit_line(ink, horiz, cands[idx][0], cands[idx][1], distance)
            if line is not None:
                lines.append(line)
        if len(lines) == 5:
            layout.staves.append(Staff(lines=lines, index=len(layout.staves)))

    # systems
    current: list[Staff] = []
    for s in layout.staves:
        if current and _connected(ink, current[-1], s, distance):
            current.append(s)
        else:
            if current:
                layout.systems.append(System(current))
            current = [s]
    if current:
        layout.systems.append(System(current))
    return layout


# ---------------------------------------------------------------------- line removal


def remove_staff_lines(ink: np.ndarray, layout: PageLayout) -> np.ndarray:
    """Erase staff line pixels wherever no symbol crosses the line."""
    out = ink.copy()
    h = ink.shape[0]
    t = layout.line_thickness
    max_run = int(np.ceil(t + max(2, 0.12 * layout.line_distance)))
    for staff in layout.staves:
        for line in staff.lines:
            xs = np.arange(line.x0, line.x1 + 1)
            yc = np.rint(line.y_at(xs)).astype(int)
            for x, y in zip(xs, yc):
                if not (0 <= y < h) or not ink[y, x]:
                    # the line may be a pixel off; look at direct neighbours
                    for dy in (-1, 1):
                        if 0 <= y + dy < h and ink[y + dy, x]:
                            y = y + dy
                            break
                    else:
                        continue
                top = y
                while top > 0 and ink[top - 1, x] and y - top <= max_run:
                    top -= 1
                bot = y
                while bot < h - 1 and ink[bot + 1, x] and bot - y <= max_run:
                    bot += 1
                if bot - top + 1 <= max_run:
                    out[top : bot + 1, x] = False
    return out


# --------------------------------------------------------------------------- debug


def draw_layout(gray: np.ndarray, layout: PageLayout) -> np.ndarray:
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    palette = [(0, 0, 230), (0, 160, 0), (220, 100, 0), (160, 0, 160)]
    for si, system in enumerate(layout.systems):
        color = palette[si % len(palette)]
        for staff in system.staves:
            for line in staff.lines:
                xs = np.arange(line.x0, line.x1 + 1, 2)
                pts = np.stack([xs, line.y_at(xs)], axis=1).round().astype(np.int32)
                cv2.polylines(img, [pts], False, color, 1)
            cv2.putText(
                img,
                f"{staff.index}",
                (max(staff.x0 - 18, 0), int(staff.bottom(staff.x0))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )
    return img
