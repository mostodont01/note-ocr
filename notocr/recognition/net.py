"""Symbol detector: a small CenterNet-style fully convolutional network.

Input: grayscale page normalised so the staff line distance is NORM_SPACING px
(white = 0, ink = 1). Output at stride 4: one centre heatmap per class, plus box
size (w, h) and sub-pixel centre offset. Small enough to train on a laptop CPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .classes import CLASS_INDEX, CLASSES, NORM_SPACING, STRIDE  # noqa: F401  (re-exported)


class Block(nn.Module):
    def __init__(self, cin, cout, stride=1, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.skip = (
            nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
            if stride != 1 or cin != cout
            else nn.Identity()
        )

    def forward(self, x):
        y = Fn.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return Fn.relu(y + self.skip(x))


class Detector(nn.Module):
    def __init__(self, n_classes: int = len(CLASSES), width: int = 16):
        super().__init__()
        w = width
        self.stem = nn.Sequential(
            nn.Conv2d(1, w, 3, 2, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU(), Block(w, w)
        )  # /2
        self.d2 = nn.Sequential(Block(w, 2 * w, stride=2), Block(2 * w, 2 * w))  # /4
        self.d3 = nn.Sequential(Block(2 * w, 4 * w, stride=2), Block(4 * w, 4 * w))  # /8
        self.d4 = nn.Sequential(
            Block(4 * w, 6 * w, stride=2), Block(6 * w, 6 * w, dilation=2), Block(6 * w, 6 * w, dilation=4)
        )  # /16
        self.u3 = Block(10 * w, 4 * w)
        self.u2 = Block(6 * w, 3 * w)
        h = 3 * w
        self.neck = nn.Sequential(nn.Conv2d(h, h, 3, 1, 1), nn.ReLU())
        self.heat = nn.Conv2d(h, n_classes, 1)
        self.size = nn.Conv2d(h, 2, 1)
        self.offset = nn.Conv2d(h, 2, 1)
        self.heat.bias.data.fill_(-4.0)

    def forward(self, x):
        x1 = self.stem(x)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        x4 = self.d4(x3)
        y = self.u3(torch.cat([Fn.interpolate(x4, size=x3.shape[-2:], mode="nearest"), x3], 1))
        y = self.u2(torch.cat([Fn.interpolate(y, size=x2.shape[-2:], mode="nearest"), x2], 1))
        y = self.neck(y)
        return self.heat(y), self.size(y), self.offset(y)


def focal_loss(logits, target, alpha=2, beta=4):
    """CenterNet penalty-reduced focal loss; target is a gaussian heatmap in [0, 1]."""
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    neg = 1 - pos
    pos_loss = torch.log(p) * (1 - p) ** alpha * pos
    neg_loss = torch.log(1 - p) * p**alpha * (1 - target) ** beta * neg
    n = pos.sum().clamp(min=1)
    return -(pos_loss.sum() + neg_loss.sum()) / n


@torch.no_grad()
def decode(heat_logits, size, offset, threshold=0.3, max_det=3000):
    """Peaks of the heatmap -> list of (class, score, cx, cy, w, h) in input pixels."""
    heat = torch.sigmoid(heat_logits)
    peaks = heat == Fn.max_pool2d(heat, 3, 1, 1)
    heat = heat * peaks
    b, c, h, w = heat.shape
    scores, idx = heat.view(b, -1).topk(min(max_det, heat.numel() // b))
    out = []
    for bi in range(b):
        dets = []
        for s, i in zip(scores[bi].tolist(), idx[bi].tolist()):
            if s < threshold:
                break
            cls, rem = divmod(i, h * w)
            y, x = divmod(rem, w)
            ox, oy = offset[bi, :, y, x].tolist()
            sw, sh = torch.exp(size[bi, :, y, x]).tolist()
            dets.append((cls, s, (x + ox) * STRIDE, (y + oy) * STRIDE, sw * NORM_SPACING, sh * NORM_SPACING))
        out.append(dets)
    return out
