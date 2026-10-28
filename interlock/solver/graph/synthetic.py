"""Synthetic puzzles for testing the graph solver without real data."""

from __future__ import annotations

import numpy as np

from .pieces import Piece, SideKind


def smooth_image(h: int, w: int, rng: np.random.Generator, n_waves: int = 14) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w, 3), np.float32)
    for ch in range(3):
        for _ in range(n_waves):
            fy, fx = rng.uniform(-1, 1, 2) * rng.uniform(0.005, 0.06)
            ph = rng.uniform(0, 2 * np.pi)
            img[..., ch] += rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * (fy * yy + fx * xx) + ph)
    img -= img.min((0, 1))
    img /= img.max((0, 1))
    return img


def make_puzzle(rows: int, cols: int, tile: int = 32, seed: int = 0,
                rotate: bool = True, image: np.ndarray | None = None):
    """Returns (pieces, truth, image).

    truth[pid] = (row, col, rot) such that rotating the scanned piece ``rot``
    quarter turns clockwise and putting it at (row, col) rebuilds the image.
    """
    rng = np.random.default_rng(seed)
    img = image if image is not None else smooth_image(rows * tile, cols * tile, rng)

    # true side kinds: TOP, RIGHT, BOTTOM, LEFT
    kinds = np.zeros((rows, cols, 4), int)
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                k = rng.integers(1, 3)  # TAB or BLANK on my right
                kinds[r, c, 1], kinds[r, c + 1, 3] = k, 3 - k
            if r + 1 < rows:
                k = rng.integers(1, 3)
                kinds[r, c, 2], kinds[r + 1, c, 0] = k, 3 - k

    order = rng.permutation(rows * cols)
    pieces, truth = [], {}
    for pid, cell in enumerate(order):
        r, c = divmod(int(cell), cols)
        q = int(rng.integers(0, 4)) if rotate else 0   # scan = truth rotated q cw
        core = np.rot90(img[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile], k=-q).copy()
        sides = tuple(SideKind(int(kinds[r, c, (s - q) % 4])) for s in range(4))
        pieces.append(Piece(pid, core, sides))
        truth[pid] = (r, c, (-q) % 4)
    return pieces, truth, img


def piece_mask(sides, tile: int = 32, pad: int | None = None) -> np.ndarray:
    """Draw a binary mask with circular tabs/blanks, for testing classify_sides."""
    pad = pad or tile // 3
    S = tile + 2 * pad
    m = np.zeros((S, S), bool)
    m[pad:pad + tile, pad:pad + tile] = True
    yy, xx = np.mgrid[0:S, 0:S]
    rad = tile / 6
    centers = {0: (pad, pad + tile / 2), 1: (pad + tile / 2, pad + tile),
               2: (pad + tile, pad + tile / 2), 3: (pad + tile / 2, pad)}
    for s, kind in enumerate(sides):
        cy, cx = centers[s]
        off = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}[s]
        if kind == SideKind.TAB:
            cy2, cx2 = cy + off[0] * rad, cx + off[1] * rad
            m |= (yy - cy2) ** 2 + (xx - cx2) ** 2 <= rad ** 2
        elif kind == SideKind.BLANK:
            cy2, cx2 = cy - off[0] * rad, cx - off[1] * rad
            m &= ~((yy - cy2) ** 2 + (xx - cx2) ** 2 <= rad ** 2)
    return m
