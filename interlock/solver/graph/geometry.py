"""Geometric rules (no learning): classify each side as FLAT / TAB / BLANK and
decide which side pairs are even allowed to be edges in the candidate graph."""

from __future__ import annotations

import numpy as np

from .pieces import Piece, SideKind, complementary


# --------------------------------------------------------------------------
# Side classification from a binary piece mask
# --------------------------------------------------------------------------
def find_core_box(mask: np.ndarray, frac: float = 0.5) -> tuple[int, int, int, int]:
    """Estimate the square body of a piece (excluding tabs).

    Rows/cols that pass through a tab are mostly empty (a tab is ~1/3 of the
    side), rows through the body are mostly full, so we threshold the coverage
    profile. Returns (y0, x0, y1, x1), exclusive ends.

    If your cutting code already knows the body box, pass that instead.
    """
    m = mask.astype(bool)
    rows, cols = m.sum(1), m.sum(0)
    ry = np.flatnonzero(rows >= frac * rows.max())
    cx = np.flatnonzero(cols >= frac * cols.max())
    return int(ry[0]), int(cx[0]), int(ry[-1]) + 1, int(cx[-1]) + 1


def classify_sides(
    mask: np.ndarray,
    core_box: tuple[int, int, int, int],
    band: int | None = None,
    thresh: float = 0.25,
) -> tuple[SideKind, SideKind, SideKind, SideKind]:
    """Classify TOP, RIGHT, BOTTOM, LEFT.

    For each side we look at the middle third of that side:
      * mask pixels *outside* the body  -> TAB
      * empty pixels *inside* the body  -> BLANK
      * neither                         -> FLAT
    """
    m = mask.astype(bool)
    y0, x0, y1, x1 = core_box
    d = band or max(2, int(0.12 * min(y1 - y0, x1 - x0)))
    body = np.zeros_like(m)
    body[y0:y1, x0:x1] = True
    out = []
    for s in range(4):
        # rot90 counter-clockwise by s moves side s to the top
        arr = np.rot90(m, k=s)
        rb = np.rot90(body, k=s)
        ys, xs = np.flatnonzero(rb.any(1)), np.flatnonzero(rb.any(0))
        by0, by1, bx0, bx1 = ys[0], ys[-1] + 1, xs[0], xs[-1] + 1
        w = bx1 - bx0
        cx0, cx1 = bx0 + w // 3, bx1 - w // 3
        outside = arr[max(0, by0 - d):by0, cx0:cx1]
        inside = arr[by0:by0 + d, cx0:cx1]
        if outside.size and outside.mean() > thresh:
            out.append(SideKind.TAB)
        elif inside.size and (1.0 - inside.mean()) > thresh:
            out.append(SideKind.BLANK)
        else:
            out.append(SideKind.FLAT)
    return tuple(out)  # type: ignore[return-value]


def _resize_nearest(img: np.ndarray, size: int) -> np.ndarray:
    ys = np.linspace(0, img.shape[0] - 1, size).round().astype(int)
    xs = np.linspace(0, img.shape[1] - 1, size).round().astype(int)
    return img[ys][:, xs]


def piece_from_mask(pid: int, image: np.ndarray, mask: np.ndarray,
                    core_size: int | None = 64, core_box=None) -> Piece:
    """Adapter: turn one cut-out piece (RGB image + binary mask) into a Piece.

    Only the square body ("green region") is kept for visual continuity, so
    every side has the same width regardless of tab/blank. ``core_size``
    resamples all bodies to the same size so strips are comparable.
    """
    box = core_box or find_core_box(mask)
    sides = classify_sides(mask, box)
    y0, x0, y1, x1 = box
    core = image[y0:y1, x0:x1].astype(np.float32)
    if core.max() > 1.5:
        core = core / 255.0
    if core.ndim == 2:
        core = core[..., None]
    if core_size:
        core = _resize_nearest(core, core_size)
    return Piece(pid, core, sides)


# --------------------------------------------------------------------------
# Which side pairs may be edges
# --------------------------------------------------------------------------
def border_consistent(pi: Piece, a: int, pj: Piece, b: int) -> bool:
    """If side a of i meets side b of j, the two sides perpendicular to the
    seam must agree on flatness (two border pieces sit on the same border).

    In i's frame, the perpendicular directions are a+1 and a+3. j's own sides
    occupying those directions are b-1 and b+1 respectively (rotation-free).
    """
    F = SideKind.FLAT
    return ((pi.sides[(a + 1) % 4] == F) == (pj.sides[(b - 1) % 4] == F)
            and (pi.sides[(a + 3) % 4] == F) == (pj.sides[(b + 1) % 4] == F))


def can_connect(pi: Piece, a: int, pj: Piece, b: int) -> bool:
    return (pi.id != pj.id
            and complementary(pi.sides[a], pj.sides[b])
            and border_consistent(pi, a, pj, b))


def validity_tensor(pieces: list[Piece]) -> np.ndarray:
    """Vectorised can_connect: bool array V[i, a, j, b]."""
    K = np.array([[int(s) for s in p.sides] for p in pieces])  # (N, 4)
    N = len(pieces)
    F = K == SideKind.FLAT
    comp = (K[:, :, None, None] + K[None, None, :, :]) == (SideKind.TAB + SideKind.BLANK)
    idx = np.arange(4)
    c1 = F[:, (idx + 1) % 4][:, :, None, None] == F[:, (idx - 1) % 4][None, None, :, :]
    c2 = F[:, (idx + 3) % 4][:, :, None, None] == F[:, (idx + 1) % 4][None, None, :, :]
    same = np.eye(N, dtype=bool)[:, None, :, None]
    return comp & c1 & c2 & ~same
