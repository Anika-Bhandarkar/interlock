"""Visual continuity score between two sides (lower = better match).

For side a of piece i we take the two pixel rows closest to that edge
(p0 = boundary row, p1 = one row in) and extrapolate the gradient one step
outward:  predicted_neighbor_row = p0 + (p0 - p1).  The error between that
prediction and piece j's actual boundary row is the score. We do it in both
directions (i -> j and j -> i) and add them, so the score is symmetric.

Tabs / blanks: we only use the square body (the "green" region). Every side
then has the same width, so all strips flatten to vectors of the same length.
The tab bulb is ignored for scoring; it is still used by the geometric rules.
"""

from __future__ import annotations

import numpy as np

from .pieces import Piece


def edge_strips(core: np.ndarray, depth: int = 2) -> np.ndarray:
    """Return (4, depth, L, C): for each side, rows ordered boundary->inward,
    pixels ordered clockwise around the piece."""
    if core.ndim == 2:
        core = core[..., None]
    # np.rot90(core, k=s) (counter-clockwise) puts side s on top, read left->right
    return np.stack([np.rot90(core, k=s)[:depth] for s in range(4)])


def score_tensor(pieces: list[Piece]) -> np.ndarray:
    """Dissimilarity D[i, a, j, b] for every pair of sides (symmetric).

    Two sides that meet run in opposite clockwise directions, so j's strip is
    reversed along its length before comparing.
    """
    S = np.stack([edge_strips(p.core) for p in pieces]).astype(np.float32)  # (N,4,2,L,C)
    N = S.shape[0]
    pred = (2 * S[:, :, 0] - S[:, :, 1]).reshape(N * 4, -1)        # i's prediction of neighbour
    actual = S[:, :, 0, ::-1].reshape(N * 4, -1)                    # boundary row, reversed
    n = pred.shape[1]
    # squared euclidean distance, all pairs, vectorised
    d = (np.sum(pred ** 2, 1)[:, None] + np.sum(actual ** 2, 1)[None, :]
         - 2.0 * pred @ actual.T) / n
    d = np.maximum(d, 0.0)
    return (d + d.T).reshape(N, 4, N, 4)
