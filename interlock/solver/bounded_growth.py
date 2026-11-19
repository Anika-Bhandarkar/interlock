"""Bounded margin growth: grow the assembly inside the board's known size, most clear-cut placement first.

Like strongest_pair, it starts from the best-scoring pair and adds one piece at a time to an open slot next to
the assembly. Two things differ:

1. Bounded: a slot is only open if placing a piece there keeps the assembly's bounding box within rows x cols.
   The assembly can't sprawl, so it ends exactly rows x cols with no frame to shrink afterwards.
2. Margin first (Paikin & Tal, CVPR 2015): each open slot's candidates are scored as in strongest_pair (mean
   score against the slot's placed neighbors), but slots are ranked by
       (best - second best) + SCORE_WEIGHT * best
   rather than by the best score alone. A correct join usually beats its runner-up clearly; a wrong join that
   scores high often has rivals that score about as high. Clear-cut decisions go first, and ambiguous slots wait
   until more neighbors are placed around them.

A swap polish (shrink_frame.swap_polish) finishes. Uses the board size, never the true positions.
"""

import numpy as np

from interlock.data_collection.pairs import OFFSETS
from interlock.solver.shrink_frame import board_shape, prepare_scores, slot_scores, swap_polish
from interlock.solver.strongest_pair import FreeAssembly

SCORE_WEIGHT = 0.5  # 0.25-1.0 all beat anchoring on 40 straight-edge puzzles; 0.5 solved the most (35/40)


def priority(means: np.ndarray, score_weight: float = SCORE_WEIGHT) -> tuple[float, int]:
    """(priority, index of the best candidate) for one slot, from its candidates' mean scores."""
    if len(means) == 1:
        return score_weight * float(means[0]), 0
    second, best = np.partition(means, -2)[-2:]
    return float(best - second) + score_weight * float(best), int(means.argmax())


def bounded_growth(logits: np.ndarray, rows: int | None = None, cols: int | None = None, *,
                   score_weight: float = SCORE_WEIGHT) -> FreeAssembly:
    """Assemble every piece inside a rows x cols box (a square by default), clearest placement first.

    Returns a FreeAssembly whose coordinates span exactly rows x cols (an arbitrary translation, as in
    strongest_pair) and `order` the order pieces were placed in.
    """
    scores = prepare_scores(logits)
    num_pieces = scores.shape[1]
    rows, cols = board_shape(num_pieces, rows, cols)
    relation, first, second = np.unravel_index(scores.argmax(), scores.shape)
    coords = np.zeros((num_pieces, 2), dtype=int)
    coords[second] = OFFSETS[int(relation)]
    occupied = {(0, 0): int(first), tuple(int(v) for v in coords[second]): int(second)}
    order = [int(first), int(second)]
    remaining = np.ones(num_pieces, dtype=bool)
    remaining[order] = False
    x0, y0 = coords[order].min(axis=0)
    x1, y1 = coords[order].max(axis=0)

    while remaining.any():
        candidates = np.flatnonzero(remaining)
        frontier = {(x + dx, y + dy) for x, y in occupied for dx, dy in OFFSETS.values()} - occupied.keys()
        best_priority, best = -np.inf, None
        for x, y in sorted(frontier, key=lambda point: (point[1], point[0])):
            if max(x1, x) - min(x0, x) + 1 > cols or max(y1, y) - min(y0, y) + 1 > rows:
                continue  # would make the assembly wider or taller than the board
            value, j = priority(slot_scores(scores, occupied, x, y, candidates), score_weight)
            if value > best_priority:
                best_priority, best = value, (int(candidates[j]), x, y)
        piece, x, y = best
        coords[piece] = (x, y)
        occupied[(x, y)] = piece
        remaining[piece] = False
        order.append(piece)
        x0, y0, x1, y1 = min(x0, x), min(y0, y), max(x1, x), max(y1, y)

    seed = {"pieces": [int(first), int(second)], "relation": int(relation),
            "score": float(scores[relation, first, second])}
    return FreeAssembly(coords, order, seed)


def solve_bounded(logits: np.ndarray, rows: int | None = None, cols: int | None = None, *, polish: bool = True,
                  score_weight: float = SCORE_WEIGHT) -> tuple[np.ndarray, FreeAssembly]:
    """(rows, cols) grid of piece ids from bounded_growth, swap-polished; and the growth itself."""
    scores = prepare_scores(logits)
    growth = bounded_growth(logits, rows, cols, score_weight=score_weight)
    local = growth.coordinates - growth.coordinates.min(axis=0)
    rows, cols = board_shape(scores.shape[1], rows, cols)
    grid = np.full((rows, cols), -1, dtype=int)
    grid[local[:, 1], local[:, 0]] = np.arange(len(local))
    if polish:
        passes = swap_polish(scores, grid)
        if passes:
            grid = passes[-1][0]
    return grid, growth

