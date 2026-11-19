"""Shrinking-frame assembly: free growth first, then a frame that closes in to the board's known size.

1. Grow: strongest_pair() assembles every piece on an open lattice, with no boundary.
2. Frame: put a rectangle around that assembly. If one side is shorter than the board, widen it.
3. Shrink: move one side of the frame in by one line at a time until it is rows x cols. Each step moves the
   side whose outermost line holds the least evidence of being right: the lowest total match probability
   (sigmoid of each piece's mean score against its neighbors). Pieces on that line are evicted and re-placed,
   most confident first, into the empty cell inside the frame where they score best against the occupied
   neighbors. Until the frame reaches its final size it has more cells than pieces, so there is always room.
4. Polish (optional): swap any two pieces whenever that raises the total score of all joins, until no swap
   helps. Without it a wrong piece already inside the frame is never challenged.

Unlike strongest_pair, this uses the board size (rows x cols, by default the square root of the piece count).
It never sees the true positions.
"""

from dataclasses import dataclass, field
import math

import numpy as np

from interlock.data_collection.pairs import OFFSETS
from interlock.solver.strongest_pair import FreeAssembly, strongest_pair

SCORE_FLOOR = -30.0  # stands in for -inf (shapes can't fit), as in updaters.py
SIDES = ("left", "right", "top", "bottom")


@dataclass
class FrameStep:
    label: str  # "Frame placed", "Shrink left", ..., "Swap polish"
    coordinates: np.ndarray  # (N, 2) piece positions after this step, on the growth lattice
    frame: tuple[int, int, int, int]  # (x0, y0, x1, y1), inclusive, on the growth lattice
    moved: list[int] = field(default_factory=list)  # pieces evicted and re-placed (or swapped) in this step


@dataclass
class ShrinkResult:
    growth: FreeAssembly  # the strongest-pair assembly the frame started from
    coordinates: np.ndarray  # (N, 2) final (x, y) of each piece, relative to the final frame's top-left
    grid: np.ndarray  # (rows, cols) piece id in each slot
    before_polish: np.ndarray  # (N, 2) like `coordinates`, before the swap polish
    steps: list[FrameStep]
    num_evicted: int  # piece moves made while shrinking
    num_swaps: int  # swaps made by the polish
    total_score: float  # sum of the scores of every join on the final board; higher = better fit


def board_shape(num_pieces: int, rows: int | None = None, cols: int | None = None) -> tuple[int, int]:
    """(rows, cols); a square board when neither is given."""
    if rows is None and cols is None:
        side = math.isqrt(num_pieces)
        if side * side != num_pieces:
            raise ValueError(f"{num_pieces} pieces is not a square board; pass rows and cols")
        return side, side
    if rows is None or cols is None or rows < 1 or cols < 1 or rows * cols != num_pieces:
        raise ValueError(f"rows x cols must equal the {num_pieces} pieces, got {rows} x {cols}")
    return rows, cols


def prepare_scores(logits: np.ndarray) -> np.ndarray:
    """Float copy with self-pairs and impossible pairs at SCORE_FLOOR, so every mean stays finite."""
    scores = np.asarray(logits, dtype=float).copy()
    for relation in range(4):
        np.fill_diagonal(scores[relation], -np.inf)
    return np.maximum(scores, SCORE_FLOOR)


def slot_scores(scores: np.ndarray, occupied: dict, x: int, y: int, candidates: np.ndarray) -> np.ndarray | None:
    """Mean score of each candidate in slot (x, y) against the slot's occupied neighbors; None if it has none."""
    evidence = [scores[relation, occupied[(x - dx, y - dy)], candidates]
                for relation, (dx, dy) in OFFSETS.items() if (x - dx, y - dy) in occupied]
    return np.mean(evidence, axis=0) if evidence else None


def keep_probability(scores: np.ndarray, occupied: dict, piece: int, x: int, y: int) -> float:
    """How likely a placed piece is right where it is: sigmoid of its mean score; 0 with no neighbors."""
    mean = slot_scores(scores, occupied, x, y, np.array([piece]))
    return 0.0 if mean is None else float(1 / (1 + np.exp(-mean[0])))


def line_cells(frame: tuple[int, int, int, int], side: str) -> list[tuple[int, int]]:
    x0, y0, x1, y1 = frame
    if side in ("left", "right"):
        x = x0 if side == "left" else x1
        return [(x, y) for y in range(y0, y1 + 1)]
    y = y0 if side == "top" else y1
    return [(x, y) for x in range(x0, x1 + 1)]


def move_in(frame: tuple[int, int, int, int], side: str) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = frame
    return {"left": (x0 + 1, y0, x1, y1), "right": (x0, y0, x1 - 1, y1),
            "top": (x0, y0 + 1, x1, y1), "bottom": (x0, y0, x1, y1 - 1)}[side]


def inside(frame: tuple[int, int, int, int], x: int, y: int) -> bool:
    x0, y0, x1, y1 = frame
    return x0 <= x <= x1 and y0 <= y <= y1


def replace_pieces(scores, occupied, coords, frame, pending: list[int]) -> None:
    """Put each pending piece back inside the frame, best (piece, empty cell) pair first."""
    pending = list(pending)
    while pending:
        candidates = np.array(pending)
        holes = {(x + dx, y + dy) for (x, y) in occupied for dx, dy in OFFSETS.values()}
        holes = sorted((h for h in holes if h not in occupied and inside(frame, *h)), key=lambda p: (p[1], p[0]))
        best_score, best = -np.inf, None
        for x, y in holes:
            means = slot_scores(scores, occupied, x, y, candidates)
            j = int(means.argmax())
            if means[j] > best_score:
                best_score, best = float(means[j]), (int(candidates[j]), x, y)
        if best is None:  # nothing placed touches an empty cell: take the first empty cell
            x0, y0, x1, y1 = frame
            x, y = next((x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1) if (x, y) not in occupied)
            best = (pending[0], x, y)
        piece, x, y = best
        occupied[(x, y)] = piece
        coords[piece] = (x, y)
        pending.remove(piece)


def join_score(scores: np.ndarray, grid: np.ndarray, cells) -> float:
    """Total score of every join (right and below) touching any of `cells` in a full grid, each counted once."""
    rows, cols = grid.shape
    edges = set()
    for x, y in cells:
        for (ax, ay), (bx, by) in (((x, y), (x + 1, y)), ((x, y), (x, y + 1)), ((x - 1, y), (x, y)), ((x, y - 1), (x, y))):
            if 0 <= ax < cols and 0 <= ay < rows and 0 <= bx < cols and 0 <= by < rows:
                edges.add((ax, ay, bx, by))
    total = 0.0
    for ax, ay, bx, by in edges:
        relation = 0 if bx > ax else 1  # RIGHT or BELOW, the first two RELATIONS
        total += scores[relation, grid[ay, ax], grid[by, bx]]
    return total


def swap_polish(scores: np.ndarray, grid: np.ndarray, max_passes: int = 10) -> list[tuple[np.ndarray, list[int]]]:
    """Swap pairs of pieces while any swap raises the total join score.

    Returns (grid after the pass, pieces swapped in it) for each pass that changed something.
    """
    grid = grid.copy()
    rows, cols = grid.shape
    cells = [(x, y) for y in range(rows) for x in range(cols)]
    passes = []
    for _ in range(max_passes):
        swapped = []
        for i, a in enumerate(cells):
            for b in cells[i + 1:]:
                before = join_score(scores, grid, (a, b))
                grid[a[1], a[0]], grid[b[1], b[0]] = grid[b[1], b[0]], grid[a[1], a[0]]
                if join_score(scores, grid, (a, b)) > before + 1e-9:
                    swapped += [int(grid[a[1], a[0]]), int(grid[b[1], b[0]])]
                else:
                    grid[a[1], a[0]], grid[b[1], b[0]] = grid[b[1], b[0]], grid[a[1], a[0]]
        if not swapped:
            break
        passes.append((grid.copy(), swapped))
    return passes


def shrink_frame(
    logits: np.ndarray,
    rows: int | None = None,
    cols: int | None = None,
    *,
    growth: FreeAssembly | None = None,
    polish: bool = True,
) -> ShrinkResult:
    """Strongest-pair growth, then a frame shrinking to rows x cols (a square by default), then swap polish.

    Args:
        logits: (4, N, N) from score_matrix or score_all_pairs.
        rows, cols: board size; both None means a square board of the piece count's square root.
        growth: a strongest_pair() result for these logits, to skip recomputing it.
        polish: run swap_polish on the final frame.
    """
    scores = prepare_scores(logits)
    num_pieces = scores.shape[1]
    rows, cols = board_shape(num_pieces, rows, cols)
    growth = growth or strongest_pair(logits)
    coords = growth.coordinates.copy()
    occupied = {tuple(int(v) for v in point): piece for piece, point in enumerate(coords)}

    # A frame around the whole assembly; widen it where it's narrower than the board, so it can only shrink.
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0)
    frame = (int(x0), int(y0), int(max(x1, x0 + cols - 1)), int(max(y1, y0 + rows - 1)))
    steps = [FrameStep("Frame placed", coords.copy(), frame)]

    num_evicted = 0
    while frame[2] - frame[0] + 1 > cols or frame[3] - frame[1] + 1 > rows:
        too_wide, too_tall = frame[2] - frame[0] + 1 > cols, frame[3] - frame[1] + 1 > rows
        options = []
        for order, side in enumerate(SIDES):
            if not (too_wide if side in ("left", "right") else too_tall):
                continue  # that dimension is already the board's size
            pieces = [occupied[cell] for cell in line_cells(frame, side) if cell in occupied]
            cost = sum(keep_probability(scores, occupied, p, *coords[p]) for p in pieces)
            options.append((cost, len(pieces), order, side, pieces))
        _, _, _, side, evicted = min(options, key=lambda option: option[:3])
        for piece in evicted:
            del occupied[tuple(int(v) for v in coords[piece])]
        frame = move_in(frame, side)
        replace_pieces(scores, occupied, coords, frame, evicted)
        num_evicted += len(evicted)
        steps.append(FrameStep(f"Shrink {side}", coords.copy(), frame, [int(p) for p in evicted]))

    origin = np.array(frame[:2])
    before_polish = coords - origin
    grid = np.full((rows, cols), -1, dtype=int)
    for piece, (x, y) in enumerate(before_polish):
        grid[y, x] = piece

    passes = swap_polish(scores, grid) if polish else []
    for polished, swapped in passes:
        steps.append(FrameStep("Swap polish", grid_coordinates(polished) + origin, frame, swapped))
    if passes:
        grid = passes[-1][0]
    return ShrinkResult(growth, grid_coordinates(grid), grid, before_polish, steps, num_evicted,
                        sum(len(swapped) // 2 for _, swapped in passes), board_score(scores, grid))


def board_score(scores: np.ndarray, grid: np.ndarray) -> float:
    """Total score of every join on a full board."""
    return float(scores[0, grid[:, :-1], grid[:, 1:]].sum() + scores[1, grid[:-1, :], grid[1:, :]].sum())


def grid_coordinates(grid: np.ndarray) -> np.ndarray:
    """(N, 2) (x, y) of each piece in a full grid."""
    rows, cols = grid.shape
    ys, xs = np.mgrid[0:rows, 0:cols]
    coords = np.zeros((grid.size, 2), dtype=int)
    coords[grid.ravel()] = np.stack([xs.ravel(), ys.ravel()], axis=1)
    return coords
