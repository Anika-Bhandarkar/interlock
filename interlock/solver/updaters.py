"""Solver stages 2-4 from the team plan ("Interlock Approach Deepdive: CNNs + Greedy Algorithms").

Terms from the plan: a *space* is a slot on the grid, its *distribution* is a probability over pieces, and
its *max-piece* is the most likely piece in that distribution.

Stage 2, first_updater (the sudoku idea: pencil in a guess for every space instead of committing).
Start at the top-left corner. A priority queue holds the empty spaces next to filled ones, ordered by the
confidence of each space's max-piece. Pop the most confident space, keep its distribution (scored against
its neighbors' max-pieces), and queue its empty neighbors. Pieces are never taken out, so a piece can be
the max-piece of several spaces; stages 3 and 4 sort that out.

Stage 3, multistage_updater. After stage 2 most spaces were scored against only 1 or 2 neighbors. Each
round re-scores every space against the max-pieces of all 4 neighbors (all spaces update together, from
the previous round's max-pieces), then applies the plan's thresholds:
  1. a piece that is the max-piece of exactly one space for STABLE_ROUNDS rounds in a row is locked into
     that space and removed from every other space's distribution;
  2. once no max-piece has moved for STABLE_ROUNDS rounds, stop;
  3. a space whose max-piece has probability > LOCK_PROB is locked (most confident first, so each piece
     locks into one space only).

Stage 4, final_assignment. Unlocked max-pieces can still repeat, so the unlocked spaces get their pieces
from the Hungarian algorithm (scipy.optimize.linear_sum_assignment) on -log(probability): the most likely
placement in which no piece is used twice.

How a space's neighbors are combined (`combine`):
- "mean" (default): average the comparator's raw scores (logits) over the neighbors, then softmax over
  candidates. Every edge counts equally however many neighbors a space has, so confidence doesn't pile up
  just because several edges agree.
- "product": multiply the neighbors' match probabilities sigmoid(logit), i.e. sum their log-probabilities,
  treating each edge as independent evidence. Sharper, so more prone to confidently locking wrong pieces.

Conventions: grid[y, x] is the id of the piece in space (x, y), or EMPTY. Piece ids index the (shuffled)
pieces that were given to score_matrix. A piece may only go in a space whose border sides are exactly the
piece's flat sides (from side_types), so corners go in corners and edge pieces along the matching edge.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from interlock.data_collection.cutting import COLS, PAD, ROWS
from interlock.data_collection.pairs import BELOW, FLAT, OFFSETS, RIGHT, cell_size

EMPTY = -1
# stands in for -inf scores (shapes can't fit), so one wrongly placed neighbor can't rule out every candidate
SCORE_FLOOR = -30.0
COMBINE_MODES = ("mean", "product")
LOCK_PROB = 0.6  # threshold 3
STABLE_ROUNDS = 2  # thresholds 1 and 2
MAX_ROUNDS = 50  # stage 3 stops here even if max-pieces keep moving (e.g. two spaces swapping back and forth)


@dataclass
class SolveResult:
    grid: np.ndarray  # (rows, cols) piece ids: the max-piece of each space, or the final assignment
    probs: np.ndarray  # (rows, cols, N) each space's distribution over pieces, from when it was last scored
    order: list[tuple[int, int]] = field(default_factory=list)  # (x, y) spaces in the order stage 2 filled them
    locked: np.ndarray | None = None  # (rows, cols) bool, from stage 3 on
    num_rounds: int = 0  # stage 3 rounds run

    @property
    def confidence(self) -> np.ndarray:
        """(rows, cols) probability of the piece actually placed in each space."""
        return np.take_along_axis(self.probs, self.grid[..., None], axis=2)[..., 0]


@dataclass
class Snapshot:
    """The board after one solver step, for replaying a solve (see scripts/animate_solve.py)."""

    label: str  # e.g. "stage 2", "stage 3, round 1", "stage 4"
    grid: np.ndarray  # (rows, cols) max-piece of each space so far, EMPTY where no distribution yet
    locked: np.ndarray  # (rows, cols) bool: spaces whose piece is set


def edge_scores(logits: np.ndarray, combine: str = "mean") -> np.ndarray:
    """(4, N, N) per-edge scores to combine across neighbors, floored at SCORE_FLOOR.

    "mean" uses the logits as they are; "product" uses log sigmoid(logits) = log P(piece j is piece i's
    r-neighbor), so adding them multiplies probabilities.
    """
    if combine not in COMBINE_MODES:
        raise ValueError(f"combine must be one of {COMBINE_MODES}, got {combine!r}")
    scores = logits if combine == "mean" else -np.logaddexp(0.0, -logits)
    return np.maximum(scores, SCORE_FLOOR)


def slot_borders(rows: int = ROWS, cols: int = COLS) -> np.ndarray:
    """(rows, cols, 4) bool: which sides of each space (in RELATIONS order) lie on the puzzle border."""
    ys, xs = np.mgrid[0:rows, 0:cols]
    borders = np.zeros((rows, cols, 4), dtype=bool)
    for relation, (dx, dy) in OFFSETS.items():
        borders[..., relation] = ~((0 <= xs + dx) & (xs + dx < cols) & (0 <= ys + dy) & (ys + dy < rows))
    return borders


def eligibility(types: np.ndarray, rows: int = ROWS, cols: int = COLS) -> np.ndarray:
    """(rows, cols, N) bool: True where a piece's flat sides are exactly the space's border sides.

    A space that no piece qualifies for (only possible if side_types misread a mask) allows every piece.
    """
    flat = types == FLAT  # (N, 4)
    allowed = (flat[None, None] == slot_borders(rows, cols)[:, :, None]).all(axis=3)
    allowed[~allowed.any(axis=2)] = True
    return allowed


def neighbor_scores(
    edges: np.ndarray, grid: np.ndarray, x: int, y: int, average: bool = True
) -> tuple[np.ndarray, int]:
    """(N,) score of each candidate for space (x, y) given its filled neighbors, and how many there are.

    Sums the candidate's edge_scores against each neighbor's max-piece, or averages them if `average`.
    """
    rows, cols = grid.shape
    scores = np.zeros(edges.shape[1])
    num_neighbors = 0
    for relation, (dx, dy) in OFFSETS.items():
        neighbor_x, neighbor_y = x + dx, y + dy
        if 0 <= neighbor_x < cols and 0 <= neighbor_y < rows and grid[neighbor_y, neighbor_x] != EMPTY:
            scores += edges[relation, :, grid[neighbor_y, neighbor_x]]  # that piece as the candidate's neighbor
            num_neighbors += 1
    if average and num_neighbors:
        scores /= num_neighbors
    return scores, num_neighbors


def masked_softmax(scores: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Softmax over the allowed entries; the others get probability 0."""
    probs = np.zeros_like(scores, dtype=float)
    if allowed.any():
        exp_scores = np.exp(scores[allowed] - scores[allowed].max())
        probs[allowed] = exp_scores / exp_scores.sum()
    return probs


def candidates(allowed: np.ndarray, x: int, y: int, unavailable: np.ndarray) -> np.ndarray:
    """(N,) bool: pieces that fit space (x, y) by shape and aren't locked elsewhere; any such piece if none fit."""
    fits = allowed[y, x] & ~unavailable
    return fits if fits.any() else ~unavailable


# ------------------------------------------------------------------ stage 2


def first_updater(
    logits: np.ndarray,
    types: np.ndarray,
    combine: str = "mean",
    rows: int = ROWS,
    cols: int = COLS,
    trace: list[Snapshot] | None = None,
) -> SolveResult:
    """Stage 2: a distribution for every space, filled best-first from the top-left corner.

    Args:
        logits: (4, N, N) from score_matrix, N = rows * cols.
        types: (N, 4) from side_types, for the same pieces.
        combine: "mean" or "product", see the module docstring.
        trace: if given, a Snapshot is appended after each space is filled.
    Returns:
        SolveResult whose grid holds each space's max-piece (pieces may repeat) and `order` the fill order.
    """
    edges, average = edge_scores(logits, combine), combine == "mean"
    allowed = eligibility(types, rows, cols)
    no_pieces_taken = np.zeros(edges.shape[1], dtype=bool)
    grid = np.full((rows, cols), EMPTY)
    probs = np.zeros((rows, cols, edges.shape[1]))
    order = []
    queue = {}  # the priority queue: (x, y) of each empty space next to a filled one -> its distribution

    def fill(x, y, distribution):
        grid[y, x], probs[y, x] = int(distribution.argmax()), distribution
        order.append((x, y))
        if trace is not None:
            trace.append(Snapshot("stage 2", grid.copy(), np.zeros((rows, cols), dtype=bool)))
        queue.pop((x, y), None)
        for dx, dy in OFFSETS.values():  # queue the empty neighbors, or re-score them now they have one more
            neighbor_x, neighbor_y = x + dx, y + dy
            if 0 <= neighbor_x < cols and 0 <= neighbor_y < rows and grid[neighbor_y, neighbor_x] == EMPTY:
                scores, _ = neighbor_scores(edges, grid, neighbor_x, neighbor_y, average)
                queue[neighbor_x, neighbor_y] = masked_softmax(
                    scores, candidates(allowed, neighbor_x, neighbor_y, no_pieces_taken)
                )

    # top-left corner, picked by shape; if several pieces have that shape, prefer the one whose best right and
    # below matches score highest
    seed_scores = edges[RIGHT].max(axis=1) + edges[BELOW].max(axis=1)
    fill(0, 0, masked_softmax(seed_scores, candidates(allowed, 0, 0, no_pieces_taken)))
    while queue:
        x, y = max(queue, key=lambda space: queue[space].max())  # space with the most confident max-piece
        fill(x, y, queue[x, y])
    return SolveResult(grid, probs, order)


# ------------------------------------------------------------------ stage 3


def rescore(
    edges: np.ndarray, allowed: np.ndarray, grid: np.ndarray, locked: np.ndarray, average: bool = True
) -> np.ndarray:
    """(rows, cols, N) distribution for every unlocked space given the max-pieces of all 4 neighbors.

    Pieces locked into other spaces are left out; locked spaces get all zeros.
    """
    rows, cols = grid.shape
    probs = np.zeros((rows, cols, edges.shape[1]))
    locked_pieces = np.zeros(edges.shape[1], dtype=bool)
    locked_pieces[grid[locked]] = True
    for y, x in zip(*np.nonzero(~locked)):
        scores, _ = neighbor_scores(edges, grid, x, y, average)
        probs[y, x] = masked_softmax(scores, candidates(allowed, x, y, locked_pieces))
    return probs


def multistage_updater(
    logits: np.ndarray,
    types: np.ndarray,
    start: SolveResult,
    combine: str = "mean",
    lock_prob: float = LOCK_PROB,
    stable_rounds: int = STABLE_ROUNDS,
    max_rounds: int = MAX_ROUNDS,
    trace: list[Snapshot] | None = None,
) -> SolveResult:
    """Stage 3: re-score every space against all 4 neighbors, round by round, locking pieces by the thresholds.

    Args:
        logits, types, combine: as for first_updater.
        start: first_updater's result (any full grid of max-pieces with distributions works).
        lock_prob: threshold 3, lock a space whose max-piece has probability above this.
        stable_rounds: rounds for threshold 1 (sole max-piece) and threshold 2 (nothing moved).
        max_rounds: stop after this many rounds regardless.
        trace: if given, a Snapshot is appended after each round.
    Returns:
        SolveResult with each space's max-piece (unlocked ones may still repeat), `locked` and `num_rounds`.
    """
    edges, average = edge_scores(logits, combine), combine == "mean"
    rows, cols = start.grid.shape
    num_pieces = edges.shape[1]
    allowed = eligibility(types, rows, cols)
    grid, probs = start.grid.copy(), start.probs.copy()
    locked = np.zeros((rows, cols), dtype=bool)
    locked_pieces = np.zeros(num_pieces, dtype=bool)
    sole_rounds = np.zeros((rows, cols), dtype=int)  # rounds in a row each space's max-piece was max nowhere else
    still_rounds = 0  # rounds in a row in which no max-piece moved

    def lock(x, y):
        locked[y, x], locked_pieces[grid[y, x]] = True, True

    num_rounds = 0
    while num_rounds < max_rounds and not locked.all():
        num_rounds += 1
        unlocked = ~locked
        rescored = rescore(edges, allowed, grid, locked, average)  # from the previous round's max-pieces
        new_grid = np.where(unlocked, rescored.argmax(axis=2), grid)
        moved = new_grid != grid
        grid, probs = new_grid, np.where(unlocked[..., None], rescored, probs)

        # threshold 1: max-piece of exactly one space, and the same piece there for stable_rounds rounds
        times_max = np.bincount(grid[unlocked], minlength=num_pieces)
        sole = unlocked & (times_max[grid] == 1)
        sole_rounds = np.where(sole, np.where(moved, 1, sole_rounds + 1), 0)
        for y, x in zip(*np.nonzero(sole & (sole_rounds >= stable_rounds))):
            lock(x, y)

        # threshold 3: confident max-pieces, most confident first, each piece into one space only
        confidence = np.take_along_axis(probs, grid[..., None], axis=2)[..., 0]
        confident = list(zip(*np.nonzero(~locked & (confidence > lock_prob))))
        for y, x in sorted(confident, key=lambda space: -confidence[space]):
            if not locked_pieces[grid[y, x]]:
                lock(x, y)

        if trace is not None:
            trace.append(Snapshot(f"stage 3, round {num_rounds}", grid.copy(), locked.copy()))

        # threshold 2: stop once nothing has moved for stable_rounds rounds
        still_rounds = 0 if moved.any() else still_rounds + 1
        if still_rounds >= stable_rounds:
            break
    return SolveResult(grid, probs, list(start.order), locked, num_rounds)


# ------------------------------------------------------------------ stage 4


def final_assignment(result: SolveResult, types: np.ndarray, trace: list[Snapshot] | None = None) -> SolveResult:
    """Stage 4: give every unlocked space a different piece with the Hungarian algorithm.

    Locked spaces keep their pieces. The remaining pieces go to the remaining spaces so that the total
    -log(probability) of the assigned pieces is as small as possible; shape rules still apply.
    If `trace` is given, the final board is appended to it with every space set.
    """
    rows, cols = result.grid.shape
    num_pieces = result.probs.shape[2]
    locked = np.zeros((rows, cols), dtype=bool) if result.locked is None else result.locked
    grid = result.grid.copy()

    free_ys, free_xs = np.nonzero(~locked)
    free_pieces = np.setdiff1d(np.arange(num_pieces), grid[locked])
    cost = -np.log(np.maximum(result.probs[free_ys, free_xs][:, free_pieces], 1e-12))
    cost += np.where(eligibility(types, rows, cols)[free_ys, free_xs][:, free_pieces], 0.0, 1e6)
    space_index, piece_index = linear_sum_assignment(cost)
    grid[free_ys[space_index], free_xs[space_index]] = free_pieces[piece_index]
    if trace is not None:
        trace.append(Snapshot("stage 4", grid.copy(), np.ones((rows, cols), dtype=bool)))
    return SolveResult(grid, result.probs, result.order, locked, result.num_rounds)


def solve(
    logits: np.ndarray, types: np.ndarray, combine: str = "mean", trace: list[Snapshot] | None = None
) -> tuple[SolveResult, ...]:
    """Run stages 2, 3 and 4. Returns all three results; the last one is the answer.

    If `trace` is given, every stage appends its Snapshots to it.
    """
    first = first_updater(logits, types, combine, trace=trace)
    refined = multistage_updater(logits, types, first, combine, trace=trace)
    return first, refined, final_assignment(refined, types, trace)


# ------------------------------------------------------------------ helpers


def shuffle_puzzle(
    pieces: np.ndarray, masks: np.ndarray, positions: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The same puzzle with its pieces in random order; positions stay the ground truth for each piece."""
    permutation = rng.permutation(len(pieces))
    return pieces[permutation], masks[permutation], positions[permutation]


def true_grid(positions: np.ndarray, rows: int = ROWS, cols: int = COLS) -> np.ndarray:
    """(rows, cols) ground-truth grid: the id of the piece that belongs in each slot."""
    grid = np.full((rows, cols), EMPTY)
    grid[positions[:, 1], positions[:, 0]] = np.arange(len(positions))
    return grid


def placement_metrics(grid: np.ndarray, positions: np.ndarray) -> dict[str, float]:
    """Scores for a full placement against the ground-truth positions of each piece.

    direct: share of slots holding the right piece.
    neighbors: share of adjacent slot pairs holding true neighbors, which still credits a correct region
        that ended up shifted.
    perfect: 1.0 if every piece is in the right slot, else 0.0.
    """
    rows, cols = grid.shape
    direct = float(np.mean(grid == true_grid(positions, rows, cols)))
    placed_at = positions[grid]  # (rows, cols, 2): true (x, y) of the piece in each slot
    right_ok = (placed_at[:, 1:] - placed_at[:, :-1] == (1, 0)).all(axis=2)
    below_ok = (placed_at[1:] - placed_at[:-1] == (0, 1)).all(axis=2)
    neighbors = float(right_ok.sum() + below_ok.sum()) / (right_ok.size + below_ok.size)
    return {"direct": direct, "neighbors": neighbors, "perfect": float(direct == 1.0)}


def assemble(pieces: np.ndarray, masks: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """(rows * cell, cols * cell, 3) uint8 image with each piece pasted into its slot.

    Where wrongly placed pieces don't interlock, the gaps stay black.
    """
    rows, cols = grid.shape
    box = pieces.shape[1]
    cell = cell_size(box)
    canvas = np.zeros((rows * cell + 2 * PAD, cols * cell + 2 * PAD, 3), dtype=np.uint8)
    for y, x in zip(*np.nonzero(grid != EMPTY)):
        piece_id = grid[y, x]
        region = canvas[y * cell : y * cell + box, x * cell : x * cell + box]  # a view into canvas
        region[masks[piece_id]] = pieces[piece_id][masks[piece_id]]
    return canvas[PAD:-PAD, PAD:-PAD]
