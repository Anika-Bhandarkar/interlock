"""Robust edge comparator experiment. Builds on interlock/ without changing it.

The stock comparator learns from pixel-exact masks, where a true neighbor always interlocks perfectly and every
wrong piece leaves a gap or an overlap, so "perfect fit" is a shortcut and color barely matters. This module
changes the training data and adds calibration:

1. perturb_masks: segmentation-style error on each piece's outline. It grows or shrinks by a few percent of the
   cell, unevenly along the outline, and the new rim takes the piece's nearest color or the color of the mat it
   was photographed on.
2. transplant: a negative that interlocks perfectly by shape. It has the true neighbor's outline but the image
   content of another cell, so only color can tell it from the real neighbor.
3. Flat-edged puzzles (flat_cut): pieces cut with straight lines only. "flat_even" is a perfect grid, so every
   edge has the same length and only color can match pieces; "flat_diagonal" slants every inner line, so sides
   vary in length and angle and the model can't learn that matching edges always have one length.
4. Temperature scaling: fit_temperature finds T so that softmax(logits / T) over candidates matches how often
   the comparator is right. fit_space_temperature does the same for the solver's space distributions.

CONDITIONS are the input conditions eval_robust.py stress-tests a comparator under; FLAT_KINDS are the
flat-edged test puzzles.
"""

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter
from scipy.optimize import minimize_scalar
from torch.utils.data import Dataset

from interlock.data_collection.cutting import SAMPLES_PER_SEGMENT, cut, extract_pieces, piece_map
from interlock.data_collection.cutting import PAD
from interlock.data_collection.dataset import PROCESSED_DIR, PuzzleDataset, random_flips_and_colors
from interlock.data_collection.grid import random_grid
from interlock.data_collection.pairs import (
    BELOW,
    DIRECTIONS,
    RELATIONS,
    RIGHT,
    WINDOW_SHAPE,
    can_fit,
    cell_size,
    neighbor_ids,
    orient,
    pad_to,
    seam_windows,
    side_types,
    to_rgba,
)
from interlock.solver.updaters import SCORE_FLOOR, first_updater, multistage_updater, true_grid

# outline error, as fractions of the cell so 30 px and 100 px pieces get the same relative error
MASK_SHIFT = 0.035  # each piece's outline grows or shrinks by up to this much overall
MASK_WOBBLE = 0.015  # std of the uneven part of the shift along the outline
# total shift is clipped here. side_types has ~0.07 of a cell of margin (cutting.MIN_TAB_HEIGHT, MAX_DIP vs
# pairs.SIDE_PROBE), less once its probe rounds to whole pixels; at 0.05 it misreads ~0.4% of sides at 30 px cells
MASK_MAX = 0.05
WOBBLE_SIGMA = 0.1  # how smoothly the uneven part varies along the outline
MIN_SHIFT_PX = 1.0  # small pieces still get at least a pixel of error
KEEP_CLEAN = 0.2  # share of pieces left exactly as cut during training, so clean accuracy doesn't drop
MAT_FILL = 0.5  # share of grown pieces whose new rim takes the mat color instead of the piece's own
TRANSPLANT_SHARE = 0.5  # share of negatives that are transplants; the rest are wrong pieces from the same puzzle
FLAT_EVEN_SHARE = 1 / 8  # share of training puzzles cut on a perfect grid of straight lines (1 in a batch of 8)
FLAT_DIAGONAL_SHARE = 1 / 8  # share cut with slanted straight lines (another 1 in 8)
FLAT_TILT = 0.2  # each end of an inner line of a "flat_diagonal" puzzle moves by up to this much of a cell
HARD_NEGATIVE_SHARE = 0.5  # flat puzzles: share of wrong pieces taken from the anchor's own row (or column)
PIXEL_NOISE = 12.0  # std of the "pixel_noise" condition, in 0-255 units
SMOOTH_SIGMA = 1.5  # px: blur of the "smoothed_outline" condition before re-thresholding

CONDITIONS = ("clean", "color_removed", "pixel_noise", "shrink_1px_half", "uneven_outline", "smoothed_outline")
FLAT_KINDS = {"flat_even": 0.0, "flat_diagonal": FLAT_TILT}  # kind -> tilt


def grid_shape(positions: np.ndarray) -> tuple[int, int]:
    """(rows, cols) of a puzzle, from its pieces' (x, y) grid positions."""
    return int(positions[:, 1].max()) + 1, int(positions[:, 0].max()) + 1


# ------------------------------------------------------------------ mask perturbation


def signed_distance(mask: np.ndarray) -> np.ndarray:
    """(B, B) float: distance to the outline, positive inside the piece and negative outside.

    The outline sits at 0, so `signed_distance(mask) > 0` is the mask itself, `> 1` shrinks it by 1 px
    (a 3x3 erosion) and `> -1` grows it by 1 px (a 3x3 dilation).
    """
    inside = distance_transform_edt(mask)
    outside = distance_transform_edt(~mask)
    return np.where(mask, inside - 0.5, 0.5 - outside)


def remask(piece: np.ndarray, mask: np.ndarray, new_mask: np.ndarray, fill_color=None) -> np.ndarray:
    """piece cut out again with new_mask. Pixels that left are 0. New rim pixels copy the nearest pixel of the
    original piece, or take fill_color (the mat showing through a loose cutout) if it is given."""
    out = np.zeros_like(piece)
    kept = new_mask & mask
    out[kept] = piece[kept]
    grown = new_mask & ~mask
    if grown.any():
        if fill_color is None:
            _, (near_y, near_x) = distance_transform_edt(~mask, return_indices=True)
            out[grown] = piece[near_y[grown], near_x[grown]]
        else:
            out[grown] = fill_color
    return out


def perturb_masks(
    pieces: np.ndarray,
    masks: np.ndarray,
    rng: np.random.Generator,
    shift: float = MASK_SHIFT,
    wobble: float = MASK_WOBBLE,
    keep_clean: float = 0.0,
    mat_fill: float = MAT_FILL,
    mat_color: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pieces and masks with segmentation-style error, each piece drawn independently.

    Each piece's outline moves by a random offset of up to shift * cell (at least MIN_SHIFT_PX; positive
    shrinks) plus a smooth uneven part with std wobble * cell, clipped to MASK_MAX * cell. A share `keep_clean`
    of pieces is left as is. mat_color is the color of the background a grown rim may show; one is drawn if not
    given.
    """
    pieces, masks = pieces.copy(), masks.copy()
    cell = cell_size(masks.shape[1])
    max_shift = max(shift * cell, MIN_SHIFT_PX) if shift else 0.0
    limit = max(MASK_MAX * cell, max_shift)
    mat_color = rng.integers(0, 256, 3, dtype=np.uint8) if mat_color is None else mat_color
    for k in range(len(masks)):
        if rng.random() < keep_clean:
            continue
        offset = rng.uniform(-max_shift, max_shift)
        if wobble:
            field = gaussian_filter(rng.normal(size=masks[k].shape), WOBBLE_SIGMA * cell)
            offset = offset + wobble * cell * field / field.std()
        new_mask = signed_distance(masks[k]) > np.clip(offset, -limit, limit)
        fill = mat_color if rng.random() < mat_fill else None
        pieces[k] = remask(pieces[k], masks[k], new_mask, fill)
        masks[k] = new_mask
    return pieces, masks


def apply_condition(
    condition: str, pieces: np.ndarray, masks: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Pieces and masks as seen under one of CONDITIONS.

    clean: as cut. color_removed: RGB zeroed, only outlines left (a diagnostic: a model that uses color
    should get worse). pixel_noise: Gaussian noise on the colors. shrink_1px_half: half the masks eroded by
    1 px. uneven_outline: perturb_masks, the same kind of error training sees, with fresh draws.
    smoothed_outline: each mask blurred and re-thresholded, which rounds corners and tab necks.
    """
    if condition == "clean":
        return pieces, masks
    if condition == "color_removed":
        return np.zeros_like(pieces), masks
    if condition == "pixel_noise":
        noisy = np.clip(pieces + rng.normal(0, PIXEL_NOISE, pieces.shape), 0, 255).astype(np.uint8)
        noisy[~masks] = 0
        return noisy, masks
    if condition == "shrink_1px_half":
        pieces, masks = pieces.copy(), masks.copy()
        for k in np.flatnonzero(rng.random(len(masks)) < 0.5):
            new_mask = binary_erosion(masks[k])
            pieces[k], masks[k] = remask(pieces[k], masks[k], new_mask), new_mask
        return pieces, masks
    if condition == "uneven_outline":
        return perturb_masks(pieces, masks, rng)
    if condition == "smoothed_outline":
        pieces, masks = pieces.copy(), masks.copy()
        for k in range(len(masks)):
            new_mask = gaussian_filter(masks[k].astype(float), SMOOTH_SIGMA) > 0.5
            pieces[k], masks[k] = remask(pieces[k], masks[k], new_mask), new_mask
        return pieces, masks
    raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")


class ConditionPuzzles:
    """A puzzle list seen under one condition: item i is (pieces, masks, positions), perturbed the same way every
    time. Works with evaluate() from interlock.models.edge_comparator."""

    def __init__(self, puzzles, condition: str, seed: int = 0):
        if condition not in CONDITIONS:
            raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")
        self.puzzles, self.condition, self.seed = puzzles, condition, seed

    def __len__(self) -> int:
        return len(self.puzzles)

    def __getitem__(self, index: int):
        pieces, masks, positions = self.puzzles[index]
        rng = np.random.default_rng([self.seed, CONDITIONS.index(self.condition), index])
        return (*apply_condition(self.condition, pieces, masks, rng), positions)


# ------------------------------------------------------------------ flat-edged puzzles


def flat_corners(height: int, width: int, rows: int, cols: int, rng: np.random.Generator | None = None,
                 tilt: float = 0.0) -> np.ndarray:
    """(rows + 1, cols + 1, 2) corners of a grid cut with straight lines, in the layout of cutting.get_corners.

    tilt 0 is the even grid. Otherwise each end of every inner line moves along the border by up to tilt * cell,
    independently, so the lines slant: pieces become irregular quadrilaterals whose sides differ in length and
    angle. The border lines stay on the image border.
    """
    cell_h, cell_w = height // rows, width // cols
    x_top = np.arange(cols + 1) * float(cell_w)
    y_left = np.arange(rows + 1) * float(cell_h)
    x_bottom, y_right = x_top.copy(), y_left.copy()
    if tilt and rng is not None:
        for line in (x_top, x_bottom):
            line[1:-1] += rng.uniform(-tilt, tilt, cols - 1) * cell_w
        for line in (y_left, y_right):
            line[1:-1] += rng.uniform(-tilt, tilt, rows - 1) * cell_h
    # vertical line j: x = a_j + b_j * y; horizontal line i: y = c_i + d_i * x; corner (i, j) is where they cross
    a, b = x_top[None, :], ((x_bottom - x_top) / height)[None, :]
    c, d = y_left[:, None], ((y_right - y_left) / width)[:, None]
    x = (a + b * c) / (1 - b * d)
    y = c + d * x
    return np.stack([x, y], axis=2)


def straight_edges(corners: np.ndarray, num_samples: int = 3 * SAMPLES_PER_SEGMENT - 2) -> tuple[np.ndarray, np.ndarray]:
    """Straight edges between neighboring corners, in get_edges' layout: horizontal (rows + 1, cols, K, 2) left to
    right and vertical (rows, cols + 1, K, 2) top to bottom."""
    t = np.linspace(0, 1, num_samples)[:, None]
    horizontal = corners[:, :-1, None] + t * (corners[:, 1:, None] - corners[:, :-1, None])
    vertical = corners[:-1, :, None] + t * (corners[1:, :, None] - corners[:-1, :, None])
    return horizontal, vertical


def flat_cut(image: np.ndarray, seed: int, rows: int, cols: int, tilt: float = 0.0):
    """Like cutting.cut, but every edge is straight: (pieces, masks, positions) in the same boxes and order."""
    height, width = image.shape[:2]
    corners = flat_corners(height, width, rows, cols, np.random.default_rng(seed), tilt)
    piece_ids = piece_map(*straight_edges(corners), height, width)
    pieces, masks = extract_pieces(image, piece_ids, rows, cols)
    positions = np.array([(piece % cols, piece // cols) for piece in range(rows * cols)])
    return pieces, masks, positions


class FlatPuzzles:
    """Item i is image i cut into a flat-edged puzzle of `kind` (a key of FLAT_KINDS), on the same random grid
    PuzzleDataset would use, so tabbed and flat test puzzles have the same sizes."""

    def __init__(self, rows: list[dict], kind: str, processed_dir=PROCESSED_DIR, seed_offset: int = 0):
        if kind not in FLAT_KINDS:
            raise ValueError(f"kind must be one of {tuple(FLAT_KINDS)}, got {kind!r}")
        self.images = PuzzleDataset(rows, processed_dir, seed_offset)
        self.kind = kind

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        seed = index + self.images.seed_offset
        crop, rows, cols = random_grid(self.images.load_image(index), np.random.default_rng([seed, 1]))
        return flat_cut(crop, seed, rows, cols, FLAT_KINDS[self.kind])


# ------------------------------------------------------------------ transplant negatives


def cell_boxes(image: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """(N, B, B, 3): the image under each cell's box (the cell plus PAD on every side), in cut()'s grid order."""
    cell_h, cell_w = image.shape[0] // rows, image.shape[1] // cols
    padded = np.pad(image, ((PAD, PAD), (PAD, PAD), (0, 0)))
    return np.stack([
        padded[row * cell_h : (row + 1) * cell_h + 2 * PAD, col * cell_w : (col + 1) * cell_w + 2 * PAD]
        for row in range(rows)
        for col in range(cols)
    ])


def interior_cells(rows: int, cols: int) -> np.ndarray:
    """Ids of cells off the image border. A piece reaches at most ~0.4 of a cell past its own cell, so any outline
    filled from an interior cell's box shows only image, never the black padding that would give it away."""
    ys, xs = np.divmod(np.arange(rows * cols), cols)
    return np.flatnonzero((ys > 0) & (ys < rows - 1) & (xs > 0) & (xs < cols - 1))


def transplant(boxes: np.ndarray, masks: np.ndarray, shape_id: int, content_id: int) -> np.ndarray:
    """(B, B, 3) fake piece: piece shape_id's outline filled with the image under cell content_id's box."""
    piece = boxes[content_id].copy()
    piece[~masks[shape_id]] = 0
    return piece


class RobustPairDataset(Dataset):
    """Training pairs like PairDataset's, with three changes that remove the perfect-fit shortcut.

    - Puzzle kind: FLAT_EVEN_SHARE of puzzles are cut on a perfect grid of straight lines, FLAT_DIAGONAL_SHARE with
      slanted straight lines, the rest with tabs as in PairDataset. Each puzzle uses PuzzleDataset's grid and seed.
    - Every piece (true neighbors included) gets perturb_masks error, so true neighbors no longer fit exactly.
    - A share `transplant_share` of negatives are transplants: the true neighbor's outline with another interior
      cell's image content, so they fit exactly where a positive would and only color can reject them. Other
      negatives: for tabbed puzzles, shape-fit wrong pieces as in PairDataset (from the clean cut's side_types);
      for flat puzzles, any other piece, half of them from the anchor's own row (or column for "below").

    Item i is puzzle i turned into 2 * len(DIRECTIONS) * pairs_per_puzzle windows (fewer for a direction with
    fewer neighbor pairs): for each direction, the positives then the negatives, one of each per anchor, padded to
    WINDOW_SHAPE. Call set_epoch(epoch) before each epoch, as for PairDataset.
    """

    def __init__(
        self,
        rows: list[dict],
        pairs_per_puzzle: int = 16,
        augment: bool = True,
        processed_dir=PROCESSED_DIR,
        seed: int = 0,
        transplant_share: float = TRANSPLANT_SHARE,
        keep_clean: float = KEEP_CLEAN,
        flat_even_share: float = FLAT_EVEN_SHARE,
        flat_diagonal_share: float = FLAT_DIAGONAL_SHARE,
    ):
        self.puzzles = PuzzleDataset(rows, processed_dir)
        self.pairs_per_puzzle = pairs_per_puzzle
        self.augment = augment
        self.seed = seed
        self.transplant_share = transplant_share
        self.keep_clean = keep_clean
        self.flat_even_share = flat_even_share
        self.flat_diagonal_share = flat_diagonal_share
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.puzzles.seed_offset = epoch * len(self.puzzles)

    def __len__(self) -> int:
        return len(self.puzzles)

    def puzzle_kind(self, rng: np.random.Generator) -> str:
        draw = rng.random()
        if draw < self.flat_even_share:
            return "flat_even"
        if draw < self.flat_even_share + self.flat_diagonal_share:
            return "flat_diagonal"
        return "tabbed"

    def __getitem__(self, index: int):
        rng = np.random.default_rng([self.seed, self.epoch, index])
        kind = self.puzzle_kind(rng)
        seed = index + self.puzzles.seed_offset
        crop, rows, cols = random_grid(self.puzzles.load_image(index), np.random.default_rng([seed, 1]))
        if kind == "tabbed":
            pieces, masks, positions = cut(crop, seed, rows, cols)
            types = side_types(masks)
        else:
            pieces, masks, positions = flat_cut(crop, seed, rows, cols, FLAT_KINDS[kind])
        boxes = cell_boxes(crop, rows, cols)
        interior = interior_cells(rows, cols)
        mat_color = rng.integers(0, 256, 3, dtype=np.uint8)  # one mat per photo
        rough_pieces, rough_masks = perturb_masks(pieces, masks, rng, keep_clean=self.keep_clean, mat_color=mat_color)
        num_pieces = len(pieces)

        all_windows, all_labels = [], []
        for direction in DIRECTIONS:
            relation = RELATIONS.index(direction)
            neighbors = neighbor_ids(positions, relation)
            num_pairs = min(self.pairs_per_puzzle, len(neighbors))
            anchors = rng.choice(list(neighbors), size=num_pairs, replace=False)
            fits = can_fit(types, relation) if kind == "tabbed" else None
            line_axis = 1 if relation == RIGHT else 0  # a right neighbor shares the anchor's row (y), below its column

            extra_pieces, extra_masks, wrong_ids = [], [], []
            for anchor in anchors:
                neighbor = neighbors[anchor]
                others = np.setdiff1d(np.arange(num_pieces), [anchor, neighbor])
                if rng.random() < self.transplant_share:
                    content = rng.choice(interior[(interior != neighbor) & (interior != anchor)])
                    fake = transplant(boxes, masks, neighbor, content)[None]
                    fake, fake_mask = perturb_masks(
                        fake, masks[neighbor][None], rng, keep_clean=self.keep_clean, mat_color=mat_color
                    )
                    wrong_ids.append(num_pieces + len(extra_pieces))
                    extra_pieces.append(fake[0])
                    extra_masks.append(fake_mask[0])
                    continue
                if kind == "tabbed":
                    candidates = np.flatnonzero(fits[anchor])
                    candidates = candidates[candidates != neighbor]
                elif rng.random() < HARD_NEGATIVE_SHARE:
                    candidates = others[positions[others, line_axis] == positions[anchor, line_axis]]
                else:
                    candidates = others
                wrong_ids.append(rng.choice(candidates if len(candidates) else others))

            all_pieces, all_masks = rough_pieces, rough_masks
            if extra_pieces:
                all_pieces = np.concatenate([rough_pieces, np.stack(extra_pieces)])
                all_masks = np.concatenate([rough_masks, np.stack(extra_masks)])
            left_ids = np.concatenate([anchors, anchors])
            right_ids = np.array([neighbors[anchor] for anchor in anchors] + wrong_ids)
            all_windows.append(seam_windows(to_rgba(*orient(all_pieces, all_masks, relation)), left_ids, right_ids))
            all_labels.append(np.repeat([1.0, 0.0], num_pairs))

        windows = torch.cat(all_windows)
        labels = torch.from_numpy(np.concatenate(all_labels).astype(np.float32))
        if self.augment:
            windows = random_flips_and_colors(windows, rng)
        return pad_to(windows, *WINDOW_SHAPE), labels


# ------------------------------------------------------------------ scoring flat puzzles


@torch.no_grad()
def score_all_pairs(model, pieces: np.ndarray, masks: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """(4, N, N) float32 logits like edge_comparator.score_matrix, but for every pair of different pieces.

    score_matrix only scores pairs whose sides can interlock (a tab facing a blank). Flat-edged pieces have no tabs,
    so it would rule out every pair; this scores them all. i == j stays -inf.
    """
    model.eval()
    device = next(model.parameters()).device
    num_pieces = len(pieces)
    logits = np.full((4, num_pieces, num_pieces), -np.inf, dtype=np.float32)
    left_ids, right_ids = np.nonzero(~np.eye(num_pieces, dtype=bool))
    for direction in DIRECTIONS:
        relation = RELATIONS.index(direction)
        rgba = to_rgba(*orient(pieces, masks, relation)).to(device)
        for start in range(0, len(left_ids), batch_size):
            batch = slice(start, start + batch_size)
            windows = pad_to(seam_windows(rgba, left_ids[batch], right_ids[batch]), *WINDOW_SHAPE)
            logits[relation, left_ids[batch], right_ids[batch]] = model(windows).float().cpu().numpy()
    logits[RELATIONS.index("left")] = logits[RIGHT].T
    logits[RELATIONS.index("above")] = logits[BELOW].T
    return logits


# ------------------------------------------------------------------ calibration


def candidate_rows(logits: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For every true right/below pair (i, j): logits[r, i] over all candidates, and j.

    Pairs whose true neighbor was ruled out by shape (-inf) are left out: no temperature can fix those.
    Returns rows (n, N) float with -inf where ruled out, and targets (n,) int.
    """
    rows, targets = [], []
    for relation in (RIGHT, BELOW):
        for piece_id, neighbor_id in neighbor_ids(positions, relation).items():
            if np.isfinite(logits[relation, piece_id, neighbor_id]):
                rows.append(logits[relation, piece_id])
                targets.append(neighbor_id)
    return np.array(rows, dtype=np.float64).reshape(-1, logits.shape[-1]), np.array(targets, dtype=int)


def stack_candidate_rows(rows_list, targets_list) -> tuple[np.ndarray, np.ndarray]:
    """candidate_rows from puzzles of different sizes as one (n, max N) array, padded with -inf.

    Padding is an impossible candidate: it gets probability 0 in softmax_rows and candidate_nll, so fitting a
    temperature on the stack is the same as fitting it over each puzzle's own candidates.
    """
    width = max(rows.shape[1] for rows in rows_list)
    padded = [np.pad(rows, ((0, 0), (0, width - rows.shape[1])), constant_values=-np.inf) for rows in rows_list]
    return np.concatenate(padded), np.concatenate(targets_list)


def softmax_rows(rows: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Softmax of each row of rows / temperature; -inf entries get probability 0."""
    scaled = rows / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def candidate_nll(rows: np.ndarray, targets: np.ndarray, temperature: float) -> float:
    """Mean -log P(true neighbor) under softmax(rows / temperature)."""
    scaled = rows / temperature
    top = scaled.max(axis=1)
    log_norm = top + np.log(np.exp(scaled - top[:, None]).sum(axis=1))
    return float(np.mean(log_norm - scaled[np.arange(len(scaled)), targets]))


def fit_temperature(rows: np.ndarray, targets: np.ndarray, bounds: tuple[float, float] = (0.05, 200.0)) -> float:
    """Temperature T > 0 that minimizes candidate_nll: softmax(logits / T) then means what it says."""
    result = minimize_scalar(
        lambda log_t: candidate_nll(rows, targets, np.exp(log_t)), bounds=np.log(bounds), method="bounded"
    )
    return float(np.exp(result.x))


def scale_logits(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Logits rescaled so the solver sees a temperature of `temperature`, keeping -inf (ruled out by shape).

    Plain logits / T doesn't do that: the solver floors every edge score at SCORE_FLOOR, so for T < 1 most
    candidates would be pushed under the floor and tie there. Instead, floor first and stretch about the floor,
    (max(l, F) - F) / T + F. That is l / T plus a constant, and every softmax the solver takes ignores constants.
    """
    floored = np.maximum(logits, SCORE_FLOOR)
    scaled = np.where(np.isneginf(logits), logits, (floored - SCORE_FLOOR) / temperature + SCORE_FLOOR)
    return scaled.astype(logits.dtype)


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, num_bins: int = 10) -> float:
    """Gap between confidence and accuracy, averaged over equal-width confidence bins weighted by their size."""
    confidence, correct = np.asarray(confidence, dtype=float), np.asarray(correct, dtype=float)
    if len(confidence) == 0:
        return 0.0
    bins = np.minimum((confidence * num_bins).astype(int), num_bins - 1)
    error = 0.0
    for b in np.unique(bins):
        in_bin = bins == b
        error += in_bin.sum() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(error / len(confidence))


def run_stages(logits: np.ndarray, types: np.ndarray, rows: int, cols: int, combine: str = "mean", **stage3):
    """Solver stages 2 and 3 for a rows x cols puzzle. updaters.solve() assumes 10 x 10, so the stages are called
    directly with the puzzle's grid. Returns (stage 2 result, stage 3 result)."""
    first = first_updater(logits, types, combine, rows, cols)
    return first, multistage_updater(logits, types, first, combine, **stage3)


def space_nll(logit_list, types_list, positions_list, temperature: float, combine: str = "mean") -> float:
    """Mean -log P(true piece) over every space of stage 3's distributions, with the logits divided by T."""
    losses = []
    for logits, types, positions in zip(logit_list, types_list, positions_list):
        rows, cols = grid_shape(positions)
        _, refined = run_stages(scale_logits(logits, temperature), types, rows, cols, combine)
        truth = true_grid(positions, rows, cols)
        true_probs = np.take_along_axis(refined.probs, truth[..., None], axis=2)[..., 0]
        losses.append(-np.log(np.maximum(true_probs, 1e-12)).ravel())
    return float(np.mean(np.concatenate(losses)))


def fit_space_temperature(
    logit_list, types_list, positions_list, grid=np.geomspace(1 / 16, 64, 31), combine: str = "mean"
) -> float:
    """Temperature from `grid` that makes the solver's stage 3 distributions most likely to hold the true piece.

    With combine="mean" a space's scores are averages of edge logits, so dividing the logits by T is a temperature
    on the space softmax too. Grid search, since the solver isn't differentiable.
    """
    losses = [space_nll(logit_list, types_list, positions_list, t, combine) for t in grid]
    return float(grid[int(np.argmin(losses))])
