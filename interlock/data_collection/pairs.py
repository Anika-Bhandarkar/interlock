"""Geometry helpers for comparing two pieces: orientation, true neighbors, side shapes, and seam windows.

Every relation is reduced to "is B the right neighbor of A?" by rotating both pieces (see the onboarding
notebook). RELATIONS is ordered so that rotating a piece counter-clockwise by r * 90 degrees brings side r
to the right, which keeps the bookkeeping below to one rule: side r of A faces side (r + 2) % 4 of B.
"""

import numpy as np
import torch

from interlock.data_collection.cutting import PAD

RELATIONS = ("right", "below", "left", "above")
RIGHT, BELOW, LEFT, ABOVE = range(4)
# solver only needs these two; left/above are the transposes (A left of B == B right of A)
DIRECTIONS = ("right", "below")
OFFSETS = {RIGHT: (1, 0), BELOW: (0, 1), LEFT: (-1, 0), ABOVE: (0, -1)}  # (dx, dy) to the neighbor

# side shapes
FLAT, TAB, BLANK = 0, 1, -1  # TAB sticks out, BLANK is cut in; two sides fit only if their product is -1
SIDE_PROBE = 0.15  # fraction of a cell; tabs reach ~0.25 cell and jitter stays under ~0.05, so 0.15 splits them

SEAM_HALF_WIDTH = 24  # window around the seam, in pixels on each side; must cover the tab reach (~12 px)


def opposite(relation: int) -> int:
    return (relation + 2) % 4


def cell_size(box: int) -> int:
    """Side of a grid cell, from the side of a piece's box (box = cell + 2 * PAD)."""
    return box - 2 * PAD


def rotate(array: np.ndarray, relation: int) -> np.ndarray:
    """Rotate (N, B, B, ...) pieces or masks counter-clockwise so that side `relation` faces right."""
    return np.rot90(array, k=relation, axes=(1, 2))


def orient(pieces: np.ndarray, masks: np.ndarray, relation: int) -> tuple[np.ndarray, np.ndarray]:
    """Pieces and masks rotated so that "B is A's `relation` neighbor" becomes "B is right of A"."""
    return rotate(pieces, relation), rotate(masks, relation)


def neighbor_ids(positions: np.ndarray, relation: int) -> dict[int, int]:
    """Dict: piece id -> id of its true neighbor in `relation` (pieces on that border have none)."""
    dx, dy = OFFSETS[relation]
    position_to_id = {(int(x), int(y)): piece_id for piece_id, (x, y) in enumerate(positions)}
    return {
        piece_id: position_to_id[(int(x) + dx, int(y) + dy)]
        for piece_id, (x, y) in enumerate(positions)
        if (int(x) + dx, int(y) + dy) in position_to_id
    }


def side_types(masks: np.ndarray) -> np.ndarray:
    """(N, 4) int array: shape of each piece's side in RELATIONS order, FLAT, TAB or BLANK.

    Read straight off the mask, so it works on shuffled pieces with no ground truth:
    TAB if the piece reaches past its cell edge by SIDE_PROBE, BLANK if a hole cuts that far into the
    middle of the side, FLAT otherwise (only border pieces have flat sides).
    """
    box = masks.shape[1]
    cell = cell_size(box)
    probe = max(1, round(SIDE_PROBE * cell))
    edge = PAD + cell  # first column past the cell
    middle = slice(PAD + cell // 4, PAD + 3 * cell // 4)  # middle of the side, away from the corners

    types = np.zeros((len(masks), 4), dtype=np.int8)
    for relation in range(4):
        rotated = rotate(masks, relation)
        sticks_out = rotated[:, :, edge + probe - 1 :].any(axis=(1, 2))
        cut_in = (~rotated[:, middle, edge - 1 - probe]).any(axis=1)
        types[sticks_out, relation] = TAB
        types[cut_in & ~sticks_out, relation] = BLANK
    return types


def can_fit(types: np.ndarray, relation: int) -> np.ndarray:
    """(N, N) bool: True where piece j could be piece i's `relation` neighbor by shape alone.

    Side `relation` of i must be a tab and the facing side of j a blank, or the other way round.
    A piece never fits next to itself.
    """
    fits = np.outer(types[:, relation], types[:, opposite(relation)]) == -1
    np.fill_diagonal(fits, False)
    return fits


def to_rgba(pieces: np.ndarray, masks: np.ndarray) -> torch.Tensor:
    """(N, 4, B, B) uint8 tensor: RGB plus the mask as a 0/255 alpha channel."""
    rgba = np.concatenate([pieces, masks[..., None].astype(np.uint8) * 255], axis=-1)
    # rot90 leaves negative strides, which torch rejects, so copy into a contiguous array first
    return torch.from_numpy(np.ascontiguousarray(rgba)).permute(0, 3, 1, 2).contiguous()


def seam_windows(
    rgba: torch.Tensor,
    left_ids: torch.Tensor | np.ndarray,
    right_ids: torch.Tensor | np.ndarray,
    half_width: int = SEAM_HALF_WIDTH,
) -> torch.Tensor:
    """The strip around the seam when piece right_ids[k] is placed as the right neighbor of left_ids[k].

    Works like the notebook's make_pair_images (B's box one cell to the right of A's), then keeps only
    `half_width` columns on each side of the seam, which is where both color and shape evidence lives.

    Args:
        rgba: (N, 4, B, B) uint8 output of to_rgba, already oriented.
        left_ids, right_ids: (n,) piece ids.
    Returns:
        (n, 5, B, 2 * half_width) uint8: RGB (B drawn over A), A's mask, B's mask (masks 0/255).
        Overlap (both masks set) or gaps (neither set) along the seam mean the shapes don't fit.
    """
    box = rgba.shape[-1]
    cell = cell_size(box)
    if not 1 <= half_width <= PAD + cell:
        raise ValueError(f"half_width must be between 1 and {PAD + cell}, got {half_width}")
    low = PAD + cell - half_width  # canvas column where the window starts
    width = 2 * half_width

    left_ids = torch.as_tensor(left_ids, device=rgba.device)
    right_ids = torch.as_tensor(right_ids, device=rgba.device)
    windows = torch.zeros((len(left_ids), 5, box, width), dtype=torch.uint8, device=rgba.device)

    # A's box covers canvas columns [0, box): canvas column c is A's column c
    a_end = min(low + width, box)
    piece_a = rgba[left_ids, :, :, low:a_end]
    windows[:, :3, :, : a_end - low] = piece_a[:, :3]  # RGB is already 0 outside the piece
    windows[:, 3, :, : a_end - low] = piece_a[:, 3]

    # B's box covers canvas columns [cell, cell + box): canvas column c is B's column c - cell
    b_start, b_end = max(low, cell), min(low + width, cell + box)
    piece_b = rgba[right_ids, :, :, b_start - cell : b_end - cell]
    region = windows[:, :, :, b_start - low : b_end - low]  # a view, so writes land in windows
    region[:, :3] = torch.where(piece_b[:, 3:4] > 0, piece_b[:, :3], region[:, :3])
    region[:, 4] = piece_b[:, 3]
    return windows
