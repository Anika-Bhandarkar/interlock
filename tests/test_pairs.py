"""Tests for interlock/data_collection/pairs.py: side shapes, shape-fit rules and seam windows."""

import numpy as np
import pytest

from interlock.data_collection.cutting import COLS, PAD, ROWS, cut
from interlock.data_collection.pairs import (
    BELOW,
    BLANK,
    DIRECTIONS,
    FLAT,
    RELATIONS,
    RIGHT,
    TAB,
    can_fit,
    cell_size,
    seam_half_width,
    neighbor_ids,
    opposite,
    orient,
    seam_windows,
    side_types,
    to_rgba,
)

SEEDS = range(20)


@pytest.fixture(scope="module")
def noise_image():
    return np.random.default_rng(123).integers(0, 256, size=(400, 400, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def puzzles(noise_image):
    return [cut(noise_image, seed) for seed in SEEDS]


def test_neighbor_ids_match_grid(puzzles):
    _, _, positions = puzzles[0]
    for relation, (dx, dy) in [(RIGHT, (1, 0)), (BELOW, (0, 1))]:
        neighbors = neighbor_ids(positions, relation)
        assert len(neighbors) == (COLS - dx) * (ROWS - dy)
        for piece_id, neighbor_id in neighbors.items():
            np.testing.assert_array_equal(positions[neighbor_id] - positions[piece_id], (dx, dy))


def test_side_types_flat_exactly_on_border(puzzles):
    for _, masks, positions in puzzles:
        types = side_types(masks)
        x, y = positions[:, 0], positions[:, 1]
        on_border = np.stack([x == COLS - 1, y == ROWS - 1, x == 0, y == 0], axis=1)  # RELATIONS order
        np.testing.assert_array_equal(types == FLAT, on_border)


def test_side_types_true_neighbors_interlock(puzzles):
    """Every shared edge is a tab on one side and a blank on the other."""
    for _, masks, positions in puzzles:
        types = side_types(masks)
        for relation in range(4):
            for piece_id, neighbor_id in neighbor_ids(positions, relation).items():
                assert {types[piece_id, relation], types[neighbor_id, opposite(relation)]} == {TAB, BLANK}


def test_can_fit_keeps_true_neighbors_and_drops_self(puzzles):
    for _, masks, positions in puzzles:
        types = side_types(masks)
        for relation in range(4):
            fits = can_fit(types, relation)
            assert not fits.diagonal().any()
            for piece_id, neighbor_id in neighbor_ids(positions, relation).items():
                assert fits[piece_id, neighbor_id]
            # shape alone should rule out a good chunk of candidates
            assert fits.mean() < 0.6


def test_seam_windows_shape(puzzles):
    pieces, masks, _ = puzzles[0]
    windows = seam_windows(to_rgba(pieces, masks), [0, 1, 2], [3, 4, 5])
    assert windows.shape == (3, 5, pieces.shape[1], 2 * seam_half_width(cell_size(pieces.shape[1])))
    assert str(windows.dtype) == "torch.uint8"


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_seam_windows_true_neighbors_rebuild_image(noise_image, puzzles, direction):
    """Through the middle of the seam, a true pair has no overlap, no gap, and the original pixels."""
    relation = RELATIONS.index(direction)
    oriented_image = np.rot90(noise_image, k=relation)  # rotating the image matches rotating every piece
    for pieces, masks, positions in puzzles[:5]:
        box = pieces.shape[1]
        cell = cell_size(box)
        half_width = seam_half_width(cell)
        rgba = to_rgba(*orient(pieces, masks, relation))
        neighbors = neighbor_ids(positions, relation)
        left_ids = np.array(list(neighbors))
        windows = seam_windows(rgba, left_ids, [neighbors[piece_id] for piece_id in left_ids]).numpy()

        middle = slice(PAD + 2 * cell // 5, PAD + 3 * cell // 5)  # clear of tabs from pieces above/below
        mask_a, mask_b = windows[:, 3, middle] > 0, windows[:, 4, middle] > 0
        assert not (mask_a & mask_b).any(), "true neighbors overlap"
        assert (mask_a | mask_b).all(), "true neighbors leave a gap"

        # grid position of each left piece after rotating the puzzle counter-clockwise
        size = noise_image.shape[0] // cell
        for window, piece_id in zip(windows, left_ids):
            x, y = positions[piece_id]
            for _ in range(relation):
                x, y = y, size - 1 - x
            top = y * cell + middle.start - PAD
            left = x * cell + cell - half_width
            expected = oriented_image[top : top + (middle.stop - middle.start), left : left + 2 * half_width]
            np.testing.assert_array_equal(window[:3, middle].transpose(1, 2, 0), expected)


def test_seam_windows_shapes_that_cant_fit_leave_overlap_or_gap(puzzles):
    pieces, masks, positions = puzzles[0]
    types = side_types(masks)
    rgba = to_rgba(pieces, masks)
    both_tabs = [(a, b) for a in range(len(pieces)) for b in range(len(pieces))
                 if a != b and types[a, RIGHT] == TAB and types[b, opposite(RIGHT)] == TAB][:20]
    windows = seam_windows(rgba, [a for a, _ in both_tabs], [b for _, b in both_tabs]).numpy()
    overlap = ((windows[:, 3] > 0) & (windows[:, 4] > 0)).sum(axis=(1, 2))
    assert np.all(overlap > 0)


@pytest.mark.parametrize("half_width", [0, PAD + 41])
def test_seam_windows_reject_bad_half_width(puzzles, half_width):
    pieces, masks, _ = puzzles[0]
    with pytest.raises(ValueError):
        seam_windows(to_rgba(pieces, masks), [0], [1], half_width=half_width)
