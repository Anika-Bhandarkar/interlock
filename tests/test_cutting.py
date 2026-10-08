"""Tests for the jigsaw cutting helpers in interlock/data_collection/cutting.py.

Checks: corner grid layout, tab control point geometry, curve sampling, edge placement, and
that get_edges/cut partition the image exactly so reassembly reproduces it.
"""

import numpy as np
import pytest

from interlock.data_collection.cutting import (
    COLS,
    PAD,
    ROWS,
    cut,
    get_corners,
    get_edges,
    place_edge,
    sample_curve,
    tab_control_points,
)

SEEDS = range(200)


def cross_2d(vector_a, vector_b):
    """z-component of the cross product of two 2D vectors (0 means parallel)."""
    return vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0]


# ---------------------------------------------------------------- get_corners


@pytest.mark.parametrize("height, width", [(400, 400), (200, 400)])
def test_corners_shape_and_extremes(height, width):
    corners = get_corners(height, width)
    assert corners.shape == (ROWS + 1, COLS + 1, 2)
    np.testing.assert_array_equal(corners[0, 0], (0, 0))
    np.testing.assert_array_equal(corners[-1, -1], (width, height))


@pytest.mark.parametrize("height, width", [(400, 400), (200, 400)])
def test_corners_evenly_spaced(height, width):
    """x steps by width/COLS along a row, y steps by height/ROWS down a column."""
    corners = get_corners(height, width)
    x_steps = np.diff(corners[..., 0], axis=1)
    y_steps = np.diff(corners[..., 1], axis=0)
    assert np.all(x_steps == width // COLS)
    assert np.all(y_steps == height // ROWS)
    # x is constant down a column and y constant along a row
    assert np.all(np.diff(corners[..., 0], axis=0) == 0)
    assert np.all(np.diff(corners[..., 1], axis=1) == 0)


@pytest.mark.parametrize("height, width", [(405, 400), (400, 395), (7, 7)])
def test_corners_reject_non_divisible_sizes(height, width):
    with pytest.raises(ValueError):
        get_corners(height, width)


# ---------------------------------------------------------- tab_control_points


def test_control_points_shape_and_endpoints():
    for seed in SEEDS:
        control_points = tab_control_points(np.random.default_rng(seed))
        assert control_points.shape == (3, 4, 2)
        np.testing.assert_array_equal(control_points[0, 0], (0, 0))
        np.testing.assert_array_equal(control_points[-1, -1], (1, 0))


def test_control_points_joints_shared():
    """Each segment starts where the previous one ends, and p2 == p7."""
    for seed in SEEDS:
        control_points = tab_control_points(np.random.default_rng(seed))
        np.testing.assert_array_equal(control_points[0, 3], control_points[1, 0])
        np.testing.assert_array_equal(control_points[1, 3], control_points[2, 0])
        np.testing.assert_array_equal(control_points[0, 2], control_points[2, 1])


def test_control_points_smooth_joints():
    """Handles on either side of each joint are collinear (p2, p3, p4 and p5, p6, p7)."""
    for seed in SEEDS:
        control_points = tab_control_points(np.random.default_rng(seed))
        p2, p3 = control_points[0, 2], control_points[0, 3]
        p4, p5 = control_points[1, 1], control_points[1, 2]
        p6, p7 = control_points[1, 3], control_points[2, 1]
        assert cross_2d(p3 - p2, p4 - p3) == pytest.approx(0, abs=1e-12)
        assert cross_2d(p6 - p5, p7 - p6) == pytest.approx(0, abs=1e-12)


def test_control_points_reproducible():
    first = tab_control_points(np.random.default_rng(7))
    second = tab_control_points(np.random.default_rng(7))
    other = tab_control_points(np.random.default_rng(8))
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)


# ---------------------------------------------------------------- sample_curve


@pytest.mark.parametrize("num_samples", [2, 5, 20])
def test_sample_curve_shape_and_endpoints(num_samples):
    control_points = tab_control_points(np.random.default_rng(0))
    curve = sample_curve(control_points, num_samples)
    assert curve.shape == (3 * num_samples - 2, 2)
    np.testing.assert_array_equal(curve[0], (0, 0))
    np.testing.assert_array_equal(curve[-1], (1, 0))


def test_sample_curve_no_repeated_points():
    """Shared joints are dropped, so no two consecutive points coincide."""
    for seed in SEEDS:
        curve = sample_curve(tab_control_points(np.random.default_rng(seed)))
        step_lengths = np.linalg.norm(np.diff(curve, axis=0), axis=1)
        assert np.all(step_lengths > 0)


# ------------------------------------------------------------------ place_edge

HORIZONTAL_START, HORIZONTAL_END = np.array([40, 80]), np.array([80, 80])
VERTICAL_START, VERTICAL_END = np.array([40, 80]), np.array([40, 120])


@pytest.fixture
def curve():
    return sample_curve(tab_control_points(np.random.default_rng(0)))


@pytest.mark.parametrize("sign", [-1, 0, 1])
@pytest.mark.parametrize(
    "start, end", [(HORIZONTAL_START, HORIZONTAL_END), (VERTICAL_START, VERTICAL_END)]
)
def test_place_edge_endpoints_exact(curve, start, end, sign):
    edge_points = place_edge(curve, start, end, sign)
    assert edge_points.shape == curve.shape
    np.testing.assert_array_equal(edge_points[0], start)
    np.testing.assert_array_equal(edge_points[-1], end)


def test_place_edge_sign_zero_is_straight(curve):
    """With sign 0 every point lies on the segment from start to end."""
    start, end = np.array([10, 20]), np.array([50, 60])
    edge_points = place_edge(curve, start, end, 0)
    offsets = edge_points - start
    assert np.allclose(cross_2d(offsets.T, end - start), 0)
    assert np.all(edge_points >= start) and np.all(edge_points <= end)


def farthest_offset(edge_points, axis, line_value):
    """Signed offset (along axis) of the point farthest from the line axis == line_value."""
    offsets = edge_points[:, axis] - line_value
    return offsets[np.argmax(np.abs(offsets))]


@pytest.mark.parametrize("sign, tab_below", [(1, True), (-1, False)])
def test_place_edge_horizontal_tab_side(curve, sign, tab_below):
    """+1 on a left-to-right edge puts the tab in the piece below (larger y)."""
    edge_points = place_edge(curve, HORIZONTAL_START, HORIZONTAL_END, sign)
    assert (farthest_offset(edge_points, axis=1, line_value=80) > 0) == tab_below


@pytest.mark.parametrize("sign, tab_left", [(1, True), (-1, False)])
def test_place_edge_vertical_tab_side(curve, sign, tab_left):
    """+1 on a top-to-bottom edge puts the tab in the piece to the left (smaller x)."""
    edge_points = place_edge(curve, VERTICAL_START, VERTICAL_END, sign)
    assert (farthest_offset(edge_points, axis=0, line_value=40) < 0) == tab_left


def test_place_edge_rejects_bad_sign(curve):
    with pytest.raises(ValueError):
        place_edge(curve, HORIZONTAL_START, HORIZONTAL_END, 2)


# ------------------------------------------------------------------- get_edges

IMAGE_HEIGHT, IMAGE_WIDTH = 200, 400  # non-square to catch row/col swaps


def make_edges(seed):
    """Corners and edges for the test image."""
    corners = get_corners(IMAGE_HEIGHT, IMAGE_WIDTH)
    horizontal_edges, vertical_edges = get_edges(corners, np.random.default_rng(seed))
    return corners, horizontal_edges, vertical_edges


def test_edges_shapes():
    _, horizontal_edges, vertical_edges = make_edges(0)
    assert horizontal_edges.shape[:2] == (ROWS + 1, COLS)
    assert vertical_edges.shape[:2] == (ROWS, COLS + 1)
    assert horizontal_edges.shape[-1] == vertical_edges.shape[-1] == 2


def test_edges_end_at_corners():
    corners, horizontal_edges, vertical_edges = make_edges(0)
    np.testing.assert_allclose(horizontal_edges[:, :, 0], corners[:, :-1])
    np.testing.assert_allclose(horizontal_edges[:, :, -1], corners[:, 1:])
    np.testing.assert_allclose(vertical_edges[:, :, 0], corners[:-1, :])
    np.testing.assert_allclose(vertical_edges[:, :, -1], corners[1:, :])


def test_border_edges_straight_inner_edges_not():
    _, horizontal_edges, vertical_edges = make_edges(0)
    np.testing.assert_allclose(horizontal_edges[0, ..., 1], 0)
    np.testing.assert_allclose(horizontal_edges[-1, ..., 1], IMAGE_HEIGHT)
    np.testing.assert_allclose(vertical_edges[:, 0, :, 0], 0)
    np.testing.assert_allclose(vertical_edges[:, -1, :, 0], IMAGE_WIDTH)
    # every inner edge leaves its grid line somewhere (has a tab)
    inner_horizontal = horizontal_edges[1:-1, ..., 1]
    inner_vertical = vertical_edges[:, 1:-1, :, 0]
    assert np.all(np.ptp(inner_horizontal, axis=-1) > 0)
    assert np.all(np.ptp(inner_vertical, axis=-1) > 0)


def test_edges_reproducible():
    _, *first = make_edges(3)
    _, *second = make_edges(3)
    for first_edges, second_edges in zip(first, second):
        np.testing.assert_array_equal(first_edges, second_edges)


# ------------------------------------------------------------------------- cut

CUT_SEED = 0


@pytest.fixture(scope="module")
def noise_image():
    """Random-noise image so any misplaced pixel shows up in reassembly."""
    return np.random.default_rng(123).integers(0, 256, size=(400, 400, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def cut_result(noise_image):
    return cut(noise_image, CUT_SEED)


def piece_layout(image, pieces):
    """Cell size and padding for the image."""
    cell_height, cell_width = image.shape[0] // ROWS, image.shape[1] // COLS
    return cell_height, cell_width, PAD


def test_cut_shapes(noise_image, cut_result):
    pieces, masks, positions = cut_result
    num_pieces = ROWS * COLS
    box_size = pieces.shape[1]
    assert pieces.shape == (num_pieces, box_size, box_size, 3)
    assert masks.shape == (num_pieces, box_size, box_size)
    assert masks.dtype == bool
    assert positions.shape == (num_pieces, 2)


def test_cut_positions_cover_grid_once(cut_result):
    # positions are (x, y) grid coordinates: x = column, y = row
    _, _, positions = cut_result
    expected = {(x, y) for x in range(COLS) for y in range(ROWS)}
    assert sorted(map(tuple, positions.tolist())) == sorted(expected)


def test_cut_pixels_outside_mask_are_zero(cut_result):
    pieces, masks, _ = cut_result
    assert np.all(pieces[~masks] == 0)


def test_cut_masks_partition_image(noise_image, cut_result):
    """Placing every mask back at its box gives a coverage count of exactly 1 inside the image."""
    pieces, masks, positions = cut_result
    cell_height, cell_width, pad = piece_layout(noise_image, pieces)
    box_height, box_width = masks.shape[1:]
    height, width = noise_image.shape[:2]
    # canvas is the image plus pad on every side, so box top-left lands at (y*cell_h, x*cell_w)
    coverage = np.zeros((height + 2 * pad, width + 2 * pad), dtype=int)
    for mask, (x, y) in zip(masks, positions):
        top, left = y * cell_height, x * cell_width
        coverage[top : top + box_height, left : left + box_width] += mask
    inside = coverage[pad : pad + height, pad : pad + width]
    assert np.all(inside == 1), f"{np.sum(inside == 0)} uncovered, {np.sum(inside > 1)} overlapping"
    assert coverage.sum() == inside.sum(), "masks extend past the image border"


def test_cut_reassembly_reproduces_image(noise_image, cut_result):
    pieces, masks, positions = cut_result
    cell_height, cell_width, pad = piece_layout(noise_image, pieces)
    box_height, box_width = masks.shape[1:]
    height, width = noise_image.shape[:2]
    canvas = np.zeros((height + 2 * pad, width + 2 * pad, 3), dtype=int)
    for piece, mask, (x, y) in zip(pieces, masks, positions):
        top, left = y * cell_height, x * cell_width
        canvas[top : top + box_height, left : left + box_width] += piece * mask[..., None]
    np.testing.assert_array_equal(canvas[pad : pad + height, pad : pad + width], noise_image)


def test_cut_reproducible(noise_image, cut_result):
    repeat = cut(noise_image, CUT_SEED)
    for first_part, second_part in zip(cut_result, repeat):
        np.testing.assert_array_equal(first_part, second_part)
