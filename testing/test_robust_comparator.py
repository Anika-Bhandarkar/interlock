"""Tests for testing/robust_comparator.py: mask perturbation, flat-edged puzzles, transplant negatives, the robust
pair dataset, scoring every pair, and temperature scaling. Run with `uv run pytest testing`."""

import csv

import numpy as np
import pytest
import torch
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
from torch.utils.data import DataLoader

from interlock.data_collection.cutting import PAD, cut, get_corners
from interlock.data_collection.dataset import PuzzleDataset, collate_pairs, load_manifest
from interlock.data_collection.pairs import (
    BELOW,
    FLAT,
    LEFT,
    RIGHT,
    ABOVE,
    WINDOW_SHAPE,
    can_fit,
    cell_size,
    neighbor_ids,
    orient,
    seam_windows,
    side_types,
    to_rgba,
)
from interlock.models.edge_comparator import EdgeComparator, evaluate
from interlock.solver.updaters import SCORE_FLOOR, edge_scores, final_assignment, slot_borders, true_grid
from robust_comparator import (
    CONDITIONS,
    FLAT_KINDS,
    FLAT_TILT,
    MASK_MAX,
    ConditionPuzzles,
    FlatPuzzles,
    RobustPairDataset,
    apply_condition,
    candidate_nll,
    candidate_rows,
    cell_boxes,
    expected_calibration_error,
    fit_space_temperature,
    fit_temperature,
    flat_corners,
    flat_cut,
    grid_shape,
    interior_cells,
    perturb_masks,
    remask,
    run_stages,
    scale_logits,
    score_all_pairs,
    signed_distance,
    softmax_rows,
    stack_candidate_rows,
    straight_edges,
    transplant,
)

SQUARE = np.zeros((40, 40), dtype=bool)
SQUARE[10:30, 12:28] = True
FULL = np.ones((3, 3), dtype=bool)


@pytest.fixture(scope="module")
def image():
    return np.random.default_rng(5).integers(0, 256, size=(400, 400, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def puzzle(image):
    return cut(image, seed=4, rows=8, cols=8)  # 50 px cells


@pytest.fixture(scope="module")
def small_puzzle(image):
    return cut(image[:390, :390], seed=6, rows=13, cols=13)  # 30 px cells, the smallest the grid draws


@pytest.fixture(scope="module")
def oracle_model():
    torch.manual_seed(0)
    return EdgeComparator(width=4).eval()  # tiny: these tests check plumbing, not accuracy


# ------------------------------------------------------------------ mask perturbation


def test_signed_distance_thresholds_are_erosion_and_dilation():
    distance = signed_distance(SQUARE)
    np.testing.assert_array_equal(distance > 0, SQUARE)
    np.testing.assert_array_equal(distance > 1, binary_erosion(SQUARE, FULL))
    np.testing.assert_array_equal(distance > -1, binary_dilation(SQUARE, FULL))


def test_remask_zeros_removed_pixels_and_fills_new_rim():
    piece = np.zeros((40, 40, 3), dtype=np.uint8)
    piece[SQUARE] = (10, 20, 30)
    grown = binary_dilation(SQUARE, FULL)
    nearest = remask(piece, SQUARE, grown)
    assert (nearest[grown] == (10, 20, 30)).all() and (nearest[~grown] == 0).all()
    mat = remask(piece, SQUARE, grown, fill_color=np.array([200, 0, 0], dtype=np.uint8))
    assert (mat[grown & ~SQUARE] == (200, 0, 0)).all() and (mat[SQUARE] == (10, 20, 30)).all()
    shrunk = binary_erosion(SQUARE, FULL)
    assert (remask(piece, SQUARE, shrunk)[SQUARE & ~shrunk] == 0).all()


def test_perturb_masks_without_error_changes_nothing(puzzle):
    pieces, masks, _ = puzzle
    rng = np.random.default_rng(0)
    same_pieces, same_masks = perturb_masks(pieces, masks, rng, shift=0.0, wobble=0.0)
    np.testing.assert_array_equal(same_masks, masks)
    np.testing.assert_array_equal(same_pieces, pieces)
    _, kept_masks = perturb_masks(pieces, masks, rng, keep_clean=1.0)
    np.testing.assert_array_equal(kept_masks, masks)


@pytest.mark.parametrize("fixture", ["puzzle", "small_puzzle"])
def test_perturb_masks_moves_outlines_by_a_share_of_the_cell(fixture, request):
    pieces, masks, _ = request.getfixturevalue(fixture)
    cell = cell_size(masks.shape[1])
    rough_pieces, rough_masks = perturb_masks(pieces, masks, np.random.default_rng(1))
    assert not np.array_equal(rough_masks, masks)
    for k in range(len(masks)):
        moved = rough_masks[k] != masks[k]
        assert np.abs(signed_distance(masks[k])[moved]).max() <= max(MASK_MAX * cell, 1.0) + 1e-9
        assert (rough_pieces[k][~rough_masks[k]] == 0).all()
    assert rough_masks.sum() == pytest.approx(masks.sum(), rel=0.1)  # grows and shrinks roughly balance


def test_perturb_masks_is_reproducible(puzzle):
    pieces, masks, _ = puzzle
    first = perturb_masks(pieces, masks, np.random.default_rng(3))
    again = perturb_masks(pieces, masks, np.random.default_rng(3))
    np.testing.assert_array_equal(first[1], again[1])
    np.testing.assert_array_equal(first[0], again[0])


@pytest.mark.parametrize("condition", [c for c in CONDITIONS if c != "uneven_outline"])
@pytest.mark.parametrize("fixture", ["puzzle", "small_puzzle"])
def test_side_types_survive_small_mask_error(fixture, condition, request):
    """These conditions move outlines by at most ~1 px, which side_types reads right even on 30 px pieces."""
    pieces, masks, positions = request.getfixturevalue(fixture)
    rows, cols = grid_shape(positions)
    _, new_masks = apply_condition(condition, pieces, masks, np.random.default_rng(2))
    types = side_types(new_masks)
    np.testing.assert_array_equal(types == FLAT, slot_borders(rows, cols)[positions[:, 1], positions[:, 0]])
    fits = can_fit(types, RIGHT)
    assert all(fits[a, b] for a, b in neighbor_ids(positions, RIGHT).items())


@pytest.mark.parametrize("rows, size", [(13, 390), (8, 400), (4, 400)])
def test_side_types_rarely_misread_uneven_outlines(image, rows, size):
    """Outline error is a share of the cell, kept under the ~0.07-cell margin side_types has, so misreads are rare
    at every piece size; eval_robust.py reports the ones that happen."""
    misread = total = 0
    for seed in range(6):
        pieces, masks, _ = cut(image[:size, :size], seed, rows, rows)
        _, new_masks = apply_condition("uneven_outline", pieces, masks, np.random.default_rng(seed))
        misread += int((side_types(new_masks) != side_types(masks)).sum())
        total += masks.shape[0] * 4
    assert misread / total < 0.01


def test_apply_condition_rejects_unknown_names(puzzle):
    with pytest.raises(ValueError):
        apply_condition("foggy", *puzzle[:2], np.random.default_rng(0))


def test_condition_puzzles_are_stable(puzzle):
    puzzles = ConditionPuzzles([puzzle], "uneven_outline")
    np.testing.assert_array_equal(puzzles[0][1], puzzles[0][1])
    assert not np.array_equal(puzzles[0][1], puzzle[1])
    with pytest.raises(ValueError):
        ConditionPuzzles([puzzle], "foggy")


# ------------------------------------------------------------------ flat-edged puzzles


def test_flat_corners_without_tilt_are_the_even_grid():
    np.testing.assert_allclose(flat_corners(400, 400, 8, 8), get_corners(400, 400, 8, 8))


def test_flat_corners_slant_inner_lines_and_keep_the_border():
    rng = np.random.default_rng(0)
    corners = flat_corners(400, 400, 8, 8, rng, tilt=FLAT_TILT)
    even = get_corners(400, 400, 8, 8)
    np.testing.assert_allclose(corners[:, 0, 0], 0)
    np.testing.assert_allclose(corners[:, -1, 0], 400)
    np.testing.assert_allclose(corners[0, :, 1], 0)
    np.testing.assert_allclose(corners[-1, :, 1], 400)
    assert np.abs(corners - even).max() <= 2 * FLAT_TILT * 50 + 1e-6
    # every grid line stays straight: the corners along it are collinear
    for j in range(9):
        line = corners[:, j]
        direction = line[-1] - line[0]
        cross = direction[0] * (line[:, 1] - line[0, 1]) - direction[1] * (line[:, 0] - line[0, 0])
        np.testing.assert_allclose(cross, 0, atol=1e-6)
    # and some do slant, so sides differ in length
    _, vertical = straight_edges(corners)
    lengths = np.linalg.norm(vertical[:, :, -1] - vertical[:, :, 0], axis=-1)
    assert lengths.std() > 1.0


def test_flat_cut_even_grid_gives_equal_squares(image):
    pieces, masks, positions = flat_cut(image, seed=1, rows=8, cols=8)
    assert pieces.shape == (64, 50 + 2 * PAD, 50 + 2 * PAD, 3)
    expected = np.zeros(masks.shape[1:], dtype=bool)
    expected[PAD : PAD + 50, PAD : PAD + 50] = True
    for mask in masks:
        np.testing.assert_array_equal(mask, expected)
    assert (side_types(masks) == FLAT).all()
    assert grid_shape(positions) == (8, 8)


def test_flat_cut_diagonal_tiles_the_image_and_neighbors_fit(image):
    pieces, masks, positions = flat_cut(image, seed=2, rows=8, cols=8, tilt=FLAT_TILT)
    assert masks.sum() == 400 * 400  # every pixel in exactly one piece
    for relation in (RIGHT, BELOW):
        neighbors = neighbor_ids(positions, relation)
        left_ids = np.array(list(neighbors))
        windows = seam_windows(to_rgba(*orient(pieces, masks, relation)), left_ids, [neighbors[a] for a in left_ids])
        overlap = ((windows[:, 3] > 0) & (windows[:, 4] > 0)).sum()
        assert overlap == 0  # true neighbors share their straight edge exactly
    areas = masks.sum(axis=(1, 2))
    assert areas.std() > 0  # slanted lines make pieces of different sizes


def test_flat_puzzles_use_puzzle_datasets_grid(processed_dir):
    rows = load_manifest("val", processed_dir)
    tabbed = PuzzleDataset(rows, processed_dir)
    for kind in FLAT_KINDS:
        flat = FlatPuzzles(rows, kind, processed_dir)
        assert len(flat) == len(tabbed)
        pieces, masks, positions = flat[1]
        tab_pieces, _, tab_positions = tabbed[1]
        assert pieces.shape == tab_pieces.shape and grid_shape(positions) == grid_shape(tab_positions)
    with pytest.raises(ValueError):
        FlatPuzzles(rows, "wavy", processed_dir)


# ------------------------------------------------------------------ transplants


def test_cell_boxes_match_cut(image, puzzle):
    pieces, masks, _ = puzzle
    boxes = cell_boxes(image, 8, 8)
    assert boxes.shape == pieces.shape
    for k in range(len(pieces)):
        np.testing.assert_array_equal(boxes[k][masks[k]], pieces[k][masks[k]])


def test_interior_cells_are_off_the_border():
    ys, xs = np.divmod(interior_cells(6, 9), 9)
    assert len(ys) == 4 * 7
    assert ys.min() == 1 and ys.max() == 4 and xs.min() == 1 and xs.max() == 7


@pytest.mark.parametrize("fixture, size, rows", [("puzzle", 400, 8), ("small_puzzle", 390, 13)])
def test_transplants_never_show_padding(fixture, size, rows, image, request):
    """Any outline filled from an interior cell shows only image pixels, at every cell size."""
    pieces, masks, positions = request.getfixturevalue(fixture)
    boxes = cell_boxes(image[:size, :size], rows, rows)
    inside = cell_boxes(np.ones((size, size, 3), dtype=np.uint8), rows, rows)  # 1 where the box shows image
    neighbor = neighbor_ids(positions, RIGHT)[0]
    fake = transplant(boxes, masks, neighbor, interior_cells(rows, rows)[3])
    np.testing.assert_array_equal(fake[masks[neighbor]], boxes[interior_cells(rows, rows)[3]][masks[neighbor]])
    for content in interior_cells(rows, rows):
        assert inside[content][masks.any(axis=0)].all()


# ------------------------------------------------------------------ dataset


@pytest.fixture(scope="module")
def processed_dir(tmp_path_factory):
    """A tiny processed/ folder: 3 train and 2 val noise images plus a manifest (as in tests/)."""
    root = tmp_path_factory.mktemp("processed")
    rng = np.random.default_rng(0)
    rows = []
    for split, count in (("train", 3), ("val", 2)):
        for index in range(count):
            path = f"{split}/sa_0/{index}.png"
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rng.integers(0, 256, size=(400, 400, 3), dtype=np.uint8)).save(root / path)
            rows.append({"path": path, "split": split, "source": "sa_000000.tar", "orig_width": 1500, "orig_height": 2250})
    with open(root / "manifest.csv", "w", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return root


@pytest.mark.parametrize("flat_even, flat_diagonal", [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])
def test_robust_pair_dataset_item(processed_dir, flat_even, flat_diagonal):
    pairs = RobustPairDataset(load_manifest("train", processed_dir), pairs_per_puzzle=5, augment=False,
                              processed_dir=processed_dir, flat_even_share=flat_even, flat_diagonal_share=flat_diagonal)
    windows, labels = pairs[0]
    assert windows.shape == (20, 5, *WINDOW_SHAPE) and windows.dtype == torch.uint8
    assert labels.tolist() == [1.0] * 5 + [0.0] * 5 + [1.0] * 5 + [0.0] * 5


def test_puzzle_kinds_follow_the_shares(processed_dir):
    pairs = RobustPairDataset(load_manifest("train", processed_dir), processed_dir=processed_dir)
    kinds = [pairs.puzzle_kind(np.random.default_rng(i)) for i in range(4000)]
    assert kinds.count("flat_even") / 4000 == pytest.approx(1 / 8, abs=0.02)
    assert kinds.count("flat_diagonal") / 4000 == pytest.approx(1 / 8, abs=0.02)


@pytest.mark.parametrize("flat_even", [0.0, 1.0])
def test_transplant_negatives_fit_exactly_like_positives(processed_dir, flat_even):
    """With no mask error and only transplant negatives, each negative has the positive's outline exactly, so
    shape alone can't separate them, for tabbed and flat puzzles alike."""
    pairs = RobustPairDataset(load_manifest("train", processed_dir), pairs_per_puzzle=6, augment=False,
                              processed_dir=processed_dir, transplant_share=1.0, keep_clean=1.0,
                              flat_even_share=flat_even, flat_diagonal_share=0.0)
    windows, _ = pairs[1]
    for start in (0, 12):  # right, then below
        positives, negatives = windows[start : start + 6], windows[start + 6 : start + 12]
        torch.testing.assert_close(negatives[:, 3:], positives[:, 3:])  # both masks identical
        assert not torch.equal(negatives[:, :3], positives[:, :3])  # colors differ
        assert torch.all(((negatives[:, 3] > 0) & (negatives[:, 4] > 0)).sum(dim=(1, 2)) == 0)


def test_robust_positives_no_longer_fit_exactly(processed_dir):
    """Clean positives never overlap (tests/test_edge_comparator.py); with mask error some do."""
    pairs = RobustPairDataset(load_manifest("train", processed_dir), pairs_per_puzzle=8, augment=False,
                              processed_dir=processed_dir, keep_clean=0.0)
    windows, labels = pairs[2]
    overlap = ((windows[:, 3] > 0) & (windows[:, 4] > 0)).sum(dim=(1, 2))
    assert (overlap[labels == 1] > 0).any()


def test_robust_pair_dataset_is_reproducible_and_changes_each_epoch(processed_dir):
    pairs = RobustPairDataset(load_manifest("train", processed_dir), pairs_per_puzzle=4, processed_dir=processed_dir)
    first, _ = pairs[1]
    torch.testing.assert_close(first, pairs[1][0])
    pairs.set_epoch(1)
    assert not torch.equal(first, pairs[1][0])


def test_robust_dataloader_with_spawned_workers(processed_dir):
    pairs = RobustPairDataset(load_manifest("train", processed_dir), pairs_per_puzzle=3, processed_dir=processed_dir)
    loader = DataLoader(pairs, batch_size=2, num_workers=2, collate_fn=collate_pairs, multiprocessing_context="spawn")
    assert [len(labels) for _, labels in loader] == [24, 12]


def test_evaluate_runs_under_a_condition(processed_dir, oracle_model):
    puzzles = ConditionPuzzles(PuzzleDataset(load_manifest("val", processed_dir), processed_dir), "uneven_outline")
    assert set(evaluate(oracle_model, puzzles, num_puzzles=1)) == {"right", "below"}


# ------------------------------------------------------------------ scoring flat puzzles


def test_score_all_pairs_scores_every_pair_of_flat_pieces(image, oracle_model):
    pieces, masks, _ = flat_cut(image[:200, :200], seed=0, rows=4, cols=4)
    logits = score_all_pairs(oracle_model, pieces, masks)
    assert logits.shape == (4, 16, 16)
    off_diagonal = ~np.eye(16, dtype=bool)
    assert np.isfinite(logits[:, off_diagonal]).all() and np.isneginf(logits[:, ~off_diagonal]).all()
    np.testing.assert_array_equal(logits[LEFT], logits[RIGHT].T)
    np.testing.assert_array_equal(logits[ABOVE], logits[BELOW].T)


# ------------------------------------------------------------------ solver on any grid


def oracle_logits(positions: np.ndarray, rng: np.random.Generator, noise: float = 0.0, signal: float = 8.0):
    """Logits that score every true neighbor `signal` higher than everything else, plus optional noise."""
    num = len(positions)
    logits = rng.normal(0, noise, size=(4, num, num)).astype(np.float32) if noise else np.zeros((4, num, num), np.float32)
    for relation in range(4):
        for piece_id, neighbor_id in neighbor_ids(positions, relation).items():
            logits[relation, piece_id, neighbor_id] += signal
    return logits


def test_run_stages_solves_grids_other_than_10x10(image):
    pieces, masks, positions = cut(image[:200, :300], seed=3, rows=4, cols=6)
    rows, cols = grid_shape(positions)
    _, refined = run_stages(oracle_logits(positions, np.random.default_rng(0)), side_types(masks), rows, cols)
    final = final_assignment(refined, side_types(masks))
    np.testing.assert_array_equal(final.grid, true_grid(positions, rows, cols))


# ------------------------------------------------------------------ calibration


def test_fit_temperature_recovers_a_known_temperature():
    rng = np.random.default_rng(0)
    rows = rng.normal(size=(4000, 30)) * 4
    true_t = 2.5
    probs = softmax_rows(rows, true_t)
    targets = np.array([rng.choice(30, p=p) for p in probs])
    assert fit_temperature(rows, targets) == pytest.approx(true_t, rel=0.1)
    assert candidate_nll(rows, targets, true_t) < candidate_nll(rows, targets, 1.0)


def test_stacking_puzzles_of_different_sizes_keeps_the_fit(image):
    """Puzzles have different piece counts; padding their candidate rows with -inf must not change the temperature
    fit or any row's probabilities."""
    rng = np.random.default_rng(0)
    rows_list, targets_list = [], []
    for rows, cols in ((4, 6), (8, 8), (5, 5)):
        _, _, positions = cut(image[: rows * 50, : cols * 50], seed=rows, rows=rows, cols=cols)
        logits = oracle_logits(positions, rng, noise=3.0, signal=4.0)
        candidates, targets = candidate_rows(logits, positions)
        rows_list.append(candidates)
        targets_list.append(targets)
    stacked, targets = stack_candidate_rows(rows_list, targets_list)
    assert stacked.shape == (sum(len(r) for r in rows_list), 64) and len(targets) == len(stacked)
    np.testing.assert_allclose(softmax_rows(stacked, 2.0)[: len(rows_list[0]), :24], softmax_rows(rows_list[0], 2.0))
    nll_each = sum(candidate_nll(r, t, 2.0) * len(r) for r, t in zip(rows_list, targets_list)) / len(stacked)
    assert candidate_nll(stacked, targets, 2.0) == pytest.approx(nll_each)
    assert np.isfinite(fit_temperature(stacked, targets))


def test_softmax_rows_and_scale_logits_keep_ruled_out_pairs_at_zero():
    probs = softmax_rows(np.array([[1.0, -np.inf, 3.0]]), 2.0)
    assert probs[0, 1] == 0 and probs.sum() == pytest.approx(1.0)
    logits = np.full((4, 3, 3), -np.inf, dtype=np.float32)
    logits[0, 0, 1] = 6.0
    scaled = scale_logits(logits, 3.0)
    assert scaled.dtype == np.float32 and np.isneginf(scaled[1]).all() and np.isneginf(scaled[0, 0, 0])


@pytest.mark.parametrize("temperature", [0.1, 0.5, 2.0, 10.0])
def test_scale_logits_is_a_true_temperature_for_the_solver(temperature):
    """The solver floors edge scores at SCORE_FLOOR and softmaxes means of them. After scale_logits, that softmax
    must equal softmax(floored / T), for T below 1 as well as above."""
    logits = np.array([[[12.0, 3.0, -8.0, -45.0, -np.inf]]] * 4, dtype=np.float32)
    seen_by_solver = edge_scores(scale_logits(logits, temperature))[0, 0]
    expected = softmax_rows(np.maximum(logits[0], SCORE_FLOOR).astype(float), temperature)[0]
    np.testing.assert_allclose(softmax_rows(seen_by_solver[None].astype(float))[0], expected, rtol=1e-5)


def test_candidate_rows_skip_pairs_ruled_out_by_shape(puzzle):
    _, masks, positions = puzzle
    logits = np.zeros((4, len(masks), len(masks)), dtype=np.float32)
    first_piece, first_neighbor = next(iter(neighbor_ids(positions, RIGHT).items()))
    logits[RIGHT, first_piece, first_neighbor] = -np.inf
    rows, targets = candidate_rows(logits, positions)
    assert rows.shape == (8 * 7 + 7 * 8 - 1, len(masks)) and len(targets) == len(rows)


def test_expected_calibration_error():
    rng = np.random.default_rng(1)
    confidence = rng.uniform(size=20000)
    calibrated = rng.uniform(size=20000) < confidence
    assert expected_calibration_error(confidence, calibrated) < 0.02
    overconfident = np.full(1000, 0.99)
    assert expected_calibration_error(overconfident, np.arange(1000) < 500) == pytest.approx(0.49)
    assert expected_calibration_error([], []) == 0.0


def test_fit_space_temperature_prefers_softer_scores_for_noisy_logits(puzzle):
    """Oracle logits with heavy noise: the solver's distributions should be softened (T > 1), not sharpened."""
    _, masks, positions = puzzle
    rng = np.random.default_rng(0)
    types = side_types(masks)
    logit_list = [oracle_logits(positions, rng, noise=30.0, signal=40.0) for _ in range(3)]
    t = fit_space_temperature(logit_list, [types] * 3, [positions] * 3, grid=np.array([0.5, 1.0, 4.0, 16.0]))
    assert t > 1.0
