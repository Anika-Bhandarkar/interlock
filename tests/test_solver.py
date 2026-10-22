"""Tests for the stage 2 / stage 3 updaters, using exact "oracle" logits built from the ground truth."""

import numpy as np
import pytest

from interlock.data_collection.cutting import COLS, ROWS, cut
from interlock.data_collection.pairs import BELOW, RELATIONS, RIGHT, can_fit, neighbor_ids, side_types
from interlock.solver.updaters import (
    EMPTY,
    MAX_ROUNDS,
    SCORE_FLOOR,
    SolveResult,
    assemble,
    edge_scores,
    eligibility,
    final_assignment,
    first_updater,
    multistage_updater,
    neighbor_scores,
    placement_metrics,
    shuffle_puzzle,
    solve,
    true_grid,
)

NUM_PIECES = ROWS * COLS


@pytest.fixture(scope="module")
def image():
    return np.random.default_rng(7).integers(0, 256, size=(400, 400, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def puzzle(image):
    """Shuffled pieces, masks and their ground-truth positions."""
    return shuffle_puzzle(*cut(image, seed=3), np.random.default_rng(0))


@pytest.fixture(scope="module")
def types(puzzle):
    return side_types(puzzle[1])


def oracle_logits(types, positions, margin=10.0):
    """+margin for true neighbors, -margin for other pairs that fit, -inf where shapes can't fit."""
    logits = np.full((4, NUM_PIECES, NUM_PIECES), -margin, dtype=np.float32)
    for relation in range(len(RELATIONS)):
        for piece_id, neighbor_id in neighbor_ids(positions, relation).items():
            logits[relation, piece_id, neighbor_id] = margin
        logits[relation][~can_fit(types, relation)] = -np.inf
    return logits


@pytest.fixture(scope="module")
def logits(puzzle, types):
    return oracle_logits(types, puzzle[2])


def assert_permutation(grid):
    assert grid.shape == (ROWS, COLS)
    assert sorted(grid.ravel().tolist()) == list(range(NUM_PIECES))


# ------------------------------------------------------------------ helpers


def test_edge_scores_floor_misfits_for_both_modes():
    logits = np.array([-np.inf, -100.0, 0.0, 5.0], dtype=np.float32)
    np.testing.assert_allclose(edge_scores(logits, "mean"), [SCORE_FLOOR, SCORE_FLOOR, 0.0, 5.0])
    expected = [SCORE_FLOOR, SCORE_FLOOR, np.log(0.5), np.log(1 / (1 + np.exp(-5.0)))]  # log sigmoid
    np.testing.assert_allclose(edge_scores(logits, "product"), expected, rtol=1e-5)
    with pytest.raises(ValueError):
        edge_scores(logits, "max")


def test_neighbor_scores_average_or_sum():
    edges = np.zeros((4, 3, 3))
    edges[RIGHT, :, 1] = [2.0, 4.0, 6.0]  # candidates scored against the piece to their right
    edges[BELOW, :, 2] = [0.0, 0.0, 6.0]  # ... and against the piece below
    grid = np.full((2, 2), EMPTY)
    grid[0, 1], grid[1, 0] = 1, 2  # right of and below slot (0, 0)
    summed, count = neighbor_scores(edges, grid, 0, 0, average=False)
    averaged, _ = neighbor_scores(edges, grid, 0, 0, average=True)
    assert count == 2
    np.testing.assert_allclose(summed, [2.0, 4.0, 12.0])
    np.testing.assert_allclose(averaged, [1.0, 2.0, 6.0])


def test_eligibility_allows_each_true_piece_and_pins_the_corners(puzzle, types):
    allowed = eligibility(types)
    grid = true_grid(puzzle[2])
    assert allowed.shape == (ROWS, COLS, NUM_PIECES)
    for (y, x), piece_id in np.ndenumerate(grid):
        assert allowed[y, x, piece_id]
    for x, y in [(0, 0), (COLS - 1, 0), (0, ROWS - 1), (COLS - 1, ROWS - 1)]:
        assert allowed[y, x].sum() == 1  # each corner shape belongs to exactly one piece


def test_true_grid_and_metrics(puzzle):
    positions = puzzle[2]
    grid = true_grid(positions)
    assert_permutation(grid)
    assert placement_metrics(grid, positions) == {"direct": 1.0, "neighbors": 1.0, "perfect": 1.0}

    swapped = grid.copy()
    swapped[3, 3], swapped[6, 6] = grid[6, 6], grid[3, 3]
    metrics = placement_metrics(swapped, positions)
    assert metrics["direct"] == pytest.approx(1 - 2 / NUM_PIECES)
    assert metrics["neighbors"] < 1.0 and metrics["perfect"] == 0.0


def test_assemble_rebuilds_the_image(image, puzzle):
    pieces, masks, positions = puzzle
    np.testing.assert_array_equal(assemble(pieces, masks, true_grid(positions)), image)


# --------------------------------------------------------------- the stages


@pytest.fixture(scope="module")
def favorite_logits(types):
    """Every interior piece's best match on every side is the same interior piece, so max-pieces repeat."""
    favorite = int(np.nonzero((types != 0).all(axis=1))[0][0])  # first piece with no flat side
    logits = np.full((4, NUM_PIECES, NUM_PIECES), -10.0, dtype=np.float32)
    logits[:, :, favorite] = 10.0
    for relation in range(4):
        np.fill_diagonal(logits[relation], -np.inf)
    return logits, favorite


@pytest.mark.parametrize("combine", ["mean", "product"])
def test_first_updater_solves_with_oracle_logits(puzzle, types, logits, combine):
    result = first_updater(logits, types, combine)
    assert_permutation(result.grid)
    assert placement_metrics(result.grid, puzzle[2])["perfect"] == 1.0
    assert result.order[0] == (0, 0)
    assert sorted(result.order) == sorted((x, y) for y in range(ROWS) for x in range(COLS))
    np.testing.assert_allclose(result.probs.sum(axis=2), 1.0)
    assert (result.confidence > 0.99).all()


def test_first_updater_lets_a_piece_be_the_max_piece_of_several_spaces(types, favorite_logits):
    logits, favorite = favorite_logits
    result = first_updater(logits, types)
    assert (result.grid == favorite).sum() > 1
    assert (result.grid != EMPTY).all()


@pytest.mark.parametrize("combine", ["mean", "product"])
def test_multistage_updater_locks_a_correct_placement_by_threshold_3(puzzle, types, logits, combine):
    result = multistage_updater(logits, types, first_updater(logits, types, combine), combine=combine)
    assert placement_metrics(result.grid, puzzle[2])["perfect"] == 1.0
    assert result.locked.all()
    assert result.num_rounds == 1  # every max-piece is far above LOCK_PROB straight away


def test_multistage_updater_threshold_1_locks_sole_stable_max_pieces(puzzle, types, logits):
    """With threshold 3 off, unique max-pieces that stay put lock after STABLE_ROUNDS rounds."""
    start = first_updater(logits, types)
    result = multistage_updater(logits, types, start, lock_prob=1.01, stable_rounds=2)
    assert result.locked.all()
    assert result.num_rounds == 2
    assert placement_metrics(result.grid, puzzle[2])["perfect"] == 1.0


def test_multistage_updater_threshold_2_stops_when_nothing_moves(types, favorite_logits):
    """Repeated max-pieces never pass threshold 1, so only threshold 2 can end the rounds."""
    logits, favorite = favorite_logits
    result = multistage_updater(logits, types, first_updater(logits, types), lock_prob=1.01)
    assert result.num_rounds < MAX_ROUNDS
    assert not result.locked.all()
    assert (result.grid[~result.locked] == favorite).sum() > 1


@pytest.mark.parametrize("combine", ["mean", "product"])
def test_multistage_updater_repairs_swapped_pieces(puzzle, types, logits, combine):
    """Two interior pieces swapped: re-scoring against all 4 neighbors finds and fixes them."""
    grid = true_grid(puzzle[2])
    grid[3, 3], grid[6, 6] = grid[6, 6], grid[3, 3]
    start = SolveResult(grid, np.zeros((ROWS, COLS, NUM_PIECES)))
    result = multistage_updater(logits, types, start, combine=combine)
    assert placement_metrics(result.grid, puzzle[2])["perfect"] == 1.0


def test_final_assignment_uses_each_piece_once_and_keeps_locked_spaces(types, favorite_logits):
    logits, _ = favorite_logits
    refined = multistage_updater(logits, types, first_updater(logits, types))
    final = final_assignment(refined, types)
    assert_permutation(final.grid)
    np.testing.assert_array_equal(final.grid[refined.locked], refined.grid[refined.locked])
    assert eligibility(types)[np.arange(ROWS)[:, None], np.arange(COLS), final.grid].all()  # shapes respected


@pytest.mark.parametrize("combine", ["mean", "product"])
@pytest.mark.parametrize("seed", range(3))
def test_solve_ends_in_a_permutation_with_random_logits(types, seed, combine):
    logits = np.random.default_rng(seed).normal(0, 5, (4, NUM_PIECES, NUM_PIECES)).astype(np.float32)
    first, refined, final = solve(logits, types, combine)
    assert (first.grid != EMPTY).all() and (refined.grid != EMPTY).all()
    assert_permutation(final.grid)


def test_solve_trace_records_every_step(types, logits):
    trace = []
    _, refined, final = solve(logits, types, trace=trace)
    assert len(trace) == NUM_PIECES + refined.num_rounds + 1
    assert [snapshot.label for snapshot in trace[:2]] == ["stage 2", "stage 2"]
    assert (trace[0].grid != EMPTY).sum() == 1 and (trace[NUM_PIECES - 1].grid != EMPTY).all()
    assert not trace[NUM_PIECES - 1].locked.any()
    assert trace[NUM_PIECES].label == "stage 3, round 1"
    assert trace[-1].label == "stage 4" and trace[-1].locked.all()
    np.testing.assert_array_equal(trace[-1].grid, final.grid)
