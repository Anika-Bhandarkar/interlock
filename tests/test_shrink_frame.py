import numpy as np
import pytest

from interlock.data_collection.pairs import OFFSETS
from interlock.solver.shrink_frame import board_shape, shrink_frame
from interlock.solver.strongest_pair import FreeAssembly


def perfect_scores(rows=3, cols=3):
    """Scores that rank every true neighbor first; piece ids are row-major, so positions are the truth."""
    positions = np.array([(x, y) for y in range(rows) for x in range(cols)])
    scores = np.full((4, len(positions), len(positions)), -5.0)
    for relation, delta in OFFSETS.items():
        for i, point in enumerate(positions):
            for j, other in enumerate(positions):
                if np.array_equal(other - point, delta):
                    scores[relation, i, j] = 8.0
    return scores, positions


def assembly(coordinates):
    coordinates = np.array(coordinates)
    return FreeAssembly(coordinates, list(range(len(coordinates))), {"pieces": [0, 1], "relation": 0, "score": 0.0})


def assert_permutation(result, num_pieces):
    assert sorted(result.grid.ravel().tolist()) == list(range(num_pieces))
    assert len(np.unique(result.coordinates, axis=0)) == num_pieces


def test_already_square_assembly_is_kept():
    scores, truth = perfect_scores()
    result = shrink_frame(scores)
    assert_permutation(result, 9)
    assert result.num_evicted == 0 and result.num_swaps == 0
    np.testing.assert_array_equal(result.coordinates, truth)


def test_piece_outside_the_board_is_pulled_into_the_hole():
    scores, truth = perfect_scores()
    coordinates = truth.copy()
    coordinates[8] = (3, 0)  # belongs at (2, 2); growth left it sticking out on the right
    result = shrink_frame(scores, growth=assembly(coordinates), polish=False)
    assert_permutation(result, 9)
    assert result.steps[0].frame == (0, 0, 3, 2)
    assert result.steps[1].label == "Shrink right" and result.steps[1].moved == [8]
    np.testing.assert_array_equal(result.coordinates, truth)


def test_side_with_the_weakest_line_moves_first():
    scores, truth = perfect_scores()
    coordinates = truth + (1, 0)
    coordinates[3] = (0, 2)  # belongs at (0, 1) of the board; a stray piece on the far left
    coordinates[6] = (1, 1)  # moved up into piece 3's spot; (1, 2) is left empty
    result = shrink_frame(scores, growth=assembly(coordinates), polish=False)
    assert result.steps[1].label == "Shrink left"  # one weak piece on the left, three strong ones on the right
    assert_permutation(result, 9)


def test_frame_narrower_than_the_board_is_widened_first():
    scores, truth = perfect_scores(2, 2)
    result = shrink_frame(scores, growth=assembly([(0, 0), (1, 0), (2, 0), (3, 0)]), polish=False)
    assert result.steps[0].frame == (0, 0, 3, 1)  # 4 wide, widened from 1 to 2 tall
    assert_permutation(result, 4)
    assert result.grid.shape == (2, 2)


def test_swap_polish_fixes_wrong_pieces_inside_the_frame():
    scores, truth = perfect_scores()
    coordinates = truth.copy()
    coordinates[[0, 8]] = coordinates[[8, 0]]
    unpolished = shrink_frame(scores, growth=assembly(coordinates), polish=False)
    np.testing.assert_array_equal(unpolished.coordinates, coordinates)
    polished = shrink_frame(scores, growth=assembly(coordinates))
    assert polished.num_swaps >= 1 and polished.steps[-1].label == "Swap polish"
    np.testing.assert_array_equal(polished.coordinates, truth)
    np.testing.assert_array_equal(polished.before_polish, coordinates)


def test_impossible_pairs_and_rectangular_boards():
    scores, truth = perfect_scores(2, 3)
    scores[scores == -5.0] = -np.inf  # e.g. tab against tab
    result = shrink_frame(scores, rows=2, cols=3)
    assert_permutation(result, 6)
    np.testing.assert_array_equal(result.coordinates, truth)


@pytest.mark.parametrize("args", [(10, None, None), (9, 2, None), (9, 2, 4)])
def test_board_shape_rejects_sizes_that_do_not_match(args):
    with pytest.raises(ValueError):
        board_shape(*args)

