import numpy as np

from interlock.data_collection.pairs import OFFSETS
from interlock.solver.bounded_growth import bounded_growth, priority, solve_bounded


def perfect_scores(rows=3, cols=3, wrong=-5.0, right=8.0):
    """Scores that rank every true neighbor first; piece ids are row-major, so positions are the truth."""
    positions = np.array([(x, y) for y in range(rows) for x in range(cols)])
    scores = np.full((4, len(positions), len(positions)), wrong)
    for relation, delta in OFFSETS.items():
        for i, point in enumerate(positions):
            for j, other in enumerate(positions):
                if np.array_equal(other - point, delta):
                    scores[relation, i, j] = right
    return scores, positions


def test_priority_prefers_a_clear_winner_over_a_higher_but_contested_score():
    clear, _ = priority(np.array([6.0, 0.0, -1.0]))
    contested, _ = priority(np.array([9.0, 8.8, 8.5]))
    assert clear > contested
    assert priority(np.array([1.0, 3.0, 2.0]))[1] == 1  # the chosen candidate is still the best-scoring one


def test_growth_never_leaves_the_board_size():
    scores, truth = perfect_scores(3, 3)
    growth = bounded_growth(scores)
    span = growth.coordinates.max(axis=0) - growth.coordinates.min(axis=0) + 1
    assert tuple(span) == (3, 3)
    assert len(np.unique(growth.coordinates, axis=0)) == 9 and sorted(growth.order) == list(range(9))


def test_bounded_growth_solves_a_clean_board_and_a_rectangle():
    for rows, cols in ((3, 3), (2, 4)):
        scores, truth = perfect_scores(rows, cols)
        grid, _ = solve_bounded(scores, rows, cols)
        np.testing.assert_array_equal(grid, np.arange(rows * cols).reshape(rows, cols))


def test_bound_stops_a_strong_false_join_from_sprawling():
    scores, truth = perfect_scores(3, 3)
    # piece 2 (top-right) strongly but wrongly "fits" right of piece 8 (bottom-right): unbounded growth could
    # put it in a 4th column; bounded growth can't, and the true layout still wins.
    scores[0, 8, 2] = scores[2, 2, 8] = 7.5
    grid, growth = solve_bounded(scores)
    span = growth.coordinates.max(axis=0) - growth.coordinates.min(axis=0) + 1
    assert tuple(span) == (3, 3)
    np.testing.assert_array_equal(grid, np.arange(9).reshape(3, 3))
