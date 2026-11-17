"""Straight tiles retain pixel scale and bypass only the jigsaw-specific constraints."""
import numpy as np
import pytest
import torch
from interlock.data_collection.cutting import PAD, cut
from interlock.data_collection.pairs import can_fit, side_types, RELATIONS, RIGHT, BELOW, LEFT, ABOVE
from interlock.models.edge_comparator import EdgeComparator, score_matrix
from interlock.solver.updaters import assemble, true_grid, eligibility, shuffle_puzzle, solve


@pytest.mark.parametrize('rows,cols,cell', [(4, 4, 30), (4, 7, 55), (4, 4, 100)])
def test_straight_cut_partitions_image_into_exact_cells(rows, cols, cell):
    image = np.random.default_rng(1).integers(0, 256, (rows*cell, cols*cell, 3), dtype=np.uint8)
    pieces, masks, positions = cut(image, 3, rows, cols, straight_edges=True)
    assert pieces.shape == (rows*cols, cell+2*PAD, cell+2*PAD, 3)
    assert (masks.sum(axis=(1,2)) == cell*cell).all()
    assert (side_types(masks) == 0).all()
    for i, (x,y) in enumerate(positions):
        np.testing.assert_array_equal(pieces[i, PAD:PAD+cell, PAD:PAD+cell], image[y*cell:(y+1)*cell, x*cell:(x+1)*cell])
    np.testing.assert_array_equal(assemble(pieces, masks, true_grid(positions, rows, cols)), image)
    with pytest.raises(ValueError):
        cut(image[:-1], 3, rows, cols, straight_edges=True)


def test_flat_matching_is_opt_in_and_never_allows_self_pairs():
    types = np.zeros((4,4), dtype=np.int8)
    types[0, RIGHT], types[1, LEFT] = 1, -1
    for relation in range(4):
        regular = can_fit(types, relation)
        straight = can_fit(types, relation, straight_edges=True)
        expected = regular | np.outer(types[:,relation] == 0, types[:,(relation+2)%4] == 0)
        np.fill_diagonal(expected, False)
        np.testing.assert_array_equal(straight, expected)
    assert not can_fit(np.zeros((4,4)), RIGHT).any()


def test_straight_scoring_reaches_cnn_for_all_distinct_pairs():
    pieces, masks, _ = cut(np.zeros((60,90,3), dtype=np.uint8), 0, 2, 3, straight_edges=True)
    model = EdgeComparator(width=1).eval()
    scored = score_matrix(model, pieces, masks, batch_size=7, straight_edges=True)
    diagonal = np.eye(6, dtype=bool)
    assert np.isneginf(scored[:,diagonal]).all()
    assert np.isfinite(scored[:,~diagonal]).all()
    np.testing.assert_array_equal(scored[LEFT], scored[RIGHT].T)
    np.testing.assert_array_equal(scored[ABOVE], scored[BELOW].T)
    assert np.isneginf(score_matrix(model, pieces, masks)).all()


@pytest.mark.parametrize('combine', ['mean', 'product'])
def test_straight_solver_retains_permutation_without_border_clues(combine):
    rows, cols = 4, 7
    n = rows*cols
    types = np.zeros((n,4), dtype=np.int8)
    assert eligibility(types, rows, cols, straight_edges=True).all()
    logits = np.random.default_rng(3).normal(0, 2, (4,n,n)).astype(np.float32)
    for relation in range(4):
        np.fill_diagonal(logits[relation], -np.inf)
    trace=[]
    *_, final = solve(logits, types, combine, trace, rows=rows, cols=cols, straight_edges=True)
    assert final.grid.shape == (rows,cols)
    assert sorted(final.grid.ravel()) == list(range(n))
    assert trace[-1].locked.all()
