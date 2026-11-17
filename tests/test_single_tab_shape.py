"""The single-tab experiment reuses exactly one canonical curve, without mirrored variants."""
import sys
from pathlib import Path
import numpy as np
import pytest
from interlock.data_collection.cutting import cut, puzzle_edges, sample_curve, tab_control_points, TAB_NOMINAL, is_valid_tab
from interlock.data_collection.pairs import side_types, can_fit, neighbor_ids, WINDOW_SHAPE
from interlock.solver.updaters import assemble, true_grid


@pytest.mark.parametrize('seed', [0, 1, 7])
def test_all_internal_edges_use_canonical_tab(seed):
    canonical = sample_curve(tab_control_points(**TAB_NOMINAL))
    assert is_valid_tab(**TAB_NOMINAL)
    horizontal, vertical = puzzle_edges(240, 300, np.random.default_rng(seed), 4, 5, single_tab_shape=True)
    for edges in (horizontal[1:-1].reshape(-1, len(canonical), 2), vertical[:,1:-1].reshape(-1,len(canonical),2)):
        for edge in edges:
            direction = edge[-1] - edge[0]
            perpendicular = np.array([-direction[1], direction[0]])
            delta = edge - edge[0]
            u = delta @ direction / (direction @ direction)
            v = delta @ perpendicular / (direction @ direction)
            v *= 1 if v.sum() > 0 else -1
            np.testing.assert_allclose(np.column_stack([u,v]), canonical, atol=1e-12)


@pytest.mark.parametrize('cell', [30, 65, 100])
def test_single_tab_cut_reconstructs_and_preserves_valid_neighbors(cell):
    image = np.random.default_rng(5).integers(0,256,(4*cell,5*cell,3),dtype=np.uint8)
    pieces, masks, positions = cut(image, 3, 4, 5, single_tab_shape=True)
    np.testing.assert_array_equal(assemble(pieces, masks, true_grid(positions,4,5)), image)
    types = side_types(masks)
    for relation in range(4):
        fits = can_fit(types, relation)
        for a,b in neighbor_ids(positions,relation).items():
            assert fits[a,b]
    with pytest.raises(ValueError, match='mutually exclusive'):
        cut(image, 3, 4, 5, single_tab_shape=True, straight_edges=True)


def test_training_default_checkpoint_is_separate(monkeypatch):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'scripts'))
    import train_comparator
    monkeypatch.setattr(sys,'argv',['train_comparator.py','--single-tab-shape'])
    args = train_comparator.parse_args()
    assert args.single_tab_shape and args.out.name == 'edge_comparator_single_tab.pt'
    monkeypatch.setattr(sys,'argv',['train_comparator.py'])
    assert train_comparator.parse_args().out.name == 'edge_comparator.pt'
