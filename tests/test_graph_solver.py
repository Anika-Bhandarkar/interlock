import numpy as np
import pytest

from interlock.solver.graph import (SideKind, build_candidate_graph, classify_sides,
                                    find_core_box, neighbor_accuracy, solve)
from interlock.solver.graph.geometry import can_connect, validity_tensor
from interlock.solver.graph.synthetic import make_puzzle, piece_mask

T, B, F = SideKind.TAB, SideKind.BLANK, SideKind.FLAT


@pytest.mark.parametrize("sides", [(F, T, B, F), (T, B, T, B), (B, B, T, F), (F, F, T, B)])
def test_classify_sides_from_mask(sides):
    m = piece_mask(sides, tile=48)
    assert classify_sides(m, find_core_box(m)) == sides


def test_validity_tensor_matches_scalar_rule():
    pieces, _, _ = make_puzzle(3, 4, tile=8, seed=1)
    V = validity_tensor(pieces)
    for pi in pieces:
        for pj in pieces:
            for a in range(4):
                for b in range(4):
                    assert V[pi.id, a, pj.id, b] == can_connect(pi, a, pj, b)


def test_true_neighbour_survives_pruning():
    pieces, truth, _ = make_puzzle(5, 6, tile=24, seed=2)
    g = build_candidate_graph(pieces, keep_frac=0.5)
    cells = {(r, c): p for p, (r, c, _) in truth.items()}
    from interlock.solver.graph.pieces import DIR_DELTA, side_facing
    missing = 0
    for p, (r, c, rot) in truth.items():
        for d, (dr, dc) in DIR_DELTA.items():
            q = cells.get((r + dr, c + dc))
            if q is None:
                continue
            a = side_facing(d, rot)
            b = side_facing((d + 2) % 4, truth[q][2])
            missing += (q, b) not in g.ranked[(p, a)]
    assert missing == 0


@pytest.mark.parametrize("rows,cols,seed", [(4, 4, 0), (6, 8, 3), (8, 10, 7)])
def test_solves_synthetic(rows, cols, seed):
    pieces, truth, _ = make_puzzle(rows, cols, tile=24, seed=seed)
    res = solve(pieces, rows, cols)
    assert (res.grid >= 0).all()
    assert neighbor_accuracy(res, truth) >= 0.95
