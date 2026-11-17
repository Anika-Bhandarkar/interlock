"""Smoke tests for scripts/solve.py, scripts/visualize_solve.py and scripts/animate_solve.py."""

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: must happen before pyplot is imported (here or in the scripts)

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from PIL import Image

from interlock.data_collection.cutting import COLS, ROWS, cut
from interlock.data_collection.pairs import ABOVE, BELOW, LEFT, RIGHT, side_types
from interlock.models.edge_comparator import EdgeComparator, save_comparator
from interlock.solver.updaters import EMPTY, Snapshot, first_updater, shuffle_puzzle, true_grid

# scripts/ isn't a package, and visualize_solve imports its neighbors by name, as it does when run directly
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solve = load_script("solve")
visualize_solve = load_script("visualize_solve")
animate_solve = load_script("animate_solve")


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def puzzle():
    image = np.random.default_rng(7).integers(0, 256, size=(400, 400, 3), dtype=np.uint8)
    return shuffle_puzzle(*cut(image, seed=3), np.random.default_rng(0))


def test_add_noise_keeps_transposes_and_misfits():
    logits = np.zeros((4, 5, 5), dtype=np.float32)
    logits[:, 0, 1] = -np.inf
    noisy = solve.add_noise(logits, 3.0, np.random.default_rng(0))
    np.testing.assert_array_equal(noisy[LEFT], noisy[RIGHT].T)
    np.testing.assert_array_equal(noisy[ABOVE], noisy[BELOW].T)
    assert np.isneginf(noisy[RIGHT, 0, 1]) and np.isneginf(noisy[BELOW, 0, 1])
    assert noisy[RIGHT][np.isfinite(noisy[RIGHT])].std() > 1.0


def test_plot_assembly_outlines_only_wrong_slots(puzzle):
    pieces, masks, positions = puzzle
    grid = true_grid(positions)
    grid[3, 3], grid[6, 6] = grid[6, 6], grid[3, 3]
    _, ax = plt.subplots()
    visualize_solve.plot_assembly(pieces, masks, grid, positions, "test", ax=ax)
    assert len(ax.images) == 1
    assert len(ax.patches) == 2


def test_plot_fill_order_and_confidence(puzzle):
    logits = np.random.default_rng(0).normal(0, 5, (4, ROWS * COLS, ROWS * COLS)).astype(np.float32)
    result = first_updater(logits, side_types(puzzle[1]))
    fig, axes = plt.subplots(1, 2)
    visualize_solve.plot_fill_order(result.order, result.grid.shape, ax=axes[0])
    visualize_solve.plot_confidence(result.confidence, np.ones((ROWS, COLS), dtype=bool), ax=axes[1])
    assert len(axes[0].texts) == ROWS * COLS
    assert len(axes[1].collections) == 1  # the locked-slot dots


@pytest.fixture
def image_and_checkpoint(tmp_path):
    image_path, checkpoint_path = tmp_path / "image.png", tmp_path / "model.pt"
    Image.fromarray(np.random.default_rng(1).integers(0, 256, (400, 400, 3), dtype=np.uint8)).save(image_path)
    torch.manual_seed(0)
    save_comparator(EdgeComparator(width=4), checkpoint_path)  # untrained: this checks plumbing only
    return ["--image", str(image_path), "--checkpoint", str(checkpoint_path), "--device", "cpu"]


def test_main_saves_figure(tmp_path, monkeypatch, image_and_checkpoint):
    out_path = tmp_path / "solve.png"
    monkeypatch.setattr(sys, "argv", ["visualize_solve.py", *image_and_checkpoint, "--out", str(out_path)])
    visualize_solve.main()
    assert out_path.exists() and out_path.stat().st_size > 0


# --------------------------------------------------------------- animation


def test_board_image_greys_pending_pieces_and_colors_set_ones(puzzle):
    pieces, masks, positions = puzzle
    grid = true_grid(positions)
    board_set = animate_solve.board_image(pieces, masks, grid, np.ones(grid.shape, dtype=bool))
    board_pending = animate_solve.board_image(pieces, masks, grid, np.zeros(grid.shape, dtype=bool))
    board_empty = animate_solve.board_image(pieces, masks, np.full(grid.shape, EMPTY), np.zeros(grid.shape, dtype=bool))
    assert board_set.shape == board_pending.shape == board_empty.shape == (400, 400, 3)
    np.testing.assert_array_equal(board_set, visualize_solve.assemble(pieces, masks, grid))  # full color
    expected_grey = animate_solve.blend(board_set, animate_solve.OVERLAY_GREY, animate_solve.OVERLAY_ALPHA)
    np.testing.assert_array_equal(board_pending, expected_grey)
    assert (board_empty >= animate_solve.GRID_LINE).all()  # just background and grid lines


def test_pile_image_fades_set_pieces(puzzle):
    pieces, masks, _ = puzzle
    is_set = np.zeros(len(pieces), dtype=bool)
    fresh = animate_solve.pile_image(pieces, masks, is_set)
    is_set[0] = True
    faded = animate_solve.pile_image(pieces, masks, is_set)
    box = pieces.shape[1]
    assert faded[:box, :box].mean() > fresh[:box, :box].mean()  # piece 0 moved toward the white background
    np.testing.assert_array_equal(faded[box:], fresh[box:])  # other rows untouched


def test_frame_plan_lingers_on_stage_3_and_4():
    blank = np.zeros((2, 2), dtype=int)
    trace = [Snapshot(label, blank, blank.astype(bool)) for label in ("stage 2", "stage 3, round 1", "stage 4")]
    plan = animate_solve.frame_plan(trace)
    hold = animate_solve.HOLD_FRAMES
    assert plan.count(-1) == hold["start"]
    assert [plan.count(index) for index in range(3)] == [1, hold["stage 3"], hold["stage 4"]]
    assert plan == sorted(plan)


def test_animate_main_saves_gif(tmp_path, monkeypatch, image_and_checkpoint):
    out_path = tmp_path / "solve.gif"
    monkeypatch.setattr(sys, "argv", ["animate_solve.py", *image_and_checkpoint, "--dpi", "20", "--out", str(out_path)])
    animate_solve.main()
    assert Image.open(out_path).n_frames > 1


@pytest.mark.parametrize('rows,cols,cell', [(4, 4, 100), (4, 7, 30)])
def test_variable_grid_script_plumbing(monkeypatch, rows, cols, cell):
    image = np.random.default_rng(4).integers(0, 256, (rows * cell, cols * cell, 3), dtype=np.uint8)
    pieces, masks, positions = shuffle_puzzle(*cut(image, 3, rows, cols), np.random.default_rng(2))
    n = len(pieces)
    logits = np.full((4, n, n), -10.0, dtype=np.float32)
    from interlock.data_collection.pairs import neighbor_ids
    for relation in range(4):
        for a, b in neighbor_ids(positions, relation).items():
            logits[relation, a, b] = 10
    monkeypatch.setattr(solve, 'score_matrix', lambda *args, **kwargs: logits)
    _, (_, _, final) = solve.solve_puzzle(None, pieces, masks, rows=rows, cols=cols)
    np.testing.assert_array_equal(animate_solve.board_image(pieces, masks, final.grid,
                                  np.ones((rows, cols), dtype=bool)), image)
    pile = animate_solve.pile_image(pieces, masks, np.zeros(n, dtype=bool), cols=cols)
    step = pieces.shape[1] + animate_solve.PILE_GAP
    assert pile.shape == (rows * step, cols * step, 3)
    _, ax = plt.subplots()
    visualize_solve.plot_assembly(pieces, masks, final.grid, positions, 'variable', ax=ax)
    assert not ax.patches


def test_solve_puzzle_rejects_mismatched_piece_masks_before_scoring(monkeypatch):
    monkeypatch.setattr(solve, 'score_matrix', lambda *args, **kwargs: pytest.fail('must validate before scoring'))
    with pytest.raises(ValueError, match='matching'):
        solve.solve_puzzle(None, np.zeros((16, 184, 184, 3)), np.zeros((15, 184, 184)), rows=4, cols=4)


@pytest.mark.parametrize("straight_edges", [False, True])
def test_solve_main_uses_dataset_dimensions_and_reports_sizes(monkeypatch, capsys, straight_edges):
    image = np.zeros((120, 210, 3), dtype=np.uint8)
    puzzle = cut(image, 3, 4, 7, straight_edges=straight_edges)
    monkeypatch.setattr(solve, 'load_manifest', lambda split: [])
    monkeypatch.setattr(solve, 'PuzzleDataset', lambda rows, **kwargs: [puzzle])
    monkeypatch.setattr(solve, 'load_comparator', lambda *args: None)
    def scores(model, pieces, masks, **kwargs):
        assert kwargs['straight_edges'] == straight_edges
        n = len(pieces)
        return np.zeros((4, n, n), dtype=np.float32)
    monkeypatch.setattr(solve, 'score_matrix', scores)
    monkeypatch.setattr(sys, 'argv', ['solve.py', '--num-puzzles', '1', '--device', 'cpu'] + (['--straight-edges'] if straight_edges else []) + (['--anchor-top-left'] if straight_edges else []))
    solve.main()
    output = capsys.readouterr().out
    assert 'cell 30px | 4x7 (28 pieces)' in output
    assert 'stage 4:' in output
    assert 'CNN overall: top1 0.000 | mean_rank 28.00 | queries 45' in output
    assert 'CNN this puzzle top1' in output
    assert 'CNN top1 0.000 rank 28.00 queries 45' in output
    assert ('anchor true-top-left' if straight_edges else 'anchor none') in output


@pytest.mark.parametrize('name', ['visualize_solve', 'animate_solve'])
@pytest.mark.parametrize('straight_edges', [False, True])
def test_visualizer_main_passes_dataset_dimensions(name, monkeypatch, tmp_path, straight_edges):
    module = globals()[name]
    puzzle = cut(np.zeros((120, 210, 3), dtype=np.uint8), 3, 4, 7, straight_edges=straight_edges)
    requested_mode = straight_edges
    monkeypatch.setattr(module, 'load_manifest', lambda split: [])
    monkeypatch.setattr(module, 'PuzzleDataset', lambda rows, **kwargs: [puzzle])
    monkeypatch.setattr(module, 'load_comparator', lambda *args: None)
    from interlock.solver.updaters import solve as solve_board
    def fake_solve(model, pieces, masks, noise, rng, combine, trace=None, *, rows, cols, straight_edges=False, anchor_piece=None):
        assert straight_edges == requested_mode
        assert (anchor_piece is not None) == requested_mode
        assert (rows, cols) == (4, 7)
        n = len(pieces)
        logits = np.zeros((4, n, n), dtype=np.float32)
        return logits, solve_board(logits, side_types(masks), combine, trace, rows=rows, cols=cols, straight_edges=straight_edges, anchor_piece=anchor_piece)
    monkeypatch.setattr(module, 'solve_puzzle', fake_solve)
    out = tmp_path / ('solve.gif' if name == 'animate_solve' else 'solve.png')
    args = [name + '.py', '--device', 'cpu', '--out', str(out)]
    if straight_edges:
        args += ['--straight-edges', '--anchor-top-left']
    if name == 'animate_solve':
        args += ['--dpi', '10']
    monkeypatch.setattr(sys, 'argv', args)
    module.main()
    assert out.exists()



def test_ranking_summary_pools_queries_not_puzzle_percentages():
    ranks = [dict(right=np.array([1]), below=np.array([2])),
             dict(right=np.array([1, 2, 3]), below=np.array([1, 1]))]
    metrics = solve.summarize_rankings(ranks)
    assert metrics['queries'] == 7
    assert metrics['top1'] == pytest.approx(4/7)
    assert metrics['mean_rank'] == pytest.approx(11/7)
    assert solve.summarize_rankings(ranks, 'right')['top1'] == .5
