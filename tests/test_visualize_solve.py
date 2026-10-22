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
