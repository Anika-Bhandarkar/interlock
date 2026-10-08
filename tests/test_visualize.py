"""Smoke tests for scripts/visualize.py: each plot draws without error and main saves a figure."""

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: must happen before pyplot is imported (here or in visualize)

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from interlock.data_collection.cutting import COLS, ROWS, get_corners, get_edges

# scripts/ isn't a package, so load visualize.py straight from its file path
VISUALIZE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "visualize.py"
spec = importlib.util.spec_from_file_location("visualize", VISUALIZE_PATH)
visualize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visualize)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_tab_gallery_draws_one_line_per_tab():
    _, ax = plt.subplots()
    visualize.plot_tab_gallery(7, seed=0, ax=ax)
    assert len(ax.lines) == 7


def test_plot_edges_draws_every_edge():
    image = np.zeros((4 * ROWS, 4 * COLS, 3), dtype=np.uint8)
    corners = get_corners(*image.shape[:2])
    horizontal_edges, vertical_edges = get_edges(corners, np.random.default_rng(0))
    _, ax = plt.subplots()
    visualize.plot_edges(image, horizontal_edges, vertical_edges, ax=ax)
    assert len(ax.lines) == (ROWS + 1) * COLS + ROWS * (COLS + 1)


def test_plot_pieces_on_synthetic_pieces():
    """Hand-built pieces, so this doesn't depend on cutting.cut."""
    box_size = 6
    pieces = np.full((2, box_size, box_size, 3), 200, dtype=np.uint8)
    masks = np.zeros((2, box_size, box_size), dtype=bool)
    masks[:, 1:-1, 1:-1] = True
    positions = np.array([(0, 0), (COLS - 1, ROWS - 1)])  # (x, y)
    _, ax = plt.subplots()
    visualize.plot_pieces(pieces, masks, positions, ax=ax)
    assert len(ax.images) == 2


def test_load_image_resizes_non_divisible(tmp_path):
    image_path = tmp_path / "odd.png"
    Image.new("RGB", (33, 21), color=(10, 20, 30)).save(image_path)
    image = visualize.load_image(image_path)
    assert image.dtype == np.uint8
    assert image.ndim == 3 and image.shape[2] == 3
    assert image.shape[0] % ROWS == 0 and image.shape[1] % COLS == 0


def test_main_saves_figure(tmp_path, monkeypatch):
    out_path = tmp_path / "viz.png"
    monkeypatch.setattr(sys, "argv", ["visualize.py", "--seed", "1", "--out", str(out_path)])
    visualize.main()
    assert out_path.exists() and out_path.stat().st_size > 0
