"""Render a cut puzzle for sanity checks: tab shapes, edges over the image, and the cut pieces.

Usage: uv run python scripts/visualize.py [--image PATH] [--seed N] [--out PATH]
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from interlock.data_collection import cutting
from interlock.data_collection.cutting import (
    COLS,
    ROWS,
    get_corners,
    place_edge,
    sample_curve,
    tab_control_points,
)

DEFAULT_SIZE = 400


def load_image(path):
    """Load an RGB image, or make a colorful gradient if no path is given."""
    if path is None:
        y_coords, x_coords = np.mgrid[0:DEFAULT_SIZE, 0:DEFAULT_SIZE] / DEFAULT_SIZE
        stripes = 0.5 + 0.5 * np.sin(12 * np.pi * (x_coords + y_coords))
        return (np.stack([x_coords, y_coords, stripes], axis=2) * 255).astype(np.uint8)
    image = Image.open(path).convert("RGB")
    if image.height % ROWS or image.width % COLS:
        print(f"Note: {image.width}x{image.height} isn't divisible by {COLS}x{ROWS}; "
              f"resizing to {DEFAULT_SIZE}x{DEFAULT_SIZE} for display.")
        image = image.resize((DEFAULT_SIZE, DEFAULT_SIZE))
    return np.asarray(image)


def plot_tab_gallery(num_tabs, seed, ax=None):
    """Overlay num_tabs random tabs on the standard edge (0, 0) -> (1, 0)."""
    ax = ax or plt.gca()
    rng = np.random.default_rng(seed)
    for _ in range(num_tabs):
        curve_points = sample_curve(tab_control_points(rng))
        ax.plot(curve_points[:, 0], curve_points[:, 1], linewidth=0.8, alpha=0.6)
    ax.set_aspect("equal")
    ax.set_title(f"{num_tabs} random tabs (seed {seed})")


def plot_edges(image, horizontal_edges, vertical_edges, ax=None):
    """Draw every edge curve over the image."""
    ax = ax or plt.gca()
    ax.imshow(image)
    for edges in (horizontal_edges, vertical_edges):
        for edge_points in edges.reshape(-1, *edges.shape[-2:]):
            ax.plot(edge_points[:, 0], edge_points[:, 1], color="black", linewidth=1)
    ax.set_title("Edges")
    ax.axis("off")


def plot_pieces(pieces, masks, positions, ax=None, gap_fraction=0.15):
    """Lay pieces out at their (x, y) grid positions with gaps; background outside each mask is transparent."""
    ax = ax or plt.gca()
    box_size = pieces.shape[1]
    spacing = box_size * (1 + gap_fraction)
    for piece, mask, (x, y) in zip(pieces, masks, positions):
        rgba_piece = np.dstack([piece, mask.astype(np.uint8) * 255])
        left, top = x * spacing, y * spacing
        ax.imshow(rgba_piece, extent=(left, left + box_size, top + box_size, top))
    ax.set_xlim(0, COLS * spacing)
    ax.set_ylim(ROWS * spacing, 0)
    ax.set_aspect("equal")
    ax.set_title("Pieces")
    ax.axis("off")


def fallback_edges(corners, rng):
    """Build edges from the existing helpers. DELETE once cutting.get_edges is implemented."""
    def make_edge(start, end, is_border):
        sign = 0 if is_border else rng.choice([-1, 1])
        return place_edge(sample_curve(tab_control_points(rng)), start, end, sign)

    horizontal_edges = np.array([
        [make_edge(corners[row, col], corners[row, col + 1], row in (0, ROWS)) for col in range(COLS)]
        for row in range(ROWS + 1)
    ])
    vertical_edges = np.array([
        [make_edge(corners[row, col], corners[row + 1, col], col in (0, COLS)) for col in range(COLS + 1)]
        for row in range(ROWS)
    ])
    return horizontal_edges, vertical_edges


def build_edges(image, seed):
    """Edges from cutting.get_edges if it's implemented, otherwise from fallback_edges."""
    corners = get_corners(*image.shape[:2])
    edges = cutting.get_edges(corners, np.random.default_rng(seed)) if hasattr(cutting, "get_edges") else None
    if isinstance(edges, tuple) and len(edges) == 2:
        return tuple(np.asarray(part) for part in edges)
    print("cutting.get_edges not implemented yet; using fallback_edges.")
    return fallback_edges(corners, np.random.default_rng(seed))


def try_cut(image, seed):
    """(pieces, masks, positions) as arrays, or None if cut isn't implemented yet."""
    result = cutting.cut(image, seed)
    if isinstance(result, tuple) and len(result) == 3 and len(result[0]) > 0:
        return tuple(np.asarray(part) for part in result)
    print("cutting.cut not implemented yet; skipping pieces plot.")
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", help="image to cut (default: synthetic gradient)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="save the figure here instead of showing it")
    args = parser.parse_args()

    image = load_image(args.image)
    horizontal_edges, vertical_edges = build_edges(image, args.seed)
    cut_result = try_cut(image, args.seed)

    num_panels = 3 if cut_result else 2
    fig, axes = plt.subplots(1, num_panels, figsize=(6 * num_panels, 6), layout="constrained")
    plot_tab_gallery(50, args.seed, ax=axes[0])
    plot_edges(image, horizontal_edges, vertical_edges, ax=axes[1])
    if cut_result:
        plot_pieces(*cut_result, ax=axes[2])

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"Saved {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
