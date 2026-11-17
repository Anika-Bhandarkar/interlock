"""Render a cut puzzle for sanity checks: tab shapes, edges over the image, and the cut pieces.

Usage: uv run python scripts/visualize.py [--image PATH] [--seed N] [--out PATH]
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from interlock.data_collection.cutting import (
    COLS,
    ROWS,
    cut,
    get_corners,
    puzzle_edges,
    sample_curve,
    tab_shapes,
    tab_control_points,
    TAB_NOMINAL,
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


def plot_tab_shapes(seed, ax=None):
    """Overlay the puzzle's set of tab shapes on the standard edge (0, 0) -> (1, 0), regular to funky.

    tab_shapes is the first thing cut draws from its rng, so these are the shapes cut(image, seed) uses.
    """
    ax = ax or plt.gca()
    shapes = tab_shapes(np.random.default_rng(seed))
    for index, shape in enumerate(shapes):
        curve_points = sample_curve(shape)
        ax.plot(curve_points[:, 0], curve_points[:, 1], linewidth=1, label=f"shape {index}")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="lower center", ncol=3)
    ax.set_title(f"{len(shapes)} tab shapes (seed {seed}), regular to funky")


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
    ax.set_xlim(0, (positions[:, 0].max() + 1) * spacing)
    ax.set_ylim((positions[:, 1].max() + 1) * spacing, 0)
    ax.set_aspect("equal")
    ax.set_title("Pieces")
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", help="image to cut (default: synthetic gradient)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="save the figure here instead of showing it")
    parser.add_argument("--straight-edges", action="store_true", help="cut square grid tiles without tabs")
    parser.add_argument("--single-tab-shape", action="store_true", help="use one fixed symmetric tab shape across all puzzles")
    args = parser.parse_args()
    if args.straight_edges and args.single_tab_shape:
        parser.error("--straight-edges and --single-tab-shape cannot be combined")

    image = load_image(args.image)
    # cut draws its edges with puzzle_edges, so with the same seed these edges match the pieces
    if args.straight_edges:
        corners = get_corners(*image.shape[:2])
        horizontal_edges = np.stack([corners[:, :-1], corners[:, 1:]], axis=2)
        vertical_edges = np.stack([corners[:-1], corners[1:]], axis=2)
    else:
        horizontal_edges, vertical_edges = puzzle_edges(*image.shape[:2], np.random.default_rng(args.seed), single_tab_shape=args.single_tab_shape)
    pieces, masks, positions = cut(image, args.seed, straight_edges=args.straight_edges, single_tab_shape=args.single_tab_shape)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), layout="constrained")
    if args.straight_edges:
        axes[0].imshow(image)
        axes[0].set_title("Original (straight grid cuts)")
        axes[0].axis("off")
    elif args.single_tab_shape:
        curve = sample_curve(tab_control_points(**TAB_NOMINAL))
        axes[0].plot(curve[:, 0], curve[:, 1])
        axes[0].set_aspect("equal")
        axes[0].set_title("Fixed symmetric tab shape")
    else:
        plot_tab_shapes(args.seed, ax=axes[0])
    plot_edges(image, horizontal_edges, vertical_edges, ax=axes[1])
    plot_pieces(pieces, masks, positions, ax=axes[2])

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"Saved {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
