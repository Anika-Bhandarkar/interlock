"""Show the solver at work on one puzzle: shuffled pieces, stage 2 fill order, and each stage's result.

Top row: original image, shuffled pieces, the order stage 2 filled the slots in.
Bottom row: stage 2's max-pieces and the final (stage 4) placement, wrong slots outlined in red, and the
probability of each final piece (slots stage 3 locked are marked with a dot). Use --noise to weaken the
comparator and see mistakes get made.

Usage: uv run python scripts/visualize_solve.py [--index 0 | --image PATH] [--combine mean] [--noise 0] [--seed 0] [--out PATH]
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from interlock.data_collection.cutting import COLS, ROWS, cut
from interlock.data_collection.dataset import PuzzleDataset, load_manifest
from interlock.data_collection.pairs import cell_size
from interlock.models.edge_comparator import load_comparator, pick_device
from interlock.solver.updaters import COMBINE_MODES, assemble, placement_metrics, shuffle_puzzle, true_grid
from solve import DEFAULT_CHECKPOINT, solve_puzzle  # scripts/ is on sys.path when running a script in it
from visualize import load_image, plot_pieces


def plot_assembly(pieces, masks, grid, positions, title, ax=None):
    """The placement as an image, with slots holding the wrong piece outlined in red."""
    ax = ax or plt.gca()
    cell = cell_size(pieces.shape[1])
    ax.imshow(assemble(pieces, masks, grid))
    wrong = grid != true_grid(positions, *grid.shape)
    for y, x in zip(*np.nonzero(wrong)):
        ax.add_patch(Rectangle((x * cell, y * cell), cell, cell, fill=False, edgecolor="red", linewidth=1.5))
    metrics = placement_metrics(grid, positions)
    ax.set_title(f"{title}: {metrics['direct']:.0%} placed right, {metrics['neighbors']:.0%} neighbors right")
    ax.axis("off")


def plot_fill_order(order, shape, ax=None):
    """Heatmap of when each slot was filled (0 = first), with the step number written in each slot."""
    ax = ax or plt.gca()
    steps = np.zeros(shape, dtype=int)
    for step, (x, y) in enumerate(order):
        steps[y, x] = step
    ax.imshow(steps, cmap="viridis")
    for (y, x), step in np.ndenumerate(steps):
        ax.text(x, y, step, ha="center", va="center", fontsize=6, color="white" if step < len(order) / 2 else "black")
    ax.set_title("Stage 2 fill order")
    ax.axis("off")


def plot_confidence(confidence, locked, ax=None):
    """Heatmap of the probability of each placed piece; dots mark slots locked by stage 3."""
    ax = ax or plt.gca()
    image = ax.imshow(confidence, cmap="magma", vmin=0, vmax=1)
    if locked is not None:
        ys, xs = np.nonzero(locked)
        ax.scatter(xs, ys, s=6, color="cyan")
    ax.figure.colorbar(image, ax=ax, fraction=0.046)
    ax.set_title("Final confidence (dots = locked by stage 3)")
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--index", type=int, default=0, help="val puzzle to solve")
    parser.add_argument("--image", help="solve this image instead of a val puzzle (cut with --seed)")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--noise", type=float, default=0.0, help="std of Gaussian noise added to the logits")
    parser.add_argument("--combine", default="mean", choices=COMBINE_MODES, help="how neighbors' scores combine")
    parser.add_argument("--device", default="auto", help="auto, mps, cuda or cpu")
    parser.add_argument("--seed", type=int, default=0, help="seed for cutting --image, shuffling and --noise")
    parser.add_argument("--out", help="save the figure here instead of showing it")
    args = parser.parse_args()

    if args.image:
        image = load_image(args.image)
        puzzle = cut(image, args.seed)
    else:
        puzzles = PuzzleDataset(load_manifest("val"))
        image, puzzle = puzzles.load_image(args.index), puzzles[args.index]
    rng = np.random.default_rng(args.seed)
    pieces, masks, positions = shuffle_puzzle(*puzzle, rng)

    model = load_comparator(args.checkpoint, pick_device(args.device))
    _, (first, _, final) = solve_puzzle(model, pieces, masks, args.noise, rng, args.combine)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), layout="constrained")
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")
    shuffled_layout = np.array([(index % COLS, index // COLS) for index in range(ROWS * COLS)])
    plot_pieces(pieces, masks, shuffled_layout, ax=axes[0, 1])
    axes[0, 1].set_title("Shuffled pieces")
    plot_fill_order(first.order, first.grid.shape, ax=axes[0, 2])
    plot_assembly(pieces, masks, first.grid, positions, "Stage 2", ax=axes[1, 0])
    plot_assembly(pieces, masks, final.grid, positions, "Final", ax=axes[1, 1])
    plot_confidence(final.confidence, final.locked, ax=axes[1, 2])

    if args.out:
        fig.savefig(args.out, dpi=120)
        print(f"Saved {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
