"""Animate a solve: the shuffled pile on the left, the board on the right filling in step by step.

Each space shows its current max-piece under a translucent grey overlay as soon as it has a distribution
(stage 2 fills them one at a time; stage 3 rounds may swap them). Once a piece is set (locked by a stage 3
threshold, or placed by the stage 4 final assignment) it turns full color and fades out of the pile.

Usage: uv run python scripts/animate_solve.py [--index 0 | --image PATH] [--noise 0] [--out solve.gif]
Without --out the animation plays in a window.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from interlock.data_collection.cutting import COLS, PAD, ROWS, cut
from interlock.data_collection.dataset import PuzzleDataset, load_manifest
from interlock.data_collection.pairs import cell_size
from interlock.models.edge_comparator import load_comparator, pick_device
from interlock.solver.updaters import COMBINE_MODES, EMPTY, Snapshot, placement_metrics, shuffle_puzzle
from solve import DEFAULT_CHECKPOINT, solve_puzzle  # scripts/ is on sys.path when running a script in it
from visualize import load_image

BACKGROUND = 255  # white
GRID_LINE = 225
OVERLAY_GREY = 150
OVERLAY_ALPHA = 0.6  # how much grey covers a max-piece that isn't set yet
PILE_FADE = 0.85  # how far a set piece fades toward the background in the pile
PILE_GAP = 8  # pixels between pieces in the pile
HOLD_FRAMES = {"start": 6, "stage 3": 8, "stage 4": 24}  # frames to linger on these steps; others get one


def blend(image: np.ndarray, color: int, amount: float) -> np.ndarray:
    """Mix uint8 pixels toward a grey level: amount 0 keeps the image, 1 gives flat color."""
    return ((1 - amount) * image + amount * color).astype(np.uint8)


def board_image(pieces: np.ndarray, masks: np.ndarray, grid: np.ndarray, locked: np.ndarray) -> np.ndarray:
    """(rows * cell, cols * cell, 3) board: set pieces in full color, other max-pieces under a grey overlay."""
    rows, cols = grid.shape
    box = pieces.shape[1]
    cell = cell_size(box)
    canvas = np.full((rows * cell + 2 * PAD, cols * cell + 2 * PAD, 3), BACKGROUND, dtype=np.uint8)
    for line in range(rows + 1):
        canvas[PAD + min(line * cell, rows * cell - 1)] = GRID_LINE
    for line in range(cols + 1):
        canvas[:, PAD + min(line * cell, cols * cell - 1)] = GRID_LINE
    # pending pieces first, so set pieces are drawn on top where their tabs overlap
    for set_pass in (False, True):
        for y, x in zip(*np.nonzero((grid != EMPTY) & (locked == set_pass))):
            piece_id = grid[y, x]
            piece = pieces[piece_id] if set_pass else blend(pieces[piece_id], OVERLAY_GREY, OVERLAY_ALPHA)
            region = canvas[y * cell : y * cell + box, x * cell : x * cell + box]  # a view into canvas
            region[masks[piece_id]] = piece[masks[piece_id]]
    return canvas[PAD:-PAD, PAD:-PAD]


def pile_image(pieces: np.ndarray, masks: np.ndarray, is_set: np.ndarray) -> np.ndarray:
    """The pieces in their shuffled order on a grid; pieces already set on the board are faded out."""
    box = pieces.shape[1]
    step = box + PILE_GAP
    num_rows = -(-len(pieces) // COLS)  # ceiling division
    canvas = np.full((num_rows * step, COLS * step, 3), BACKGROUND, dtype=np.uint8)
    for piece_id, (piece, mask) in enumerate(zip(pieces, masks)):
        top, left = (piece_id // COLS) * step, (piece_id % COLS) * step
        shown = blend(piece, BACKGROUND, PILE_FADE) if is_set[piece_id] else piece
        canvas[top : top + box, left : left + box][mask] = shown[mask]
    return canvas


def frame_plan(trace: list[Snapshot]) -> list[int]:
    """Snapshot index for each animation frame (-1 = the empty board), lingering per HOLD_FRAMES."""
    plan = [-1] * HOLD_FRAMES["start"]
    for index, snapshot in enumerate(trace):
        stage = snapshot.label.split(",")[0]
        plan += [index] * HOLD_FRAMES.get(stage, 1)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--index", type=int, default=0, help="val puzzle to solve")
    parser.add_argument("--image", help="solve this image instead of a val puzzle (cut with --seed)")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--noise", type=float, default=0.0, help="std of Gaussian noise added to the logits")
    parser.add_argument("--combine", default="mean", choices=COMBINE_MODES, help="how neighbors' scores combine")
    parser.add_argument("--device", default="auto", help="auto, mps, cuda or cpu")
    parser.add_argument("--seed", type=int, default=0, help="seed for cutting --image, shuffling and --noise")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=60, help="resolution of the saved .gif (60 -> 720x384)")
    parser.add_argument("--out", help="save a .gif here instead of playing the animation in a window")
    args = parser.parse_args()

    puzzle = cut(load_image(args.image), args.seed) if args.image else PuzzleDataset(load_manifest("val"))[args.index]
    rng = np.random.default_rng(args.seed)
    pieces, masks, positions = shuffle_puzzle(*puzzle, rng)
    model = load_comparator(args.checkpoint, pick_device(args.device))
    trace = []
    _, (*_, final) = solve_puzzle(model, pieces, masks, args.noise, rng, args.combine, trace)
    metrics = placement_metrics(final.grid, positions)

    fig, (pile_ax, board_ax) = plt.subplots(1, 2, figsize=(12, 6.4), layout="constrained")
    empty_grid = np.full((ROWS, COLS), EMPTY)
    no_locks = np.zeros((ROWS, COLS), dtype=bool)
    pile = pile_ax.imshow(pile_image(pieces, masks, np.zeros(len(pieces), dtype=bool)))
    board = board_ax.imshow(board_image(pieces, masks, empty_grid, no_locks))
    pile_ax.set_title("Pieces")
    for ax in (pile_ax, board_ax):
        ax.axis("off")

    def draw(snapshot_index):
        if snapshot_index < 0:
            grid, locked, label = empty_grid, no_locks, "start"
        else:
            snapshot = trace[snapshot_index]
            grid, locked, label = snapshot.grid, snapshot.locked, snapshot.label
        is_set = np.zeros(len(pieces), dtype=bool)
        is_set[grid[locked]] = True
        pile.set_data(pile_image(pieces, masks, is_set))
        board.set_data(board_image(pieces, masks, grid, locked))
        title = f"{label}: {(grid != EMPTY).sum()} spaces guessed (grey), {locked.sum()} set (color)"
        if label == "stage 4":
            title = f"final: {metrics['direct']:.0%} of pieces in the right space"
        board_ax.set_title(title)
        return pile, board

    animation = FuncAnimation(fig, draw, frames=frame_plan(trace), interval=1000 / args.fps, blit=False)
    if args.out:
        animation.save(args.out, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
        print(f"Saved {args.out} ({len(trace)} solver steps)")
    else:
        plt.show()


if __name__ == "__main__":
    main()
