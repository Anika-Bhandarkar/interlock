"""Solve shuffled val puzzles with the edge comparator and solver stages 2-4, and report accuracy.

For each puzzle: shuffle the pieces, score every pair with score_matrix, then run first_updater (stage 2),
multistage_updater (stage 3) and final_assignment (stage 4). Prints the mean of placement_metrics per stage:
direct (pieces in the right space), neighbors (adjacent pairs that are true neighbors) and perfect (share
of puzzles solved exactly), plus how many spaces hold a repeated max-piece (stages 2 and 3 allow repeats).

CNN neighbor top-1 and mean rank are computed from the same score matrix (no extra inference),
pooled over true-neighbor queries in right/below directions. Shape filtering remains active; with
--noise these metrics include the injected noise. Anchoring does not affect CNN ranking.

Progress is printed and flushed after each puzzle, including cumulative solved counts per stage.

--noise adds Gaussian noise to the logits, to see how the updaters cope with a weaker comparator.

Usage: uv run python scripts/solve.py [--num-puzzles 30] [--checkpoint PATH] [--combine mean] [--noise 0]
"""

import argparse
import time
from pathlib import Path

import numpy as np

from interlock.data_collection.cutting import COLS, ROWS
from interlock.data_collection.dataset import PuzzleDataset, load_manifest
from interlock.data_collection.pairs import ABOVE, BELOW, DIRECTIONS, LEFT, RELATIONS, RIGHT, cell_size, side_types
from interlock.models.edge_comparator import load_comparator, neighbor_ranks, pick_device, score_matrix
from interlock.solver.updaters import COMBINE_MODES, placement_metrics, puzzle_shape, top_left_anchor, shuffle_puzzle, solve, validate_dimensions

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "edge_comparator.pt"


def add_noise(logits: np.ndarray, noise: float, rng: np.random.Generator) -> np.ndarray:
    """Logits plus N(0, noise^2) on "right" and "below", keeping left/above their transposes and -inf as is."""
    noisy = logits.copy()
    for direction in DIRECTIONS:
        relation = RELATIONS.index(direction)
        noisy[relation] += rng.normal(0, noise, noisy[relation].shape).astype(noisy.dtype)
    noisy[LEFT], noisy[ABOVE] = noisy[RIGHT].T, noisy[BELOW].T
    return noisy


def solve_puzzle(
    model,
    pieces,
    masks,
    noise: float = 0.0,
    rng: np.random.Generator | None = None,
    combine: str = "mean",
    trace: list | None = None,
    *, rows: int = ROWS, cols: int = COLS, straight_edges: bool = False, anchor_piece: int | None = None,
):
    """Score already shuffled pieces and run stages 2-4. Returns (logits, (stage 2, stage 3, stage 4 results)).

    Pass the known board dimensions as rows and cols (10x10 by default).
    Pass a list as `trace` to collect a Snapshot of the board after every solver step.
    """
    validate_dimensions(rows, cols)
    if len(pieces) != rows * cols:
        raise ValueError(f"{len(pieces)} pieces do not fill a {rows}x{cols} board")
    if pieces.ndim != 4 or pieces.shape[-1] != 3 or masks.shape != pieces.shape[:3]:
        raise ValueError("pieces must be (N, B, B, 3) with matching (N, B, B) masks")
    if pieces.shape[1] != pieces.shape[2]:
        raise ValueError("the comparator requires square piece boxes")
    logits = score_matrix(model, pieces, masks, straight_edges=straight_edges)
    if noise:
        logits = add_noise(logits, noise, rng or np.random.default_rng())
    return logits, solve(logits, side_types(masks), combine, trace, rows=rows, cols=cols, straight_edges=straight_edges, anchor_piece=anchor_piece)


def summarize_rankings(rank_sets: list[dict[str, np.ndarray]], direction: str | None = None) -> dict:
    """Pool neighbor queries, weighting each query equally; ties count against the true neighbor."""
    directions = DIRECTIONS if direction is None else (direction,)
    values = [ranks[name] for ranks in rank_sets for name in directions]
    ranks = np.concatenate(values) if values else np.array([])
    return {
        "top1": float(np.mean(ranks == 1)) if len(ranks) else float("nan"),
        "mean_rank": float(ranks.mean()) if len(ranks) else float("nan"),
        "queries": len(ranks),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--num-puzzles", type=int, default=30, help="first N val puzzles")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--noise", type=float, default=0.0, help="std of Gaussian noise added to the logits")
    parser.add_argument("--combine", default="mean", choices=COMBINE_MODES, help="how neighbors' scores combine")
    parser.add_argument("--device", default="auto", help="auto, mps, cuda or cpu")
    parser.add_argument("--seed", type=int, default=0, help="seed for shuffling pieces and for --noise")
    parser.add_argument("--straight-edges", action="store_true", help="use square grid tiles without tabs; board dimensions remain known")
    parser.add_argument("--anchor-top-left", action="store_true", help="evaluation only: fix the true top-left piece using ground-truth metadata")
    parser.add_argument("--single-tab-shape", action="store_true", help="use one fixed symmetric tab shape across all puzzles")
    args = parser.parse_args()
    if args.straight_edges and args.single_tab_shape:
        parser.error("--straight-edges and --single-tab-shape cannot be combined")

    model = load_comparator(args.checkpoint, pick_device(args.device))
    puzzles = PuzzleDataset(load_manifest("val"), straight_edges=args.straight_edges, single_tab_shape=args.single_tab_shape)
    rng = np.random.default_rng(args.seed)
    stages = ("stage 2", "stage 3", "stage 4")
    metrics = {stage: [] for stage in stages}

    size_metrics = {}
    cnn_ranks = []
    num_puzzles = min(args.num_puzzles, len(puzzles))
    if num_puzzles <= 0:
        raise ValueError("no validation puzzles selected")
    start_time = time.time()
    print(f"Evaluating {num_puzzles} val puzzles | cut {'straight' if args.straight_edges else 'single-tab' if args.single_tab_shape else 'jigsaw'} | "
          f"anchor {'true-top-left' if args.anchor_top_left else 'none'} | combine {args.combine} | noise {args.noise}", flush=True)
    for index in range(num_puzzles):
        puzzle = puzzles[index]
        rows, cols = puzzle_shape(puzzle[2])
        pieces, masks, positions = shuffle_puzzle(*puzzle, rng)
        puzzle_start = time.time()
        logits, results = solve_puzzle(model, pieces, masks, args.noise, rng, args.combine, rows=rows, cols=cols, straight_edges=args.straight_edges, anchor_piece=top_left_anchor(positions) if args.anchor_top_left else None)
        ranks = neighbor_ranks(logits, positions)
        cnn_ranks.append(ranks)
        current_cnn = summarize_rankings([ranks])
        key = (cell_size(pieces.shape[1]), rows, cols)
        size_metrics.setdefault(key, []).append(placement_metrics(results[-1].grid, positions)
                                               | {"seconds": time.time() - puzzle_start, "cnn_ranks": ranks})
        for stage, result in zip(stages, results):
            repeated = {"repeated": float(result.grid.size - len(np.unique(result.grid)))}
            metrics[stage].append(placement_metrics(result.grid, positions) | repeated)

        completed = index + 1
        solved = " | ".join(
            f"{stage} solved {sum(int(m['perfect']) for m in metrics[stage])}/{completed}"
            for stage in stages
        )
        print(f"Puzzle {completed}/{num_puzzles} (index {index}, {rows}x{cols}, cell {key[0]}px) | "
              f"{time.time() - puzzle_start:.1f}s | "
              f"CNN this puzzle top1 {current_cnn['top1']:.1%} rank {current_cnn['mean_rank']:.2f} | {solved} | "
              f"final placed {np.mean([m['direct'] for m in metrics['stage 4']]):.1%} | "
              f"elapsed {(time.time() - start_time) / 60:.1f}m", flush=True)

    print(f"{len(metrics['stage 2'])} val puzzles | cut {'straight' if args.straight_edges else 'single-tab' if args.single_tab_shape else 'jigsaw'} | anchor {'true-top-left' if args.anchor_top_left else 'none'} | combine {args.combine} | noise {args.noise} | "
          f"{time.time() - start_time:.0f}s")
    print("CNN rankings: existing shape-filtered candidates, ties count against truth; "
          "queries pooled across puzzles; anchor not used in ranking")
    if args.noise:
        print("CNN rankings include the configured score noise (same scores used by the solver)")
    for direction in (None, *DIRECTIONS):
        cnn = summarize_rankings(cnn_ranks, direction)
        print(f"CNN {direction or 'overall'}: top1 {cnn['top1']:.3f} | "
              f"mean_rank {cnn['mean_rank']:.2f} | queries {cnn['queries']}")
    print("Assembly metrics (neighbors = correctly adjacent pairs on the assembled board):")
    for stage, rows in metrics.items():
        summary = " | ".join(f"{name} {np.mean([row[name] for row in rows]):.3f}" for name in rows[0])
        print(f"{stage}: {summary}")

    for (cell, rows, cols), samples in sorted(size_metrics.items()):
        cnn = summarize_rankings([sample["cnn_ranks"] for sample in samples])
        print(f"cell {cell}px | {rows}x{cols} ({rows * cols} pieces) | {len(samples)} puzzles | "
              f"placed {np.mean([m['direct'] for m in samples]):.3f} | "
              f"perfect {np.mean([m['perfect'] for m in samples]):.3f} | "
              f"CNN top1 {cnn['top1']:.3f} rank {cnn['mean_rank']:.2f} queries {cnn['queries']} | "
              f"seconds {np.mean([m['seconds'] for m in samples]):.2f}")


if __name__ == "__main__":
    main()
