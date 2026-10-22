"""Solve shuffled val puzzles with the edge comparator and solver stages 2-4, and report accuracy.

For each puzzle: shuffle the pieces, score every pair with score_matrix, then run first_updater (stage 2),
multistage_updater (stage 3) and final_assignment (stage 4). Prints the mean of placement_metrics per stage:
direct (pieces in the right space), neighbors (adjacent pairs that are true neighbors) and perfect (share
of puzzles solved exactly), plus how many spaces hold a repeated max-piece (stages 2 and 3 allow repeats).

--noise adds Gaussian noise to the logits, to see how the updaters cope with a weaker comparator.

Usage: uv run python scripts/solve.py [--num-puzzles 30] [--checkpoint PATH] [--combine mean] [--noise 0]
"""

import argparse
import time
from pathlib import Path

import numpy as np

from interlock.data_collection.dataset import PuzzleDataset, load_manifest
from interlock.data_collection.pairs import ABOVE, BELOW, DIRECTIONS, LEFT, RELATIONS, RIGHT, side_types
from interlock.models.edge_comparator import load_comparator, pick_device, score_matrix
from interlock.solver.updaters import COMBINE_MODES, placement_metrics, shuffle_puzzle, solve

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
):
    """Score already shuffled pieces and run stages 2-4. Returns (logits, (stage 2, stage 3, stage 4 results)).

    Pass a list as `trace` to collect a Snapshot of the board after every solver step.
    """
    logits = score_matrix(model, pieces, masks)
    if noise:
        logits = add_noise(logits, noise, rng or np.random.default_rng())
    return logits, solve(logits, side_types(masks), combine, trace)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--num-puzzles", type=int, default=30, help="first N val puzzles")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--noise", type=float, default=0.0, help="std of Gaussian noise added to the logits")
    parser.add_argument("--combine", default="mean", choices=COMBINE_MODES, help="how neighbors' scores combine")
    parser.add_argument("--device", default="auto", help="auto, mps, cuda or cpu")
    parser.add_argument("--seed", type=int, default=0, help="seed for shuffling pieces and for --noise")
    args = parser.parse_args()

    model = load_comparator(args.checkpoint, pick_device(args.device))
    puzzles = PuzzleDataset(load_manifest("val"))
    rng = np.random.default_rng(args.seed)
    stages = ("stage 2", "stage 3", "stage 4")
    metrics = {stage: [] for stage in stages}

    start_time = time.time()
    for index in range(min(args.num_puzzles, len(puzzles))):
        pieces, masks, positions = shuffle_puzzle(*puzzles[index], rng)
        _, results = solve_puzzle(model, pieces, masks, args.noise, rng, args.combine)
        for stage, result in zip(stages, results):
            repeated = {"repeated": float(result.grid.size - len(np.unique(result.grid)))}
            metrics[stage].append(placement_metrics(result.grid, positions) | repeated)

    print(f"{len(metrics['stage 2'])} val puzzles | combine {args.combine} | noise {args.noise} | "
          f"{time.time() - start_time:.0f}s")
    for stage, rows in metrics.items():
        summary = " | ".join(f"{name} {np.mean([row[name] for row in rows]):.3f}" for name in rows[0])
        print(f"{stage}: {summary}")


if __name__ == "__main__":
    main()
