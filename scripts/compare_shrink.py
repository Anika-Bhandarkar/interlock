"""Compare the shrinking-frame and bounded-growth solvers (both with swap polish) on validation puzzles.

Categories are the studio's: regular jigsaw cuts under each condition (clean, uneven_outline, pixel_noise,
shrink_1px_half, smoothed_outline, color_removed) and the flat cuts (flat_even, flat_diagonal). Uses the same
puzzles, shuffles, damage and images as the validation studio (val images 100-199, so preprocess at least
200 val images). Scores are cached per checkpoint in artifacts/shrink_compare/, so only the first run per image
uses the GPU; later runs (e.g. after changing a solver) take seconds.

Both solvers are scored with translation alignment (aligned_metrics). Both are given the board size, never the
true positions.

Usage: uv run python scripts/compare_shrink.py [--count 10] [--start 0] [--categories uneven_outline flat_even ...]
                                               [--per-image] [--checkpoint PATH] [--device auto]
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "frontend"))  # experiment.py: flat cuts and all-pairs scoring

from experiment import CONDITIONS, FlatPuzzles, apply_condition, score_all_pairs  # noqa: E402

from interlock.data_collection.dataset import PuzzleDataset, load_manifest  # noqa: E402
from interlock.models.edge_comparator import load_comparator, pick_device, score_matrix  # noqa: E402
from interlock.solver.bounded_growth import solve_bounded  # noqa: E402
from interlock.solver.shrink_frame import grid_coordinates, shrink_frame  # noqa: E402
from interlock.solver.strongest_pair import aligned_metrics  # noqa: E402
from interlock.solver.updaters import puzzle_shape, shuffle_puzzle  # noqa: E402

CACHE = REPO_ROOT / "artifacts" / "shrink_compare"
CHECKPOINTS = [REPO_ROOT / "checkpoints" / "robust_comparator_hf.pt", REPO_ROOT / "checkpoints" / "edge_comparator.pt"]
FLAT = ("flat_even", "flat_diagonal")
CATEGORIES = CONDITIONS + FLAT  # the studio's eight
MODES = ("shrink + polish", "bounded + polish")


def puzzle_scores(category, index, checkpoint, model_loader, rows_meta):
    """(logits, positions) for viewer image `index`, cut, shuffled and damaged exactly like the studio does;
    cached. Flat cuts score every pair; jigsaw cuts score only pairs whose tabs and blanks can interlock, as read
    off the (possibly damaged) outlines."""
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()[:12]
    path = CACHE / f"{digest}-{category}-{index:03d}.npz"
    if path.exists():
        with np.load(path) as cached:
            return cached["logits"], cached["positions"]
    flat = category in FLAT
    dataset = FlatPuzzles(rows_meta, category, seed_offset=100) if flat else PuzzleDataset(rows_meta, seed_offset=100)
    pieces, masks, positions = shuffle_puzzle(*dataset[index], np.random.default_rng(10000 + index))
    if flat:
        logits = score_all_pairs(model_loader(), pieces, masks, batch_size=128)
    else:
        rng = np.random.default_rng([10000, CONDITIONS.index(category), index])
        pieces, masks = apply_condition(category, pieces, masks, rng)
        logits = score_matrix(model_loader(), pieces, masks, batch_size=128)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, logits=logits, positions=positions)
    return logits, positions


def run_modes(logits, positions):
    rows, cols = puzzle_shape(positions)
    results = {
        "shrink + polish": shrink_frame(logits, rows, cols).coordinates,
        "bounded + polish": grid_coordinates(solve_bounded(logits, rows, cols)[0]),
    }
    return {name: aligned_metrics(coords, positions) for name, coords in results.items()}


def failures(args, rows_meta, model_loader):
    """Per-image accuracy for every image either solver didn't solve, to see where they differ."""
    for category in args.categories:
        print(category)
        for index in range(args.start, args.start + args.count):
            logits, positions = puzzle_scores(category, index, args.checkpoint, model_loader, rows_meta)
            results = run_modes(logits, positions)
            if all(m["perfect"] for m in results.values()):
                continue
            cells = " | ".join(f"{name} {m['direct']:.0%}" for name, m in results.items())
            print(f"  image {index + 1:>3} ({len(positions):>3} pieces): {cells}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=10, help="first N studio images (1-100)")
    parser.add_argument("--categories", nargs="+", default=["uneven_outline", "clean"], choices=CATEGORIES,
                        help="studio categories; uneven_outline is the closest to photographed pieces")
    parser.add_argument("--checkpoint", type=Path, default=next((p for p in CHECKPOINTS if p.exists()), CHECKPOINTS[0]))
    parser.add_argument("--start", type=int, default=0, help="skip the first N studio images (e.g. to tune on others)")
    parser.add_argument("--per-image", action="store_true", help="list each image that some mode failed")
    parser.add_argument("--device", default="auto", help="auto, mps, cuda or cpu (only used for uncached scores)")
    args = parser.parse_args()

    rows_meta = load_manifest("val")[100:200]
    if len(rows_meta) < args.start + args.count:
        sys.exit(f"only {len(rows_meta)} val images past index 100; rerun scripts/preprocess.py with --num-val 200")
    model = None

    def model_loader():
        nonlocal model
        if model is None:
            model = load_comparator(args.checkpoint, pick_device(args.device))
        return model

    if args.per_image:
        return failures(args, rows_meta, model_loader)
    print(f"checkpoint {args.checkpoint.relative_to(REPO_ROOT)} | studio images {args.start + 1}-{args.start + args.count}")
    for category in args.categories:
        totals = {mode: [] for mode in MODES}
        start = time.time()
        for index in range(args.start, args.start + args.count):
            logits, positions = puzzle_scores(category, index, args.checkpoint, model_loader, rows_meta)
            for mode, metrics in run_modes(logits, positions).items():
                totals[mode].append(metrics)
            print(f"\r{category}: {index + 1}/{args.count} images", end="", flush=True)
        print(f" ({time.time() - start:.0f}s)")
        print(f"  {'mode':<20} {'placed right':>12} {'solved':>8} {'neighbors right':>16}")
        for mode, rows in totals.items():
            direct = np.mean([m["direct"] for m in rows])
            solved = sum(m["perfect"] for m in rows)
            neighbors = np.mean([m["neighbors"] for m in rows])
            print(f"  {mode:<20} {direct:>11.1%} {int(solved):>5}/{len(rows):<2} {neighbors:>15.1%}")


if __name__ == "__main__":
    main()
