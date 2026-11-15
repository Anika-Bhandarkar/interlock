"""Stress-test comparators and the solver on held-out val puzzles, before and after temperature scaling.

For each model (default: the baseline trained by scripts/train_comparator.py and the robust one):
1. Fit temperatures on dev puzzles (val 0..num_dev-1) under "uneven_outline" mask error:
   T_edge by likelihood of each piece's true neighbor (fit_temperature), and T_space by likelihood of the true
   piece in the solver's stage 3 distributions (fit_space_temperature).
2. Score test puzzles (val 100..100+num_test-1) under every condition in CONDITIONS, and flat-edged test puzzles
   (FLAT_KINDS, every pair scored since nothing can be ruled out by shape). Report comparator top-1, true pairs
   ruled out by shape (side_types misreads, which no comparator can recover), and calibration error of the
   comparator's candidate softmax before and after T_edge.
3. Run solver stages 2-4 on each test puzzle, on its own grid size, with the stock settings and with each
   temperature, with LOCK_PROB 0.6 (stock) and 0.95, and report placement accuracy and how trustworthy the locks
   were.

Logits are cached in data/eval/robust_cache/, keyed by a hash of the data manifest and the checkpoint, so a new
dataset or a retrained model never reuses old scores. Results go to testing/results/robust_eval.json and .md.

Usage: uv run python testing/eval_robust.py [--num-test 100] [--num-flat 30] [--num-dev 50]
                                           [--models baseline=PATH robust=PATH]
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from interlock.data_collection.dataset import DATA_DIR, PROCESSED_DIR, load_manifest
from interlock.data_collection.pairs import side_types
from interlock.models.edge_comparator import load_comparator, neighbor_ranks, pick_device, score_matrix
from interlock.solver.updaters import final_assignment, placement_metrics, shuffle_puzzle, true_grid
from robust_comparator import (
    CONDITIONS,
    FLAT_KINDS,
    FlatPuzzles,
    apply_condition,
    candidate_rows,
    expected_calibration_error,
    fit_space_temperature,
    fit_temperature,
    grid_shape,
    run_stages,
    scale_logits,
    score_all_pairs,
    softmax_rows,
    stack_candidate_rows,
)
from train_robust_comparator import NUM_DEV, dev_puzzles, test_puzzles

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = DATA_DIR / "eval" / "robust_cache"
CALIBRATION_CONDITION = "uneven_outline"
# solver variants: (which temperature, LOCK_PROB); "stock" is the repo's current behavior
VARIANTS = {
    "stock": (None, 0.6),
    "T_edge": ("edge", 0.6),
    "T_edge, lock 0.95": ("edge", 0.95),
    "T_space": ("space", 0.6),
    "T_space, lock 0.95": ("space", 0.95),
}
NUM_BINS = 10


def cache_key(checkpoint: Path) -> str:
    """Hash of the data manifest and the checkpoint: cached scores are reused only for the same data and weights."""
    digest = hashlib.sha1((PROCESSED_DIR / "manifest.csv").read_bytes())
    digest.update(Path(checkpoint).read_bytes())
    return digest.hexdigest()[:12]


def scored_puzzles(model, key: str, puzzles, condition: str, split: str, count: int, seed_base: int):
    """[(logits, types, positions)] for the first `count` puzzles, shuffled and seen under `condition` (one of
    CONDITIONS, or a flat kind, which `puzzles` then already is). Cached. Also returns seconds per puzzle scored,
    or None when read from the cache."""
    path = CACHE_DIR / f"{key}_{split}_{condition}_{count}.npz"
    if path.exists():
        data = np.load(path)
        return [(data[f"logits_{i}"], data[f"types_{i}"], data[f"pos_{i}"]) for i in range(count)], None
    out, store, start = [], {}, time.time()
    for i in range(count):
        pieces, masks, positions = shuffle_puzzle(*puzzles[i], np.random.default_rng(seed_base + i))
        if condition in FLAT_KINDS:
            logits = score_all_pairs(model, pieces, masks)
            # every side is flat, so the solver gets no shape rules (a slanted side can read as a tab or blank)
            types = np.zeros((len(pieces), 4), dtype=np.int8)
        else:
            pieces, masks = apply_condition(
                condition, pieces, masks, np.random.default_rng([seed_base, CONDITIONS.index(condition), i])
            )
            logits, types = score_matrix(model, pieces, masks), side_types(masks)
        out.append((logits, types, positions))
        store |= {f"logits_{i}": logits, f"types_{i}": types, f"pos_{i}": positions}
    seconds = (time.time() - start) / count
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **store)
    return out, seconds


def comparator_metrics(scored, t_edge: float) -> dict:
    ranks, ruled_out, pairs = [], 0, 0
    conf = {"raw": [], "T_edge": []}
    correct = []
    for logits, _, positions in scored:
        puzzle_ranks = list(neighbor_ranks(logits, positions).values())
        ranks += puzzle_ranks
        rows, targets = candidate_rows(logits, positions)
        pairs += sum(len(r) for r in puzzle_ranks)
        ruled_out += sum(len(r) for r in puzzle_ranks) - len(rows)  # true neighbor is -inf: shape misread
        if len(rows):
            correct.append(rows.argmax(axis=1) == targets)
            conf["raw"].append(softmax_rows(rows).max(axis=1))
            conf["T_edge"].append(softmax_rows(rows, t_edge).max(axis=1))
    ranks = np.concatenate(ranks)
    correct = np.concatenate(correct)
    return {
        "top1": float(np.mean(ranks == 1)),
        "mean_rank": float(ranks.mean()),
        "ruled_out": int(ruled_out),
        "pairs": int(pairs),
        "puzzles": len(scored),
        "pieces_per_puzzle": float(np.mean([len(positions) for _, _, positions in scored])),
        "edge_ece_raw": expected_calibration_error(np.concatenate(conf["raw"]), correct),
        "edge_ece_T_edge": expected_calibration_error(np.concatenate(conf["T_edge"]), correct),
    }


def solver_metrics(scored, temperatures: dict) -> dict:
    results = {}
    for variant, (which, lock_prob) in VARIANTS.items():
        rows, lock_conf, lock_correct = [], [], []
        start = time.time()
        for logits, types, positions in scored:
            num_rows, num_cols = grid_shape(positions)
            scaled = logits if which is None else scale_logits(logits, temperatures[which])
            first, refined = run_stages(scaled, types, num_rows, num_cols, lock_prob=lock_prob)
            final = final_assignment(refined, types)
            truth = true_grid(positions, num_rows, num_cols)
            confidence = np.take_along_axis(refined.probs, refined.grid[..., None], axis=2)[..., 0]
            lock_conf.append(confidence[refined.locked])
            lock_correct.append((refined.grid == truth)[refined.locked])
            metrics = placement_metrics(final.grid, positions)
            metrics["stage2_direct"] = placement_metrics(first.grid, positions)["direct"]
            metrics["locked"] = float(refined.locked.mean())
            metrics["stage4_changed"] = float((final.grid != refined.grid).any())
            rows.append(metrics)
        lock_conf, lock_correct = np.concatenate(lock_conf), np.concatenate(lock_correct)
        bins = np.minimum((lock_conf * NUM_BINS).astype(int), NUM_BINS - 1)
        results[variant] = {name: float(np.mean([row[name] for row in rows])) for name in rows[0]} | {
            "locked_wrong": float(1 - lock_correct.mean()) if len(lock_correct) else 0.0,
            "lock_ece": expected_calibration_error(lock_conf, lock_correct),
            "lock_bins": [[int((bins == b).sum()), int(lock_correct[bins == b].sum()), float(lock_conf[bins == b].sum())]
                          for b in range(NUM_BINS)],
            "ms_per_solve": 1000 * (time.time() - start) / len(scored),
        }
    return results


def write_markdown(results: dict, path: Path, args) -> None:
    models = list(results)
    sets = list(next(iter(results.values()))["conditions"])
    pct = lambda v: f"{100 * v:.1f}%"
    lines = ["# Robust comparator evaluation", "",
             f"{args.num_test} tabbed and {args.num_flat} flat-edged test puzzles per set (val {NUM_DEV}+), temperatures "
             f"fit on {args.num_dev} dev puzzles (val 0-{args.num_dev - 1}) under `{CALIBRATION_CONDITION}`. Grids vary "
             "per puzzle (cell 30-100 px).", ""]
    lines += ["Temperatures: " + ", ".join(f"{m}: T_edge {results[m]['T_edge']:.2f}, T_space {results[m]['T_space']:.2f}"
                                           for m in models), ""]

    lines += ["## Comparator top-1 (true neighbor ranked first)", "",
              "| test set | " + " | ".join(models) + " | true pairs ruled out by shape | pieces per puzzle |",
              "|---|" + "---|" * (len(models) + 2)]
    for c in sets:
        cells = [pct(results[m]["conditions"][c]["top1"]) for m in models]
        first = results[models[0]]["conditions"][c]
        lines.append(f"| {c} | " + " | ".join(cells) + f" | {first['ruled_out']}/{first['pairs']} | "
                     f"{first['pieces_per_puzzle']:.0f} |")

    lines += ["", "## Solver with stock settings: pieces placed right / neighbor pairs right / puzzles solved", "",
              "Flat-edged puzzles have no corner or edge shapes to anchor the grid, so a correct assembly can come out "
              "shifted; neighbor pairs right is the measure that matters there.", "",
              "| test set | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    for c in sets:
        cells = [f"{pct(results[m]['conditions'][c]['solver']['stock']['direct'])} / "
                 f"{pct(results[m]['conditions'][c]['solver']['stock']['neighbors'])} / "
                 f"{pct(results[m]['conditions'][c]['solver']['stock']['perfect'])}" for m in models]
        lines.append(f"| {c} | " + " | ".join(cells) + " |")

    for m in models:
        lines += ["", f"## Calibration and solver variants: {m}", "",
                  "| test set | edge ECE raw → T_edge | variant | placed right | neighbors right | solved | locked | locks wrong | lock ECE | stage 4 changed answer |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for c in sets:
            cond = results[m]["conditions"][c]
            edge = f"{cond['edge_ece_raw']:.3f} → {cond['edge_ece_T_edge']:.3f}"
            for k, (variant, v) in enumerate(cond["solver"].items()):
                lines.append(f"| {c if k == 0 else ''} | {edge if k == 0 else ''} | {variant} | {pct(v['direct'])} | "
                             f"{pct(v['neighbors'])} | {pct(v['perfect'])} | {pct(v['locked'])} | {pct(v['locked_wrong'])} | {v['lock_ece']:.3f} | "
                             f"{pct(v['stage4_changed'])} |")
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--num-test", type=int, default=100, help="tabbed test puzzles per condition")
    parser.add_argument("--num-flat", type=int, default=30, help="flat-edged test puzzles per kind (every pair is scored)")
    parser.add_argument("--num-dev", type=int, default=50, help="dev puzzles for fitting temperatures")
    parser.add_argument("--models", nargs="+", default=[
        f"baseline={REPO_ROOT / 'checkpoints' / 'baseline_comparator.pt'}",
        f"robust={REPO_ROOT / 'checkpoints' / 'robust_comparator.pt'}",
    ])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "testing" / "results")
    args = parser.parse_args()

    device = pick_device(args.device)
    dev, test = dev_puzzles(), test_puzzles(args.num_test)
    flat_rows = load_manifest("val")[NUM_DEV : NUM_DEV + args.num_flat]
    flat = {kind: FlatPuzzles(flat_rows, kind, seed_offset=NUM_DEV) for kind in FLAT_KINDS}
    results = {}
    for spec in args.models:
        name, path = spec.split("=", 1)
        model, key = load_comparator(path, device), cache_key(path)
        dev_scored, _ = scored_puzzles(model, key, dev, CALIBRATION_CONDITION, "dev", args.num_dev, seed_base=0)
        rows, targets = stack_candidate_rows(*zip(*[candidate_rows(logits, positions) for logits, _, positions in dev_scored]))
        t_edge = fit_temperature(rows, targets)
        t_space = fit_space_temperature(*zip(*dev_scored))
        temperatures = {"edge": t_edge, "space": t_space}
        print(f"{name}: T_edge {t_edge:.3f}, T_space {t_space:.3f}", flush=True)

        results[name] = {"checkpoint": str(path), "T_edge": t_edge, "T_space": t_space, "conditions": {}}
        test_sets = [(c, test, args.num_test) for c in CONDITIONS] + [(k, flat[k], args.num_flat) for k in FLAT_KINDS]
        for condition, puzzles, count in test_sets:
            scored, seconds = scored_puzzles(model, key, puzzles, condition, "test", count, seed_base=10_000)
            entry = comparator_metrics(scored, t_edge) | {"solver": solver_metrics(scored, temperatures)}
            if seconds is not None:
                entry["score_seconds_per_puzzle"] = seconds
            results[name]["conditions"][condition] = entry
            stock = entry["solver"]["stock"]
            print(f"  {condition}: top1 {entry['top1']:.4f} | ruled out {entry['ruled_out']}/{entry['pairs']} | "
                  f"solver placed {stock['direct']:.3f} solved {stock['perfect']:.2f}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "robust_eval.json").write_text(json.dumps(results, indent=1))
    write_markdown(results, args.out / "robust_eval.md", args)
    print(f"wrote {args.out / 'robust_eval.json'} and robust_eval.md")


if __name__ == "__main__":
    main()
