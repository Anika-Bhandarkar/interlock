"""Train the robust edge comparator: same EdgeComparator as scripts/train_comparator.py, on RobustPairDataset.

Differences from scripts/train_comparator.py:
- training pairs come from RobustPairDataset: outline error on every piece, half the negatives transplants, and
  1 puzzle in 8 cut on an even grid of straight lines plus 1 in 8 with slanted straight lines;
- val is split: the first 100 val puzzles are "dev" (pick the best epoch, fit temperatures) and puzzles 100-199
  are "test" (eval_robust.py only), so the reported numbers come from puzzles nothing was tuned on;
- each epoch reports dev top-1 on clean pieces and under "uneven_outline" mask error, and the best epoch by the
  latter is saved, with the training settings and a hash of the data manifest (for the model card).

Runs on Apple Silicon (MPS) by default, else CUDA, else CPU.

Usage: uv run python testing/train_robust_comparator.py [--epochs 15] [--out checkpoints/robust_comparator.pt]
"""

import argparse
import hashlib
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from interlock.data_collection.dataset import PROCESSED_DIR, PuzzleDataset, collate_pairs, load_manifest
from interlock.models.edge_comparator import EdgeComparator, evaluate, pick_device, save_comparator
from robust_comparator import (
    FLAT_DIAGONAL_SHARE,
    FLAT_EVEN_SHARE,
    TRANSPLANT_SHARE,
    ConditionPuzzles,
    RobustPairDataset,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKERS = max(0, min(8, (os.cpu_count() or 1) - 2))
NUM_DEV = 100  # val puzzles 0-99 tune; 100-199 test


def dev_puzzles() -> PuzzleDataset:
    return PuzzleDataset(load_manifest("val")[:NUM_DEV])


def test_puzzles(count: int = 100) -> PuzzleDataset:
    """Val puzzles NUM_DEV.., cut with their usual val seeds (seed = val index)."""
    return PuzzleDataset(load_manifest("val")[NUM_DEV : NUM_DEV + count], seed_offset=NUM_DEV)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--pairs-per-puzzle", type=int, default=16, help="positives per puzzle per direction")
    parser.add_argument("--puzzles-per-batch", type=int, default=8, help="batch = this x 4 x pairs-per-puzzle")
    parser.add_argument("--lr", type=float, default=2e-3, help="peak learning rate (one-cycle schedule)")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=32, help="channels in the first conv block")
    parser.add_argument("--num-train", type=int, default=None, help="use only the first N train images")
    parser.add_argument("--eval-puzzles", type=int, default=20, help="dev puzzles scored after each epoch")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="DataLoader worker processes")
    parser.add_argument("--device", default="auto", help="auto, mps, cuda or cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--transplant-share", type=float, default=TRANSPLANT_SHARE, help="share of negatives that are transplants")
    parser.add_argument("--flat-even-share", type=float, default=FLAT_EVEN_SHARE, help="share of puzzles on an even straight grid")
    parser.add_argument("--flat-diagonal-share", type=float, default=FLAT_DIAGONAL_SHARE,
                        help="share of puzzles cut with slanted straight lines")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "checkpoints" / "robust_comparator.pt")
    return parser.parse_args()


def manifest_hash() -> str:
    """Short hash of data/processed/manifest.csv, so a checkpoint records which images it was trained on."""
    return hashlib.sha1((PROCESSED_DIR / "manifest.csv").read_bytes()).hexdigest()[:12]


def dataset_name() -> str:
    """Which image set data/processed holds: SA-1B manifests list each image's source tar, Imagenette's its class."""
    header = (PROCESSED_DIR / "manifest.csv").read_text().splitlines()[0].split(",")
    return "SA-1B" if "source" in header else "Imagenette" if "class_id" in header else "unknown"


def mean_top1(metrics: dict[str, dict[str, float]]) -> float:
    return float(np.mean([direction_metrics["top1"] for direction_metrics in metrics.values()]))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    train_rows = load_manifest("train")[: args.num_train]
    dev = dev_puzzles()
    dev_by_condition = {"clean": dev, "uneven_outline": ConditionPuzzles(dev, "uneven_outline")}
    train_set = RobustPairDataset(
        train_rows, args.pairs_per_puzzle, augment=True, seed=args.seed, transplant_share=args.transplant_share,
        flat_even_share=args.flat_even_share, flat_diagonal_share=args.flat_diagonal_share,
    )
    training = {key: value for key, value in vars(args).items() if key not in ("out", "device", "workers")}
    training |= {"num_train_images": len(train_rows), "manifest_sha1": manifest_hash(), "dataset": dataset_name()}
    loader = DataLoader(
        train_set,
        batch_size=args.puzzles_per_batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_pairs,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
    )

    window_shape = tuple(train_set[0][0].shape[-2:])
    model = EdgeComparator(width=args.width, input_shape=window_shape).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epochs * len(loader))
    loss_fn = nn.BCEWithLogitsLoss()

    print(f"device {device} | {len(train_rows)} train puzzles, {min(args.eval_puzzles, len(dev))} dev puzzles | "
          f"window {window_shape} | {args.workers} workers", flush=True)

    best = -1.0
    for epoch in range(args.epochs):
        train_set.set_epoch(epoch)
        model.train()
        start_time = time.time()
        total_loss, total_correct, total_pairs = 0.0, 0, 0
        for windows, labels in loader:
            windows, labels = windows.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            logits = model(windows)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * len(labels)
            total_correct += ((logits > 0) == (labels > 0.5)).sum().item()
            total_pairs += len(labels)
        train_time = time.time() - start_time

        scores = {name: evaluate(model, puzzles, args.eval_puzzles) for name, puzzles in dev_by_condition.items()}
        summary = " | ".join(f"{name} top1 {mean_top1(metrics):.3f}" for name, metrics in scores.items())
        print(f"epoch {epoch + 1}/{args.epochs}: loss {total_loss / total_pairs:.4f} acc {total_correct / total_pairs:.3f} "
              f"({train_time:.0f}s) | dev {summary} ({time.time() - start_time - train_time:.0f}s)", flush=True)

        if mean_top1(scores["uneven_outline"]) > best:
            best = mean_top1(scores["uneven_outline"])
            save_comparator(model, args.out, epoch=epoch + 1, dev_metrics=scores, training=training)
            print(f"  saved {args.out} (dev uneven_outline top-1 {best:.3f})", flush=True)


if __name__ == "__main__":  # required on macOS: DataLoader workers re-import this file when they start
    main()
