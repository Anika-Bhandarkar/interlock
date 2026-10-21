"""Train the CNN edge comparator: scores whether piece B is the right neighbor of piece A.

Uses the Apple Silicon GPU (Metal / MPS) when available, else CUDA, else CPU. Each epoch re-cuts every
training image with new tab shapes, then reports top-1 accuracy and mean rank of true neighbors on val
puzzles (same metric as the onboarding notebook). The best epoch by val top-1 is saved to --out.

Usage: uv run python scripts/train_comparator.py [--epochs 15] [--device auto] [--workers 6]
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from interlock.data_collection.dataset import PairDataset, PuzzleDataset, collate_pairs, load_manifest
from interlock.models.edge_comparator import EdgeComparator, evaluate, pick_device, save_comparator

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKERS = max(0, min(8, (os.cpu_count() or 1) - 2))  # cutting runs on CPU; keep a couple of cores free


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--pairs-per-puzzle", type=int, default=16, help="positives per puzzle per direction")
    parser.add_argument("--puzzles-per-batch", type=int, default=8, help="batch = this x 4 x pairs-per-puzzle")
    parser.add_argument("--lr", type=float, default=2e-3, help="peak learning rate (one-cycle schedule)")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=32, help="channels in the first conv block")
    parser.add_argument("--num-train", type=int, default=None, help="use only the first N train images")
    parser.add_argument("--eval-puzzles", type=int, default=30, help="val puzzles scored after each epoch")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="DataLoader worker processes")
    parser.add_argument("--device", default="auto", help="auto, mps, cuda or cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "checkpoints" / "edge_comparator.pt")
    return parser.parse_args()


def mean_top1(metrics: dict[str, dict[str, float]]) -> float:
    return float(np.mean([direction_metrics["top1"] for direction_metrics in metrics.values()]))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    train_rows = load_manifest("train")[: args.num_train]
    val_puzzles = PuzzleDataset(load_manifest("val"))
    train_set = PairDataset(train_rows, args.pairs_per_puzzle, augment=True, seed=args.seed)
    loader = DataLoader(
        train_set,
        batch_size=args.puzzles_per_batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_pairs,
        pin_memory=device.type == "cuda",  # pinned memory only helps CUDA copies
        generator=torch.Generator().manual_seed(args.seed),
    )

    window_shape = tuple(train_set[0][0].shape[-2:])
    model = EdgeComparator(width=args.width, input_shape=window_shape).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epochs * len(loader))
    loss_fn = nn.BCEWithLogitsLoss()

    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"device {device} | {len(train_rows)} train puzzles, {min(args.eval_puzzles, len(val_puzzles))} val "
          f"puzzles | window {window_shape} | {num_parameters:,} parameters | {args.workers} workers")

    best_top1 = -1.0
    for epoch in range(args.epochs):
        train_set.set_epoch(epoch)  # new cuts and pairs; workers pick this up when the epoch starts
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

        metrics = evaluate(model, val_puzzles, args.eval_puzzles)
        summary = " | ".join(f"{name} top1 {m['top1']:.3f} rank {m['mean_rank']:.2f}" for name, m in metrics.items())
        print(f"epoch {epoch + 1}/{args.epochs}: loss {total_loss / total_pairs:.4f} "
              f"acc {total_correct / total_pairs:.3f} ({train_time:.0f}s) | val {summary} "
              f"({time.time() - start_time - train_time:.0f}s)")

        if mean_top1(metrics) > best_top1:
            best_top1 = mean_top1(metrics)
            save_comparator(model, args.out, epoch=epoch + 1, metrics=metrics)
            print(f"  saved {args.out} (val top-1 {best_top1:.3f})")


if __name__ == "__main__":  # required on macOS: DataLoader workers re-import this file when they start
    main()
