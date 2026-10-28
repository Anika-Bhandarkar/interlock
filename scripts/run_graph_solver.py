"""Run the graph solver on real puzzles from data/processed (via PuzzleDataset).

    uv run python scripts/run_graph_solver.py --split val --n 5
    uv run python scripts/run_graph_solver.py --split val --n 5 --rotate   # also randomly rotate pieces
"""

import argparse

import numpy as np

from interlock.data_collection.dataset import PuzzleDataset, load_manifest
from interlock.solver.graph import neighbor_accuracy, piece_from_mask, render, solve


def to_grid(positions: np.ndarray) -> np.ndarray:
    """Turn positions into (row, col) grid indices, whether cut() gives grid
    indices or pixel offsets (each unique value is ranked)."""
    pos = np.asarray(positions).reshape(len(positions), -1)[:, :2]
    out = np.zeros_like(pos, dtype=int)
    for axis in range(2):
        _, out[:, axis] = np.unique(pos[:, axis], return_inverse=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=5, help="how many puzzles to try")
    ap.add_argument("--rotate", action="store_true", help="randomly rotate each piece before solving")
    ap.add_argument("--core-size", type=int, default=64)
    ap.add_argument("--save", default="", help="save the first solved puzzle to this PNG")
    args = ap.parse_args()

    ds = PuzzleDataset(load_manifest(args.split))
    rng = np.random.default_rng(0)
    accs = []
    for idx in range(min(args.n, len(ds))):
        pieces_img, masks, positions = ds[idx]
        grid = to_grid(positions)
        rows, cols = int(grid[:, 0].max()) + 1, int(grid[:, 1].max()) + 1

        pieces, truth = [], {}
        for k in range(len(pieces_img)):
            img, mask = np.asarray(pieces_img[k]), np.asarray(masks[k]).astype(bool)
            if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
                img = np.moveaxis(img, 0, -1)  # CHW -> HWC
            img = img[..., :3]
            q = int(rng.integers(0, 4)) if args.rotate else 0
            img, mask = np.rot90(img, k=-q), np.rot90(mask, k=-q)  # scan = truth rotated q cw
            pieces.append(piece_from_mask(k, img, mask, core_size=args.core_size))
            truth[k] = (int(grid[k, 0]), int(grid[k, 1]), (-q) % 4)

        res = solve(pieces, rows, cols)
        acc = neighbor_accuracy(res, truth)
        accs.append(acc)
        print(f"puzzle {idx}: {rows}x{cols} = {len(pieces)} pieces, neighbour accuracy {acc:.3f}")

        if args.save and idx == 0:
            import matplotlib.pyplot as plt
            plt.imsave(args.save, np.clip(render(res, pieces), 0, 1))
            print(f"saved {args.save}")

    print(f"mean neighbour accuracy over {len(accs)} puzzles: {np.mean(accs):.3f}")


if __name__ == "__main__":
    main()