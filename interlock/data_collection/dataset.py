"""torch Datasets: read data/processed/manifest.csv and cut puzzles on the fly in __getitem__.

PuzzleDataset gives whole puzzles (for evaluation and the solver). PairDataset gives labeled seam windows
for training the edge comparator.

Both are plain picklable classes so DataLoader workers work on macOS, where workers are spawned fresh and
receive a pickled copy of the dataset.
"""

import csv
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from interlock.data_collection.cutting import cut
from interlock.data_collection.grid import random_grid
from interlock.data_collection.pairs import (
    DIRECTIONS,
    RELATIONS,
    WINDOW_SHAPE,
    can_fit,
    neighbor_ids,
    orient,
    pad_to,
    seam_windows,
    side_types,
    to_rgba,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("INTERLOCK_DATA_DIR", REPO_ROOT / "data"))
PROCESSED_DIR = DATA_DIR / "processed"


def load_manifest(split: str, processed_dir: Path = PROCESSED_DIR) -> list[dict]:
    """Manifest rows for one split ("train" or "val"), in manifest order."""
    manifest_path = Path(processed_dir) / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found; run scripts/download.py and scripts/preprocess.py")
    with open(manifest_path) as manifest_file:
        return [row for row in csv.DictReader(manifest_file) if row["split"] == split]


class PuzzleDataset(Dataset):
    """Item i is image i of `rows` cut into a puzzle: (pieces, masks, positions) as returned by cut().

    Each puzzle gets its own grid from random_grid (cell size and crop), so puzzles differ in piece count and
    piece size; the grid is positions.max(axis=0) + 1 and the cell is pairs.cell_size(box).

    Seed rule (same as the onboarding notebook): puzzle i uses seed i + seed_offset, for both its grid and its
    cut, so seed_offset=0 always gives the same puzzles. PairDataset changes the offset every epoch to get new
    grids and tab shapes.
    """

    def __init__(self, rows: list[dict], processed_dir: Path = PROCESSED_DIR, seed_offset: int = 0):
        self.rows = rows
        self.processed_dir = Path(processed_dir)
        self.seed_offset = seed_offset

    def __len__(self) -> int:
        return len(self.rows)

    def load_image(self, index: int) -> np.ndarray:
        return np.array(Image.open(self.processed_dir / self.rows[index]["path"]).convert("RGB"))

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        seed = index + self.seed_offset
        # a separate stream from cut's default_rng(seed), so the grid draw isn't correlated with the tab shapes
        crop, rows, cols = random_grid(self.load_image(index), np.random.default_rng([seed, 1]))
        return cut(crop, seed, rows, cols)


class PairDataset(Dataset):
    """Training pairs for the edge comparator. Item i is puzzle i turned into 4 * pairs_per_puzzle windows.

    For each direction in DIRECTIONS (rotated so it reads as "right"), we pick pairs_per_puzzle anchor
    pieces A and make two pairs from each: (A, its true neighbor) labeled 1 and (A, a wrong piece) labeled 0.
    Wrong pieces come from the same puzzle and have a shape that fits (a tab facing a blank). Pairs whose
    shapes can't fit are ruled out by score_matrix before the model sees them, so training on them would
    only teach the model what it never needs to know.

    Call set_epoch(epoch) before each epoch: puzzles are re-cut with new tab shapes and new pairs are
    drawn. DataLoader copies the dataset into its workers when an epoch starts, so don't use
    persistent_workers or they'll keep the old epoch.

    Returns windows (4 * pairs_per_puzzle, 5, *WINDOW_SHAPE) uint8, padded so every puzzle gives the same
    shape whatever its cell size, and labels (same length,) float32. Use collate_pairs to batch several puzzles.
    """

    def __init__(
        self,
        rows: list[dict],
        pairs_per_puzzle: int = 16,
        augment: bool = True,
        processed_dir: Path = PROCESSED_DIR,
        seed: int = 0,
    ):
        self.puzzles = PuzzleDataset(rows, processed_dir)
        self.pairs_per_puzzle = pairs_per_puzzle
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.puzzles.seed_offset = epoch * len(self.puzzles)  # every (epoch, puzzle) gets its own seed

    def __len__(self) -> int:
        return len(self.puzzles)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng([self.seed, self.epoch, index])
        pieces, masks, positions = self.puzzles[index]
        types = side_types(masks)
        num_pieces = len(pieces)

        all_windows, all_labels = [], []
        for direction in DIRECTIONS:
            relation = RELATIONS.index(direction)
            neighbors = neighbor_ids(positions, relation)
            fits = can_fit(types, relation)
            num_pairs = min(self.pairs_per_puzzle, len(neighbors))
            anchors = rng.choice(list(neighbors), size=num_pairs, replace=False)

            wrong_ids = []
            for anchor in anchors:
                candidates = np.flatnonzero(fits[anchor])
                candidates = candidates[candidates != neighbors[anchor]]
                if len(candidates) == 0:  # rare: nothing else fits, so fall back to any other piece
                    candidates = np.setdiff1d(np.arange(num_pieces), [anchor, neighbors[anchor]])
                wrong_ids.append(rng.choice(candidates))

            left_ids = np.concatenate([anchors, anchors])
            right_ids = np.array([neighbors[anchor] for anchor in anchors] + wrong_ids)
            all_windows.append(seam_windows(to_rgba(*orient(pieces, masks, relation)), left_ids, right_ids))
            all_labels.append(np.repeat([1.0, 0.0], num_pairs))

        windows = torch.cat(all_windows)
        labels = torch.from_numpy(np.concatenate(all_labels).astype(np.float32))
        if self.augment:
            windows = random_flips_and_colors(windows, rng)
        return pad_to(windows, *WINDOW_SHAPE), labels


def random_flips_and_colors(windows: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Augmentations that keep the label: flips and an RGB channel shuffle.

    Flipping top to bottom is a pair from a mirrored image. Flipping left to right is too, but the right
    piece is now on the left, so the two mask channels swap. The channel shuffle is one per puzzle.
    """
    windows = windows.clone()
    flip_rows = torch.from_numpy(rng.random(len(windows)) < 0.5)
    windows[flip_rows] = windows[flip_rows].flip(-2)
    flip_columns = torch.from_numpy(rng.random(len(windows)) < 0.5)
    windows[flip_columns] = windows[flip_columns].flip(-1)[:, [0, 1, 2, 4, 3]]
    color_order = [*rng.permutation(3).tolist(), 3, 4]
    return windows[:, color_order]


def collate_pairs(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate PairDataset items (each a puzzle's worth of pairs) into one batch."""
    windows, labels = zip(*batch)
    return torch.cat(windows), torch.cat(labels)
