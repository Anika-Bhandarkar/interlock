"""Tests for the edge comparator model, puzzle scoring, and the training datasets."""

import csv

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from interlock.data_collection.cutting import COLS, ROWS, cut
from interlock.data_collection.dataset import PairDataset, PuzzleDataset, collate_pairs, load_manifest
from interlock.data_collection.pairs import (
    ABOVE,
    BELOW,
    LEFT,
    RIGHT,
    SEAM_HALF_WIDTH,
    can_fit,
    neighbor_ids,
    side_types,
)
from interlock.models.edge_comparator import (
    EdgeComparator,
    evaluate,
    load_comparator,
    neighbor_ranks,
    pick_device,
    save_comparator,
    score_matrix,
)

NUM_PIECES = ROWS * COLS


@pytest.fixture(scope="module")
def puzzle():
    image = np.random.default_rng(7).integers(0, 256, size=(400, 400, 3), dtype=np.uint8)
    return cut(image, seed=3)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return EdgeComparator(width=4).eval()  # tiny: these tests check plumbing, not accuracy


@pytest.fixture(scope="module")
def logits(model, puzzle):
    pieces, masks, _ = puzzle
    return score_matrix(model, pieces, masks)


# ----------------------------------------------------------------------- model


def test_forward_accepts_uint8_and_float(model):
    windows = torch.randint(0, 256, (6, 5, 72, 2 * SEAM_HALF_WIDTH), dtype=torch.uint8)
    from_uint8 = model(windows)
    from_float = model(windows.float() / 255)
    assert from_uint8.shape == (6,)
    torch.testing.assert_close(from_uint8, from_float)


def test_pick_device_respects_explicit_choice():
    assert pick_device("cpu").type == "cpu"
    assert pick_device().type in {"mps", "cuda", "cpu"}


# --------------------------------------------------------------- score_matrix


def test_score_matrix_shape_and_transposes(logits):
    assert logits.shape == (4, NUM_PIECES, NUM_PIECES)
    assert logits.dtype == np.float32
    np.testing.assert_array_equal(logits[LEFT], logits[RIGHT].T)
    np.testing.assert_array_equal(logits[ABOVE], logits[BELOW].T)


def test_score_matrix_rules_out_self_and_misfits(puzzle, logits):
    _, masks, _ = puzzle
    types = side_types(masks)
    for relation in range(4):
        assert np.all(np.isneginf(logits[relation].diagonal()))
        fits = can_fit(types, relation)
        assert np.all(np.isfinite(logits[relation][fits]))
        assert np.all(np.isneginf(logits[relation][~fits]))


def test_score_matrix_true_neighbors_never_ruled_out(puzzle, logits):
    _, _, positions = puzzle
    for relation in range(4):
        for piece_id, neighbor_id in neighbor_ids(positions, relation).items():
            assert np.isfinite(logits[relation, piece_id, neighbor_id])


def test_score_matrix_ignores_piece_order(model, puzzle, logits):
    """Shuffling the pieces just shuffles the matrix: nothing leaks from the order pieces arrive in."""
    pieces, masks, _ = puzzle
    order = np.random.default_rng(0).permutation(NUM_PIECES)
    shuffled = score_matrix(model, pieces[order], masks[order])
    np.testing.assert_allclose(shuffled, logits[:, order][:, :, order], rtol=1e-4, atol=1e-4)


def test_score_matrix_batch_size_does_not_matter(model, puzzle, logits):
    pieces, masks, _ = puzzle
    np.testing.assert_allclose(score_matrix(model, pieces, masks, batch_size=97), logits, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------- evaluation


def test_neighbor_ranks_perfect_scores(puzzle):
    _, _, positions = puzzle
    oracle = np.full((4, NUM_PIECES, NUM_PIECES), -np.inf, dtype=np.float32)
    for relation in range(4):
        for piece_id, neighbor_id in neighbor_ids(positions, relation).items():
            oracle[relation, piece_id, neighbor_id] = 1.0
    ranks = neighbor_ranks(oracle, positions)
    assert set(ranks) == {"right", "below"}
    assert all(np.all(direction_ranks == 1) for direction_ranks in ranks.values())
    assert len(ranks["right"]) == ROWS * (COLS - 1)


def test_neighbor_ranks_ties_count_against_true_neighbor(puzzle):
    _, _, positions = puzzle
    ranks = neighbor_ranks(np.zeros((4, NUM_PIECES, NUM_PIECES), dtype=np.float32), positions)
    assert np.all(ranks["right"] == NUM_PIECES)  # a constant score ranks the true neighbor last, not first


def test_save_and_load_roundtrip(tmp_path, model, puzzle, logits):
    path = tmp_path / "nested" / "comparator.pt"
    save_comparator(model, path, epoch=3, metrics={"right": {"top1": 0.5}})
    loaded = load_comparator(path)
    pieces, masks, _ = puzzle
    np.testing.assert_allclose(score_matrix(loaded, pieces, masks), logits, rtol=1e-5, atol=1e-5)
    assert torch.load(path, weights_only=True)["epoch"] == 3


# ------------------------------------------------------------------ datasets


@pytest.fixture(scope="module")
def processed_dir(tmp_path_factory):
    """A tiny processed/ folder: 3 train and 2 val noise images plus a manifest."""
    root = tmp_path_factory.mktemp("processed")
    rng = np.random.default_rng(0)
    rows = []
    for split, count in [("train", 3), ("val", 2)]:
        for index in range(count):
            path = f"{split}/sa_{index}.png"
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rng.integers(0, 256, size=(400, 400, 3), dtype=np.uint8)).save(root / path)
            rows.append({"path": path, "split": split, "source": "sa_000000", "orig_width": 400, "orig_height": 400})
    with open(root / "manifest.csv", "w", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return root


def test_load_manifest_filters_split(processed_dir):
    assert len(load_manifest("train", processed_dir)) == 3
    assert len(load_manifest("val", processed_dir)) == 2


def test_puzzle_dataset_follows_seed_rule(processed_dir):
    puzzles = PuzzleDataset(load_manifest("val", processed_dir), processed_dir)
    pieces, masks, positions = puzzles[1]
    expected = cut(puzzles.load_image(1), seed=1)
    np.testing.assert_array_equal(masks, expected[1])
    assert pieces.shape[0] == NUM_PIECES


def test_pair_dataset_item(processed_dir):
    pairs = PairDataset(load_manifest("train", processed_dir), pairs_per_puzzle=5, augment=False,
                        processed_dir=processed_dir)
    windows, labels = pairs[0]
    assert windows.shape == (20, 5, 72, 2 * SEAM_HALF_WIDTH) and windows.dtype == torch.uint8
    assert labels.tolist() == [1.0] * 5 + [0.0] * 5 + [1.0] * 5 + [0.0] * 5
    # positives interlock: no pixel claimed by both pieces; every negative overlaps or leaves a gap somewhere
    overlap = ((windows[:, 3] > 0) & (windows[:, 4] > 0)).sum(dim=(1, 2))
    assert torch.all(overlap[labels == 1] == 0)


def test_pair_dataset_is_reproducible_and_changes_each_epoch(processed_dir):
    pairs = PairDataset(load_manifest("train", processed_dir), pairs_per_puzzle=4, processed_dir=processed_dir)
    first, _ = pairs[1]
    again, _ = pairs[1]
    torch.testing.assert_close(first, again)
    pairs.set_epoch(1)
    later, _ = pairs[1]
    assert not torch.equal(first, later)


def test_augmentation_keeps_masks_consistent(processed_dir):
    """Flips swap or mirror masks, but a positive pair still never overlaps."""
    pairs = PairDataset(load_manifest("train", processed_dir), pairs_per_puzzle=8, augment=True,
                        processed_dir=processed_dir)
    windows, labels = pairs[2]
    overlap = ((windows[:, 3] > 0) & (windows[:, 4] > 0)).sum(dim=(1, 2))
    assert torch.all(overlap[labels == 1] == 0)


def test_dataloader_with_spawned_workers(processed_dir):
    """macOS starts DataLoader workers with spawn; make sure everything pickles and batches."""
    pairs = PairDataset(load_manifest("train", processed_dir), pairs_per_puzzle=3, processed_dir=processed_dir)
    loader = DataLoader(pairs, batch_size=2, num_workers=2, collate_fn=collate_pairs,
                        multiprocessing_context="spawn")
    sizes = [len(labels) for _, labels in loader]
    assert sizes == [24, 12]


def test_evaluate_reports_both_directions(model, processed_dir):
    metrics = evaluate(model, PuzzleDataset(load_manifest("val", processed_dir), processed_dir), num_puzzles=1)
    assert set(metrics) == {"right", "below"}
    for direction_metrics in metrics.values():
        assert 0 <= direction_metrics["top1"] <= 1
        assert direction_metrics["mean_rank"] >= 1
