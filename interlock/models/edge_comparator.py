"""CNN edge comparator: scores how likely piece B is piece A's right neighbor.

This is the first stage of the greedy solver. score_matrix runs it over every ordered pair of pieces and
returns logits[r, i, j] for every relation r in RELATIONS, i.e. "how likely piece j is piece i's r-neighbor"
(the D[r][i][j] table from the brainstorm, except higher = better fit):
- sigmoid(logits) is the probability that a pair matches
- softmax(logits[r, i]) is a distribution over which piece sits on side r of piece i, for the updaters
- pairs whose shapes can't interlock (and i == j) are -inf, so they get probability 0
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from interlock.data_collection.pairs import (
    ABOVE,
    BELOW,
    DIRECTIONS,
    LEFT,
    RELATIONS,
    RIGHT,
    SEAM_HALF_WIDTH,
    can_fit,
    neighbor_ids,
    orient,
    seam_windows,
    side_types,
    to_rgba,
)


def pick_device(preference: str = "auto") -> torch.device:
    """Apple Silicon GPU (Metal) if available, else CUDA, else CPU. Pass a name to force one."""
    if preference != "auto":
        return torch.device(preference)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Two 3x3 conv + batchnorm + ReLU layers, then halve the resolution."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class EdgeComparator(nn.Module):
    """Logit that B is A's right neighbor, from the seam window between them (pairs.seam_windows).

    Input channels: RGB, A's mask, B's mask. Color covers the picture continuing across the seam; the masks
    cover shape, since true neighbors interlock exactly and any overlap or gap is evidence against a match.

    Args:
        width: channels in the first block (doubles in each of the 3 blocks).
        input_shape: (height, width) of a window; (72, 48) for 400 px images cut 10 x 10.
    """

    def __init__(self, width: int = 32, input_shape: tuple[int, int] = (72, 2 * SEAM_HALF_WIDTH)):
        super().__init__()
        self.config = {"width": width, "input_shape": tuple(input_shape)}
        self.features = nn.Sequential(
            conv_block(5, width), conv_block(width, 2 * width), conv_block(2 * width, 4 * width)
        )
        # size the head from a dummy pass instead of adaptive pooling, which MPS only supports for even splits
        with torch.no_grad():
            num_features = self.features(torch.zeros(1, 5, *input_shape)).numel()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """windows: (n, 5, H, W), uint8 straight from seam_windows or float in [0, 1]. Returns (n,) logits."""
        x = windows.float().div(255) if windows.dtype == torch.uint8 else windows
        return self.head(self.features(x)).squeeze(1)


@torch.no_grad()
def score_matrix(model: EdgeComparator, pieces: np.ndarray, masks: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """(4, N, N) float32 logits: logits[r, i, j] scores piece j as piece i's RELATIONS[r] neighbor.

    Only "right" and "below" go through the model (pieces rotated so "below" reads as "right");
    "left" and "above" are their transposes. Pairs that can't interlock and i == j are -inf.

    Args:
        pieces, masks: as returned by cut(), in any order (the solver sees them shuffled).
    """
    model.eval()
    device = next(model.parameters()).device
    num_pieces = len(pieces)
    types = side_types(masks)
    logits = np.full((4, num_pieces, num_pieces), -np.inf, dtype=np.float32)

    for direction in DIRECTIONS:
        relation = RELATIONS.index(direction)
        rgba = to_rgba(*orient(pieces, masks, relation)).to(device)
        left_ids, right_ids = np.nonzero(can_fit(types, relation))
        for start in range(0, len(left_ids), batch_size):
            batch = slice(start, start + batch_size)
            windows = seam_windows(rgba, left_ids[batch], right_ids[batch])
            logits[relation, left_ids[batch], right_ids[batch]] = model(windows).float().cpu().numpy()

    logits[LEFT] = logits[RIGHT].T  # j left of i  <=>  i right of j
    logits[ABOVE] = logits[BELOW].T
    return logits


def neighbor_ranks(logits: np.ndarray, positions: np.ndarray) -> dict[str, np.ndarray]:
    """Rank (1 = best) of each piece's true neighbor among all candidates, for "right" and "below".

    Same metric as the onboarding notebook, except ties count against the true neighbor, so a model that
    scores everything the same can't look perfect.
    """
    ranks = {}
    for direction in DIRECTIONS:
        relation = RELATIONS.index(direction)
        ranks[direction] = np.array([
            np.sum(logits[relation, piece_id] >= logits[relation, piece_id, neighbor_id])  # includes itself
            for piece_id, neighbor_id in neighbor_ids(positions, relation).items()
        ])
    return ranks


def evaluate(model: EdgeComparator, puzzles, num_puzzles: int | None = None) -> dict[str, dict[str, float]]:
    """Top-1 accuracy and mean rank of true neighbors over the first num_puzzles of a PuzzleDataset."""
    num_puzzles = len(puzzles) if num_puzzles is None else min(num_puzzles, len(puzzles))
    all_ranks = {direction: [] for direction in DIRECTIONS}
    for index in range(num_puzzles):
        pieces, masks, positions = puzzles[index]
        for direction, ranks in neighbor_ranks(score_matrix(model, pieces, masks), positions).items():
            all_ranks[direction].append(ranks)
    metrics = {}
    for direction, ranks in all_ranks.items():
        ranks = np.concatenate(ranks)
        metrics[direction] = {"top1": round(float(np.mean(ranks == 1)), 4), "mean_rank": round(float(ranks.mean()), 3)}
    return metrics


def save_comparator(model: EdgeComparator, path: str | Path, **extra) -> None:
    """Save weights and config (plus any extra plain values, e.g. metrics) so load_comparator can rebuild it."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state_dict = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    torch.save({"config": model.config, "state_dict": state_dict, **extra}, path)


def load_comparator(path: str | Path, device: str | torch.device = "cpu") -> EdgeComparator:
    """Rebuild a saved EdgeComparator on `device`, ready for score_matrix."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = EdgeComparator(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()
