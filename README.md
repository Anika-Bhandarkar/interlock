# interlock
Training a model to solve a jigsaw puzzle

## Setup
Install [uv](https://docs.astral.sh/uv/) once:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Then, from the repo root:
```bash
uv sync
```
This creates `.venv/` with the right Python and all dependencies. Run things with `uv run`, e.g. `uv run python scripts/preprocess.py` or `uv run pytest`. In VS Code, select `.venv/bin/python` as the interpreter.

Add a dependency with `uv add <package>` (or `uv add --dev <package>` for dev-only tools) and commit both `pyproject.toml` and `uv.lock`.

## Data
Images come from [Imagenette](https://github.com/fastai/imagenette), a free 10-class subset of ImageNet (no account needed). Everything under `data/` is gitignored.

1. Download (~1.45 GB) into `data/raw/imagenette2/`. Safe to re-run; it skips steps that are already done.
   ```bash
   uv run python scripts/download.py
   ```
2. Preprocess into 400x400 RGB PNGs in `data/processed/`, plus `data/processed/manifest.csv` listing every image (`path`, `split`, `class_id`, `orig_width`, `orig_height`). Images are resized so the shorter side is 400 and center-cropped; images smaller than 400px are skipped.
   ```bash
   uv run python scripts/preprocess.py
   ```
   Defaults: 1500 train and 300 val images, seed 0. Change with `--num-train`, `--num-val`, `--seed`, `--size` (must be divisible by the puzzle's rows and columns). The manifest is rewritten on every run and is the source of truth for which images are in the dataset.

To keep data somewhere other than the repo's `data/` folder (e.g. on a GPU machine), set `INTERLOCK_DATA_DIR` before running either script:
```bash
INTERLOCK_DATA_DIR=/path/to/data uv run python scripts/preprocess.py
```

scripts/visualize.py shows what happens when interlock/data_collection/cutting.py runs. Run with `uv run python scripts/visualize.py --image path/to/photo.jpg`

Onboarding notebook is located at `notebooks/00_intro_adjacency.ipynb`.

## Edge comparator (solver stage 1)
A CNN that scores whether piece B is piece A's right neighbor, built from the onboarding notebook's
approach. "Below" is handled by rotating both pieces, so one model covers all four directions.

Train it (uses the Apple Silicon GPU via Metal/MPS when available, else CUDA, else CPU):
```bash
uv run python scripts/train_comparator.py              # defaults: 15 epochs, all train images
uv run python scripts/train_comparator.py --epochs 2 --num-train 200   # quick check
```
The best epoch by val top-1 is saved to `checkpoints/edge_comparator.pt` (gitignored). Each epoch prints
top-1 accuracy and mean rank of true neighbors on val puzzles, the same metric as the notebook.

Use it in the solver:
```python
from interlock.models.edge_comparator import load_comparator, pick_device, score_matrix

model = load_comparator("checkpoints/edge_comparator.pt", pick_device())
logits = score_matrix(model, pieces, masks)  # (4, N, N); RELATIONS = right, below, left, above
# logits[r, i, j]: how likely piece j is piece i's r-neighbor (higher = better)
# sigmoid -> match probability; softmax over j -> distribution for the updaters
# -inf where shapes can't interlock (tab must meet blank) or i == j
```

How it works:
- `interlock/data_collection/pairs.py`: reads each side's shape (flat / tab / blank) straight off the
  mask, and builds the seam window for a pair: RGB plus each piece's mask, cropped around the seam.
  Masks let the model see tab fit; colors let it see the picture continue.
- `interlock/data_collection/dataset.py`: `PuzzleDataset` (whole puzzles, notebook seed rule) and
  `PairDataset` (training pairs: true neighbors vs. wrong pieces from the same puzzle whose shape fits,
  re-cut with new tab shapes every epoch).
- `interlock/models/edge_comparator.py`: the model, `score_matrix`, `evaluate`, save/load.
## Solver stages 2-4
`interlock/solver/updaters.py` follows the team plan ("Interlock Approach Deepdive: CNNs + Greedy
Algorithms"). A *space* is a slot on the board, its *distribution* is a probability over pieces, and its
*max-piece* is the most likely piece.
- **Stage 2, `first_updater`:** pencil in a guess for every space, like sudoku. Start at the top-left
  corner; a priority queue of empty spaces next to filled ones is ordered by the confidence of each space's
  max-piece. Each space is scored against its neighbors' max-pieces. Pieces aren't taken out, so a piece
  can be the max-piece of several spaces.
- **Stage 3, `multistage_updater`:** each round re-scores every space against the max-pieces of all 4
  neighbors, then applies the plan's thresholds: (1) a piece that is the max-piece of exactly one space for
  2 rounds is locked there and removed from other spaces; (2) once no max-piece has moved for 2 rounds,
  stop; (3) a space whose max-piece has probability > 0.6 is locked.
- **Stage 4, `final_assignment`:** the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) fills
  the unlocked spaces so that no piece is used twice.

A space's distribution comes from its neighbors' edge scores: by default the comparator's raw scores
(logits) are averaged over the neighbors, then softmaxed over candidates (`--combine mean`).
`--combine product` multiplies the neighbors' match probabilities instead. Corner and edge pieces only go
in border spaces with matching flat sides.

```bash
uv run python scripts/solve.py                  # accuracy on the first 30 val puzzles, per stage
uv run python scripts/solve.py --noise 8        # same, with noise added to the comparator's scores
uv run python scripts/animate_solve.py --index 3 --out solve.gif     # animated solve (or no --out: window)
uv run python scripts/visualize_solve.py --index 5 --noise 10 --out solve.png   # static summary figure
```
With the trained comparator all 30 val puzzles are solved perfectly, in about 3 ms per puzzle after the
~0.7 s `score_matrix` call. Adding noise to the scores shows how the solver copes with a weaker comparator
(30 val puzzles, final result):

| `--noise` | placed right | puzzles solved | locked but wrong |
|---|---|---|---|
| 6 | 99.7% | 97% | 0.3% |
| 8 | 83.1% | 60% | 16.9% |
| 10 | 57.8% | 10% | 42.2% |

Stage 3 locks pieces its distribution is sure about, so it relies on the comparator's scores being
calibrated: when they are confidently wrong (high `--noise`), it locks wrong pieces too. Raising threshold
3 or turning it off changed accuracy by under 1.5 points in these tests. If a future comparator is less
accurate, temperature-scale its logits on val data before solving.

`animate_solve.py` shows the shuffled pieces on the left and the board on the right: each space shows its
max-piece under a grey overlay once it has a distribution, and turns full color once set (locked by a
threshold, or placed by stage 4), when the piece also fades out of the pile.
