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