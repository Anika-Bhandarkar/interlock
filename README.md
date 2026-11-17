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
Images come from [SA-1B](https://ai.meta.com/datasets/segment-anything-downloads/), Meta's dataset of high-resolution images. Everything under `data/` is gitignored.

SA-1B is license-gated, so everyone who uses the repo downloads it themselves:

1. Accept the license on the SA-1B page above and download the links file it gives you. Save it as `data/raw/sa1b_links.txt`. The links are signed and tied to you, so don't commit or share the file. They also expire; if the download fails with a 403, get a fresh links file.
2. Download (~1.45 GB) into `data/raw/sa1b/`. Each tar is ~11 GB, but it is streamed: only the `.jpg` images are kept, the full tar is never written to disk, and the download stops once `--max-images` images (default 1800) have arrived. Use `--num-tars` to read from more than one tar. Safe to re-run; it skips work that's already done.
   ```bash
   uv run python scripts/download.py
   ```
3. Preprocess into 400x400 RGB PNGs in `data/processed/`, plus `data/processed/manifest.csv` listing every image (`path`, `split`, `source` (the tar it came from), `orig_width`, `orig_height`). Images are resized so the shorter side is 400 and center-cropped; images smaller than 400px are skipped.
   ```bash
   uv run python scripts/preprocess.py
   ```
   Defaults: 1500 train and 300 val images, seed 0. Change with `--num-train`, `--num-val`, `--seed`, `--size` (must be divisible by the default 10x10 grid). The manifest is rewritten on every run and is the source of truth for which images are in the dataset.

SA-1B is licensed for research use only, so don't redistribute the images.

### Puzzles
Puzzles are cut on the fly when the dataset is read, so each image can become a different puzzle every epoch (`interlock/data_collection/dataset.py`, `PuzzleDataset`):
- `grid.py` draws a cell size from 30, 35, ..., 100 px (`CELL_MIN`, `CELL_STEP`, `CELL_MAX` in `cutting.py`) and crops the image to the largest whole grid of square cells, so puzzle size and piece count vary (at least 4 pieces per side).
- `cutting.py` first draws the puzzle's set of 6 tab shapes (`tab_shapes`), from textbook to funky, then cuts the crop into that grid: every inner edge uses one shape from the set, mirrored or not, with the tab on a random side. Sharing shapes means the model can't match pieces by shape alone. Grid-line jitter is currently disabled (`GRID_JITTER = 0.0`), so cells within a puzzle have equal dimensions. Each piece comes in a box of `cell + 2 * PAD` pixels placed on the unjittered grid (`PAD = 42`, enough for the tabs of the largest pieces plus the jitter).
- Puzzle `i` uses seed `i + seed_offset` for both the grid and the tabs, so the val set is reproducible and training gets new grids and tabs every epoch.
- Before anything goes into a model it is zero-padded, centered, to one fixed size with `pairs.pad_to`, so puzzles of every cell size batch together: whole pieces to `PIECE_SIZE` x `PIECE_SIZE` (184 x 184), seam windows to `WINDOW_SHAPE` (184 x 80). Pieces are never resized, so pixels keep their real scale.

To get the grid of a puzzle from the dataset: `rows, cols = positions.max(axis=0)[::-1] + 1`, and the cell size is `pairs.cell_size(pieces.shape[1])`.

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
  mask, and builds the seam window for a pair: RGB plus each piece's mask, cropped around the seam
  (40 px each side, less for cells under 70 px so the window stops short of the pieces' far sides), then
  padded to `WINDOW_SHAPE`. Masks let the model see tab fit; colors let it see the picture continue.
- `interlock/data_collection/dataset.py`: `PuzzleDataset` (whole puzzles with random grids, see
  [Puzzles](#puzzles)) and `PairDataset` (training pairs: true neighbors vs. wrong pieces from the same
  puzzle whose shape fits, re-cut with new grids and tab shapes every epoch).
- `interlock/models/edge_comparator.py`: the model, `score_matrix`, `evaluate`, save/load.
## Solver stages 2-4
The solver supports variable square and rectangular boards. Dataset entry points read the known board
size from metadata before shuffling and pass only `rows` and `cols` to the solver; ground-truth piece
positions are used only to evaluate the answer. Direct calls can use
`solve(logits, types, rows=rows, cols=cols)`; omitted dimensions retain the 10x10 default.
Input dimensions are checked against the score matrix and piece count before solving. The visualizers
use the actual board dimensions, and `solve.py` also reports final accuracy and total scoring/solving
time grouped by cell size and grid dimensions.

Comparator checkpoints from before the variable-grid change need retraining (the input is now 184 x 80).
**Historical results below are from the earlier Imagenette, fixed 10x10 setup; they are not validation of
the current variable-grid dataset.**

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
### Single-tab-shape experiment

`--single-tab-shape` replaces the six randomized curves with one fixed, symmetric canonical curve,
shared by all internal edges and all puzzles; mirroring is disabled. Tab/blank directions remain random,
and flat borders and shape-compatibility filtering remain available. This reduces exact-curve clues
without removing the border anchor or changing to straight seams. It does not eliminate all geometric
information or prove that earlier performance was overfitting.

First evaluate the existing checkpoint on the changed geometry:

```bash
uv run python scripts/solve.py --single-tab-shape --num-puzzles 5
uv run python scripts/visualize.py --single-tab-shape --out single_tab_pieces.png
```

For a matched training/evaluation experiment, train a separate model (training and validation both use
the canonical shape), then select that checkpoint explicitly:

```bash
uv run python scripts/train_comparator.py --single-tab-shape --epochs 2 --num-train 200 --puzzles-per-batch 2 --workers 2
uv run python scripts/solve.py --single-tab-shape --checkpoint checkpoints/edge_comparator_single_tab.pt --num-puzzles 30
uv run python scripts/animate_solve.py --single-tab-shape --checkpoint checkpoints/edge_comparator_single_tab.pt --index 0 --out single_tab.gif
```

The default output for single-tab training is `checkpoints/edge_comparator_single_tab.pt`, preserving
the ordinary training checkpoint unless you explicitly override `--out`. Use the same seed, image subset,
and training settings for comparisons. `--single-tab-shape` cannot be combined with `--straight-edges`.

### Straight-edge experiment

Use `--straight-edges` to cut each image into rectangular grid tiles instead of tabbed jigsaw pieces.
The same crop and grid seeds are used in both modes for comparison. Existing checkpoints can run unchanged:
seam inputs retain RGB plus two masks and the same 184x80 padded dimensions. Flat-to-flat pairs are
scored in this mode, and flat-side border restrictions are disabled in every solver stage. Known board
dimensions and upright piece orientation remain assumptions; no corner identity is supplied. The initial
seed is still chosen by its best right/below scores, so it can start with the wrong corner without border
clues. This experiment measures transfer from tabbed training data; accuracy may be lower.

```bash
uv run python scripts/solve.py --straight-edges --num-puzzles 30
uv run python scripts/visualize.py --straight-edges --out straight_pieces.png
uv run python scripts/visualize_solve.py --straight-edges --index 0 --out straight_solve.png
uv run python scripts/animate_solve.py --straight-edges --index 0 --out straight_solve.gif
```

Omit the flag for ordinary jigsaw behavior. Dataset puzzles vary their cell size; `--image` in the
visualizers retains the default 10x10 grid. Straight tiles allow more candidate pairs, which may increase
scoring time. Training remains on tabbed pieces; this flag is for generation, evaluation, and display.

To isolate the initial-corner choice, add `--anchor-top-left`:

```bash
uv run python scripts/solve.py --straight-edges --anchor-top-left --num-puzzles 5
uv run python scripts/animate_solve.py --straight-edges --anchor-top-left --index 0 --out anchored.gif
```

This evaluation mode reads only the true top-left piece ID from the shuffled ground-truth metadata,
fixes it at (0, 0), and reserves it throughout all solver stages. Other positions are not supplied to the
solver. Summaries and visualizers label anchored runs explicitly. Compare with the same command without
`--anchor-top-left`, using the same checkpoint and seed. This is an assisted assembly diagnostic; direct
neighbor-ranking evaluation is still the more isolated measure of CNN matching ability.

### Historical fixed-grid results

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


`solve.py` reports CNN neighbor `top1`, `mean_rank`, and query counts overall, separately for right/below,
and by cell/grid size, reusing the score matrix already computed. Each progress line also reports this
puzzle's CNN ranking. Ranking queries are pooled (larger puzzles contribute more queries); assembly
percentages remain averages over puzzles. Ties count against the true neighbor. These rankings include
the existing geometry-based candidate filtering, so straight and tabbed modes have different candidate
sets; they are not an RGB-only controlled comparison. The anchor is not used to improve CNN rankings.
With `--noise`, rankings describe the perturbed scores supplied to the solver. Assembly `neighbors`
continues to mean correct adjacencies on the assembled board, not CNN top-1 accuracy.
