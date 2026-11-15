# Robust comparator experiment

An experiment that makes the edge comparator hold up on imperfect pieces, without changing anything under
`interlock/`, `scripts/` or `tests/`. Everything here imports the existing code and adds to it. It lives on the
`greedy` branch, which tracks `main`.

The stock comparator learns from pixel-exact masks, where a true neighbor always interlocks perfectly. That makes
"perfect fit" a shortcut: in the first round (Imagenette, 10×10, results below) it still ranked the true neighbor
first 99.98% of the time with all color removed, and fell to 25% when half the outlines were shrunk by one pixel.

This round moves to `main`'s data and cutting:
- **SA-1B** images (Meta), preprocessed to 400 × 400 by `scripts/preprocess.py`.
- **New cutting:** every puzzle gets its own random grid (30-100 px pieces, 4×4 to 13×13) and a set of 6 tab
  shapes that all its inner edges reuse. Model inputs are padded to 184 × 80.

## What's here

| File | What it does |
|---|---|
| `robust_comparator.py` | Outline error (`perturb_masks`), flat-edged puzzles (`flat_cut`, `FlatPuzzles`), transplant negatives, `RobustPairDataset`, scoring every pair (`score_all_pairs`), stress-test `CONDITIONS`, and temperature scaling |
| `train_robust_comparator.py` | Trains the same `EdgeComparator` on `RobustPairDataset` (MPS by default). Writes `checkpoints/robust_comparator.pt` with its training settings |
| `eval_robust.py` | Scores the baseline and robust checkpoints on held-out puzzles under every condition and on flat-edged puzzles, fits temperatures, runs solver stages 2-4 on each puzzle's own grid. Writes `results/robust_eval.{json,md}` |
| `publish_hf.py` | Builds a Hugging Face release folder (weights, config, model card from the measured results); uploads only with `--upload` |
| `run_all.sh` | Data, both trainings, evaluation and the release folder in one command |
| `test_robust_comparator.py` | Tests for all of the above (`uv run pytest testing`) |
| `results/` | Evaluation output; `results/imagenette_10x10/` holds the first round |

### Training data (`RobustPairDataset`)
- **Outline error on every piece** ("masked pieces"). Each outline grows or shrinks by up to 3.5% of the cell,
  plus an uneven part (std 1.5%), capped at 5%; at least 1 px. New rim pixels take the piece's nearest color, or for
  half the pieces the "mat" color. 20% of pieces stay exact. The cap keeps `side_types` readable: it misreads about
  0.4% of sides at 30 px pieces, almost none above 40 px.
- **Transplant negatives (half of all negatives).** The true neighbor's exact outline filled with another interior
  cell of the same picture. They fit exactly where the positive would, so only color can reject them. With 6 shared
  tab shapes some ordinary wrong pieces also fit exactly; transplants make it half of them.
- **Flat-edged puzzles (1 in 8 + 1 in 8).**
  - `flat_even`: a perfect grid of straight lines. Every edge has the same length, so only color can match.
  - `flat_diagonal`: each end of every inner line slides by up to 20% of a cell, so the lines slant and pieces become
    irregular quadrilaterals with sides of different lengths and angles. Length and angle stop being a reliable cue.
  - Half their wrong pieces come from the anchor's own row (or column), so "same height" can't decide a match.
  - Labels come from the known layout: `side_types` would call every flat side a border, so `PairDataset`'s
    shape-fit rule doesn't apply. The solver doesn't support flat puzzles; they're for training, and the evaluation
    shows how well the comparator matches them (`score_all_pairs` scores every pair, since nothing can be ruled out
    by shape).
- Everything else is as in `PairDataset`: same grids and seeds, flips and channel shuffles, 16 pairs per direction.

### Calibration
- `T_edge`: one temperature for the logits, fitted by maximum likelihood of each piece's true neighbor.
- `T_space`: a grid search for the temperature that makes the solver's stage 3 distributions most likely to hold
  the true piece. `scale_logits` applies either one so the solver sees an exact temperature despite its score floor.
- Both are fitted on dev puzzles under outline error, never on clean cuts.

### Evaluation
- Dev: val images 0-99 (best epoch: 20 of them; temperatures: 50). Test: val images 100-199, never tuned on.
- Tabbed test sets: clean, color removed (a check: a model that uses color should drop), pixel noise, half the
  outlines shrunk 1 px, uneven outline error, smoothed outlines. Flat test sets: `flat_even`, `flat_diagonal`
  (30 puzzles each by default; scoring every pair is slower).
- The solver runs on each puzzle's own grid (`run_stages`); `updaters.solve()` still assumes 10×10.
- Scores are cached in `data/eval/robust_cache/`, keyed by a hash of the data manifest and the checkpoint.

## Run it

```bash
bash testing/run_all.sh              # data, baseline, robust model, evaluation, release folder
SKIP_DATA=1 bash testing/run_all.sh  # data already in data/processed/
```

Before the first run, accept the SA-1B license at https://ai.meta.com/datasets/segment-anything-downloads/ and save
the links file as `data/raw/sa1b_links.txt`. It's personal and expires; never commit it. The steps one at a time:

```bash
uv run python scripts/download.py && uv run python scripts/preprocess.py
uv run python scripts/train_comparator.py --out checkpoints/baseline_comparator.pt   # baseline (main's PairDataset)
uv run python testing/train_robust_comparator.py                                      # robust model
uv run python testing/eval_robust.py
uv run python testing/publish_hf.py
```

On an M4 Pro (24 GB) each model trains in about an hour at 2 puzzles per batch (`run_all.sh`'s default; see the
training notes below for why not 8), and the evaluation takes about an hour.

## Hugging Face

Published: **[dm157/interlock-edge-comparator](https://huggingface.co/dm157/interlock-edge-comparator)** (the robust
model, `T_edge = 0.73`), for research and non-commercial use only under the SA-1B license.

`publish_hf.py` builds `checkpoints/hf_release/` (gitignored): `edge_comparator.pt` (loads with
`load_comparator`), `config.json` (architecture, temperatures, training settings, manifest hash), `robust_eval.md` and
a `README.md` model card written from the measured results. Review it, then:

```bash
hf auth login   # once, with a write token
uv run --with huggingface_hub python testing/publish_hf.py --repo USER/interlock-edge-comparator --upload
```

Add `--private` to create the repo private. SA-1B's license allows research on a non-commercial basis and says
nothing about trained models, so the model card marks the weights research and non-commercial only and links the
license. The license also requires deleting the images within 3 months after the research ends. The local copies were
deleted on 2026-11-11, so retraining or re-running the evaluation needs a fresh SA-1B download first
(`scripts/download.py`, then `scripts/preprocess.py`).

## Results

SA-1B, `main`'s cutting (random 30-100 px grids, 6 tab shapes per puzzle), both models trained the same way
(15 epochs, 2 puzzles per batch, peak learning rate 1e-3) on Apple Silicon. 100 tabbed test puzzles per set and 30
flat-edged ones (val 100+, about 58 pieces each). "Baseline" is `scripts/train_comparator.py` on `main`'s
`PairDataset`; "robust" is `train_robust_comparator.py`. Full tables: `results/robust_eval.md`.

| test set | top-1, baseline | top-1, robust | placed right, baseline | placed right, robust | solved, baseline | solved, robust |
|---|---|---|---|---|---|---|
| clean | 98.6% | 95.5% | 100% | 99.9% | 100 | 99 |
| pixel noise σ=12 | 95.7% | 89.8% | 100% | 99.5% | 100 | 94 |
| half the outlines shrunk 1 px | 28.9% | **91.2%** | 42.3% | **99.0%** | 3 | **95** |
| uneven outline error (≤5% of a cell) | 4.5% | **86.3%** | 30.4% | **97.5%** | 0 | **91** |
| smoothed outlines | 96.9% | 93.9% | 100% | 98.8% | 100 | 95 |
| color removed (diagnostic) | 33.9% | 30.5% | 85.7% | 75.4% | 77 | 64 |

- **Outline error still breaks the baseline, and the robust model fixes it.** At 1 px of error it goes from 3 to
  95 puzzles solved; with uneven error from 0 to 91.
- **The cost on clean pieces is small:** 95.5% top-1 instead of 98.6%, 1 puzzle in 100 not solved. Pixel noise costs
  more (94 solved instead of 100), as in the first round.
- **The new cutting already makes the baseline use color.** With color removed it falls to 33.9% top-1; the old
  stock model stayed at 99.99%. Six shared tab shapes per puzzle mean shape alone often can't decide, so the
  perfect-fit shortcut is mostly gone. What's left is fragility to outline error, which is what the mask error fixes.

### Flat-edged puzzles

| test set | top-1, baseline | top-1, robust | neighbor pairs right, baseline | neighbor pairs right, robust |
|---|---|---|---|---|
| flat, even grid | 84.2% | **86.8%** | 44.1% | 48.3% |
| flat, diagonal cuts | **97.1%** | 90.3% | 50.4% | 47.5% |

- **Both models match flat edges by color,** the baseline without ever having seen one, because `main`'s cutting
  already pushes it onto color. Training on flat puzzles adds a little on the even grid.
- **On diagonal cuts the baseline is better.** With clean outlines a slanted edge only lines up with its true
  neighbor, and the baseline trusts geometry fully; the robust model learned that outlines are noisy and leans on it
  less. Under outline error that trade pays off (the tables above); on perfectly clean slanted cuts it costs ~7 points.
- **The solver can't assemble flat puzzles yet:** about half the neighbor pairs come out right and almost no puzzle is
  solved, because there are no corner or edge shapes to anchor the grid, and stage 3 locks early guesses. That's a
  solver change, not a comparator one.

### Calibration

| model, test set | edge ECE raw → T_edge | lock ECE: stock / T_edge / T_space | placed right: stock / T_edge / T_space | solved: stock / T_edge / T_space |
|---|---|---|---|---|
| baseline, shrunk 1 px | 0.530 → 0.243 | 0.653 / 0.202 / 0.188 | 42.3% / 43.0% / 43.2% | 3 / 4 / 3 |
| baseline, uneven outlines | 0.760 → 0.009 | 0.679 / 0.015 / 0.012 | 30.4% / 30.8% / 31.2% | 0 / 0 / 0 |
| robust, shrunk 1 px | 0.052 → 0.014 | 0.022 / 0.021 / 0.005 | 99.0% / 99.2% / 99.7% | 95 / 97 / 97 |
| robust, uneven outlines | 0.047 → 0.003 | 0.060 / 0.046 / 0.034 | 97.5% / 98.1% / 98.0% | 91 / 92 / 91 |

Temperatures fitted on dev: baseline T_edge 200 and T_space 64 (both at the top of their search range: under outline
error its scores carry almost no information, so the best fit flattens them); robust T_edge 0.73, T_space 1.59.

- **The robust model comes out nearly calibrated** and T_edge takes it the rest of the way (edge ECE 0.047 → 0.003);
  solves go 91 → 92 under uneven outlines, 95 → 97 at 1 px.
- **Calibrating the baseline fixes its confidence but not its answers,** as in the first round.
- **Stage 3 still locks ~100% of spaces for the robust model,** so most of the remaining solver gain is in its
  threshold 1 (to-do below).

### Recommendation
Publish `checkpoints/robust_comparator.pt` with `T_edge = 0.73` (`checkpoints/hf_release/`, built by `publish_hf.py`).

### Notes from training
- `scripts/train_comparator.py`'s default of 8 puzzles per batch (512 windows) needs ~15-20 GB of GPU memory at the
  new 184 × 80 input and swapped on a 24 GB Mac: 35 min per epoch. At 2 puzzles per batch an epoch takes ~3.5 min,
  so each model trains in about an hour. `run_all.sh` passes `--puzzles-per-batch 2 --lr 1e-3` to both models.
- Baseline best: epoch 15, val top-1 98.0% (clean). Robust best: epoch 15, dev top-1 85.1% under uneven outlines and
  94.7% clean. Both were still improving slowly, so more epochs may help.

### First round: Imagenette, fixed 10×10 grid, 40 px pieces

The same ideas on the old data and cutting, without flat-edged puzzles (`results/imagenette_10x10/robust_eval.md`).
100 test puzzles.

| condition | top-1, stock | top-1, robust | placed right, stock | placed right, robust | solved, stock | solved, robust |
|---|---|---|---|---|---|---|
| clean | 99.99% | 97.6% | 100% | 99.2% | 100% | 99% |
| pixel noise σ=12 | 99.99% | 94.9% | 100% | 99.5% | 100% | 96% |
| half the masks shrunk 1 px | 25.4% | **95.1%** | 16.2% | **99.0%** | 0% | **98%** |
| uneven outlines (≤2.5 px) | 6.1% | **85.8%** | 10.7% | **89.2%** | 0% | **64%** |
| color removed (diagnostic) | 99.99% | 68.2% | 100% | 88.8% | 100% | 42% |

Calibration there: a temperature fitted under outline error cut the stock model's lock error (ECE) from 0.71 to 0.03.
The robust model came out nearly calibrated (edge ECE 0.035, 0.004 with T = 0.80). Placements barely changed, because
stage 3's threshold 1 locks almost every space whatever its confidence.

## To do
- [x] Download SA-1B, train both models on `main`'s cutting, run the evaluation, fill in the results above.
- [x] Review the SA-1B license and publish the robust model to Hugging Face (research and non-commercial only).
- [ ] Solver support for flat-edged puzzles (anchor the grid without corner shapes).
- [ ] Fix stage 3's threshold 1 (locks ~every space regardless of confidence) as a new solver variant here.
- [ ] `side_types` has ~0.07 of a cell of margin, about 2 px on 30 px pieces; outline error past that misreads
  tabs and blanks. Needs a change in `interlock/data_collection/pairs.py`.
- [ ] Solve flat-edged puzzles end to end: `score_matrix` and the solver's shape rules assume tabs.
- [ ] Real photos: fit the temperature on real segmentations once there are some.
