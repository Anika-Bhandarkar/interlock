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