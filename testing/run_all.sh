#!/usr/bin/env bash
# The whole experiment in one command, on this machine (Apple Silicon MPS by default, else CUDA, else CPU):
# SA-1B data -> baseline comparator -> robust comparator -> evaluation -> Hugging Face release folder (not uploaded).
#
# Before the first run, accept the SA-1B license at https://ai.meta.com/datasets/segment-anything-downloads/ and save
# the links file it gives you as data/raw/sa1b_links.txt (it's personal and expires; never commit it).
#
# Usage: bash testing/run_all.sh            # everything
#        SKIP_DATA=1 bash testing/run_all.sh # data already downloaded and preprocessed
#        EPOCHS=5 bash testing/run_all.sh    # shorter training
set -euo pipefail
cd "$(dirname "$0")/.."

EPOCHS="${EPOCHS:-15}"
# 184 x 80 windows take ~4x the memory of the old 72 x 48 ones. The scripts' default of 8 puzzles per batch
# (512 windows) needs ~15-20 GB of GPU memory and swaps on a 24 GB Mac, which made an epoch take 35 min.
# 2 puzzles per batch (128 windows) fits; the peak learning rate drops with it.
PUZZLES_PER_BATCH="${PUZZLES_PER_BATCH:-2}"
LR="${LR:-1e-3}"
TRAIN_ARGS=(--epochs "$EPOCHS" --puzzles-per-batch "$PUZZLES_PER_BATCH" --lr "$LR")
DATA_DIR="${INTERLOCK_DATA_DIR:-data}"

if [ "${SKIP_DATA:-0}" != "1" ]; then
  if [ ! -f "$DATA_DIR/raw/sa1b_links.txt" ]; then
    echo "Missing $DATA_DIR/raw/sa1b_links.txt: accept the SA-1B license and save its links file there first." >&2
    exit 1
  fi
  uv run python scripts/download.py
  uv run python scripts/preprocess.py
fi

uv run pytest -q tests testing
uv run python scripts/train_comparator.py "${TRAIN_ARGS[@]}" --out checkpoints/baseline_comparator.pt \
  | tee checkpoints/baseline_train.log
uv run python testing/train_robust_comparator.py "${TRAIN_ARGS[@]}" --out checkpoints/robust_comparator.pt \
  | tee checkpoints/robust_train.log
uv run python testing/eval_robust.py
uv run python testing/publish_hf.py
echo "Done. Results: testing/results/robust_eval.md. Release folder: checkpoints/hf_release (not uploaded)."
