"""Turn raw Imagenette photos into square RGB PNGs that cut() accepts, plus a manifest listing them.

Each kept image: converted to RGB, resized so its shorter side is --size, center-cropped to
--size x --size, and saved to data/processed/<split>/<class_id>/<name>.png. Images smaller than
--size are skipped rather than upscaled. manifest.csv is rewritten from scratch on every run and is
the source of truth for which images are in the dataset.

Usage: uv run python scripts/preprocess.py [--size 400] [--num-train 1500] [--num-val 300] [--seed 0]
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from PIL import Image

from interlock.data_collection.cutting import COLS, ROWS

DATA_DIR = Path(os.environ.get("INTERLOCK_DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
RAW_DATASET_DIR = DATA_DIR / "raw" / "imagenette2"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFEST_COLUMNS = ["path", "split", "class_id", "orig_width", "orig_height"]


def list_images(split: str) -> list[Path]:
    """Sorted raw image paths for a split, so a seeded shuffle picks the same images on every machine."""
    return sorted((RAW_DATASET_DIR / split).rglob("*.JPEG"))


def process_image(path: Path, size: int) -> tuple[Image.Image, int, int] | None:
    """Resize shorter side to size and center-crop to size x size. None if the image is too small."""
    image = Image.open(path).convert("RGB")  # handles grayscale and CMYK images
    orig_width, orig_height = image.size  # PIL sizes are (width, height)
    if min(orig_width, orig_height) < size:
        return None  # skip rather than upscale, which would blur edges

    scale = size / min(orig_width, orig_height)
    new_width, new_height = round(orig_width * scale), round(orig_height * scale)
    image = image.resize((new_width, new_height), Image.LANCZOS)

    left, top = (new_width - size) // 2, (new_height - size) // 2
    image = image.crop((left, top, left + size, top + size))  # (left, top, right, bottom)
    return image, orig_width, orig_height


def process_split(split: str, num_images: int, size: int, rng: np.random.Generator) -> list[dict]:
    """Process randomly chosen images from a split until num_images are kept; returns manifest rows."""
    candidate_paths = list_images(split)
    if not candidate_paths:
        raise FileNotFoundError(f"no images in {RAW_DATASET_DIR / split}; run scripts/download.py first")
    rng.shuffle(candidate_paths)  # shuffle before filtering so every class is represented

    rows, num_skipped = [], 0
    for raw_path in candidate_paths:
        if len(rows) == num_images:
            break
        result = process_image(raw_path, size)
        if result is None:
            num_skipped += 1
            continue

        image, orig_width, orig_height = result
        class_id = raw_path.parent.name
        relative_path = Path(split) / class_id / f"{raw_path.stem}.png"  # relative to PROCESSED_DIR
        (PROCESSED_DIR / relative_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(PROCESSED_DIR / relative_path)  # PNG: lossless, so no JPEG artifacts along cut lines
        rows.append({"path": relative_path.as_posix(), "split": split, "class_id": class_id,
                     "orig_width": orig_width, "orig_height": orig_height})

    print(f"{split}: kept {len(rows)}, skipped {num_skipped} smaller than {size}px")
    if len(rows) < num_images:
        print(f"  warning: only {len(rows)} of the requested {num_images} images were large enough")
    return rows


def write_manifest(rows: list[dict]) -> Path:
    """Write manifest.csv from scratch; files not listed in it are ignored."""
    manifest_path = PROCESSED_DIR / "manifest.csv"
    with open(manifest_path, "w", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=400, help="output side length in pixels")
    parser.add_argument("--num-train", type=int, default=1500)
    parser.add_argument("--num-val", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0, help="which images get picked")
    args = parser.parse_args()

    # fail before processing anything if cut() would reject these images
    if args.size % ROWS != 0 or args.size % COLS != 0:
        parser.error(f"--size must be divisible by ROWS={ROWS} and COLS={COLS}, got {args.size}")

    rng = np.random.default_rng(args.seed)
    rows = process_split("train", args.num_train, args.size, rng)
    rows += process_split("val", args.num_val, args.size, rng)
    manifest_path = write_manifest(rows)
    print(f"Wrote {len(rows)} images to {PROCESSED_DIR}; manifest at {manifest_path}")


if __name__ == "__main__":
    main()
