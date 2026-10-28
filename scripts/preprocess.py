"""Turn raw SA-1B photos into square RGB PNGs that cut() accepts, plus a manifest listing them.

SA-1B has no train/val split, so all downloaded images are shuffled once with --seed and train is filled first,
then val from the images train didn't use. Each kept image: converted to RGB, resized so its shorter side is
--size, center-cropped to --size x --size, and saved to data/processed/<split>/<name>.png. Images smaller than
--size are skipped rather than upscaled. manifest.csv is rewritten from scratch on every run and is the source of
truth for which images are in the dataset.

Usage: uv run python scripts/preprocess.py [--size 400] [--num-train 1500] [--num-val 300] [--seed 0]
"""

import argparse
import csv
import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from PIL import Image

from interlock.data_collection.cutting import COLS, ROWS

DATA_DIR = Path(os.environ.get("INTERLOCK_DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
RAW_DATASET_DIR = DATA_DIR / "raw" / "sa1b"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFEST_COLUMNS = ["path", "split", "source", "orig_width", "orig_height"]


def list_images() -> list[Path]:
    """Sorted raw image paths across every downloaded tar, so a seeded shuffle picks the same images on every
    machine."""
    return sorted(RAW_DATASET_DIR.glob("sa_*/*.jpg"))


def process_image(path: Path, size: int) -> tuple[Image.Image, int, int] | None:
    """Resize shorter side to size and center-crop to size x size. None if the image is too small."""
    image = Image.open(path)
    orig_width, orig_height = image.size  # PIL sizes are (width, height)
    if min(orig_width, orig_height) < size:
        return None  # skip rather than upscale, which would blur edges

    # let the JPEG decoder skip detail we'd throw away anyway; it never goes below size x size
    image.draft("RGB", (size, size))
    image = image.convert("RGB")  # handles grayscale and CMYK images
    scale = size / min(image.size)
    new_width, new_height = round(image.width * scale), round(image.height * scale)
    image = image.resize((new_width, new_height), Image.LANCZOS)

    left, top = (new_width - size) // 2, (new_height - size) // 2
    image = image.crop((left, top, left + size, top + size))  # (left, top, right, bottom)
    return image, orig_width, orig_height


def process_split(split: str, candidate_paths: Iterator[Path], num_images: int, size: int) -> list[dict]:
    """Process images from candidate_paths until num_images are kept; returns manifest rows. Images it reads
    are used up, so the next split continues where this one stopped."""
    rows, num_skipped = [], 0
    while len(rows) < num_images:
        raw_path = next(candidate_paths, None)
        if raw_path is None:
            break  # ran out of images
        result = process_image(raw_path, size)
        if result is None:
            num_skipped += 1
            continue

        image, orig_width, orig_height = result
        relative_path = Path(split) / f"{raw_path.stem}.png"  # relative to PROCESSED_DIR
        (PROCESSED_DIR / relative_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(PROCESSED_DIR / relative_path)  # PNG: lossless, so no JPEG artifacts along cut lines
        rows.append({"path": relative_path.as_posix(), "split": split, "source": raw_path.parent.name,
                     "orig_width": orig_width, "orig_height": orig_height})

    print(f"{split}: kept {len(rows)}, skipped {num_skipped} smaller than {size}px")
    if len(rows) < num_images:
        print(f"  warning: only {len(rows)} of the requested {num_images} images were available; "
              f"download more with scripts/download.py --max-images")
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
    parser.add_argument("--seed", type=int, default=0, help="which images get picked, and for which split")
    args = parser.parse_args()

    # fail before processing anything if cut() would reject these images
    if args.size % ROWS != 0 or args.size % COLS != 0:
        parser.error(f"--size must be divisible by ROWS={ROWS} and COLS={COLS}, got {args.size}")

    candidate_paths = list_images()
    if not candidate_paths:
        raise FileNotFoundError(f"no images in {RAW_DATASET_DIR}; run scripts/download.py first")
    np.random.default_rng(args.seed).shuffle(candidate_paths)

    # one shared iterator, so no image lands in both splits
    remaining_paths = iter(candidate_paths)
    rows = process_split("train", remaining_paths, args.num_train, args.size)
    rows += process_split("val", remaining_paths, args.num_val, args.size)
    manifest_path = write_manifest(rows)
    print(f"Wrote {len(rows)} images to {PROCESSED_DIR}; manifest at {manifest_path}")


if __name__ == "__main__":
    main()
