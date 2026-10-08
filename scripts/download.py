"""Download Imagenette (a free 10-class ImageNet subset) into data/raw/, and skips download/extraction if they have 
already been downloaded.
Set INTERLOCK_DATA_DIR to put data somewhere other than the repo's data/ folder.

Usage: uv run python scripts/download.py
"""

import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"  # full size, ~1.45 GB
DATA_DIR = Path(os.environ.get("INTERLOCK_DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
RAW_DIR = DATA_DIR / "raw"


def download(url: str, destination: Path) -> None:
    """Download url to destination, skipping if it already exists."""
    if destination.exists():
        print(f"Already downloaded: {destination}")
        return

    def report_progress(blocks_done, block_size, total_size):
        if total_size <= 0:  # server didn't send a size
            return
        percent = min(100, blocks_done * block_size * 100 // total_size)
        print(f"\rDownloading {destination.name}: {percent}%", end="", flush=True)

    # download to a temporary name so an interrupted download isn't mistaken for a finished one
    partial_path = destination.with_name(destination.name + ".partial")
    urllib.request.urlretrieve(url, partial_path, reporthook=report_progress)
    partial_path.rename(destination)
    print()


def safe_extract(archive_path: Path, target_dir: Path) -> None:
    """Extract a .tgz into target_dir, refusing entries that would land outside it.
    """
    target_dir = target_dir.resolve()
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            member_path = (target_dir / member.name).resolve()
            if not member_path.is_relative_to(target_dir) or member.issym() or member.islnk():
                raise RuntimeError(f"unsafe entry in {archive_path.name}: {member.name}")
        archive.extractall(target_dir)


def extract(archive_path: Path, output_dir: Path) -> None:
    """Extract the archive so its contents end up at output_dir, skipping if already done."""
    if output_dir.exists():
        print(f"Already extracted: {output_dir}")
        return

    # extract into a scratch folder first so an interrupted extraction isn't mistaken for a finished one
    scratch_dir = output_dir.with_name(output_dir.name + ".partial")
    shutil.rmtree(scratch_dir, ignore_errors=True)
    print(f"Extracting {archive_path.name}...")
    safe_extract(archive_path, scratch_dir)
    (scratch_dir / output_dir.name).rename(output_dir)  # archive's top-level folder is imagenette2/
    scratch_dir.rmdir()


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = RAW_DIR / "imagenette2.tgz"
    dataset_dir = RAW_DIR / "imagenette2"

    download(IMAGENETTE_URL, archive_path)
    extract(archive_path, dataset_dir)

    num_images = sum(1 for _ in dataset_dir.rglob("*.JPEG"))
    print(f"Done: {num_images} images in {dataset_dir}")


if __name__ == "__main__":
    main()
