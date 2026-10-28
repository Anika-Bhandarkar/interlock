"""Download SA-1B (Meta's Segment Anything dataset) images into data/raw/sa1b/, one folder per tar, stopping once
--max-images images have been kept and skipping work that has already been done.

SA-1B is license-gated, so first accept the license at https://ai.meta.com/datasets/segment-anything-downloads/
and save the links file it gives you as data/raw/sa1b_links.txt. Each tar is ~11 GB and holds ~11k images of ~1 MB
each; tars are streamed and only the .jpg images are kept (the .json mask annotations are skipped), so the full tar
is never written to disk, and the download stops as soon as enough images have arrived.
Set INTERLOCK_DATA_DIR to put data somewhere other than the repo's data/ folder.

Usage: uv run python scripts/download.py [--max-images 1800] [--num-tars 1]
"""

import argparse
import os
import re
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

LICENSE_URL = "https://ai.meta.com/datasets/segment-anything-downloads/"
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("INTERLOCK_DATA_DIR", REPO_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
LINKS_PATH = RAW_DIR / "sa1b_links.txt"
DATASET_DIR = RAW_DIR / "sa1b"
TAR_NAME = re.compile(r"sa_\d{6}\.tar")  # image tars; the links file also lists sa_images_ids.txt and sa_co_gold.tar
COMPLETE_MARKER = ".complete"  # written into a tar's folder once every image in the tar has been kept


def read_links(links_path: Path) -> dict[str, str]:
    """Image tar name -> download URL, from the tab-separated links file (header: file_name, cdn_link)."""
    if not links_path.exists():
        # show the path relative to the repo when it's inside it (the default), so the message matches the README
        shown_path = links_path.relative_to(REPO_ROOT) if links_path.is_relative_to(REPO_ROOT) else links_path
        raise FileNotFoundError(f"ERROR: expected SA-1B links file at {shown_path}. Download from {LICENSE_URL} "
                                f"and move and rename accordingly.")
    links = {}
    with open(links_path) as links_file:
        next(links_file)  # header
        for line in links_file:
            file_name, url = line.strip().split("\t")
            if TAR_NAME.fullmatch(file_name):
                links[file_name] = url
    return links


class ByteCounter:
    """Wraps a download and counts the bytes read from it, for progress messages."""

    def __init__(self, response):
        self.response = response
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self.response.read(size)
        self.bytes_read += len(chunk)
        return chunk


def count_images(folder: Path) -> int:
    return sum(1 for _ in folder.glob("*.jpg"))


def download_images(url: str, output_dir: Path, max_images: int | None) -> int:
    """Stream the tar at url and write up to max_images of its .jpg images (all of them if None) into output_dir.
    Skips the download if output_dir already holds enough images, or the whole tar. Returns the number of images
    in output_dir."""
    if output_dir.exists():
        num_existing = count_images(output_dir)
        if (output_dir / COMPLETE_MARKER).exists() or (max_images is not None and num_existing >= max_images):
            print(f"Already downloaded: {num_existing} images in {output_dir}")
            return num_existing
        # too few images for this request; a stream can't resume, so download this tar again from the start

    # write into a scratch folder first so an interrupted download isn't mistaken for a finished one
    scratch_dir = output_dir.with_name(output_dir.name + ".partial")
    shutil.rmtree(scratch_dir, ignore_errors=True)
    scratch_dir.mkdir(parents=True)

    try:
        response = urllib.request.urlopen(url, timeout=60)
    except urllib.error.HTTPError as error:
        if error.code == 403:  # signed links expire
            raise RuntimeError(f"download refused (403); the links may have expired. Get a fresh links file "
                               f"from {LICENSE_URL} and save it to {LINKS_PATH}.") from error
        raise

    num_images, read_whole_tar = 0, True
    total_gb = int(response.headers.get("Content-Length", 0)) / 1e9
    stream = ByteCounter(response)
    # "r|*" reads entries in order as they arrive, without seeking, and detects compression: despite the .tar
    # name, SA-1B's tars are gzipped
    with response, tarfile.open(fileobj=stream, mode="r|*") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".jpg"):
                continue  # skips the .json mask annotations, folders, and links
            if num_images == max_images:
                read_whole_tar = False  # stop early; closing the response ends the download
                break
            # keep only the file name, so no entry can be written outside scratch_dir
            with archive.extractfile(member) as source, open(scratch_dir / Path(member.name).name, "wb") as target:
                shutil.copyfileobj(source, target)
            num_images += 1
            print(f"\rDownloading {output_dir.name}: {num_images}/{max_images or 'all'} images, "
                  f"{stream.bytes_read / 1e9:.1f} of {total_gb:.1f} GB", end="", flush=True)
    print()

    if read_whole_tar:
        (scratch_dir / COMPLETE_MARKER).touch()
    shutil.rmtree(output_dir, ignore_errors=True)  # an earlier, smaller download of this tar
    scratch_dir.rename(output_dir)
    return num_images


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-images", type=int, default=1800,
                        help="stop after this many images in total; 0 keeps every image in the tars")
    parser.add_argument("--num-tars", type=int, default=1, help="most tars to read from (~11k images each)")
    args = parser.parse_args()

    try:
        links = read_links(LINKS_PATH)
    except FileNotFoundError as error:
        sys.exit(str(error))  # prints the message to stderr and exits with status 1, without a traceback
    tar_names = sorted(links)[:args.num_tars]  # links file isn't in order; sort so everyone gets the same tars
    num_remaining = args.max_images or None
    for tar_name in tar_names:
        num_images = download_images(links[tar_name], DATASET_DIR / tar_name.removesuffix(".tar"), num_remaining)
        if num_remaining is not None:
            num_remaining -= num_images
            if num_remaining <= 0:
                break

    print(f"Done: {sum(1 for _ in DATASET_DIR.rglob('*.jpg'))} images in {DATASET_DIR}")


if __name__ == "__main__":
    main()
