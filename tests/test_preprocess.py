"""Tests for scripts/preprocess.py, on small generated JPEGs laid out like download.py's data/raw/sa1b/."""

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# scripts/ isn't a package, so load preprocess.py straight from its file path
PREPROCESS_PATH = Path(__file__).resolve().parents[1] / "scripts" / "preprocess.py"
spec = importlib.util.spec_from_file_location("preprocess", PREPROCESS_PATH)
preprocess = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocess)

SIZE = 40


@pytest.fixture
def data_dirs(tmp_path, monkeypatch):
    """Point preprocess at an empty raw/ and processed/ under tmp_path."""
    raw_dir, processed_dir = tmp_path / "raw" / "sa1b", tmp_path / "processed"
    monkeypatch.setattr(preprocess, "RAW_DATASET_DIR", raw_dir)
    monkeypatch.setattr(preprocess, "PROCESSED_DIR", processed_dir)
    return raw_dir, processed_dir


def save_jpg(path: Path, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.random.default_rng(0).integers(0, 256, (height, width, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)


def test_process_image_crops_non_square_to_square(tmp_path):
    save_jpg(tmp_path / "wide.jpg", 3 * SIZE, 2 * SIZE)
    image, orig_width, orig_height = preprocess.process_image(tmp_path / "wide.jpg", SIZE)
    assert image.size == (SIZE, SIZE)
    assert image.mode == "RGB"
    assert (orig_width, orig_height) == (3 * SIZE, 2 * SIZE)  # before draft() shrank it


def test_process_image_skips_small(tmp_path):
    save_jpg(tmp_path / "small.jpg", SIZE - 1, 2 * SIZE)
    assert preprocess.process_image(tmp_path / "small.jpg", SIZE) is None


def test_main_splits_without_overlap(data_dirs, monkeypatch):
    raw_dir, processed_dir = data_dirs
    for tar_name, image_ids in [("sa_000000", range(0, 6)), ("sa_000001", range(6, 10))]:
        for image_id in image_ids:
            save_jpg(raw_dir / tar_name / f"sa_{image_id}.jpg", 2 * SIZE, 3 * SIZE)
    save_jpg(raw_dir / "sa_000000" / "sa_99.jpg", SIZE // 2, SIZE // 2)  # too small: skipped, not counted
    monkeypatch.setattr("sys.argv", ["preprocess.py", "--size", str(SIZE), "--num-train", "6", "--num-val", "3"])

    preprocess.main()

    with open(processed_dir / "manifest.csv") as manifest_file:
        rows = list(csv.DictReader(manifest_file))
    assert list(rows[0]) == preprocess.MANIFEST_COLUMNS
    train = [row for row in rows if row["split"] == "train"]
    val = [row for row in rows if row["split"] == "val"]
    assert (len(train), len(val)) == (6, 3)
    assert not {Path(row["path"]).name for row in train} & {Path(row["path"]).name for row in val}
    assert "sa_99.png" not in {Path(row["path"]).name for row in rows}
    for row in rows:
        assert row["source"] in {"sa_000000", "sa_000001"}
        assert Image.open(processed_dir / row["path"]).size == (SIZE, SIZE)


def test_main_warns_when_too_few_images(data_dirs, monkeypatch, capsys):
    raw_dir, _ = data_dirs
    for image_id in range(3):
        save_jpg(raw_dir / "sa_000000" / f"sa_{image_id}.jpg", SIZE, SIZE)
    monkeypatch.setattr("sys.argv", ["preprocess.py", "--size", str(SIZE), "--num-train", "2", "--num-val", "5"])

    preprocess.main()

    assert "only 1 of the requested 5" in capsys.readouterr().out


def test_main_without_images_says_to_download(data_dirs, monkeypatch):
    monkeypatch.setattr("sys.argv", ["preprocess.py"])
    with pytest.raises(FileNotFoundError, match="download.py"):
        preprocess.main()
