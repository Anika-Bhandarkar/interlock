"""Tests for scripts/download.py, using a small local tar served from a file:// URL instead of Meta's CDN."""

import importlib.util
import io
import tarfile
from pathlib import Path

import pytest

# scripts/ isn't a package, so load download.py straight from its file path
DOWNLOAD_PATH = Path(__file__).resolve().parents[1] / "scripts" / "download.py"
spec = importlib.util.spec_from_file_location("download", DOWNLOAD_PATH)
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


def make_tar(path: Path, mode: str = "w:gz") -> None:
    """A tar shaped like an SA-1B tar (gzipped, despite the name): image/annotation pairs, plus entries that must
    not be extracted."""
    with tarfile.open(path, mode) as archive:
        for name, data in [("sa_1.jpg", b"image 1"), ("sa_1.json", b"{}"),
                           ("./sa_2.jpg", b"image 2"), ("sa_2.json", b"{}"),
                           ("../escape.jpg", b"outside")]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        link = tarfile.TarInfo("link.jpg")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        archive.addfile(link)


def test_read_links_keeps_only_image_tars(tmp_path):
    links_path = tmp_path / "links.txt"
    links_path.write_text("file_name\tcdn_link\n"
                          "sa_000001.tar\thttps://example.com/1\n"
                          "sa_images_ids.txt\thttps://example.com/ids\n"
                          "sa_co_gold.tar\thttps://example.com/gold\n"
                          "sa_000000.tar\thttps://example.com/0\n")
    assert download.read_links(links_path) == {"sa_000001.tar": "https://example.com/1",
                                               "sa_000000.tar": "https://example.com/0"}


def test_read_links_missing_file_says_where_to_get_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="ERROR: expected SA-1B links file at .*missing.txt"):
        download.read_links(tmp_path / "missing.txt")


def test_main_exits_with_error_when_links_file_missing(monkeypatch, capsys):
    monkeypatch.setattr(download, "LINKS_PATH", download.REPO_ROOT / "data" / "raw" / "does_not_exist.txt")
    monkeypatch.setattr("sys.argv", ["download.py"])
    with pytest.raises(SystemExit) as exit_info:
        download.main()
    assert exit_info.value.code == ("ERROR: expected SA-1B links file at data/raw/does_not_exist.txt. Download from "
                                    "https://ai.meta.com/datasets/segment-anything-downloads/ and move and rename "
                                    "accordingly.")


@pytest.mark.parametrize("mode", ["w:gz", "w"])
def test_download_images_keeps_only_jpgs(tmp_path, mode):
    tar_path = tmp_path / "sa_000000.tar"
    make_tar(tar_path, mode)
    output_dir = tmp_path / "sa1b" / "sa_000000"

    assert download.download_images(tar_path.as_uri(), output_dir, max_images=None) == 3

    assert sorted(p.name for p in output_dir.glob("*.jpg")) == ["escape.jpg", "sa_1.jpg", "sa_2.jpg"]
    assert (output_dir / "sa_1.jpg").read_bytes() == b"image 1"
    assert not (tmp_path / "escape.jpg").exists()  # "../escape.jpg" was written inside output_dir, not above it
    assert (output_dir / download.COMPLETE_MARKER).exists()
    assert not output_dir.with_name("sa_000000.partial").exists()


def test_download_images_stops_at_max_images(tmp_path):
    tar_path = tmp_path / "sa_000000.tar"
    make_tar(tar_path)
    output_dir = tmp_path / "sa_000000"

    assert download.download_images(tar_path.as_uri(), output_dir, max_images=2) == 2
    assert sorted(p.name for p in output_dir.glob("*.jpg")) == ["sa_1.jpg", "sa_2.jpg"]
    assert not (output_dir / download.COMPLETE_MARKER).exists()  # stopped early, so more images remain


def test_download_images_skips_when_enough_images(tmp_path):
    output_dir = tmp_path / "sa_000000"
    output_dir.mkdir()
    for name in ["sa_1.jpg", "sa_2.jpg"]:
        (output_dir / name).write_bytes(b"image")
    # would raise if it tried to download
    assert download.download_images("file:///does/not/exist.tar", output_dir, max_images=2) == 2


def test_download_images_skips_complete_tar(tmp_path):
    output_dir = tmp_path / "sa_000000"
    output_dir.mkdir()
    (output_dir / "sa_1.jpg").write_bytes(b"image")
    (output_dir / download.COMPLETE_MARKER).touch()
    # the tar had only one image, so asking for more can't get more
    assert download.download_images("file:///does/not/exist.tar", output_dir, max_images=5) == 1


def test_download_images_redownloads_when_too_few(tmp_path):
    tar_path = tmp_path / "sa_000000.tar"
    make_tar(tar_path)
    output_dir = tmp_path / "sa_000000"
    download.download_images(tar_path.as_uri(), output_dir, max_images=1)

    assert download.download_images(tar_path.as_uri(), output_dir, max_images=3) == 3
    assert (output_dir / download.COMPLETE_MARKER).exists()
