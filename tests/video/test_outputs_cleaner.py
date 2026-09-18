"""`make clean-outputs` must not take a product directory or the history.

The cleaner walked the outputs root and removed every path it did not
expect, and the age cutoff applied to files only, so a directory it did not
expect went whole whatever its age. It expected only the four configured
global directories, so a fresh product directory, a topic directory and the
performance history were all "unexpected". The cleaner had never run for
real on a working tree, which is why this cost nothing before it was found.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from src.utils.outputs_paths import GLOBAL_DIR_NAMES

A_YEAR_AGO = time.time() - 60 * 60 * 24 * 400


def _config_on(root: Path):
    from src.video.config_adapter import load_video_config_modular

    config = load_video_config_modular()
    config.global_output_root_path = root
    return config


def _age(path: Path) -> None:
    os.utime(path, (A_YEAR_AGO, A_YEAR_AGO))


def _product(root: Path, name: str) -> Path:
    product = root / name
    (product / "images").mkdir(parents=True)
    (product / "temp").mkdir()
    (product / "data.json").write_text("{}", encoding="utf-8")
    (product / f"video_{name}_slideshow_images1.mp4").write_text("x")
    (product / "images" / "1.jpg").write_text("x")
    (product / "temp" / "scratch.jpg").write_text("x")
    return product


@pytest.fixture
def root(tmp_path: Path) -> Path:
    for name in ("cache", "logs", "reports", "temp", "state"):
        (tmp_path / name).mkdir()
    return tmp_path


class TestTheProbeFromTheIssue:
    def test_a_dry_run_names_nothing_fresh(self, root: Path):
        _product(root, "B0FRESH001")
        history = root / "performance_history"
        history.mkdir()
        (history / "performance_history.jsonl").write_text("{}\n")

        report = _config_on(root).cleanup_outputs_directory(dry_run=True)

        assert report["actions"] == []


class TestARealRunKeepsWhatIsNotItsToTake:
    def test_a_fresh_product_directory_survives(self, root: Path):
        product = _product(root, "B0FRESH001")

        _config_on(root).cleanup_outputs_directory(dry_run=False)

        assert (product / "data.json").exists()
        assert (product / "video_B0FRESH001_slideshow_images1.mp4").exists()
        assert (product / "images" / "1.jpg").exists()

    def test_an_aged_product_directory_keeps_its_files_too(self, root: Path):
        """Only temp/ files age out; data.json and the render are the product."""
        product = _product(root, "B0OLD000001")
        for path in product.rglob("*"):
            _age(path)
        _age(product)

        _config_on(root).cleanup_outputs_directory(dry_run=False)

        assert (product / "data.json").exists()
        assert (product / "video_B0OLD000001_slideshow_images1.mp4").exists()
        assert (product / "images" / "1.jpg").exists()
        assert not (product / "temp" / "scratch.jpg").exists()
        assert not (product / "temp").exists(), "the emptied temp/ is swept"

    def test_a_topic_directory_survives(self, root: Path):
        topic = _product(root, "topic-why-wifi-drops-1a2b3c4d")

        _config_on(root).cleanup_outputs_directory(dry_run=False)

        assert (topic / "data.json").exists()

    def test_every_global_directory_survives_with_its_files(self, root: Path):
        for name in GLOBAL_DIR_NAMES:
            (root / name).mkdir(exist_ok=True)
            old = root / name / "kept.dat"
            old.write_text("x")
            _age(old)

        _config_on(root).cleanup_outputs_directory(dry_run=False)

        for name in GLOBAL_DIR_NAMES:
            assert (root / name / "kept.dat").exists(), name

    def test_an_aged_stray_file_at_the_root_still_goes(self, root: Path):
        """The cleaner's remaining job."""
        stray = root / "stray.dat"
        stray.write_text("x")
        _age(stray)

        _config_on(root).cleanup_outputs_directory(dry_run=False)

        assert not stray.exists()
