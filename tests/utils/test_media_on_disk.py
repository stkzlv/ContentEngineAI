"""Both media summaries count the files a product has on disk.

The scraper's summary counted the image URLs on the page (15 where 10 were
saved) and the batch counted what one download step reported, which misses
files an earlier run already saved.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.outputs_paths import media_on_disk


@pytest.mark.req("REQ-BAT-055", "REQ-SCR-024")
def test_it_counts_files_on_disk(tmp_path: Path) -> None:
    images = tmp_path / "B0MEDIA001" / "images"
    images.mkdir(parents=True)
    for i in range(3):
        (images / f"img_{i}.jpg").write_bytes(b"x")
    (images / "nested").mkdir()
    videos = tmp_path / "B0MEDIA001" / "videos"
    videos.mkdir()
    (videos / "v.mp4").write_bytes(b"x")

    assert media_on_disk("B0MEDIA001", str(tmp_path)) == (3, 1)


def test_a_product_without_media_dirs_counts_zero(tmp_path: Path) -> None:
    assert media_on_disk("B0NONE0001", str(tmp_path)) == (0, 0)
    assert not (tmp_path / "B0NONE0001").exists()
