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


@pytest.mark.req("REQ-BAT-055", "REQ-SCR-024")
def test_empty_and_unlisted_files_do_not_count(tmp_path: Path) -> None:
    """Only non-empty files of the scraper's media types are validated media."""
    images = tmp_path / "B0MEDIA002" / "images"
    images.mkdir(parents=True)
    (images / "a.jpg").write_bytes(b"x")
    (images / "empty.png").write_bytes(b"")
    (images / "b.webp").write_bytes(b"x")
    (images / "notes.json").write_text("{}")

    assert media_on_disk("B0MEDIA002", str(tmp_path)) == (1, 0)


@pytest.mark.asyncio
async def test_a_failed_download_leaves_no_file(tmp_path: Path) -> None:
    """An empty response is removed, so no count can read it as media."""
    from unittest.mock import MagicMock

    from src.scraper.amazon.download_async import download_file_async

    empty: tuple[bytes, ...] = ()

    async def no_chunks(_size):
        for chunk in empty:
            yield chunk

    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.content.iter_chunked = no_chunks
    context = MagicMock()
    context.__aenter__.return_value = response
    context.__aexit__.return_value = False
    session = MagicMock()
    session.get.return_value = context

    target = tmp_path / "videos" / "v.mp4"
    ok = await download_file_async(session, "https://example.com/v.mp4", target)

    assert ok is False
    assert not target.exists()
