"""Scraped media is found under the render's outputs root (REQ-OPS-031)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.video.producer.steps import _scraped_media


def _ctx(tmp_path: Path, images: list[str] | None) -> SimpleNamespace:
    return SimpleNamespace(
        run_paths={"run_root": tmp_path / "scratch-out" / "B0X"},
        config=SimpleNamespace(project_root=tmp_path / "repo"),
        product=SimpleNamespace(asin="B0X", downloaded_images=images),
    )


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


@pytest.mark.req("REQ-OPS-031")
def test_a_recorded_path_resolves_under_the_outputs_root(tmp_path: Path) -> None:
    image = _touch(tmp_path / "scratch-out" / "B0X" / "images" / "a.jpg")
    ctx = _ctx(tmp_path, ["B0X/images/a.jpg"])

    found = _scraped_media(ctx, ctx.product.downloaded_images, "images", ("*.jpg",))

    assert found == [image]


@pytest.mark.req("REQ-OPS-031")
def test_without_a_recorded_path_the_outputs_roots_directory_is_scanned(
    tmp_path: Path,
) -> None:
    image = _touch(tmp_path / "scratch-out" / "B0X" / "images" / "b.png")
    # The repository's own outputs tree holds another copy, not this render's.
    _touch(tmp_path / "repo" / "outputs" / "B0X" / "images" / "c.png")

    found = _scraped_media(_ctx(tmp_path, []), [], "images", ("*.jpg", "*.png"))

    assert found == [image]


@pytest.mark.req("REQ-OPS-031")
def test_a_path_relative_to_the_repository_is_still_accepted(tmp_path: Path) -> None:
    image = _touch(tmp_path / "repo" / "outputs" / "B0X" / "images" / "a.jpg")
    ctx = _ctx(tmp_path, ["outputs/B0X/images/a.jpg"])

    found = _scraped_media(ctx, ctx.product.downloaded_images, "images", ("*.jpg",))

    assert found == [image]


@pytest.mark.req("REQ-OPS-031")
def test_the_gather_step_uses_the_helper() -> None:
    import inspect

    from src.video.producer import steps

    source = inspect.getsource(steps.step_gather_visuals)
    assert "_scraped_media(" in source and '"outputs" / ctx.product.asin' not in source
