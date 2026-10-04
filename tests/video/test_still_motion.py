"""Slow motion on every still image (design 0001), held off by default."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from src.video.assembler.visual_builder import (
    VisualFilterBuilder,
    _build_image_placement,
    _build_still_motion_scale,
    pick_still_move,
)
from src.video.config import config
from src.video.config.visual_models import StillMotionSettings, VideoSettings

PROFILE = "slideshow_images1"
MOVES = ["push_in", "pull_out", "pan_left", "pan_right", "pan_up"]


async def _chain(
    tmp_path: Path,
    motion: StillMotionSettings,
    product_id="B0X",
    pre_motion: bool | None = None,
):
    """The filter graph for three stills, through the merged profile."""
    cfg = config.model_copy(deep=True)
    cfg.video_profiles[PROFILE].still_motion = motion
    cfg.video_profiles[PROFILE].first_frame_pre_motion = pre_motion
    stills = []
    for i in range(3):
        path = tmp_path / f"still_{i}.png"
        path.write_bytes(b"")
        stills.append(path)
    inspector = MagicMock()
    inspector.is_video.return_value = False
    inspector.get_media_dimensions = AsyncMock(return_value=(1000, 1000))
    settings = cfg.get_profile_merged_settings(PROFILE)
    builder = VisualFilterBuilder(
        media_inspector=inspector,
        config=cfg,
        strategy_factory=None,
        profile_settings=settings,
        product_id=product_id,
    )
    parts, *_ = await builder.build_visual_chain(
        visual_inputs=stills,
        total_video_duration=9.0,
        is_relative_mode=True,
        video_settings_dict=settings.video_settings.model_dump(),
    )
    return parts


@pytest.mark.req("REQ-VID-010")
def test_the_shipped_config_keeps_still_motion_off() -> None:
    assert config.video_settings.still_motion.enabled is False
    for name in config.video_profiles:
        merged = config.get_profile_merged_settings(name).video_settings
        assert merged.still_motion.enabled is False, name


@pytest.mark.req("REQ-VID-010")
@pytest.mark.asyncio
async def test_every_still_moves_and_the_moves_vary(tmp_path: Path) -> None:
    parts = await _chain(tmp_path, StillMotionSettings(enabled=True))

    previous = None
    expected = []
    for i in range(3):
        previous = pick_still_move("B0X", i, MOVES, previous)
        expected.append(previous)
    clause = {
        "push_in": "(1.0000+(0.1500)*min(t/",
        "pull_out": "(1.1500+(-0.1500)*min(t/",
        "pan_left": "*(1-min(t/",
        "pan_right": ")*min(t/",
        "pan_up": "'(ih-817)*(1-min(t/",
    }
    for i, move in enumerate(expected):
        (part,) = (p for p in parts if p.startswith(f"[{i}:v]"))
        assert clause[move] in part, (i, move)
        assert ":exact=1" in part
    assert len(set(expected)) >= 2


@pytest.mark.req("REQ-VID-010")
@pytest.mark.asyncio
async def test_off_renders_the_plain_scale(tmp_path: Path) -> None:
    with patch(
        "src.video.assembler.visual_builder._build_still_motion_scale"
    ) as motion:
        parts = await _chain(tmp_path, StillMotionSettings(enabled=False))

    motion.assert_not_called()
    graph = "\n".join(parts)
    for i in range(3):
        assert f"[fg_{i}]scale=817:817,setsar=1[fgs_{i}]" in graph
    assert "eval=frame" not in graph
    assert "exact=1" not in graph


@pytest.mark.req("REQ-VID-010")
@pytest.mark.asyncio
async def test_the_first_image_keeps_its_settle_zoom(tmp_path: Path) -> None:
    parts = await _chain(tmp_path, StillMotionSettings(enabled=True), pre_motion=True)

    (first,) = (p for p in parts if p.startswith("[0:v]"))
    assert "zoompan=" in first
    assert "[fg_0]scale=817:817,setsar=1" in first
    assert "eval=frame" not in first and "exact=1" not in first
    for i in (1, 2):
        (part,) = (p for p in parts if p.startswith(f"[{i}:v]"))
        assert ":exact=1" in part


@pytest.mark.req("REQ-VID-010")
def test_the_draw_is_stable_differs_by_product_and_never_repeats() -> None:
    def run(pid: str) -> list[str]:
        previous = None
        out = []
        for i in range(12):
            previous = pick_still_move(pid, i, MOVES, previous)
            out.append(previous)
        return out

    assert run("B0AAA") == run("B0AAA")
    assert run("B0AAA") != run("B0BBB")
    for seq in (run("B0AAA"), run("B0BBB")):
        assert all(a != b for a, b in zip(seq, seq[1:], strict=False))
    assert pick_still_move("B0AAA", 3, ["pan_up"], "pan_up") == "pan_up"


def test_a_zoom_range_upside_down_is_rejected() -> None:
    StillMotionSettings(min_zoom=1.0, max_zoom=1.2)
    with pytest.raises(ValidationError):
        StillMotionSettings(min_zoom=1.3, max_zoom=1.2)
    assert StillMotionSettings(moves=["push_in", "push_in", "pan_up"]).moves == [
        "push_in",
        "pan_up",
    ]
    with pytest.raises(ValidationError):
        StillMotionSettings(moves=["spin"])  # type: ignore[list-item]
    with pytest.raises(ValidationError):
        VideoSettings(resolution=(1080, 1920), frame_rate=30, still_motion={"speed": 2})


def _render(tmp_path: Path, move: str, box: tuple[int, int]) -> np.ndarray:
    """Two seconds of one still, as grey frames of its image box."""
    bw, bh = box
    img = np.zeros((bh, bw, 3), np.uint8)
    img[:, int(bw * 0.3) : int(bw * 0.3) + 4] = 255
    img[:, int(bw * 0.7) : int(bw * 0.7) + 4] = 255
    img[bh // 2 - 20 : bh // 2 - 16, :] = 255
    src = tmp_path / "lines.jpg"
    Image.fromarray(img).save(src, quality=95)
    scale = _build_still_motion_scale(
        box_w=bw, box_h=bh, duration_sec=2.0, move=move, min_zoom=1.0, max_zoom=1.15
    )
    # The frame is the box rounded up to even, so only the box is decoded.
    fw, fh = bw + bw % 2, bh + bh % 2
    graph = _build_image_placement(
        index=0,
        vf_scale=scale,
        width=fw,
        height=fh,
        target_y=0,
        pad_color="black",
        pix_fmt="yuv420p",
        background_fill="color",
        blur_sigma=20.0,
        blur_darken=0.6,
        out_label="[v]",
    )
    out = tmp_path / f"{move}.mkv"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-loop", "1", "-framerate", "30"]
        + ["-t", "2", "-i", str(src), "-filter_complex", graph, "-map", "[v]"]
        + ["-c:v", "ffv1", str(out)],
        check=True,
    )
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(out), "-f", "rawvideo"]
        + ["-pix_fmt", "gray", "-"],
        check=True,
        capture_output=True,
    ).stdout
    x0 = (fw - bw) // 2
    return np.frombuffer(raw, np.uint8).reshape(-1, fh, fw)[:, :bh, x0 : x0 + bw]


def _tracks(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per frame: the gap between the two lines, their midpoint, the bar's row."""
    _, bh, bw = frames.shape
    gaps, mids, rows = [], [], []
    for f in frames.astype(np.float32):
        cols = np.where(f[bh // 4 : bh * 3 // 4].mean(axis=0) > 120)[0]
        left, right = cols[cols < bw // 2].mean(), cols[cols >= bw // 2].mean()
        gaps.append(right - left)
        mids.append((left + right) / 2)
        rows.append(np.where(f[:, bw // 10 : bw // 4].mean(axis=1) > 120)[0].mean())
    return np.array(gaps), np.array(mids), np.array(rows)


def _monotonic(values: np.ndarray) -> bool:
    steps = np.diff(values)
    return bool((steps >= -0.3).all() or (steps <= 0.3).all())


@pytest.mark.req("REQ-VID-010")
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
@pytest.mark.parametrize("move", MOVES)
@pytest.mark.parametrize("box", [(1000, 800), (999, 801)])
def test_a_rendered_still_moves_every_frame_without_oscillating(
    tmp_path: Path, move: str, box: tuple[int, int]
) -> None:
    frames = _render(tmp_path, move, box)
    gaps, mids, rows = _tracks(frames)

    assert len(frames) == 60
    changed = [
        not np.array_equal(a, b) for a, b in zip(frames, frames[1:], strict=False)
    ]
    assert all(changed), changed.index(False)
    for track in (gaps, mids, rows):
        assert _monotonic(track), (move, box)
    if move in ("push_in", "pull_out"):
        assert abs(gaps[-1] - gaps[0]) > 0.1 * box[0] * 0.4
        assert abs(mids[-1] - mids[0]) < 1.5
    elif move == "pan_up":
        assert abs(rows[-1] - rows[0]) > 0.1 * box[1]
    else:
        assert abs(mids[-1] - mids[0]) > 0.1 * box[0]
