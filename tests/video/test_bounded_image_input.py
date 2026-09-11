"""Oversized images are bounded before entering the filtergraph (#414).

FFmpeg buffers decoded frames at source resolution per input stream, so a
handful of full-resolution photos in one assembly exceeded a 6 GB memory
cap on the decoder side alone: the kernel OOM-killed the encode with the
mjpeg decoder thread on the stack. The images only ever render inside a
1080x1920 frame, so nothing above the configured edge bound ever reaches
the screen.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from src.video.assembler.visual_builder import bounded_image_input


def _jpeg(path: Path, w: int, h: int) -> Path:
    Image.new("RGB", (w, h), (200, 40, 40)).save(path, quality=90)
    return path


class TestTheHelper:
    def test_an_oversized_image_gets_a_bounded_copy(self, tmp_path):
        src = _jpeg(tmp_path / "big.jpg", 4000, 3000)
        out = bounded_image_input(src, 4000, 3000, 2560, tmp_path)

        assert out != src
        assert out.parent.name == "scaled_inputs"
        with Image.open(out) as img:
            assert max(img.size) == 2560
            # Aspect preserved: 4:3 in, 4:3 out.
            assert img.size == (2560, 1920)

    def test_a_small_image_is_untouched(self, tmp_path):
        src = _jpeg(tmp_path / "small.jpg", 1200, 900)
        assert bounded_image_input(src, 1200, 900, 2560, tmp_path) == src

    def test_zero_disables_the_bound(self, tmp_path):
        src = _jpeg(tmp_path / "big.jpg", 4000, 3000)
        assert bounded_image_input(src, 4000, 3000, 0, tmp_path) == src

    def test_no_temp_dir_keeps_the_original(self, tmp_path):
        src = _jpeg(tmp_path / "big.jpg", 4000, 3000)
        assert bounded_image_input(src, 4000, 3000, 2560, None) == src

    def test_an_unreadable_source_falls_back_to_itself(self, tmp_path):
        """A render that might exceed a cap still beats no render."""
        src = tmp_path / "corrupt.jpg"
        src.write_bytes(b"not an image")
        assert bounded_image_input(src, 4000, 3000, 2560, tmp_path) == src

    def test_an_existing_copy_is_reused(self, tmp_path):
        src = _jpeg(tmp_path / "big.jpg", 4000, 3000)
        first = bounded_image_input(src, 4000, 3000, 2560, tmp_path)
        stamp = first.stat().st_mtime_ns
        second = bounded_image_input(src, 4000, 3000, 2560, tmp_path)
        assert second == first
        assert second.stat().st_mtime_ns == stamp, "the copy was rewritten"


class TestTheBuilderSubstitutesIt:
    @pytest.mark.asyncio
    async def test_the_input_command_references_the_bounded_copy(self, tmp_path):
        from src.video.assembler.visual_builder import VisualFilterBuilder
        from src.video.config import config as video_config

        big = _jpeg(tmp_path / "photo.jpg", 4000, 3000)

        inspector = MagicMock()
        inspector.is_video = MagicMock(return_value=False)
        inspector.get_media_dimensions = AsyncMock(return_value=(4000, 3000))

        settings = video_config.get_profile_merged_settings("slideshow_images1")
        builder = VisualFilterBuilder(
            media_inspector=inspector,
            config=video_config,
            strategy_factory=None,
            profile_settings=settings,
        )
        _, input_cmd_parts, *_ = await builder.build_visual_chain(
            visual_inputs=[big],
            total_video_duration=6.0,
            is_relative_mode=False,
            video_settings_dict=settings.video_settings.model_dump(),
            temp_dir=tmp_path,
        )

        inputs = [
            input_cmd_parts[i + 1]
            for i, tok in enumerate(input_cmd_parts)
            if tok == "-i"
        ]
        assert inputs, "no image input built"
        assert (
            "scaled_inputs" in inputs[0]
        ), f"the filtergraph was handed the full-resolution source: {inputs[0]}"
        with Image.open(inputs[0]) as img:
            assert max(img.size) == settings.video_settings.max_image_input_edge

    @pytest.mark.asyncio
    async def test_without_a_temp_dir_the_original_is_used(self, tmp_path):
        """The default parameter keeps every other caller working."""
        from src.video.assembler.visual_builder import VisualFilterBuilder
        from src.video.config import config as video_config

        big = _jpeg(tmp_path / "photo.jpg", 4000, 3000)

        inspector = MagicMock()
        inspector.is_video = MagicMock(return_value=False)
        inspector.get_media_dimensions = AsyncMock(return_value=(4000, 3000))

        settings = video_config.get_profile_merged_settings("slideshow_images1")
        builder = VisualFilterBuilder(
            media_inspector=inspector,
            config=video_config,
            strategy_factory=None,
            profile_settings=settings,
        )
        _, input_cmd_parts, *_ = await builder.build_visual_chain(
            visual_inputs=[big],
            total_video_duration=6.0,
            is_relative_mode=False,
            video_settings_dict=settings.video_settings.model_dump(),
        )
        inputs = [
            input_cmd_parts[i + 1]
            for i, tok in enumerate(input_cmd_parts)
            if tok == "-i"
        ]
        assert inputs and inputs[0] == str(big)


class TestTheAssemblerPassesItsTempDir:
    def test_the_call_site_forwards_temp_dir(self):
        """The bound is inert if the assembler never hands over the dir."""
        import ast

        tree = ast.parse(Path("src/video/assembler/core.py").read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "build_visual_chain"
            ):
                assert any(k.arg == "temp_dir" for k in node.keywords), (
                    "core.assemble_video must pass temp_dir to " "build_visual_chain"
                )
                return
        pytest.fail("no build_visual_chain call found in core.py")
