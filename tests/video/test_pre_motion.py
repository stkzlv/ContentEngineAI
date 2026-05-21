"""Tests for the Phase 1.2 first-frame pre-motion (Ken Burns) wiring."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.video.assembler.visual_builder import _build_ken_burns_filter
from src.video.config.visual_models import VideoProfile, VideoSettings


class TestVideoSettingsField:
    def test_default_is_off(self) -> None:
        vs = VideoSettings(resolution=(1080, 1920), frame_rate=30)
        assert vs.first_frame_pre_motion is False
        assert vs.pre_motion_peak_zoom == 1.10

    def test_peak_zoom_floor(self) -> None:
        with pytest.raises(ValidationError):
            VideoSettings(
                resolution=(1080, 1920), frame_rate=30, pre_motion_peak_zoom=0.5
            )

    def test_peak_zoom_ceiling(self) -> None:
        with pytest.raises(ValidationError):
            VideoSettings(
                resolution=(1080, 1920), frame_rate=30, pre_motion_peak_zoom=1.6
            )


class TestVideoProfileOverride:
    def test_partial_accepts_none_default(self) -> None:
        p = VideoProfile(description="x")
        assert p.first_frame_pre_motion is None
        assert p.pre_motion_peak_zoom is None

    def test_partial_accepts_override(self) -> None:
        p = VideoProfile(
            description="x", first_frame_pre_motion=True, pre_motion_peak_zoom=1.15
        )
        assert p.first_frame_pre_motion is True
        assert p.pre_motion_peak_zoom == 1.15


class TestKenBurnsFilter:
    def test_filter_string_shape(self) -> None:
        f = _build_ken_burns_filter(
            width=1080,
            height=1920,
            duration_sec=2.0,
            fps=30,
            peak_zoom=1.10,
            in_label="[v_temp_0]",
            out_label="[v_motion_0]",
        )
        # input/output labels
        assert f.startswith("[v_temp_0]zoompan=")
        assert f.endswith("[v_motion_0]")
        # frame count: 2.0s * 30fps = 60
        assert "d=60" in f
        # target resolution
        assert "s=1080x1920" in f
        # fps
        assert "fps=30" in f
        # zoom expression carries the peak
        assert "1.100" in f

    def test_zoom_step_is_consistent(self) -> None:
        """(peak - 1.0) / frames lands the zoom at 1.0 on the last frame."""
        f = _build_ken_burns_filter(
            width=1080,
            height=1920,
            duration_sec=1.0,
            fps=30,
            peak_zoom=1.30,
            in_label="[in]",
            out_label="[out]",
        )
        # 0.30 / 30 = 0.010000
        assert "zoom-0.010000" in f

    def test_short_segment_clamps_frames(self) -> None:
        """A 0.01s segment still produces a valid filter with at least 2 frames."""
        f = _build_ken_burns_filter(
            width=1080,
            height=1920,
            duration_sec=0.01,
            fps=30,
            peak_zoom=1.10,
            in_label="[in]",
            out_label="[out]",
        )
        # ceil(0.01 * 30) = 0.3 -> clamped to 2
        assert "d=2" in f
