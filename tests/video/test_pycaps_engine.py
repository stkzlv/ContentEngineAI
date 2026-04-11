"""Unit tests for the pycaps subtitle engine integration.

These tests cover the logic that lives in ``src/video/pycaps_engine/`` plus
the config merge wiring. All heavy pycaps calls are stubbed so the tests
run without the optional ``pycaps`` Poetry group installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src.video.config.subtitle_models import PycapsSettings
from src.video.config.visual_models import MergedSubtitleSettings
from src.video.pycaps_engine import (
    PycapsRenderer,
    PycapsRenderResult,
    PycapsUnavailableError,
    load_whisper_transcript,
    save_whisper_transcript,
    select_template_for_product,
)
from src.video.subtitle_positioning import VisualBounds


class TestSelectTemplateForProduct:
    def test_pool_with_multiple_entries_deterministic(self):
        settings = PycapsSettings(
            template_pool=["word-focus", "hype", "minimalist", "vibrant"]
        )
        first = select_template_for_product("B0ABC12345", settings)
        second = select_template_for_product("B0ABC12345", settings)
        assert first == second

    def test_pool_distributes_across_entries(self):
        settings = PycapsSettings(
            template_pool=["word-focus", "hype", "minimalist", "vibrant"]
        )
        picks = {
            select_template_for_product(f"B0TEST{i:04d}", settings) for i in range(200)
        }
        # Not a strict uniformity test — just confirm the selection isn't
        # pinned to a single entry.
        assert len(picks) >= 2

    def test_empty_pool_uses_template_name(self):
        settings = PycapsSettings(template_pool=[], template_name="hype")
        assert select_template_for_product("anything", settings) == "hype"

    def test_single_entry_pool_always_returns_that_entry(self):
        settings = PycapsSettings(template_pool=["word-focus"], template_name="hype")
        assert select_template_for_product("B0ANY", settings) == "word-focus"


class TestTranscriptAdapter:
    @pytest.fixture
    def raw_whisper(self) -> dict[str, Any]:
        return {
            "language": "en",
            "text": "Hello world this is a test",
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 1.2,
                    "text": "Hello world",
                    "words": [
                        {"word": "Hello", "start": 0.0, "end": 0.6},
                        {"word": "world", "start": 0.6, "end": 1.2},
                    ],
                },
            ],
        }

    def test_save_and_load_round_trip(self, tmp_path: Path, raw_whisper):
        out = tmp_path / "transcript.json"
        save_whisper_transcript(raw_whisper, out)
        assert out.exists()
        loaded = load_whisper_transcript(out)
        assert loaded == raw_whisper

    def test_save_creates_parent_directories(self, tmp_path: Path, raw_whisper):
        nested = tmp_path / "nested" / "deeper" / "transcript.json"
        save_whisper_transcript(raw_whisper, nested)
        assert nested.exists()

    def test_save_rejects_invalid_input(self, tmp_path: Path):
        with pytest.raises(ValueError, match="segments"):
            save_whisper_transcript({"no": "segments"}, tmp_path / "bad.json")

    def test_load_rejects_non_whisper_json(self, tmp_path: Path):
        broken = tmp_path / "broken.json"
        broken.write_text(json.dumps({"not": "whisper"}))
        with pytest.raises(ValueError, match="whisper_json"):
            load_whisper_transcript(broken)

    def test_load_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_whisper_transcript(tmp_path / "missing.json")


class TestPycapsRendererFallback:
    """The renderer should surface missing-dependency state via PycapsUnavailableError."""

    def test_render_raises_when_pycaps_not_installed(self, tmp_path: Path):
        renderer = PycapsRenderer()
        settings = PycapsSettings()
        bounds = VisualBounds(x=0.075, y=0.10, width=0.85, height=0.75)

        # Craft a minimal valid whisper transcript so the file-exists checks pass.
        transcript = tmp_path / "transcript.json"
        transcript.write_text(
            json.dumps(
                {
                    "segments": [
                        {
                            "start": 0.0,
                            "end": 1.0,
                            "text": "hi",
                            "words": [{"word": "hi", "start": 0.0, "end": 1.0}],
                        }
                    ]
                }
            )
        )
        input_video = tmp_path / "in.mp4"
        input_video.write_bytes(b"fake")  # content doesn't matter — we stub before use
        output_video = tmp_path / "out.mp4"

        # Force an ImportError inside the deferred import path so the test
        # doesn't depend on whether the optional group is actually absent.
        with (
            patch.object(
                PycapsRenderer,
                "_build_pipeline",
                side_effect=ImportError("No module named 'pycaps'"),
            ),
            pytest.raises(PycapsUnavailableError),
        ):
            renderer.render(
                input_video=input_video,
                transcript_path=transcript,
                output_video=output_video,
                product_id="B0TEST",
                visual_bounds=bounds,
                settings=settings,
            )

    def test_render_returns_failure_result_on_library_error(self, tmp_path: Path):
        renderer = PycapsRenderer()
        settings = PycapsSettings()

        transcript = tmp_path / "transcript.json"
        transcript.write_text(
            json.dumps(
                {
                    "segments": [
                        {
                            "start": 0.0,
                            "end": 1.0,
                            "text": "hi",
                            "words": [{"word": "hi", "start": 0.0, "end": 1.0}],
                        }
                    ]
                }
            )
        )
        input_video = tmp_path / "in.mp4"
        input_video.write_bytes(b"fake")
        output_video = tmp_path / "out.mp4"

        with patch.object(
            PycapsRenderer,
            "_build_pipeline",
            side_effect=RuntimeError("boom"),
        ):
            result = renderer.render(
                input_video=input_video,
                transcript_path=transcript,
                output_video=output_video,
                product_id="B0TEST",
                visual_bounds=None,
                settings=settings,
            )

        assert isinstance(result, PycapsRenderResult)
        assert result.success is False
        assert result.error and "boom" in result.error
        assert result.template_used  # renderer still picked a template
        assert result.renderer_used == "css"


class TestConfigMergePycapsLayer:
    """The 3-level config merge must honour CLI and profile overrides."""

    def test_cli_override_creates_pycaps_settings(self):
        from src.video.config import config as video_config

        merged = video_config.get_profile_merged_settings(
            "slideshow_images1",
            cli_overrides={
                "subtitle_settings.subtitle_engine": "pycaps",
                "subtitle_settings.pycaps.template_name": "hype",
                "subtitle_settings.pycaps.renderer": "pictex",
            },
        )
        assert merged.subtitle_settings.subtitle_engine == "pycaps"
        assert merged.subtitle_settings.pycaps is not None
        assert merged.subtitle_settings.pycaps.template_name == "hype"
        assert merged.subtitle_settings.pycaps.renderer == "pictex"

    def test_cli_template_pool_override(self):
        from src.video.config import config as video_config

        merged = video_config.get_profile_merged_settings(
            "slideshow_images1",
            cli_overrides={
                "subtitle_settings.subtitle_engine": "pycaps",
                "subtitle_settings.pycaps.template_pool": ["word-focus", "hype"],
            },
        )
        assert merged.subtitle_settings.pycaps is not None
        assert merged.subtitle_settings.pycaps.template_pool == [
            "word-focus",
            "hype",
        ]

    def test_default_engine_is_ffmpeg(self):
        settings = MergedSubtitleSettings()
        assert settings.subtitle_engine == "ffmpeg"
        assert settings.pycaps is None

    def test_pycaps_settings_in_dict_form_constructs_model(self):
        settings = MergedSubtitleSettings(
            subtitle_engine="pycaps",
            pycaps={"template_name": "vibrant", "renderer": "pictex"},
        )
        assert isinstance(settings.pycaps, PycapsSettings)
        assert settings.pycaps.template_name == "vibrant"
        assert settings.pycaps.renderer == "pictex"
