"""The disclosure is in the script's language, on the frame and the caption.

The FTC wants the disclosure in the language of the endorsement, and Spain's
Royal Decree 444/2024 asks for `#publi`. The text was one setting, `#ad`, so a
Spanish render would have disclosed in English.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.video.assembler.core import VideoAssembler
from src.video.config import VideoConfig, load_video_config_modular
from src.video.config_validator import VideoConfigValidator


def config_in(language_code: str, **overlay: object) -> VideoConfig:
    config = load_video_config_modular()
    data = config.model_dump()
    data["tts_config"]["google_cloud"]["language_code"] = language_code
    data["video_settings"]["disclosure_overlay"].update(overlay)
    return VideoConfig.model_validate(data)


@pytest.mark.req("REQ-CMP-022")
@pytest.mark.parametrize(
    ("language_code", "expected"), [("en-US", "#ad"), ("es-ES", "#publi")]
)
def test_the_disclosure_follows_the_script_language(
    language_code: str, expected: str
) -> None:
    config = config_in(language_code)

    assert config.disclosure_text() == expected
    assert VideoAssembler(config)._disclosure_settings().text == expected


@pytest.mark.req("REQ-CMP-022")
def test_the_caption_record_carries_the_same_text(tmp_path: Path) -> None:
    from src.video.producer.steps import _generate_unified_metadata

    ctx = MagicMock()
    ctx.config = config_in("es-ES")
    ctx.product.title = "Lámpara"
    ctx.product.asin = "B0SPANISH1"
    ctx.description = None
    ctx.run_paths = {"run_root": tmp_path, "description_file": tmp_path / "d.txt"}
    ctx.state = {}

    with patch(
        "src.video.producer.steps.generate_ai_description",
        new=AsyncMock(return_value="Una lámpara."),
    ):
        asyncio.run(_generate_unified_metadata(ctx))

    written = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert written["disclosure"] == "#publi"


@pytest.mark.req("REQ-CMP-023")
def test_a_disclosure_in_another_language_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        config = config_in("es-ES", language="en")

    assert config.disclosure_text() == "#ad"
    assert "the voice language is es" in caplog.text


@pytest.mark.req("REQ-CMP-023")
def test_a_language_with_no_variant_falls_back_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        config = config_in("fr-FR")

    assert config.disclosure_text() == "#ad"
    assert "no entry for fr" in caplog.text


@pytest.mark.req("REQ-CMP-023")
def test_a_custom_text_a_variant_shadows_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`text` was the only setting; an English `#sponsored` now loses to `en`."""
    with caplog.at_level(logging.WARNING):
        config = config_in("en-US", text="#sponsored")

    assert config.disclosure_text() == "#ad"
    assert "'#sponsored'" in caplog.text


@pytest.mark.req("REQ-CMP-022")
def test_the_assembler_draws_the_resolved_text(tmp_path: Path) -> None:
    """Driven through `assemble_video` as far as the overlay call."""

    class ReachedError(Exception):
        pass

    seen: list[str] = []

    def capture(filters, settings, *args, **kwargs):
        seen.append(settings.text)
        raise ReachedError

    assembler = VideoAssembler(config_in("es-ES"))
    assembler.carries_affiliate_content = True
    assembler.visual_builder = MagicMock()
    assembler.visual_builder.build_visual_chain = AsyncMock(return_value=MagicMock())
    assembler.subtitle_builder = MagicMock()
    assembler.subtitle_builder.build_subtitle_graph = AsyncMock(
        return_value=(["[0:v]copy[v_out]"], [])
    )
    with (
        patch("src.video.assembler.core.apply_hook_overlay", lambda f, *a, **k: f),
        patch(
            "src.video.assembler.core.apply_upper_line_overlay",
            lambda f, *a, **k: f,
        ),
        patch("src.video.assembler.core.apply_disclosure_overlay", capture),
        pytest.raises(ReachedError),
    ):
        asyncio.run(
            assembler.assemble_video(
                visual_inputs=[tmp_path / "v.mp4"],
                voiceover_audio_path=None,
                music_track_path=None,
                output_path=tmp_path / "out.mp4",
                subtitle_path=None,
                total_video_duration=5.0,
                temp_dir=tmp_path,
            )
        )

    assert seen == ["#publi"]


@pytest.mark.req("REQ-CMP-022")
@pytest.mark.asyncio
async def test_the_platform_records_carry_the_resolved_text(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from src.video.producer import steps

    ctx = SimpleNamespace(
        config=config_in("es-ES"),
        product=MagicMock(topic=None, pillar=None, title="Lámpara"),
        run_paths={
            "run_root": tmp_path,
            "description_file": tmp_path / "text" / "description.txt",
            "script_file": tmp_path / "text" / "script.txt",
            "final_video_output": tmp_path / "video.mp4",
        },
        state={},
        secrets={},
        session=None,
        debug_mode=False,
    )
    save = MagicMock()
    with (
        patch(
            "src.ai.platform_metadata.PlatformMetadataFactory.generate_multi_platform",
            AsyncMock(return_value={"youtube": MagicMock()}),
        ),
        patch("src.ai.platform_metadata.save_metadata_to_file", save),
        patch(
            "src.ai.platform_metadata.text_formatter.format_upload_instructions",
            return_value="",
        ),
    ):
        await steps._generate_optimized_metadata(ctx)

    assert save.call_args.kwargs["disclosure"] == "#publi"


@pytest.mark.req("REQ-CMP-022")
def test_a_stale_record_gets_the_current_text(tmp_path: Path) -> None:
    from src.video.producer.steps import _check_existing_metadata

    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {"description": "d", "carries_affiliate_content": True, "disclosure": "#ad"}
        ),
        encoding="utf-8",
    )
    ctx = MagicMock()
    ctx.config = config_in("es-ES")
    ctx.state = {}
    ctx.run_paths = {"run_root": tmp_path, "description_file": tmp_path / "d.txt"}

    assert _check_existing_metadata(ctx) is True
    written = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert written["disclosure"] == "#publi"


@pytest.mark.parametrize("language_code", ["en-US", "es-ES"])
def test_the_bundled_config_loads_without_a_warning(
    caplog: pytest.LogCaptureFixture, language_code: str
) -> None:
    """Switching only the voice is enough; nothing was customised to warn about."""
    config = load_video_config_modular()
    data = config.model_dump(exclude_unset=True)
    with caplog.at_level(logging.WARNING):
        tts = config.tts_config.model_copy(deep=True)
        assert tts.google_cloud is not None
        tts.google_cloud.language_code = language_code
        VideoConfig.model_validate(
            {**data, "tts_config": tts.model_dump(exclude_unset=True)}
        )

    assert "disclosure_overlay" not in caplog.text


@pytest.mark.req("REQ-CMP-022")
def test_a_stale_platform_record_gets_the_current_text(tmp_path: Path) -> None:
    """Optimized mode writes only these; the publisher falls back to them."""
    from src.video.producer.steps import _check_existing_metadata

    (tmp_path / "metadata_youtube.json").write_text(
        json.dumps({"title": "t", "disclosure": "#ad"}), encoding="utf-8"
    )
    ctx = MagicMock()
    ctx.config = config_in("es-ES")
    ctx.state = {}
    ctx.run_paths = {"run_root": tmp_path, "description_file": tmp_path / "d.txt"}

    assert _check_existing_metadata(ctx) is True
    written = json.loads((tmp_path / "metadata_youtube.json").read_text("utf-8"))
    assert written["disclosure"] == "#publi"


@pytest.mark.req("REQ-CMP-024")
def test_every_variant_is_checked_for_glyphs() -> None:
    """The check read a removed `enabled` key, so it never ran."""
    config = config_in("en-US")
    seen: list[str] = []

    def record(text: str, **_: object) -> None:
        seen.append(text)

    with patch("src.video.assembler.font_resolver.fontfile_for_text", record):
        VideoConfigValidator()._validate_overlay_glyph_coverage(config)

    assert "#publi" in seen
    assert "#ad" in seen
