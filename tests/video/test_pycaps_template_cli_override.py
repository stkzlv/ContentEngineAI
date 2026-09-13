"""CLI override semantics for `--pycaps-template` and `--pycaps-template-pool`.

The deterministic template selector falls through to ``template_name`` only
when ``template_pool`` is empty or single-entry. With a multi-entry pool the
md5 hash always wins, so ``--pycaps-template`` alone would silently no-op.
``_build_cli_overrides`` (producer) and ``_build_producer_overrides`` (global
batch) compensate by clearing the pool whenever ``--pycaps-template`` is
passed without an explicit ``--pycaps-template-pool``.
"""

from __future__ import annotations

import argparse


def _make_producer_args(**kwargs) -> argparse.Namespace:
    """Minimal namespace for src.video.producer.cli._build_cli_overrides.

    Only fields the override builder reads are populated; the rest default
    to None / False to satisfy the `getattr(..., None)` checks.
    """
    defaults: dict = {
        "topic": None,
        "topic_description": None,
        "topic_keywords": None,
        "topics_file": None,
        "voice_profile": None,
        "script_template": None,
        "pillar": None,
        "subtitle_engine": None,
        "pycaps_template": None,
        "pycaps_template_pool": None,
        "pycaps_renderer": None,
        "subtitle_anchor": None,
        "subtitle_margin": None,
        "content_aware": False,
        "no_content_aware": False,
        "font_size_scale": None,
        "max_subtitle_width_fraction": None,
        "subtitle_alignment": None,
        "max_line_length": None,
        "max_words_per_line": None,
        "max_duration": None,
        "min_duration": None,
        "preset": None,
        "ass_karaoke": False,
        "ass_fade": False,
        "subtitle_format": None,
        "randomize_fonts": False,
        "no_randomize_fonts": False,
        "randomize_colors": False,
        "no_randomize_colors": False,
        "randomize_effects": False,
        "no_randomize_effects": False,
        "image_width_percent": None,
        "image_top_position_percent": None,
        "target_platform": None,
        "metadata_mode": None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestProducerCliPycapsTemplateOverride:
    """`src.video.producer.cli._build_cli_overrides` template/pool semantics."""

    def test_template_alone_clears_pool(self):
        from src.video.producer.cli import _build_cli_overrides

        overrides = _build_cli_overrides(
            _make_producer_args(pycaps_template="neo-minimal")
        )

        assert overrides["subtitle_settings.pycaps.template_name"] == "neo-minimal"
        assert overrides["subtitle_settings.pycaps.template_pool"] == []

    def test_pool_alone_does_not_set_template_name(self):
        from src.video.producer.cli import _build_cli_overrides

        overrides = _build_cli_overrides(
            _make_producer_args(pycaps_template_pool=["neo-minimal", "explosive"])
        )

        assert "subtitle_settings.pycaps.template_name" not in overrides
        assert overrides["subtitle_settings.pycaps.template_pool"] == [
            "neo-minimal",
            "explosive",
        ]

    def test_explicit_pool_wins_over_implicit_clear(self):
        """When both flags are passed, --pycaps-template-pool wins.

        --pycaps-template still sets template_name, but the pool override
        from --pycaps-template-pool must replace the implicit empty list.
        """
        from src.video.producer.cli import _build_cli_overrides

        overrides = _build_cli_overrides(
            _make_producer_args(
                pycaps_template="neo-minimal",
                pycaps_template_pool=["word-focus", "hype"],
            )
        )

        assert overrides["subtitle_settings.pycaps.template_name"] == "neo-minimal"
        assert overrides["subtitle_settings.pycaps.template_pool"] == [
            "word-focus",
            "hype",
        ]

    def test_neither_flag_omits_both_keys(self):
        from src.video.producer.cli import _build_cli_overrides

        overrides = _build_cli_overrides(_make_producer_args())

        assert "subtitle_settings.pycaps.template_name" not in overrides
        assert "subtitle_settings.pycaps.template_pool" not in overrides
