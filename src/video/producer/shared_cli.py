"""Render-override flags shared by the producer CLI and the global batch.

Declaration and application live together here because they drifted apart
when each parser carried its own copy: a flag declared in both parsers but
applied in only one path is how `--cta` shipped inert on the batch, with
tests that grepped for the symbols and passed. One definition makes
declare-and-apply one unit.

A new flag joins `layout_render_overrides`, which the batch stores as
`GlobalBatchConfig.render_overrides`, and needs nothing else. A flag that
`pipeline.yaml` can also set goes through `subtitle_render_overrides`
instead and needs two batch-side hops: a `GlobalBatchConfig` field and its
copy in `src/pipeline/config.py`'s loader. That helper's `getattr(...,
None)` masks a missing config attribute -- the flag would parse on both
CLIs, work on the producer, and be silently inert on the batch.
"""

from __future__ import annotations

from typing import Any, Protocol


class _AddsArguments(Protocol):
    def add_argument(self, *args: Any, **kwargs: Any) -> Any: ...


def add_shared_render_args(target: _AddsArguments) -> None:
    """Declare the shared render-override flags on a parser or group.

    Both entry points must accept the same names with the same choices and
    semantics (the Module/Batch Alignment Rule); the help texts live once
    here so they cannot disagree either.
    """
    target.add_argument(
        "--voice-profile",
        type=str,
        metavar="NAME",
        help="Override voice profile selection.",
    )
    target.add_argument(
        "--script-template",
        type=str,
        metavar="NAME",
        help="Override script template (name without .md).",
    )
    target.add_argument(
        "--cta",
        type=str,
        metavar="LINE",
        help=(
            "Override the closing call to action (must be one of the "
            "configured options; otherwise selection proceeds normally)."
        ),
    )
    target.add_argument(
        "--pillar",
        type=str,
        metavar="NAME",
        help=(
            "Content pillar for the run (e.g. value, novelty, utility). "
            "Prepends the pillar preamble to the LLM prompt and picks the "
            "pillar audience; on a product render it also narrows the "
            "script template pool to that pillar's templates."
        ),
    )
    target.add_argument(
        "--subtitle-format",
        choices=["srt", "ass"],
        help=(
            "Subtitle format: srt or ass (with animations). The pycaps "
            "engine ignores it, and the bundled YAML selects pycaps, so "
            "pair this with --subtitle-engine ffmpeg to have it apply."
        ),
    )
    target.add_argument(
        "--subtitle-engine",
        choices=["ffmpeg", "pycaps"],
        help=(
            "Subtitle rendering engine. The bundled YAML selects pycaps. "
            "'ffmpeg' burns SRT/ASS via libass; 'pycaps' runs the pycaps "
            "library as a post-assembly step for animated captions "
            "(install the optional group first: `poetry install --with "
            "pycaps`)."
        ),
    )
    target.add_argument(
        "--pycaps-template",
        type=str,
        metavar="NAME",
        help=(
            "Pycaps template name (e.g. word-focus, hype, minimalist). "
            "Forces this template for every product by clearing the "
            "template pool. To use a custom multi-entry pool, pass "
            "--pycaps-template-pool instead."
        ),
    )
    target.add_argument(
        "--pycaps-template-pool",
        nargs="+",
        type=str,
        metavar="NAME",
        help=(
            "Pool of pycaps templates for deterministic per-product "
            "selection. Example: --pycaps-template-pool word-focus hype "
            "vibrant"
        ),
    )
    target.add_argument(
        "--pycaps-renderer",
        choices=["css", "pictex"],
        help=(
            "Pycaps renderer backend. 'css' = Playwright+Chromium "
            "(default, the only production-safe option). 'pictex' = "
            "browserless Skia path; PREVIEW ONLY, it renders words with "
            "no gaps between them."
        ),
    )

    target.add_argument(
        "--preset",
        choices=["minimal", "modern", "bold", "animated", "random"],
        help="Override subtitle style preset: minimal, modern, bold, animated, random.",
    )

    # Subtitle positioning arguments
    target.add_argument(
        "--subtitle-anchor",
        choices=["top", "center", "bottom", "above_content", "below_content"],
        help="Subtitle anchor position.",
    )
    target.add_argument(
        "--subtitle-margin",
        type=float,
        help="Subtitle margin as fraction of frame height (0.0-0.5).",
    )
    target.add_argument(
        "--content-aware",
        action="store_true",
        dest="subtitle_content_aware",
        default=None,
        help="Enable content-aware subtitle positioning.",
    )
    target.add_argument(
        "--no-content-aware",
        action="store_false",
        dest="subtitle_content_aware",
        default=None,
        help="Disable content-aware subtitle positioning.",
    )

    # Subtitle styling arguments
    target.add_argument(
        "--font-size-scale",
        type=float,
        help="Font size scale factor (0.5-2.0).",
    )
    target.add_argument(
        "--max-subtitle-width-fraction",
        type=float,
        help="Max subtitle width as fraction of frame width (0.0-1.0).",
    )
    target.add_argument(
        "--subtitle-alignment",
        choices=["left", "center", "right"],
        help="Horizontal text alignment.",
    )

    # Subtitle text formatting arguments
    target.add_argument(
        "--max-line-length",
        type=int,
        help="Maximum characters per subtitle line.",
    )
    target.add_argument(
        "--max-words-per-line",
        type=int,
        help="Maximum words per subtitle line (0 to disable).",
    )
    target.add_argument(
        "--max-duration",
        type=float,
        help="Maximum subtitle duration in seconds.",
    )
    target.add_argument(
        "--min-duration",
        type=float,
        help="Minimum subtitle duration in seconds.",
    )

    # Randomization arguments
    target.add_argument(
        "--randomize-fonts",
        action="store_true",
        dest="subtitle_randomize_fonts",
        default=None,
        help="Enable font randomization.",
    )
    target.add_argument(
        "--no-randomize-fonts",
        action="store_false",
        dest="subtitle_randomize_fonts",
        default=None,
        help="Disable font randomization.",
    )
    target.add_argument(
        "--randomize-colors",
        action="store_true",
        dest="subtitle_randomize_colors",
        default=None,
        help="Enable color randomization.",
    )
    target.add_argument(
        "--no-randomize-colors",
        action="store_false",
        dest="subtitle_randomize_colors",
        default=None,
        help="Disable color randomization.",
    )
    target.add_argument(
        "--randomize-effects",
        action="store_true",
        dest="subtitle_randomize_effects",
        default=None,
        help="Enable effect randomization.",
    )
    target.add_argument(
        "--no-randomize-effects",
        action="store_false",
        dest="subtitle_randomize_effects",
        default=None,
        help="Disable effect randomization.",
    )

    # Image positioning arguments
    target.add_argument(
        "--image-width-percent",
        type=float,
        help="Override image width as percentage of frame (0.0-1.0).",
    )
    target.add_argument(
        "--image-top-position-percent",
        type=float,
        help="Override image top position as percentage from top (0.0-1.0).",
    )

    # Metadata mode argument
    target.add_argument(
        "--metadata-mode",
        choices=["unified", "optimized"],
        help=(
            "Metadata generation mode. "
            "unified: Single title/description/hashtags for all platforms (default). "
            "optimized: Platform-specific SEO-tailored metadata."
        ),
    )


def subtitle_render_overrides(source: Any) -> dict[str, Any]:
    """The dotted subtitle-override keys, from an argparse namespace or the
    batch config -- both carry the same attribute names.

    Includes the pool-clear semantics: `--pycaps-template` clears the pool
    so the deterministic selector falls through to the named template
    (a multi-entry pool would otherwise still win via md5 hash and
    silently ignore the flag), while an explicit `--pycaps-template-pool`
    wins over that clear when both are passed.
    """
    overrides: dict[str, Any] = {}
    if getattr(source, "subtitle_format", None):
        overrides["subtitle_settings.subtitle_format"] = source.subtitle_format
    if getattr(source, "subtitle_engine", None):
        overrides["subtitle_settings.subtitle_engine"] = source.subtitle_engine
    if getattr(source, "pycaps_template", None):
        overrides["subtitle_settings.pycaps.template_name"] = source.pycaps_template
        overrides["subtitle_settings.pycaps.template_pool"] = []
    if getattr(source, "pycaps_template_pool", None):
        overrides["subtitle_settings.pycaps.template_pool"] = (
            source.pycaps_template_pool
        )
    if getattr(source, "pycaps_renderer", None):
        overrides["subtitle_settings.pycaps.renderer"] = source.pycaps_renderer
    return overrides


def layout_render_overrides(source: Any) -> dict[str, Any]:
    """The subtitle layout, image and metadata overrides, as dotted keys.

    Reads an argparse namespace; the batch stores the result on its config
    as `render_overrides`, since none of these has a pipeline.yaml key.
    """
    names = {
        "preset": "subtitle_settings.style_preset",
        "subtitle_anchor": "subtitle_settings.anchor",
        "subtitle_margin": "subtitle_settings.margin",
        "subtitle_content_aware": "subtitle_settings.content_aware",
        "font_size_scale": "subtitle_settings.font_size_scale",
        "max_subtitle_width_fraction": "subtitle_settings.max_subtitle_width_fraction",
        "subtitle_alignment": "subtitle_settings.horizontal_alignment",
        "max_line_length": "subtitle_settings.max_line_length",
        "max_words_per_line": "subtitle_settings.max_words_per_line",
        "max_duration": "subtitle_settings.max_duration",
        "min_duration": "subtitle_settings.min_duration",
        "subtitle_randomize_fonts": "subtitle_settings.randomize_fonts",
        "subtitle_randomize_colors": "subtitle_settings.randomize_colors",
        "subtitle_randomize_effects": "subtitle_settings.randomize_effects",
        "image_width_percent": "video_settings.image_width_percent",
        "image_top_position_percent": "video_settings.image_top_position_percent",
        "metadata_mode": "description_settings.metadata_mode",
    }
    # None is "not passed"; False and 0 are real values (--no-content-aware,
    # --max-words-per-line 0).
    return {
        key: getattr(source, name)
        for name, key in names.items()
        if getattr(source, name, None) is not None
    }
