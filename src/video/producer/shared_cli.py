"""Render-override flags shared by the producer CLI and the global batch.

Declaration and application live together here because they drifted apart
when each parser carried its own copy: a flag declared in both parsers but
applied in only one path is how `--cta` shipped inert on the batch, with
tests that grepped for the symbols and passed. One definition makes
declare-and-apply one unit.
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
