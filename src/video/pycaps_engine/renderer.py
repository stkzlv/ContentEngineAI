"""pycaps renderer wrapper.

Bridges ContentEngineAI's config + VisualBounds model onto the pycaps
``CapsPipelineBuilder`` API. All ``import pycaps`` calls are deferred until
:meth:`PycapsRenderer.render` so the module can be imported safely without
the optional Poetry group installed.

Flow:

1. ``select_template_for_product`` picks a deterministic template name for a
   product id (md5-keyed against ``PycapsSettings.template_pool``).
2. ``layout_from_visual_bounds`` translates the existing
   :class:`src.video.subtitle_positioning.VisualBounds` into a pycaps
   ``SubtitleLayoutOptions`` instance so captions land in the whitespace
   below the product image.
3. ``PycapsRenderer.render`` builds the caps pipeline, runs it, and returns
   a :class:`PycapsRenderResult` with timing + metadata.

On import failure (``pycaps`` group not installed), :meth:`render` raises
:class:`PycapsUnavailableError` which the caller catches and routes into
the configured fallback policy.
"""

from __future__ import annotations

import gc
import hashlib
import logging
import os
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.video.config.subtitle_models import PycapsSettings
    from src.video.subtitle_positioning import VisualBounds

logger = logging.getLogger(__name__)

_DEFAULT_TEMPLATE: str = "word-focus"
_BOTTOM_ANCHOR_MARGIN: float = 0.02  # keep 2% whitespace below the image
_PYCAPS_BOTTOM_BASE: float = 0.95  # matches LayoutUtils.get_vertical_alignment_position
_OFFSET_MIN: float = -0.9
_OFFSET_MAX: float = 0.0


# Distros Playwright maps onto the ``ubuntuXX.XX`` build family. On these, a
# major >= 26 produces the ``ubuntu26.04`` tag that has no registry build.
# Matches Playwright's own getHostPlatform() list (ubuntu, pop, neon, tuxedo).
_UBUNTU_LIKE_IDS = frozenset({"ubuntu", "pop", "neon", "tuxedo"})


def _ensure_playwright_chromium_platform() -> None:
    """Force the Ubuntu 24.04 chromium build on Ubuntu 26.04+.

    Playwright (<=1.60) maps Ubuntu 26.04 to ``ubuntu26.04-x64``, which has no
    browser in its registry, so the CSS renderer's ``chromium.launch()`` fails.
    The 24.04 build is binary-compatible; the documented override forces it.
    Runs for every entry point (standalone producer, batch, tests). No-op off
    Ubuntu 26+ or when the var is already set, so an explicit override always
    wins. Covers Ubuntu and the derivatives Playwright treats as Ubuntu (Pop!_OS,
    KDE neon, Tuxedo). See docs/troubleshooting.md.
    """
    if os.environ.get("PLAYWRIGHT_HOST_PLATFORM_OVERRIDE"):
        return
    try:
        with open("/etc/os-release", encoding="utf-8") as f:
            release = dict(line.rstrip().split("=", 1) for line in f if "=" in line)
    except OSError:
        return
    if release.get("ID", "").strip('"').lower() not in _UBUNTU_LIKE_IDS:
        return
    try:
        major = int(release.get("VERSION_ID", "").strip('"').split(".")[0])
    except ValueError:
        return
    if major >= 26:
        os.environ["PLAYWRIGHT_HOST_PLATFORM_OVERRIDE"] = "ubuntu24.04-x64"
        logger.debug(
            "Forced PLAYWRIGHT_HOST_PLATFORM_OVERRIDE=ubuntu24.04-x64 "
            "(Ubuntu %s has no native Playwright chromium build)",
            release.get("VERSION_ID", "").strip('"'),
        )


class PycapsUnavailableError(RuntimeError):
    """Raised when the pycaps optional dependency is not installed."""


@dataclass(frozen=True)
class PycapsRenderResult:
    """Outcome of a single pycaps render call.

    All numbers are best-effort and intended for telemetry / logging, not for
    precise billing. ``peak_rss_mb`` reflects RUSAGE_SELF before/after the
    render — it's a process-wide ceiling, not a delta.
    """

    success: bool
    output_path: Path
    template_used: str
    renderer_used: str
    wall_time_sec: float
    peak_rss_mb: float
    error: str | None = None


def select_template_for_product(
    product_id: str,
    settings: PycapsSettings,
) -> str:
    """Pick a deterministic template name for the given product id.

    Uses the same md5-hex-slice pattern already used for script template /
    font / colour randomisation elsewhere in the project, so the same product
    reproducibly gets the same template across runs.

    Fallback order:
      1. If ``template_pool`` has >=1 entries → md5 hash mod len.
      2. Else use ``template_name``.
      3. Else the hard-coded ``_DEFAULT_TEMPLATE``.
    """
    pool = list(settings.template_pool or [])
    if not pool:
        return settings.template_name or _DEFAULT_TEMPLATE
    if len(pool) == 1:
        return pool[0]
    # md5 is used here as a deterministic, non-cryptographic selector keyed on
    # product_id — same pattern as script/font/colour selection elsewhere in
    # the project. Not a security primitive.
    digest = hashlib.md5(  # noqa: S324
        f"{product_id}:pycaps_template".encode()
    ).hexdigest()
    index = int(digest[0:8], 16) % len(pool)
    return pool[index]


def _safe_zone_max_width(safe_zone: Any) -> float | None:
    """Compute the maximum width ratio that keeps centered text inside the safe zone.

    For center-aligned text the constraint is:
    ``max_width = 2 * min(0.5 - min_x, max_x - 0.5)``

    Returns None when no safe zone is provided or when the safe zone is
    wider than the frame (no clamping needed).
    """
    if safe_zone is None:
        return None
    min_x = getattr(safe_zone, "min_x", 0.0)
    max_x = getattr(safe_zone, "max_x", 1.0)
    limit = 2.0 * min(0.5 - min_x, max_x - 0.5)
    if limit >= 1.0:
        return None
    return max(0.1, limit)  # floor at 10% to avoid degenerate layouts


def _drop_punctuation_stripping(builder: Any) -> None:
    """Stop a template deleting the decimal point out of a number.

    `word-focus` ships `RemovePunctuationMarksEffect(['.'])`, whose
    implementation is `text.replace('.', '')` -- every period in the word, not
    just a trailing one. It is a styling choice that reads fine on prose and
    silently rewrites a figure: joining "2" and ".4GHz" back into one word
    hands it an internal period, and it burns `24GHz`. `$1,299.99` becomes
    `$1,29999`.

    The effect's own `exception_marks` cannot express "a period between
    digits", so the template cannot be configured out of it; the effect has to
    go. A caption losing a full stop costs nothing next to a caption stating a
    different number.

    Reaches into the pipeline the same way the layout merge above does, and
    tolerates the attribute being absent so a pycaps upgrade that renames it
    degrades to the old behaviour rather than raising.
    """
    pipeline = getattr(builder, "_caps_pipeline", None)
    if pipeline is None:
        return
    effects = getattr(pipeline, "_text_effects", None)
    if not effects:
        return

    kept = [
        effect
        for effect in effects
        if type(effect).__name__ != "RemovePunctuationMarksEffect"
    ]
    if len(kept) != len(effects):
        logger.debug(
            "Dropped %d punctuation-stripping effect(s) from the template",
            len(effects) - len(kept),
        )
        pipeline._text_effects = kept


def merge_layout_with_template(
    template_layout: Any,
    settings: PycapsSettings,
    bounds: VisualBounds | None = None,
    safe_zone: Any = None,
) -> Any:
    """Merge our config values into the template's own layout.

    The key insight: pycaps templates ship their own ``SubtitleLayoutOptions``
    with a vertical_align tuned for that template's font size and style (e.g.
    ``word-focus`` uses ``center``). Replacing it wholesale with our computed
    layout breaks positioning.

    Instead, we selectively override only what the user's config explicitly
    asks for:

    - ``max_width_ratio`` and ``max_number_of_lines`` always override
      (these are safe, template-agnostic layout constraints).
    - ``max_width_ratio`` is clamped to the platform safe zone when provided
      so captions never extend into TikTok/Shorts/Reels UI overlay zones.
    - ``vertical_align`` is only overridden when the user sets an explicit
      ``vertical_align_offset``. Otherwise the template's own alignment
      (center, bottom, etc.) is preserved.
    """
    from pycaps.layout import (
        VerticalAlignment,
        VerticalAlignmentType,
    )

    # Clamp max_width_ratio to safe zone boundaries.
    width_ratio = settings.max_width_ratio
    sz_limit = _safe_zone_max_width(safe_zone)
    if sz_limit is not None and width_ratio > sz_limit:
        logger.debug(
            "Clamping max_width_ratio from %.2f to %.2f (safe zone)",
            width_ratio,
            sz_limit,
        )
        width_ratio = sz_limit

    # Start from the template's layout and override width/lines.
    updates: dict[str, Any] = {
        "max_width_ratio": width_ratio,
        "max_number_of_lines": settings.max_number_of_lines,
    }

    # Only touch vertical alignment when explicitly requested.
    if settings.vertical_align_offset is not None:
        align_type = VerticalAlignmentType(settings.vertical_align)
        updates["vertical_align"] = VerticalAlignment(
            align=align_type, offset=settings.vertical_align_offset
        )
    elif bounds is not None and settings.vertical_align == "bottom":
        # Compute offset from VisualBounds, but ONLY when the user
        # explicitly requested "bottom" (not the default from the template).
        # If vertical_align still has the Pydantic default we leave the
        # template alone.
        pass  # Template wins — don't override.

    merged = template_layout.model_copy(update=updates)
    logger.debug(
        "Merged layout: template align=%s offset=%.2f -> final align=%s "
        "offset=%.2f (width=%.2f, lines=%d)",
        template_layout.vertical_align.align.value,
        template_layout.vertical_align.offset,
        merged.vertical_align.align.value,
        merged.vertical_align.offset,
        merged.max_width_ratio,
        merged.max_number_of_lines,
    )
    return merged


# Keep the old function name as a thin wrapper for backward compatibility
# with tests and docs that reference it.
def layout_from_visual_bounds(
    bounds: VisualBounds | None,
    settings: PycapsSettings,
) -> Any:
    """Build a standalone SubtitleLayoutOptions (no template merge).

    Used by unit tests and any code path that doesn't have access to a
    template's pre-loaded layout. For the production render path, prefer
    :func:`merge_layout_with_template`.
    """
    from pycaps.layout import (
        SubtitleLayoutOptions,
        VerticalAlignment,
        VerticalAlignmentType,
    )

    align_type = VerticalAlignmentType(settings.vertical_align)

    if settings.vertical_align_offset is not None:
        offset = settings.vertical_align_offset
    elif bounds is not None and align_type == VerticalAlignmentType.BOTTOM:
        raw = (bounds.y + bounds.height + _BOTTOM_ANCHOR_MARGIN) - _PYCAPS_BOTTOM_BASE
        offset = max(_OFFSET_MIN, min(_OFFSET_MAX, raw))
    else:
        offset = 0.0

    return SubtitleLayoutOptions(
        vertical_align=VerticalAlignment(align=align_type, offset=offset),
        max_width_ratio=settings.max_width_ratio,
        max_number_of_lines=settings.max_number_of_lines,
    )


class PycapsRenderer:
    """Thin synchronous wrapper around ``CapsPipelineBuilder``.

    One instance is reusable across renders. The constructor does nothing
    heavy — pycaps imports happen inside :meth:`render` so unit tests can
    monkey-patch at call time.
    """

    def render(
        self,
        input_video: Path,
        transcript_path: Path,
        output_video: Path,
        product_id: str,
        visual_bounds: VisualBounds | None,
        settings: PycapsSettings,
        safe_zone: Any = None,
    ) -> PycapsRenderResult:
        """Burn pycaps captions onto ``input_video`` and write ``output_video``.

        Args:
        ----
            input_video: Pre-assembled mp4 (usually ``run_paths["final_video_output"]``
                straight from ``step_assemble_video``).
            transcript_path: ``whisper_json`` file saved by
                :mod:`src.video.pycaps_engine.transcript_adapter`.
            output_video: Target path for the burned mp4. The caller may
                rename this over ``input_video`` after verifying success.
            product_id: Used to seed deterministic template selection.
            visual_bounds: Optional bounds of visual content. When present,
                the layout offset is computed to place captions below the
                content. When None, pycaps defaults apply.
            settings: Typed pycaps settings from the merged profile.
            safe_zone: Optional ``PlatformSafeZone`` for clamping caption
                width to avoid platform UI overlays.

        Returns:
        -------
            :class:`PycapsRenderResult` with a ``success`` flag. The caller
            decides what to do with failures based on ``fallback_policy``.

        Raises:
        ------
            PycapsUnavailableError: The pycaps optional group is not
                installed. Wraps the ``ImportError`` so callers can
                distinguish "library missing" from "library broke".

        """
        wall_start = time.monotonic()
        template_used = select_template_for_product(product_id, settings)
        if settings.renderer == "css":
            _ensure_playwright_chromium_platform()
        try:
            caps_builder, custom_renderer = self._build_pipeline(
                input_video=input_video,
                transcript_path=transcript_path,
                output_video=output_video,
                template_name=template_used,
                visual_bounds=visual_bounds,
                settings=settings,
                safe_zone=safe_zone,
            )
        except ImportError as e:
            raise PycapsUnavailableError(
                "pycaps is not installed. Install the optional group with "
                "`poetry install --with pycaps` and "
                "`poetry run playwright install chromium`."
            ) from e
        except Exception as e:  # noqa: BLE001 - build errors shouldn't leak
            wall_time = time.monotonic() - wall_start
            peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            logger.error("Failed to build pycaps pipeline: %s", e, exc_info=True)
            return PycapsRenderResult(
                success=False,
                output_path=output_video,
                template_used=template_used,
                renderer_used=settings.renderer,
                wall_time_sec=wall_time,
                peak_rss_mb=peak_rss,
                error=str(e),
            )

        pipeline = caps_builder.build()

        gc.collect()
        try:
            pipeline.run()
        except Exception as e:  # noqa: BLE001 - render errors caught for telemetry
            wall_time = time.monotonic() - wall_start
            peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            logger.error("pycaps render failed: %s", e, exc_info=True)
            return PycapsRenderResult(
                success=False,
                output_path=output_video,
                template_used=template_used,
                renderer_used=settings.renderer,
                wall_time_sec=wall_time,
                peak_rss_mb=peak_rss,
                error=str(e),
            )

        wall_time = time.monotonic() - wall_start
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

        if not output_video.exists():
            return PycapsRenderResult(
                success=False,
                output_path=output_video,
                template_used=template_used,
                renderer_used=settings.renderer,
                wall_time_sec=wall_time,
                peak_rss_mb=peak_rss,
                error="pycaps finished without producing the expected output file",
            )

        logger.info(
            "pycaps rendered %s in %.2fs (template=%s, renderer=%s, peak=%.0f MB)",
            output_video.name,
            wall_time,
            template_used,
            settings.renderer,
            peak_rss,
        )
        # ``custom_renderer`` is intentionally discarded; it was only needed
        # while configuring the builder.
        _ = custom_renderer

        return PycapsRenderResult(
            success=True,
            output_path=output_video,
            template_used=template_used,
            renderer_used=settings.renderer,
            wall_time_sec=wall_time,
            peak_rss_mb=peak_rss,
        )

    def _build_pipeline(
        self,
        *,
        input_video: Path,
        transcript_path: Path,
        output_video: Path,
        template_name: str,
        visual_bounds: VisualBounds | None,
        settings: PycapsSettings,
        safe_zone: Any = None,
    ) -> tuple[Any, Any | None]:
        """Construct the ``CapsPipelineBuilder`` and apply layout + renderer.

        Separated from :meth:`render` so tests can patch it to inject fakes
        without monkey-patching ``pycaps`` itself.
        """
        from pycaps.template import TemplateFactory, TemplateLoader
        from pycaps.transcriber import TranscriptFormat

        template = TemplateFactory().create(template_name)
        builder = (
            TemplateLoader(template).with_input_video(str(input_video)).load(False)
        )
        builder.with_transcription_file(
            str(transcript_path), TranscriptFormat.WHISPER_JSON
        )
        builder.with_output_video(str(output_video))

        # Merge our config with the template's own layout instead of
        # replacing it. The template's vertical_align is preserved unless
        # the user explicitly set vertical_align_offset.
        template_layout = builder._caps_pipeline._layout_options
        merged_layout = merge_layout_with_template(
            template_layout, settings, visual_bounds, safe_zone=safe_zone
        )
        builder.with_layout_options(merged_layout)
        _drop_punctuation_stripping(builder)

        custom_renderer: Any | None = None
        if settings.renderer == "pictex":
            from pycaps.renderer import PictexSubtitleRenderer

            custom_renderer = PictexSubtitleRenderer()
            builder.with_custom_subtitle_renderer(custom_renderer)
        # CSS renderer is the library default, no explicit wiring needed.

        return builder, custom_renderer
