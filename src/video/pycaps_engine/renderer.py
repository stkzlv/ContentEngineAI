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


def layout_from_visual_bounds(
    bounds: VisualBounds | None,
    settings: PycapsSettings,
) -> Any:
    """Translate VisualBounds into a pycaps ``SubtitleLayoutOptions``.

    The bottom-anchor offset is derived so caption top falls just below
    the visual content. Formula:

        offset = (bounds.y + bounds.height + margin) - 0.95

    clamped to [-0.9, 0]. Values outside that range get clamped — they would
    push captions above centre or below frame. When ``bounds`` is None or
    ``vertical_align_offset`` is set manually, the manual value wins.
    """
    # Deferred import so the main app can import this module without pycaps.
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
        try:
            caps_builder, custom_renderer = self._build_pipeline(
                input_video=input_video,
                transcript_path=transcript_path,
                output_video=output_video,
                template_name=template_used,
                visual_bounds=visual_bounds,
                settings=settings,
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

        layout = layout_from_visual_bounds(visual_bounds, settings)
        builder.with_layout_options(layout)

        custom_renderer: Any | None = None
        if settings.renderer == "pictex":
            from pycaps.renderer import PictexSubtitleRenderer

            custom_renderer = PictexSubtitleRenderer()
            builder.with_custom_subtitle_renderer(custom_renderer)
        # CSS renderer is the library default, no explicit wiring needed.

        return builder, custom_renderer
