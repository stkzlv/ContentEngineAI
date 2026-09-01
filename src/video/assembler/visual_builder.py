"""Visual filter chain construction.

This module provides utilities for building FFmpeg visual filter chains with
aspect ratio handling, positioning, and transitions.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.video.assembler.media_inspector import MediaInspector
from src.video.assembler.video_strategies import VideoStrategyFactory

if TYPE_CHECKING:
    from src.video.config.visual_models import MergedProfileSettings
from src.video.config import VideoConfig

logger = logging.getLogger(__name__)


def _label_body(label: str) -> str:
    """The inner text of an FFmpeg stream label, safe to build new labels from.

    Strips the brackets and replaces the `:` in a pad-style label such as
    `[0:v]`, which FFmpeg would otherwise read as an argument separator.
    """
    return label.strip("[]").replace(":", "_")


# How far the `blur-fill` background is downscaled before the gaussian runs.
# The upscale afterwards does most of the blurring for free.
_BLUR_DOWNSCALE = 6


def _build_image_placement(
    *,
    index: int,
    vf_scale: str,
    width: int,
    height: int,
    target_y: int,
    pad_color: str,
    pix_fmt: str,
    background_fill: str,
    blur_sigma: float,
    blur_darken: float,
    out_label: str,
) -> str:
    """Put a scaled image on the frame, and decide what surrounds it.

    `color` pads with a solid colour, which is what this always did. A product
    photo is square or landscape against a 9:16 frame, so that leaves roughly
    half the screen empty -- measured 42-52% black across four frames of a
    real render.

    `blur` fills the remainder with a scaled, blurred copy of the same image:
    the standard short-form treatment, and it needs no second asset. The copy
    is scaled to *cover* rather than fit (`force_original_aspect_ratio=
    increase` then a centre crop), so it reaches the edges whichever way the
    source is oriented.

    The blurred copy is then darkened by `blur_darken`, because it is what the
    captions sit on. `docs/subtitle-best-practices.md` puts the base style at
    white fill with a black stroke, and the 21:1 it quotes is the fill against
    that stroke, which is what keeps captions legible over anything. What a
    bright backdrop costs is the margin around it: measured on a real render
    the band ran 102-165 of 255, and white fill against the light end is
    2.5:1, so the stroke is doing all the separating on its own.

    `colorlevels` scales rather than subtracts, which matters: `eq=brightness`
    took a dark backdrop to solid black in testing and lost the surround.
    """
    if background_fill != "blur":
        return (
            f"[{index}:v]{vf_scale},setsar=1,"
            f"pad={width}:{height}:(ow-iw)/2:{target_y}:"
            f"color={pad_color},"
            f"format={pix_fmt}{out_label}"
        )

    # `colorlevels` is RGB-only, so it is placed on the source-resolution copy
    # and followed straight back to YUV. Leaving it after the crop measured
    # 15s -> 38s of filter CPU on a 5s clip, because the whole backdrop then
    # runs at frame size in RGB and converts back before the overlay.
    # The leading `format=` is load-bearing, not tidiness. `colorlevels` on a
    # packed RGBA frame produces striped garbage on ffmpeg 8.0.1 and still
    # exits 0, and a PNG with alpha reaches here through both the scraper and
    # the stock provider. Measured against the pre-move placement on an RGBA
    # gradient: maxdiff 84 without it, 3 with it.
    darken = (
        f"format={pix_fmt},"
        f"colorlevels=romax={blur_darken}:gomax={blur_darken}:"
        f"bomax={blur_darken},format={pix_fmt},"
        if blur_darken < 1.0
        else ""
    )
    return (
        f"[{index}:v]split=2[bg_{index}][fg_{index}];"
        f"[bg_{index}]{darken}scale={width}:{height}:"
        f"force_original_aspect_ratio=increase,"
        f"crop={width}:{height},gblur=sigma={blur_sigma},"
        f"setsar=1[bgb_{index}];"
        f"[fg_{index}]{vf_scale},setsar=1[fgs_{index}];"
        f"[bgb_{index}][fgs_{index}]overlay=(W-w)/2:{target_y},"
        f"format={pix_fmt}{out_label}"
    )


def _build_ken_burns_filter(
    *,
    width: int,
    height: int,
    duration_sec: float,
    fps: int,
    peak_zoom: float,
    in_label: str,
    out_label: str,
) -> str:
    """Build a settle-zoom (Ken Burns) FFmpeg zoompan filter for the first image.

    The output starts at `peak_zoom` on frame 0 and decreases to 1.0 over the
    segment duration, keeping the centre of the frame fixed. Frame 1 is
    mid-zoom rather than a static still — Phase 1.2 pattern-interrupt.

    Returns the filter as a single ``[in]...[out]`` clause; the caller chains
    it with the existing trim/setpts steps. No-op math (peak <= 1.0) is the
    caller's responsibility; the field validator already enforces peak >= 1.0.
    """
    total_frames = max(int(round(duration_sec * fps)), 2)
    zoom_step = (peak_zoom - 1.0) / total_frames
    return (
        f"{in_label}zoompan="
        f"z='if(eq(on,0),{peak_zoom:.3f},max(1.0,zoom-{zoom_step:.6f}))':"
        f"d={total_frames}:"
        f"x='iw/2-(iw/zoom/2)':"
        f"y='ih/2-(ih/zoom/2)':"
        f"s={width}x{height}:"
        f"fps={fps}{out_label}"
    )


@dataclass
class VisualGeometry:
    """Position and dimensions of visual element in video.

    This class stores the coordinates and size of a visual element after it
    has been positioned and scaled within the output video frame.

    Attributes
    ----------
        rendered_x: X-coordinate of the top-left corner
        rendered_y: Y-coordinate of the top-left corner
        rendered_w: Width of the rendered visual
        rendered_h: Height of the rendered visual

    """

    rendered_x: int
    rendered_y: int
    rendered_w: int
    rendered_h: int


class VisualFilterBuilder:
    """Build FFmpeg visual filter chains with aspect ratio handling."""

    def __init__(
        self,
        media_inspector: MediaInspector,
        config: VideoConfig,
        strategy_factory: VideoStrategyFactory | None,
        profile_settings: "MergedProfileSettings | None",
        debug_mode: bool = False,
        normalize_video_callback: (Callable[[Path], Awaitable[Path]] | None) = None,
    ):
        """Initialize VisualFilterBuilder.

        Args:
        ----
            media_inspector: MediaInspector instance for media operations
            config: VideoConfig containing settings
            strategy_factory: VideoStrategyFactory for video mode strategies
            profile_settings: Merged profile settings (may be None initially)
            debug_mode: Enable debug logging
            normalize_video_callback: Async callback for video format normalization

        """
        self.inspector = media_inspector
        self.config = config
        self.strategy_factory = strategy_factory
        self.profile_settings = profile_settings
        self.debug_mode = debug_mode
        self.normalize_video_callback = normalize_video_callback

    def _get_effective_subtitle_settings(self) -> dict[str, Any]:
        """Get effective subtitle settings with profile overrides applied."""
        if self.profile_settings:
            return self.profile_settings.subtitle_settings.model_dump()
        return dict(self.config.subtitle_settings)

    def apply_aspect_ratio_mode(
        self,
        input_label: str,
        aspect_mode: str,
        target_width: int,
        target_height: int,
        video_width: int,
        video_height: int,
        output_label: str | None = None,
        video_top_percent: float | None = None,
        target_content_height: int | None = None,
        blur_sigma: float | None = None,
        blur_darken: float | None = None,
    ) -> tuple[str, str, VisualGeometry | None]:
        """Apply aspect ratio transformation based on configured mode.

        Generates FFmpeg filter strings for aspect ratio handling:
        - letterbox: Maintain aspect ratio with black padding (centered)
        - crop-to-fit: Scale to fill frame and crop edges (centered)
        - blur-fill: Letterbox geometry, but the bars carry a scaled and
          blurred copy of the same frame instead of black
        - smart-scale: Auto-select based on aspect ratio similarity

        `blur-fill` reports the same geometry as `letterbox`: the content band
        is placed identically and only what surrounds it differs, so subtitle
        and disclosure placement are unaffected by the choice between them.

        Args:
        ----
            input_label: FFmpeg input label (e.g., "[v0]")
            aspect_mode: Mode ("letterbox", "crop-to-fit", "blur-fill",
                "smart-scale")
            target_width: Target output width in pixels
            target_height: Target output height in pixels (full frame)
            video_width: Source video width in pixels
            video_height: Source video height in pixels
            output_label: Optional output label override
            video_top_percent: Optional vertical position override (0.0-1.0)
            target_content_height: Optional content height limit
            blur_sigma: Blur strength for `blur-fill`; the caller passes the
                profile-merged value, so the merge stays in one place
            blur_darken: Multiplier applied to the blurred backdrop only, so
                captions keep their contrast when the surround is a bright
                shot rather than the black bars `letterbox` left

        Returns:
        -------
            Tuple of (filter_string, output_label, geometry)
            geometry contains actual rendered position and dimensions

        """
        # Calculate aspect ratios
        target_aspect = target_width / target_height
        video_aspect = video_width / video_height

        # Smart-scale: auto-select mode based on aspect ratio similarity.
        #
        # The far branch is `blur-fill`, not `letterbox`. For a 16:9 source in
        # a 9:16 frame the difference is 2.16 against a tolerance of 0.10, so
        # no landscape clip can ever reach `crop-to-fit` and this resolved to
        # `letterbox` unconditionally -- a 608px content band in a 1920px
        # frame, 68% of it black. `blur-fill` keeps that geometry and every
        # pixel of the source, and fills the bars instead of leaving them
        # empty, so it dominates `letterbox` on the axis smart-scale is
        # choosing along. `letterbox` remains reachable by naming it.
        if aspect_mode == "smart-scale":
            aspect_diff = abs(target_aspect - video_aspect) / target_aspect
            aspect_tolerance = self.config.aspect_ratio.get(
                "smart_scale_tolerance", 0.10
            )
            aspect_mode = (
                "crop-to-fit" if aspect_diff <= aspect_tolerance else "blur-fill"
            )

        # Use provided output_label or generate one from input_label.
        #
        # `_label_body` is what makes the generated one a valid label. The
        # old default was `f"{input_label}_scaled"`, which yields
        # `[0:v]_scaled` -- brackets in the middle, and a `:` that FFmpeg
        # reads as an argument separator. Every mode returns that label and
        # the caller appends it, so every mode was rejected the same way
        # (`Trailing garbage after a filter`); blur-fill additionally derived
        # four internal labels from it. No production caller omits the
        # argument, which is why nothing had hit it.
        if output_label is None:
            output_label = f"[{_label_body(input_label)}_scaled]"

        # Letterbox and blur-fill: identical placement, different surround.
        geometry: VisualGeometry | None = None
        if aspect_mode in ("letterbox", "blur-fill"):
            # Determine scaling target height
            if target_content_height is not None:
                scale_height = target_content_height
            else:
                scale_height = target_height

            # Calculate vertical position
            if video_top_percent is not None:
                y_offset = int(target_height * video_top_percent)
                pad_y = str(y_offset)
            else:
                pad_y = "(oh-ih)/2"

            if aspect_mode == "letterbox":
                filter_string = (
                    f"{input_label}scale={target_width}:{scale_height}:"
                    f"force_original_aspect_ratio=decrease,"
                    f"pad={target_width}:{target_height}:"
                    f"(ow-iw)/2:{pad_y}:black"
                )
            else:
                # Same placement, filled surroundings. The background copy is
                # scaled to *cover* the whole frame and centre-cropped, so it
                # reaches every edge whichever way the source is oriented --
                # the treatment `_build_image_placement` already applies to
                # images. The labels are namespaced by the caller's output
                # label, which is unique per visual, so two segments in one
                # filtergraph cannot collide.
                tag = _label_body(output_label)
                overlay_y = "(H-h)/2" if video_top_percent is None else pad_y
                sigma = 20.0 if blur_sigma is None else blur_sigma
                # Same reasoning as the image chain: the backdrop is what
                # the captions sit on, and a bright one leaves the black
                # stroke doing all the separating.
                darken = 0.6 if blur_darken is None else blur_darken
                # Must stay after `gblur`. That filter accepts only planar
                # formats, so negotiation converts before `colorlevels` runs,
                # which is what keeps the video chain clear of the packed-RGBA
                # striping the image chain guards against with a leading
                # `format=`. Moving this ahead of the blur for speed would
                # reintroduce it.
                #
                # Applied at 1/6 scale and returned to YUV before the upscale,
                # for the same reason the blur is: `colorlevels` is RGB-only,
                # and running it at full frame measured 21s -> 139s of filter
                # CPU on a 30s clip. Omitted entirely at 1.0, so the
                # documented opt-out costs nothing.
                darken_clause = (
                    f"colorlevels=romax={darken}:gomax={darken}:bomax={darken},"
                    "format=yuv420p,"
                    if darken < 1.0
                    else ""
                )

                # Blur small, then upscale. A gaussian at full 1080x1920 runs
                # on every frame of every video segment and measured 15.8s of
                # added encode time on a 30s clip against letterbox; blurring
                # at 1/6 scale and letting the upscale carry the rest costs
                # 5.3s for a frame that is visually indistinguishable.
                #
                # `sigma` stays in full-frame terms -- it is divided by the
                # factor here rather than being reinterpreted -- so raising it
                # in YAML raises the blur by the same proportion it always
                # did. The upscale does contribute a blur of its own, on the
                # order of the factor, so values near the configured floor of
                # 1.0 are not distinguishable from each other. Nothing in the
                # bundled config sits near that floor; the default is 20.0.
                small_w = max(2, round(target_width / _BLUR_DOWNSCALE) // 2 * 2)
                small_h = max(2, round(target_height / _BLUR_DOWNSCALE) // 2 * 2)
                filter_string = (
                    f"{input_label}split=2[{tag}_bg][{tag}_fg];"
                    f"[{tag}_bg]scale={small_w}:{small_h}:"
                    f"force_original_aspect_ratio=increase,"
                    f"crop={small_w}:{small_h},"
                    f"gblur=sigma={sigma / _BLUR_DOWNSCALE:.4f},"
                    f"{darken_clause}"
                    f"scale={target_width}:{target_height},"
                    f"setsar=1[{tag}_bgb];"
                    f"[{tag}_fg]scale={target_width}:{scale_height}:"
                    f"force_original_aspect_ratio=decrease,setsar=1[{tag}_fgs];"
                    f"[{tag}_bgb][{tag}_fgs]overlay=(W-w)/2:{overlay_y}"
                )

            # Compute actual scaled dimensions (force_original_aspect_ratio=decrease)
            # Scale to fit within target_width x scale_height while maintaining aspect
            scale_by_width = target_width / video_width
            scale_by_height = scale_height / video_height
            scale_factor = min(scale_by_width, scale_by_height)
            actual_w = int(video_width * scale_factor)
            actual_h = int(video_height * scale_factor)

            # Compute actual position within padded frame
            actual_x = (target_width - actual_w) // 2
            if video_top_percent is not None:
                # Content area starts at y_offset, video centered within scale_height
                content_offset = (scale_height - actual_h) // 2
                actual_y = y_offset + content_offset
            else:
                # Centered in full frame
                actual_y = (target_height - actual_h) // 2

            geometry = VisualGeometry(
                rendered_x=actual_x,
                rendered_y=actual_y,
                rendered_w=actual_w,
                rendered_h=actual_h,
            )

            # Debug logging
            if target_content_height is not None:
                label = aspect_mode.upper()
                logger.debug(
                    f"[{label}] Constrained video: scale to "
                    f"{target_width}x{scale_height}, "
                    f"place in {target_width}x{target_height} at Y={pad_y}"
                )
                logger.debug(
                    f"[{label}] Actual geometry: {actual_w}x{actual_h} "
                    f"at ({actual_x}, {actual_y})"
                )

        # Crop-to-fit mode: scale to fill, crop excess
        elif aspect_mode == "crop-to-fit":
            filter_string = (
                f"{input_label}scale={target_width}:{target_height}:"
                f"force_original_aspect_ratio=increase,"
                f"crop={target_width}:{target_height}"
            )

        else:
            # Invalid mode - fallback to letterbox with warning
            logger.warning(
                f"Invalid aspect_mode '{aspect_mode}', falling back to letterbox"
            )
            filter_string = (
                f"{input_label}scale={target_width}:{target_height}:"
                f"force_original_aspect_ratio=decrease,"
                f"pad={target_width}:{target_height}:"
                f"(ow-iw)/2:(oh-ih)/2:black"
            )

        return filter_string, output_label, geometry

    def _calculate_subtitle_reserved_space(self, height: int) -> int:
        """Calculate subtitle space reservation to prevent overlap.

        Args:
        ----
            height: Frame height in pixels

        Returns:
        -------
            Reserved space in pixels for subtitles

        """
        subtitle_reserved_space = 0
        try:
            # Check if subtitles are enabled at the profile level
            if self.profile_settings:
                sub_settings = self.profile_settings.subtitle_settings
                subtitle_enabled = getattr(sub_settings, "enabled", False)
            else:
                subtitle_enabled = False

            if subtitle_enabled:
                subtitle_settings = self._get_effective_subtitle_settings()
                # Estimate subtitle height based on font size and margins
                font_size_scale = subtitle_settings.get("font_size_scale", 1.0)
                base_pct = self.config.video_settings.base_font_height_percent
                base_font_height = height * base_pct
                font_height = base_font_height * font_size_scale

                margin = subtitle_settings.get("margin", 0.05)
                margin_pixels = height * margin

                # Reserve space for font + outline + margin + buffer
                multiplier = self.config.video_settings.reserved_space_font_multiplier
                subtitle_reserved_space = int(font_height * multiplier + margin_pixels)
        except Exception as e:
            # If we can't get subtitle settings, use conservative default
            default_space = self.config.video_settings.default_subtitle_reserved_space
            logger.debug(
                f"Could not calculate subtitle space from settings ({e}), "
                f"using default {default_space * 100}% reservation"
            )
            subtitle_reserved_space = int(height * default_space)

        return (
            subtitle_reserved_space
            if subtitle_reserved_space > 0
            else int(height * default_space)
        )

    async def build_visual_chain(
        self,
        visual_inputs: list[Path],
        total_video_duration: float,
        is_relative_mode: bool,
        video_settings_dict: dict,
    ) -> tuple[
        list[str],
        list[str],
        list[tuple[Path, float, bool]],
        str,
        list[VisualGeometry],
        bool,
    ]:
        """Build visual filter chain and return geometry.

        Args:
        ----
            visual_inputs: List of visual input file paths
            total_video_duration: Target video duration in seconds
            is_relative_mode: Whether to use relative positioning
            video_settings_dict: Video settings from profile

        Returns:
        -------
            Tuple of (filter_parts, input_cmd_parts, timed_visuals,
                     final_video_stream_label, geometries,
                     image_positioning_overridden)

        """
        # Profile-merged, not global. Bound to `self.config.video_settings`,
        # every profile-overridable field read below silently took the global
        # value: `video_aspect_mode` and `video_assembly_mode` on three
        # bundled profiles each, and `first_frame_pre_motion` on one. The
        # merged object is the same `VideoSettings` type with the profile
        # folded in, so it carries every field the global does.
        #
        # The nearby `vs` binding a hundred lines down did this correctly for
        # three positioning fields, which is why the aspect mode reading the
        # global three lines later was easy to miss.
        video_settings = (
            self.profile_settings.video_settings
            if self.profile_settings
            else self.config.video_settings
        )
        video_files = [path for path in visual_inputs if self.inspector.is_video(path)]
        image_files = [
            path for path in visual_inputs if not self.inspector.is_video(path)
        ]

        # Apply format normalization to videos when enabled
        if video_files and video_settings.enable_format_normalization:
            if self.debug_mode:
                logger.debug(
                    f"Normalizing {len(video_files)} videos to H.264/30fps/yuv420p"
                )
            if self.normalize_video_callback:
                video_files = list(
                    await asyncio.gather(
                        *[self.normalize_video_callback(vf) for vf in video_files]
                    )
                )

        # Assemble videos using configured mode
        timed_visuals: list[tuple[Path, float, bool]] = []
        mode_info = ""

        if video_files and self.strategy_factory:
            # Call video assembly mode dispatcher
            strategy = self.strategy_factory.get_strategy(
                video_settings.video_assembly_mode
            )
            timed_visuals, mode_info = await strategy.assemble(
                video_files, image_files, total_video_duration
            )
        elif image_files:
            # Backward compatibility: image-only behavior
            num_visuals_total = len(image_files)
            if num_visuals_total > 1:
                num_transitions = num_visuals_total - 1
                transition_duration = video_settings.transition_duration_sec
                total_gross_image_duration = total_video_duration + (
                    num_transitions * transition_duration
                )
                if total_gross_image_duration > 0:
                    image_segment_duration = total_gross_image_duration / len(
                        image_files
                    )
                    if (
                        image_segment_duration
                        < video_settings.min_visual_segment_duration_sec
                    ):
                        image_segment_duration = (
                            video_settings.min_visual_segment_duration_sec
                        )
                    for path in image_files:
                        timed_visuals.append((path, image_segment_duration, False))
            elif num_visuals_total == 1:
                timed_visuals.append((image_files[0], total_video_duration, False))
            mode_info = f"image_only ({len(image_files)} images)"

        if not timed_visuals:
            raise ValueError("No visual media could be prepared for the timeline.")

        if self.debug_mode and mode_info:
            logger.debug(f"Visual assembly mode: {mode_info}")

        # Detect no-video scenario
        has_any_videos = any(is_video for _, _, is_video in timed_visuals)
        image_positioning_overridden = False
        if not has_any_videos and self.profile_settings:
            vs_model = self.profile_settings.video_settings
            has_video_positioning = (
                vs_model.video_top_position_percent
                != self.config.video_settings.video_top_position_percent
                or vs_model.video_content_height_percent
                != self.config.video_settings.video_content_height_percent
            )

            if has_video_positioning:
                # Deliberately global: neither fallback is declared on
                # VideoProfile, so there is nothing to merge. Named to say so
                # -- a second binding whose name did not distinguish it from
                # the merged one is what hid the aspect-mode bug above.
                global_settings = self.config.video_settings
                fallback_top = global_settings.fallback_image_top_percent
                fallback_width = global_settings.fallback_image_width_percent
                video_settings_dict["image_top_position_percent"] = fallback_top
                video_settings_dict["image_width_percent"] = fallback_width
                image_positioning_overridden = True
                logger.info(
                    "No videos detected in video-centric profile - "
                    "applying image-optimized positioning "
                    f"(top={fallback_top:.0%}, width={fallback_width:.0%})"
                )

        # Build filter chain
        input_cmd_parts: list[str] = []
        filter_parts: list[str] = []
        stream_labels: list[str] = []
        geometries: list[VisualGeometry] = []
        width, height = video_settings.resolution
        pix_fmt = video_settings.output_pixel_format

        all_visuals_dims = await asyncio.gather(
            *[self.inspector.get_media_dimensions(p) for p, _, _ in timed_visuals]
        )

        uniform_height = -1
        if not is_relative_mode:
            scaled_heights = []
            for orig_w, orig_h in all_visuals_dims:
                if orig_w > 0 and orig_h > 0:
                    scaled_h = int(
                        (width * video_settings_dict["image_width_percent"])
                        * (orig_h / orig_w)
                    )
                    scaled_heights.append(scaled_h)
            if scaled_heights:
                uniform_height = min(scaled_heights)

        for i, (path, duration, is_video_item) in enumerate(timed_visuals):
            if is_video_item:
                input_cmd_parts.extend(["-i", str(path)])
            else:
                input_cmd_parts.extend(
                    [
                        "-loop",
                        str(video_settings.image_loop),
                        "-framerate",
                        str(video_settings.frame_rate),
                        "-t",
                        str(duration),
                        "-i",
                        str(path),
                    ]
                )

            proc_label = f"[v_proc_{i}]"
            orig_w, orig_h = all_visuals_dims[i]

            # Apply aspect ratio handling for videos
            if is_video_item:
                # Get video positioning from profile or global settings
                video_top_percent = video_settings.video_top_position_percent
                video_height_percent = video_settings.video_content_height_percent
                video_valign = video_settings.video_vertical_align
                logger.debug(
                    f"[VIDEO POS] top={video_top_percent:.2%}, "
                    f"height={video_height_percent:.2%}, align={video_valign}"
                )

                target_content_height = int(height * video_height_percent)

                # When centering, pass video_top_percent=None so
                # apply_aspect_ratio_mode uses FFmpeg's (oh-ih)/2 expression
                effective_top = None if video_valign == "center" else video_top_percent

                aspect_filter, aspect_label, actual_geom = self.apply_aspect_ratio_mode(
                    f"[{i}:v]",
                    video_settings.video_aspect_mode,
                    width,
                    height,
                    orig_w,
                    orig_h,
                    output_label=f"[v{i}_scaled]",
                    video_top_percent=effective_top,
                    target_content_height=target_content_height,
                    blur_sigma=video_settings.video_background_blur_sigma,
                    blur_darken=video_settings.video_background_blur_darken,
                )

                vf_string = (
                    f"{aspect_filter}{aspect_label};"
                    f"{aspect_label}format={pix_fmt}[v_temp_{i}];"
                    f"[v_temp_{i}]trim=duration={duration},"
                    f"setpts=PTS-STARTPTS{proc_label}"
                )

                # Use actual geometry from apply_aspect_ratio_mode if available
                if actual_geom:
                    logger.debug(
                        f"Video {i}: Actual geometry "
                        f"y={actual_geom.rendered_y}px, "
                        f"height={actual_geom.rendered_h}px"
                    )
                    geometries.append(actual_geom)
                else:
                    # Fallback: compute from config (for non-letterbox modes)
                    video_height_pixels = int(height * video_height_percent)
                    if video_valign == "center":
                        video_top_pixels = (height - video_height_pixels) // 2
                    else:
                        video_top_pixels = int(height * video_top_percent)
                    logger.debug(
                        f"Video {i}: Config-based geometry "
                        f"y={video_top_pixels}px, height={video_height_pixels}px"
                    )
                    geometries.append(
                        VisualGeometry(
                            rendered_x=0,
                            rendered_y=video_top_pixels,
                            rendered_w=width,
                            rendered_h=video_height_pixels,
                        )
                    )
            else:
                # Image handling logic
                scaled_w_base = int(width * video_settings_dict["image_width_percent"])
                scaled_w, scaled_h = 0, 0
                vertical_align = video_settings_dict.get(
                    "image_vertical_align", "center"
                )

                # Calculate subtitle space reservation
                subtitle_reserved_space = self._calculate_subtitle_reserved_space(
                    height
                )

                # For centering, we calculate Y after knowing scaled_h
                # For top alignment, use the configured top position
                top_offset = video_settings_dict["image_top_position_percent"] * height
                max_available_height = height - top_offset - subtitle_reserved_space
                logger.debug(
                    f"Image {i}: Reserved {subtitle_reserved_space}px for "
                    f"subtitles, max available height: {max_available_height}px "
                    f"(frame: {height}px, align: {vertical_align})"
                )

                if not is_relative_mode and uniform_height > 0:
                    scaled_h = uniform_height
                    scaled_w = (
                        int(scaled_h * (orig_w / orig_h))
                        if orig_h > 0
                        else scaled_w_base
                    )
                    vf_scale = f"scale={scaled_w}:{scaled_h}"
                else:
                    scaled_w = scaled_w_base
                    scaled_h = int(scaled_w * (orig_h / orig_w)) if orig_w > 0 else -1

                    # Ensure scaled height doesn't exceed available space
                    if scaled_h > max_available_height:
                        scaled_h = int(max_available_height)
                        scaled_w = (
                            int(scaled_h * (orig_w / orig_h))
                            if orig_h > 0
                            else scaled_w_base
                        )

                    vf_scale = f"scale={scaled_w}:{scaled_h}"

                # Calculate Y position based on alignment
                if vertical_align == "center":
                    # Center image vertically in frame
                    target_y_pos = (height - scaled_h) / 2
                else:
                    # Use configured top offset
                    target_y_pos = top_offset

                geometries.append(
                    VisualGeometry(
                        rendered_x=int((width - scaled_w) / 2),
                        rendered_y=int(target_y_pos),
                        rendered_w=scaled_w,
                        rendered_h=scaled_h,
                    )
                )

                pre_motion = i == 0 and video_settings.first_frame_pre_motion
                if pre_motion:
                    zoom_filter = _build_ken_burns_filter(
                        width=width,
                        height=height,
                        duration_sec=duration,
                        fps=video_settings.frame_rate,
                        peak_zoom=video_settings.pre_motion_peak_zoom,
                        in_label=f"[v_temp_{i}]",
                        out_label=f"[v_motion_{i}]",
                    )
                    placement = _build_image_placement(
                        index=i,
                        vf_scale=vf_scale,
                        width=width,
                        height=height,
                        target_y=int(target_y_pos),
                        pad_color=video_settings.pad_color,
                        pix_fmt=pix_fmt,
                        background_fill=video_settings_dict.get(
                            "image_background_fill", "color"
                        ),
                        blur_sigma=video_settings_dict.get(
                            "image_background_blur_sigma", 20.0
                        ),
                        blur_darken=video_settings_dict.get(
                            "image_background_blur_darken", 0.6
                        ),
                        out_label=f"[v_temp_{i}]",
                    )
                    vf_string = (
                        f"{placement};"
                        f"{zoom_filter};"
                        f"[v_motion_{i}]trim=duration={duration},"
                        f"setpts=PTS-STARTPTS{proc_label}"
                    )
                else:
                    placement = _build_image_placement(
                        index=i,
                        vf_scale=vf_scale,
                        width=width,
                        height=height,
                        target_y=int(target_y_pos),
                        pad_color=video_settings.pad_color,
                        pix_fmt=pix_fmt,
                        background_fill=video_settings_dict.get(
                            "image_background_fill", "color"
                        ),
                        blur_sigma=video_settings_dict.get(
                            "image_background_blur_sigma", 20.0
                        ),
                        blur_darken=video_settings_dict.get(
                            "image_background_blur_darken", 0.6
                        ),
                        out_label=f"[v_temp_{i}]",
                    )
                    vf_string = (
                        f"{placement};"
                        f"[v_temp_{i}]trim=duration={duration},"
                        f"setpts=PTS-STARTPTS{proc_label}"
                    )

            filter_parts.append(vf_string)
            stream_labels.append(proc_label)

        # Apply transitions
        if len(stream_labels) > 1:
            transition_duration = video_settings.transition_duration_sec
            current_stream = stream_labels[0]
            current_offset = timed_visuals[0][1] - transition_duration
            for i in range(1, len(stream_labels)):
                next_stream = stream_labels[i]
                output_stream_label = f"[v_chain_{i}]"
                filter_parts.append(
                    f"{current_stream}{next_stream}xfade=transition=fade:"
                    f"duration={transition_duration}:"
                    f"offset={current_offset:.4f}{output_stream_label}"
                )
                current_stream = output_stream_label
                if i < len(timed_visuals) - 1:
                    current_offset += timed_visuals[i][1] - transition_duration
            final_video_stream_label = current_stream
        else:
            final_video_stream_label = stream_labels[0]

        return (
            filter_parts,
            input_cmd_parts,
            timed_visuals,
            final_video_stream_label,
            geometries,
            image_positioning_overridden,
        )
