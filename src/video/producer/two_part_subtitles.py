# src/video/producer/two_part_subtitles.py
"""Two-part subtitle system handler.

Handles generation of dual subtitles (upper static + lower voiceover) for video
production with content-aware positioning.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.video.producer.constants import (
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_TOP_POSITION,
    DEFAULT_VIDEO_WIDTH,
)

if TYPE_CHECKING:
    from src.video.producer.context import PipelineContext
    from src.video.subtitle_positioning import VisualBounds

logger = logging.getLogger(__name__)


@dataclass
class TwoPartConfig:
    """Parsed two-part subtitle configuration."""

    enabled: bool
    upper_enabled: bool
    upper_config: dict[str, Any]
    lower_enabled: bool
    lower_config: dict[str, Any]


class TwoPartSubtitleHandler:
    """Handles two-part (upper + lower) subtitle generation.

    This class encapsulates the logic for generating dual subtitles:
    - Upper line: Static text (product URL, custom text)
    - Lower line: Dynamic voiceover-synced subtitles
    """

    def __init__(
        self,
        ctx: "PipelineContext",
        profile_subtitle_settings: dict[str, Any],
        merged_profile_settings: dict[str, Any],
        two_part_config: dict[str, Any],
    ):
        """Initialize the handler.

        Args:
        ----
            ctx: Pipeline context with product data and configuration.
            profile_subtitle_settings: Subtitle-specific settings from profile.
            merged_profile_settings: Full merged profile settings.
            two_part_config: Two-part subtitle configuration dict.

        """
        self.ctx = ctx
        self.profile_subtitle_settings = profile_subtitle_settings
        self.merged_profile_settings = merged_profile_settings
        self.two_part_config = two_part_config
        self.config = self._parse_config()

    def _parse_config(self) -> TwoPartConfig:
        """Parse configuration supporting both nested dict and flat key structures."""
        # Parse upper line config
        if (
            isinstance(self.two_part_config, dict)
            and "upper_line" in self.two_part_config
        ):
            upper_config = self.two_part_config.get("upper_line", {})
            upper_enabled = upper_config.get("enabled", True)
        else:
            upper_enabled = self.merged_profile_settings.get(
                "two_part_subtitles_upper_enabled", True
            )
            upper_config = {
                "enabled": upper_enabled,
                "source_field": self.merged_profile_settings.get(
                    "two_part_subtitles_upper_source_field",
                    "shortened_affiliate_link",
                ),
                "anchor": self.merged_profile_settings.get(
                    "two_part_subtitles_upper_anchor", "above_content"
                ),
                "margin": self.merged_profile_settings.get(
                    "two_part_subtitles_upper_margin", 0.08
                ),
                "font_size_scale": self.merged_profile_settings.get(
                    "two_part_subtitles_upper_font_size_scale", 0.75
                ),
                "style_preset": self.merged_profile_settings.get(
                    "two_part_subtitles_upper_style_preset", "minimal"
                ),
                "use_full_duration": self.merged_profile_settings.get(
                    "two_part_subtitles_upper_use_full_duration", False
                ),
                "randomize_effects": self.merged_profile_settings.get(
                    "two_part_subtitles_upper_randomize_effects", False
                ),
            }

        # Parse lower line config
        if (
            isinstance(self.two_part_config, dict)
            and "lower_line" in self.two_part_config
        ):
            lower_config = self.two_part_config.get("lower_line", {})
            lower_enabled = lower_config.get("enabled", True)
        else:
            lower_enabled = self.profile_subtitle_settings.get(
                "two_part_subtitles_lower_enabled", True
            )
            lower_config = {
                "enabled": lower_enabled,
                "anchor": self.profile_subtitle_settings.get(
                    "two_part_subtitles_lower_anchor", "below_content"
                ),
                "margin": self.profile_subtitle_settings.get(
                    "two_part_subtitles_lower_margin", 0.05
                ),
            }

        return TwoPartConfig(
            enabled=True,
            upper_enabled=upper_enabled,
            upper_config=upper_config,
            lower_enabled=lower_enabled,
            lower_config=lower_config,
        )

    def calculate_visual_bounds(self) -> "VisualBounds":
        """Calculate visual bounds based on media type present in context."""
        from src.video.subtitle_positioning import VisualBounds

        has_videos = self.ctx.scraped_videos and len(self.ctx.scraped_videos) > 0
        has_images = self.ctx.scraped_images and len(self.ctx.scraped_images) > 0

        # If ctx.scraped_images is empty, try to find images from outputs directory
        # (subtitle generation may run before gather_visuals completes)
        if not has_images and not has_videos:
            # run_root is the product directory (e.g., outputs/B0BPH6S4DN/)
            images_dir = Path(self.ctx.run_paths["run_root"]) / "images"
            if images_dir.exists():
                image_files = list(images_dir.glob("*.jpg")) + list(
                    images_dir.glob("*.png")
                )
                if image_files:
                    self.ctx.scraped_images = image_files
                    has_images = True
                    logger.debug(
                        "Found %d images in outputs directory for visual bounds",
                        len(image_files),
                    )

        if has_videos:
            video_top = (
                self.ctx.profile.video_top_position_percent
                or DEFAULT_VIDEO_TOP_POSITION
            )
            video_height = (
                self.ctx.profile.video_content_height_percent or DEFAULT_VIDEO_HEIGHT
            )
            video_width = self.ctx.profile.image_width_percent or DEFAULT_VIDEO_WIDTH
            logger.debug("Using video positioning for visual bounds (videos present)")
        elif has_images:
            video_width = self.ctx.profile.image_width_percent or DEFAULT_VIDEO_WIDTH
            # Get vertical_align from video_settings (nested in merged_profile_settings)
            video_settings = self.merged_profile_settings.get("video_settings", {})
            vertical_align = video_settings.get("image_vertical_align", "center")
            logger.debug(
                "video_settings keys=%s, vertical_align=%s",
                list(video_settings.keys())[:5],
                vertical_align,
            )

            if vertical_align == "center":
                # Estimate centered image position based on typical aspect ratio
                video_top, video_height = self._estimate_centered_image_bounds(
                    video_width
                )
                logger.debug(
                    "Centered image bounds: video_top=%.4f (%.1f%%), "
                    "video_height=%.4f (%.1f%%)",
                    video_top,
                    video_top * 100,
                    video_height,
                    video_height * 100,
                )
            else:
                video_top = (
                    self.ctx.profile.image_top_position_percent
                    or DEFAULT_VIDEO_TOP_POSITION
                )
                video_height = DEFAULT_VIDEO_HEIGHT
                logger.debug("Using top-aligned image positioning for visual bounds")
        else:
            video_top = DEFAULT_VIDEO_TOP_POSITION
            video_height = DEFAULT_VIDEO_HEIGHT
            video_width = DEFAULT_VIDEO_WIDTH
            logger.debug("No media found, using default visual bounds")

        logger.debug(
            "Visual bounds for subtitles: y=%.2f%%, height=%.2f%%",
            video_top * 100,
            video_height * 100,
        )

        return VisualBounds(
            x=(1.0 - video_width) / 2,
            y=video_top,
            width=video_width,
            height=video_height,
        )

    def _estimate_centered_image_bounds(
        self, image_width_percent: float
    ) -> tuple[float, float]:
        """Estimate visual bounds for a vertically centered image.

        Args:
        ----
            image_width_percent: Image width as fraction of frame width.

        Returns:
        -------
            Tuple of (y_position, height) as fractions of frame height.

        """
        # Get frame dimensions
        frame_width, frame_height = self.ctx.config.video_settings.resolution

        # Try to get actual image dimensions from first scraped image
        if self.ctx.scraped_images:
            try:
                from PIL import Image

                with Image.open(self.ctx.scraped_images[0]) as img:
                    orig_w, orig_h = img.size
                    # Calculate scaled dimensions
                    scaled_w = int(frame_width * image_width_percent)
                    scaled_h = int(scaled_w * (orig_h / orig_w)) if orig_w > 0 else 0

                    # Calculate centered Y position
                    if scaled_h > 0 and scaled_h < frame_height:
                        centered_y = (frame_height - scaled_h) / 2
                        video_top = centered_y / frame_height
                        video_height = scaled_h / frame_height
                        logger.debug(
                            "Calculated centered bounds from image: "
                            "y=%.2f%%, h=%.2f%%",
                            video_top * 100,
                            video_height * 100,
                        )
                        return (video_top, video_height)
            except (OSError, ValueError) as e:
                logger.debug("Could not read image dimensions: %s", e)

        # Fallback: estimate using typical portrait aspect ratio (4:5)
        # Most product images are portrait
        typical_aspect = 5 / 4  # height / width ratio
        est_scaled_w = frame_width * image_width_percent
        estimated_h = est_scaled_w * typical_aspect

        if estimated_h < frame_height:
            centered_y = (frame_height - estimated_h) / 2
            video_top = centered_y / frame_height
            video_height = estimated_h / frame_height
        else:
            # Image would fill frame vertically
            video_top = 0.0
            video_height = 1.0

        return (video_top, video_height)

    def _get_upper_text(self) -> str | None:
        """Get text for upper subtitle from product data or custom URL."""
        # Check for custom URL first
        custom_url: str | None = self.profile_subtitle_settings.get(
            "two_part_subtitles_upper_custom_url"
        )
        if custom_url:
            logger.info("Using custom URL for upper subtitle: %s", custom_url)
            return custom_url

        # Get from product data
        source_field = self.config.upper_config.get(
            "source_field", "shortened_affiliate_link"
        )
        product_data_dict = self.ctx.product.__dict__
        upper_text: str = str(product_data_dict.get(source_field, "") or "")

        if not upper_text:
            # Fallback to other URL fields
            for fallback_field in [
                "shortened_affiliate_link",
                "affiliate_link",
                "url",
            ]:
                upper_text = str(product_data_dict.get(fallback_field, "") or "")
                if upper_text:
                    logger.info(
                        "Using fallback field '%s' for upper subtitle",
                        fallback_field,
                    )
                    break

        if not upper_text:
            logger.warning("No data found for upper subtitle field '%s'", source_field)
            return None

        # Apply URL prefix replacement if configured
        prefix_replace = self.merged_profile_settings.get(
            "two_part_subtitles_upper_prefix_replace"
        )
        if prefix_replace:
            if upper_text.startswith("https://"):
                upper_text = prefix_replace + upper_text[8:]
            elif upper_text.startswith("http://"):
                upper_text = prefix_replace + upper_text[7:]

        return upper_text

    async def generate_lower_subtitle(
        self,
        voiceover_path: Path,
        visual_bounds: "VisualBounds",
        product_id: str,
    ) -> Path | None:
        """Generate lower (voiceover-synced) subtitle.

        Args:
        ----
            voiceover_path: Path to voiceover audio file.
            visual_bounds: Visual bounds for positioning.
            product_id: Product identifier for deterministic effects.

        Returns:
        -------
            Path to generated subtitle file, or None if disabled/failed.

        """
        if not self.config.lower_enabled:
            return None

        from src.video.subtitle_utils import create_unified_subtitles

        lower_subtitle_settings = self.profile_subtitle_settings.copy()
        lower_subtitle_settings["anchor"] = self.config.lower_config.get(
            "anchor", "below_content"
        )
        lower_subtitle_settings["margin"] = self.config.lower_config.get("margin", 0.05)

        if self.config.lower_config.get("custom_style"):
            lower_subtitle_settings.update(self.config.lower_config["custom_style"])

        lower_path = await create_unified_subtitles(
            voiceover_path,
            self.ctx.run_paths["subtitle_file"],
            lower_subtitle_settings,
            self.ctx.config.whisper_settings,
            self.ctx.config.google_cloud_stt_settings,
            self.ctx.secrets,
            self.ctx.script,
            self.ctx.voiceover_duration,
            self.ctx.debug_mode,
            self.ctx.config,
            Path(self.ctx.run_paths["run_root"])
            / self.ctx.config.output_structure.product_subdirs.temp,
            product_id,
            visual_bounds,
        )

        if lower_path and lower_path.exists():
            logger.info("Lower subtitle created: %s", lower_path.name)
            return lower_path
        return None

    def generate_upper_subtitle(
        self,
        visual_bounds: "VisualBounds",
        product_id: str,
        lower_path: Path | None,
    ) -> Path | None:
        """Generate upper (static) subtitle.

        Args:
        ----
            visual_bounds: Visual bounds for positioning.
            product_id: Product identifier for deterministic effects.
            lower_path: Path to lower subtitle for CTA detection.

        Returns:
        -------
            Path to generated subtitle file, or None if disabled/failed.

        """
        if not self.config.upper_enabled:
            return None

        from src.video.subtitle_utils import create_static_upper_subtitle

        upper_text = self._get_upper_text()
        if not upper_text:
            return None

        logger.debug("Upper subtitle enabled: %s", self.config.upper_enabled)

        subtitle_format = self.profile_subtitle_settings.get("subtitle_format", "srt")
        upper_output_path = self.ctx.run_paths["subtitle_file"].with_name(
            f"subtitle_upper.{subtitle_format}"
        )

        upper_path = create_static_upper_subtitle(
            text=upper_text,
            output_path=upper_output_path,
            subtitle_settings=self.profile_subtitle_settings,
            video_config=self.ctx.config,
            format_type=subtitle_format,
            product_id=product_id,
            voiceover_duration=self.ctx.voiceover_duration,
            visual_bounds=visual_bounds,
            lower_subtitle_path=lower_path,
        )

        if upper_path and upper_path.exists():
            logger.info("Upper subtitle created: %s", upper_path.name)
            return upper_path

        logger.warning("Failed to generate upper subtitle, continuing with lower only")
        return None

    async def generate(
        self,
        voiceover_path: Path,
        product_id: str,
    ) -> tuple[Path | None, Path | None]:
        """Generate both upper and lower subtitles.

        Args:
        ----
            voiceover_path: Path to voiceover audio file.
            product_id: Product identifier for deterministic effects.

        Returns:
        -------
            Tuple of (lower_path, upper_path), either may be None.

        """
        visual_bounds = self.calculate_visual_bounds()

        # Generate lower first (needed for upper CTA detection)
        lower_path = await self.generate_lower_subtitle(
            voiceover_path, visual_bounds, product_id
        )

        # Generate upper
        upper_path = self.generate_upper_subtitle(visual_bounds, product_id, lower_path)

        return lower_path, upper_path
