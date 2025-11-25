"""Font and color management system for ContentEngineAI.

This module implements the randomization requirements from REQUIREMENTS.md:
- Random font selection from 5 curated fonts per video
- Random color selection from 5 coordinated text/outline pairs per video
- Font availability verification with fallback options
- Compatibility with SRT, ASS, and FFmpeg formats
"""

import hashlib
import logging
import os
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FontFamily(str, Enum):
    """Available font families with proper names for FFmpeg."""

    MONTSERRAT = "Montserrat-Bold.ttf"
    POPPINS = "Poppins-Bold.ttf"
    GABARITO = "Gabarito-Bold.ttf"
    RUBIK = "Rubik-Bold.ttf"
    DM_SERIF = "DMSerifDisplay-Regular.ttf"


class ColorPair(str, Enum):
    """Coordinated text and outline color combinations."""

    CLASSIC = "classic"  # White text + Black outline
    HIGH_CONTRAST = "high_contrast"  # Yellow text + Dark Blue outline
    VIBRANT = "vibrant"  # Light Blue text + Dark Red outline
    WARM = "warm"  # Orange text + Dark Green outline
    MODERN = "modern"  # Pink text + Purple outline


@dataclass
class FontInfo:
    """Font information with paths and fallbacks."""

    name: str
    file_path: str
    ffmpeg_name: str
    system_fallback: str


@dataclass
class ColorInfo:
    """Color pair information for text and outline."""

    name: str
    font_color: str  # ASS hex format (&H00RRGGBB)
    outline_color: str  # ASS hex format (&H00RRGGBB)
    description: str


class FontManager:
    """Manages font selection, verification, and fallbacks."""

    def __init__(self, static_fonts_dir: str = "static/fonts"):
        """Initialize font manager with font directory.

        Args:
        ----
            static_fonts_dir: Path to static fonts directory

        """
        self.fonts_dir = Path(static_fonts_dir)
        self._font_cache: dict[str, FontInfo] = {}
        self._initialize_fonts()

    def _initialize_fonts(self) -> None:
        """Initialize the curated font collection."""
        self._curated_fonts = {
            FontFamily.MONTSERRAT: FontInfo(
                name="Montserrat",
                file_path=str(self.fonts_dir / FontFamily.MONTSERRAT),
                ffmpeg_name="Montserrat-Bold",
                system_fallback="Arial",
            ),
            FontFamily.POPPINS: FontInfo(
                name="Poppins",
                file_path=str(self.fonts_dir / FontFamily.POPPINS),
                ffmpeg_name="Poppins-Bold",
                system_fallback="Arial",
            ),
            FontFamily.GABARITO: FontInfo(
                name="Gabarito",
                file_path=str(self.fonts_dir / FontFamily.GABARITO),
                ffmpeg_name="Gabarito-Bold",
                system_fallback="Arial",
            ),
            FontFamily.RUBIK: FontInfo(
                name="Rubik",
                file_path=str(self.fonts_dir / FontFamily.RUBIK),
                ffmpeg_name="Rubik-Bold",
                system_fallback="Arial",
            ),
            FontFamily.DM_SERIF: FontInfo(
                name="DM Serif Display",
                file_path=str(self.fonts_dir / FontFamily.DM_SERIF),
                ffmpeg_name="DMSerifDisplay-Regular",
                system_fallback="Times New Roman",
            ),
        }

    def verify_font_availability(self, font_family: FontFamily) -> bool:
        """Verify if a font file exists and is accessible.

        Args:
        ----
            font_family: Font family to verify

        Returns:
        -------
            True if font file exists and is readable

        """
        if font_family not in self._curated_fonts:
            return False

        font_info = self._curated_fonts[font_family]
        font_path = Path(font_info.file_path)

        if font_path.exists() and font_path.is_file():
            try:
                # Test readability
                with font_path.open("rb"):
                    pass
                logger.debug(f"Font verified: {font_info.name} at {font_path}")
                return True
            except (OSError, PermissionError) as e:
                logger.warning(f"Font file not readable: {font_path} - {e}")
                return False
        else:
            logger.warning(f"Font file not found: {font_path}")
            return False

    def get_available_fonts(self) -> list[FontFamily]:
        """Get list of available and verified fonts.

        Returns
        -------
            List of verified font families

        """
        available = []
        for font_family in FontFamily:
            if self.verify_font_availability(font_family):
                available.append(font_family)

        if not available:
            logger.error("No fonts available! Check static/fonts directory")
            # Return all fonts as fallback - system fonts will be used
            available = list(FontFamily)

        logger.info(f"Available fonts: {len(available)}/{len(FontFamily)}")
        return available

    def get_font_info(self, font_family: FontFamily) -> FontInfo:
        """Get font information for a specific font family.

        Args:
        ----
            font_family: Font family to get info for

        Returns:
        -------
            FontInfo with paths and fallback information

        """
        return self._curated_fonts[font_family]

    def select_random_font(self, seed: str) -> FontFamily:
        """Select a random font using deterministic seeding.

        Args:
        ----
            seed: Seed for consistent randomization (e.g., product_id)

        Returns:
        -------
            Selected font family

        """
        available_fonts = self.get_available_fonts()

        # Create deterministic random generator
        hash_object = hashlib.md5(seed.encode(), usedforsecurity=False)
        random_seed = int(hash_object.hexdigest()[:8], 16)
        rng = random.Random(random_seed)  # noqa: S311

        selected = rng.choice(available_fonts)
        logger.info(f"Selected font for '{seed}': {selected.value}")
        return selected


class ColorManager:
    """Manages color pair selection for subtitle styling."""

    def __init__(self):
        """Initialize color manager with coordinated color pairs."""
        self._initialize_color_pairs()

    def _initialize_color_pairs(self) -> None:
        """Initialize the coordinated color pair collection."""
        self._color_pairs = {
            ColorPair.CLASSIC: ColorInfo(
                name="Classic",
                font_color="&H00FFFFFF",  # White
                outline_color="&H00000000",  # Black
                description="White text with black outline - maximum readability",
            ),
            ColorPair.HIGH_CONTRAST: ColorInfo(
                name="High Contrast",
                font_color="&H0000FFFF",  # Yellow
                outline_color="&H00800000",  # Dark Blue
                description="Yellow text with dark blue outline - high visibility",
            ),
            ColorPair.VIBRANT: ColorInfo(
                name="Vibrant",
                font_color="&H00FFAA00",  # Light Blue
                outline_color="&H00000080",  # Dark Red
                description="Light blue text with dark red outline - vibrant contrast",
            ),
            ColorPair.WARM: ColorInfo(
                name="Warm",
                font_color="&H000080FF",  # Orange
                outline_color="&H00008000",  # Dark Green
                description="Orange text with dark green outline - warm palette",
            ),
            ColorPair.MODERN: ColorInfo(
                name="Modern",
                font_color="&H00FF80FF",  # Pink
                outline_color="&H00800080",  # Purple
                description="Pink text with purple outline - modern aesthetic",
            ),
        }

    def get_color_info(self, color_pair: ColorPair) -> ColorInfo:
        """Get color information for a specific color pair.

        Args:
        ----
            color_pair: Color pair to get info for

        Returns:
        -------
            ColorInfo with ASS color codes and description

        """
        return self._color_pairs[color_pair]

    def get_available_color_pairs(self) -> list[ColorPair]:
        """Get list of all available color pairs.

        Returns
        -------
            List of all color pairs

        """
        return list(ColorPair)

    def select_random_color_pair(self, seed: str) -> ColorPair:
        """Select a random color pair using deterministic seeding.

        Args:
        ----
            seed: Seed for consistent randomization (e.g., product_id)

        Returns:
        -------
            Selected color pair

        """
        available_pairs = self.get_available_color_pairs()

        # Create deterministic random generator
        hash_object = hashlib.md5(seed.encode(), usedforsecurity=False)
        # Different slice for colors
        random_seed = int(hash_object.hexdigest()[8:16], 16)
        rng = random.Random(random_seed)  # noqa: S311

        selected = rng.choice(available_pairs)
        logger.info(f"Selected color pair for '{seed}': {selected.value}")
        return selected


class RandomizationEngine:
    """Coordinates font and color randomization for video production."""

    def __init__(self, static_fonts_dir: str = "static/fonts"):
        """Initialize randomization engine.

        Args:
        ----
            static_fonts_dir: Path to static fonts directory

        """
        self.font_manager = FontManager(static_fonts_dir)
        self.color_manager = ColorManager()

    def generate_randomized_style(
        self,
        product_id: str,
        enable_font_randomization: bool = False,
        enable_color_randomization: bool = False,
        base_style: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate a randomized style configuration.

        Args:
        ----
            product_id: Product identifier for consistent seeding
            enable_font_randomization: Whether to randomize font selection
            enable_color_randomization: Whether to randomize color pairs
            base_style: Base style configuration to extend

        Returns:
        -------
            Style configuration with randomized elements

        """
        if base_style is None:
            base_style = {}

        style = base_style.copy()

        # Randomize font if enabled
        if enable_font_randomization:
            selected_font = self.font_manager.select_random_font(product_id)
            font_info = self.font_manager.get_font_info(selected_font)

            if self.font_manager.verify_font_availability(selected_font):
                style["font_name"] = font_info.ffmpeg_name
                style["font_path"] = font_info.file_path
            else:
                style["font_name"] = font_info.system_fallback
                logger.warning(
                    f"Font file not available, using fallback: "
                    f"{font_info.system_fallback}"
                )

        # Randomize colors if enabled
        if enable_color_randomization:
            selected_pair = self.color_manager.select_random_color_pair(product_id)
            color_info = self.color_manager.get_color_info(selected_pair)

            style["font_color"] = color_info.font_color
            style["outline_color"] = color_info.outline_color

            logger.info(
                f"Applied color pair '{color_info.name}': {color_info.description}"
            )

        return style

    def get_system_info(self) -> dict[str, Any]:
        """Get system information about available fonts and colors.

        Returns
        -------
            Dictionary with font and color availability information

        """
        available_fonts = self.font_manager.get_available_fonts()
        available_colors = self.color_manager.get_available_color_pairs()

        return {
            "fonts": {
                "total": len(FontFamily),
                "available": len(available_fonts),
                "families": [f.value for f in available_fonts],
            },
            "colors": {
                "total": len(ColorPair),
                "pairs": [c.value for c in available_colors],
            },
        }
