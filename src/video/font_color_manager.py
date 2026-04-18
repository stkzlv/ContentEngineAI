"""Font and color randomization driven by YAML pools.

Pools live on `VideoConfig.font_pool` / `VideoConfig.color_pool`. Selection is
deterministic per product_id via md5 — same input, same output across runs.

The legacy `FontFamily` and `ColorPair` Python enums were removed; pool
membership is data, not code. Old pair names that no longer exist
(`vibrant`, `warm`, `modern`, `dm_serif`) silently fall back to `classic` /
the first font, with a warning.
"""

import hashlib
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.video.config.subtitle_models import ColorPoolEntry, FontPoolEntry

logger = logging.getLogger(__name__)


# Names that used to live in the old enums. Resolved to safe defaults so old
# YAML / profile overrides keep loading instead of crashing.
_DEPRECATED_FONT_FILES = {"DMSerifDisplay-Regular.ttf", "DM_SERIF"}
_DEPRECATED_COLOR_NAMES = {"vibrant", "warm", "modern", "VIBRANT", "WARM", "MODERN"}


@dataclass
class FontInfo:
    """Resolved font paths used by ASS/FFmpeg downstream."""

    name: str
    file_path: str
    ffmpeg_name: str
    system_fallback: str


@dataclass
class ColorInfo:
    """Resolved color pair used by ASS/FFmpeg downstream."""

    name: str
    font_color: str
    outline_color: str
    description: str


def _load_default_pools() -> tuple[list[FontPoolEntry], list[ColorPoolEntry], str]:
    """Pull pools and font directory from the global VideoConfig singleton."""
    from src.video.config import config as _global

    font_dir = _global.subtitle_settings.get("font_directory", "static/fonts")
    return list(_global.font_pool), list(_global.color_pool), font_dir


class FontManager:
    """Picks fonts from a YAML-defined pool and verifies file availability."""

    def __init__(
        self,
        font_pool: list[FontPoolEntry] | None = None,
        static_fonts_dir: str | None = None,
    ):
        if font_pool is None or static_fonts_dir is None:
            default_pool, _, default_dir = _load_default_pools()
            font_pool = font_pool or default_pool
            static_fonts_dir = static_fonts_dir or default_dir

        self.fonts_dir = Path(static_fonts_dir)
        self._font_by_name: dict[str, FontInfo] = {
            entry.name: FontInfo(
                name=entry.name,
                file_path=str(self.fonts_dir / entry.file),
                ffmpeg_name=entry.ffmpeg_name,
                system_fallback=entry.system_fallback,
            )
            for entry in font_pool
        }
        if not self._font_by_name:
            raise ValueError(
                "FontManager requires a non-empty font_pool. "
                "Check config/subtitles.yaml font_pool section."
            )

    def verify_font_availability(self, font_name: str) -> bool:
        """Return True if the font's TTF exists and is readable."""
        info = self._font_by_name.get(font_name)
        if info is None:
            return False

        font_path = Path(info.file_path)
        if not (font_path.exists() and font_path.is_file()):
            logger.warning("Font file not found: %s", font_path)
            return False
        try:
            with font_path.open("rb"):
                pass
        except (OSError, PermissionError) as e:
            logger.warning("Font file not readable: %s - %s", font_path, e)
            return False
        logger.debug("Font verified: %s at %s", info.name, font_path)
        return True

    def get_available_fonts(self) -> list[str]:
        """Names of fonts whose TTFs are present on disk; falls back to all."""
        available = [
            name for name in self._font_by_name if self.verify_font_availability(name)
        ]
        if not available:
            logger.error("No font files available. System fallbacks will be used.")
            available = list(self._font_by_name)
        logger.info("Available fonts: %d/%d", len(available), len(self._font_by_name))
        return available

    def get_font_info(self, font_name: str) -> FontInfo:
        if font_name in _DEPRECATED_FONT_FILES:
            logger.warning(
                "Font '%s' was removed from the pool; falling back to first available.",
                font_name,
            )
            return next(iter(self._font_by_name.values()))
        info = self._font_by_name.get(font_name)
        if info is None:
            logger.warning(
                "Unknown font '%s'; falling back to first available pool entry.",
                font_name,
            )
            return next(iter(self._font_by_name.values()))
        return info

    def select_random_font(self, seed: str) -> str:
        """Pick one font name deterministically from the available pool."""
        available_fonts = self.get_available_fonts()
        hash_object = hashlib.md5(seed.encode(), usedforsecurity=False)
        random_seed = int(hash_object.hexdigest()[:8], 16)
        rng = random.Random(random_seed)  # noqa: S311
        selected = rng.choice(available_fonts)
        logger.info("Selected font for '%s': %s", seed, selected)
        return selected


class ColorManager:
    """Picks coordinated color pairs from a YAML-defined pool."""

    def __init__(self, color_pool: list[ColorPoolEntry] | None = None):
        if color_pool is None:
            _, default_pool, _ = _load_default_pools()
            color_pool = default_pool

        self._color_by_name: dict[str, ColorInfo] = {
            entry.name: ColorInfo(
                name=entry.display_name or entry.name,
                font_color=entry.font_color,
                outline_color=entry.outline_color,
                description=entry.description,
            )
            for entry in color_pool
        }
        if not self._color_by_name:
            raise ValueError(
                "ColorManager requires a non-empty color_pool. "
                "Check config/subtitles.yaml color_pool section."
            )

    def get_color_info(self, color_pair: str) -> ColorInfo:
        if color_pair in _DEPRECATED_COLOR_NAMES:
            fallback = (
                "classic"
                if "classic" in self._color_by_name
                else next(iter(self._color_by_name))
            )
            logger.warning(
                "Color pair '%s' was removed from the pool; falling back to '%s'.",
                color_pair,
                fallback,
            )
            return self._color_by_name[fallback]
        info = self._color_by_name.get(color_pair)
        if info is None:
            fallback = (
                "classic"
                if "classic" in self._color_by_name
                else next(iter(self._color_by_name))
            )
            logger.warning(
                "Unknown color pair '%s'; falling back to '%s'.",
                color_pair,
                fallback,
            )
            return self._color_by_name[fallback]
        return info

    def get_available_color_pairs(self) -> list[str]:
        return list(self._color_by_name)

    def select_random_color_pair(self, seed: str) -> str:
        """Pick one color pair name deterministically from the pool."""
        available_pairs = self.get_available_color_pairs()
        hash_object = hashlib.md5(seed.encode(), usedforsecurity=False)
        random_seed = int(hash_object.hexdigest()[8:16], 16)
        rng = random.Random(random_seed)  # noqa: S311
        selected = rng.choice(available_pairs)
        logger.info("Selected color pair for '%s': %s", seed, selected)
        return selected


class RandomizationEngine:
    """Coordinates font and color randomization for a single video."""

    def __init__(
        self,
        video_config: Any = None,
        static_fonts_dir: str | None = None,
    ):
        """Build managers from `video_config.font_pool` / `color_pool`.

        When `video_config` is None the global `src.video.config.config`
        singleton is used. `static_fonts_dir` overrides the YAML
        `subtitle_settings.font_directory` value.
        """
        font_pool: list[FontPoolEntry] | None = None
        color_pool: list[ColorPoolEntry] | None = None
        fonts_dir = static_fonts_dir

        if video_config is not None:
            raw_font_pool = getattr(video_config, "font_pool", None)
            if isinstance(raw_font_pool, list) and raw_font_pool:
                font_pool = list(raw_font_pool)
            raw_color_pool = getattr(video_config, "color_pool", None)
            if isinstance(raw_color_pool, list) and raw_color_pool:
                color_pool = list(raw_color_pool)
            if fonts_dir is None:
                ss = getattr(video_config, "subtitle_settings", None)
                if isinstance(ss, dict):
                    fonts_dir = ss.get("font_directory")
                elif ss is not None:
                    candidate = getattr(ss, "font_directory", None)
                    if isinstance(candidate, str):
                        fonts_dir = candidate

        self.font_manager = FontManager(
            font_pool=font_pool,
            static_fonts_dir=fonts_dir,
        )
        self.color_manager = ColorManager(color_pool=color_pool)

    def generate_randomized_style(
        self,
        product_id: str,
        enable_font_randomization: bool = False,
        enable_color_randomization: bool = False,
        base_style: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Apply font/color randomization to a base style dict."""
        style = dict(base_style or {})

        if enable_font_randomization:
            selected = self.font_manager.select_random_font(product_id)
            info = self.font_manager.get_font_info(selected)
            if self.font_manager.verify_font_availability(selected):
                style["font_name"] = info.ffmpeg_name
                style["font_path"] = info.file_path
            else:
                style["font_name"] = info.system_fallback
                logger.warning(
                    "Font file unavailable, using fallback: %s", info.system_fallback
                )

        if enable_color_randomization:
            selected_pair = self.color_manager.select_random_color_pair(product_id)
            color_info = self.color_manager.get_color_info(selected_pair)
            style["font_color"] = color_info.font_color
            style["outline_color"] = color_info.outline_color
            logger.info(
                "Applied color pair '%s': %s", color_info.name, color_info.description
            )

        return style

    def get_system_info(self) -> dict[str, Any]:
        """Snapshot of pool sizes and available font names."""
        available_fonts = self.font_manager.get_available_fonts()
        available_colors = self.color_manager.get_available_color_pairs()
        return {
            "fonts": {
                "total": len(self.font_manager._font_by_name),
                "available": len(available_fonts),
                "families": available_fonts,
            },
            "colors": {
                "total": len(self.color_manager._color_by_name),
                "pairs": available_colors,
            },
        }
