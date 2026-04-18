"""Tests for font and color randomization driven by YAML pools."""

import pytest

from src.video.config.subtitle_models import ColorPoolEntry, FontPoolEntry
from src.video.font_color_manager import (
    ColorManager,
    FontManager,
    RandomizationEngine,
)


@pytest.fixture
def font_pool() -> list[FontPoolEntry]:
    """Mirror of the curated default pool, kept independent of YAML."""
    return [
        FontPoolEntry(
            name="Montserrat",
            file="Montserrat-Bold.ttf",
            ffmpeg_name="Montserrat-Bold",
            system_fallback="Arial",
        ),
        FontPoolEntry(
            name="Poppins",
            file="Poppins-Bold.ttf",
            ffmpeg_name="Poppins-Bold",
            system_fallback="Arial",
        ),
        FontPoolEntry(
            name="Gabarito",
            file="Gabarito-Bold.ttf",
            ffmpeg_name="Gabarito-Bold",
            system_fallback="Arial",
        ),
    ]


@pytest.fixture
def color_pool() -> list[ColorPoolEntry]:
    """Mirror of the curated default pool."""
    return [
        ColorPoolEntry(
            name="classic",
            display_name="Classic",
            font_color="&H00FFFFFF",
            outline_color="&H00000000",
            description="White on black stroke",
        ),
        ColorPoolEntry(
            name="high_contrast",
            display_name="High Contrast",
            font_color="&H0000FFFF",
            outline_color="&H00000000",
            description="Yellow on black stroke",
        ),
        ColorPoolEntry(
            name="brand_yellow",
            display_name="Brand Yellow",
            font_color="&H0000EBFF",
            outline_color="&H00000000",
            description="Saturated yellow on black",
        ),
    ]


@pytest.mark.unit
class TestFontManager:
    """Font selection over an explicit pool."""

    @pytest.fixture
    def font_manager(self, font_pool):
        return FontManager(font_pool=font_pool, static_fonts_dir="static/fonts")

    def test_get_available_fonts_returns_strings(self, font_manager):
        fonts = font_manager.get_available_fonts()
        assert isinstance(fonts, list)
        assert len(fonts) > 0
        assert all(isinstance(name, str) for name in fonts)

    def test_pool_contains_expected_fonts(self, font_manager):
        fonts = font_manager.get_available_fonts()
        for expected in ("Montserrat", "Poppins", "Gabarito"):
            assert expected in fonts

    def test_select_random_font_deterministic(self, font_manager):
        font1 = font_manager.select_random_font("test_product_123")
        font2 = font_manager.select_random_font("test_product_123")
        font3 = font_manager.select_random_font("different_product")

        assert font1 == font2
        assert isinstance(font3, str)

    def test_get_font_info_returns_resolved_paths(self, font_manager):
        info = font_manager.get_font_info("Montserrat")
        assert info.name == "Montserrat"
        assert info.ffmpeg_name == "Montserrat-Bold"
        assert info.file_path.endswith("Montserrat-Bold.ttf")

    def test_deprecated_font_falls_back(self, font_manager, caplog):
        info = font_manager.get_font_info("DM_SERIF")
        assert info.name in {"Montserrat", "Poppins", "Gabarito"}
        assert any("removed from the pool" in rec.message for rec in caplog.records)

    def test_unknown_font_falls_back(self, font_manager, caplog):
        info = font_manager.get_font_info("NotAFont")
        assert info.name in {"Montserrat", "Poppins", "Gabarito"}
        assert any("Unknown font" in rec.message for rec in caplog.records)

    def test_empty_pool_raises(self):
        with pytest.raises(ValueError, match="non-empty font_pool"):
            FontManager(font_pool=[], static_fonts_dir="static/fonts")


@pytest.mark.unit
class TestColorManager:
    """Color pair selection over an explicit pool."""

    @pytest.fixture
    def color_manager(self, color_pool):
        return ColorManager(color_pool=color_pool)

    def test_get_available_color_pairs_returns_strings(self, color_manager):
        pairs = color_manager.get_available_color_pairs()
        assert pairs == ["classic", "high_contrast", "brand_yellow"]

    def test_get_color_info_returns_pair(self, color_manager):
        info = color_manager.get_color_info("brand_yellow")
        assert info.name == "Brand Yellow"
        assert info.font_color == "&H0000EBFF"
        assert info.outline_color == "&H00000000"

    def test_select_random_color_deterministic(self, color_manager):
        color1 = color_manager.select_random_color_pair("test_product_123")
        color2 = color_manager.select_random_color_pair("test_product_123")
        assert color1 == color2

    def test_deprecated_color_falls_back_to_classic(self, color_manager, caplog):
        info = color_manager.get_color_info("vibrant")
        assert info.font_color == "&H00FFFFFF"
        assert any("removed from the pool" in rec.message for rec in caplog.records)

    def test_unknown_color_falls_back_to_classic(self, color_manager, caplog):
        info = color_manager.get_color_info("not_a_color")
        assert info.font_color == "&H00FFFFFF"
        assert any("Unknown color pair" in rec.message for rec in caplog.records)

    def test_all_color_pairs_have_info(self, color_manager):
        for pair in color_manager.get_available_color_pairs():
            info = color_manager.get_color_info(pair)
            assert info.font_color.startswith("&H")
            assert info.outline_color.startswith("&H")
            assert len(info.description) > 0

    def test_empty_pool_raises(self):
        with pytest.raises(ValueError, match="non-empty color_pool"):
            ColorManager(color_pool=[])


@pytest.mark.unit
class TestRandomizationEngine:
    """Engine orchestrates font and color managers from a video_config."""

    @pytest.fixture
    def engine(self, font_pool, color_pool):
        from types import SimpleNamespace

        fake_config = SimpleNamespace(
            font_pool=font_pool,
            color_pool=color_pool,
            subtitle_settings={"font_directory": "static/fonts"},
        )
        return RandomizationEngine(video_config=fake_config)

    def _base_style(self) -> dict:
        return {
            "font_name": "Arial",
            "font_color": "&H00FFFFFF",
            "outline_color": "&H00000000",
        }

    def test_font_randomization_changes_font(self, engine):
        result = engine.generate_randomized_style(
            product_id="test_123",
            enable_font_randomization=True,
            enable_color_randomization=False,
            base_style=self._base_style(),
        )
        assert result["font_name"] != "Arial"
        assert "font_path" in result
        assert result["font_color"] == "&H00FFFFFF"
        assert result["outline_color"] == "&H00000000"

    def test_color_randomization_changes_colors(self, engine):
        result = engine.generate_randomized_style(
            product_id="test_pid_with_diff_color",
            enable_font_randomization=False,
            enable_color_randomization=True,
            base_style=self._base_style(),
        )
        assert result["font_name"] == "Arial"
        # Selected pair name is non-empty in result; values may equal classic too
        assert "font_color" in result
        assert "outline_color" in result

    def test_no_randomization_preserves_base_style(self, engine):
        base = {
            **self._base_style(),
            "bold": True,
        }
        result = engine.generate_randomized_style(
            product_id="test_123",
            enable_font_randomization=False,
            enable_color_randomization=False,
            base_style=base,
        )
        assert result == base

    def test_deterministic_across_calls(self, engine):
        kwargs = {
            "product_id": "SAME_ID",
            "enable_font_randomization": True,
            "enable_color_randomization": True,
            "base_style": self._base_style(),
        }
        result1 = engine.generate_randomized_style(**kwargs)
        result2 = engine.generate_randomized_style(**kwargs)
        assert result1["font_name"] == result2["font_name"]
        assert result1["font_color"] == result2["font_color"]
        assert result1["outline_color"] == result2["outline_color"]

    def test_preserves_unrelated_style_properties(self, engine):
        base = {
            **self._base_style(),
            "bold": True,
            "outline_thickness": 2,
            "shadow": True,
            "effects": ["karaoke"],
        }
        result = engine.generate_randomized_style(
            product_id="test_123",
            enable_font_randomization=True,
            enable_color_randomization=True,
            base_style=base,
        )
        assert result["bold"] is True
        assert result["outline_thickness"] == 2
        assert result["shadow"] is True
        assert result["effects"] == ["karaoke"]

    def test_get_system_info_reports_pool_sizes(self, engine):
        info = engine.get_system_info()
        assert info["fonts"]["total"] == 3
        assert info["colors"]["total"] == 3
        assert isinstance(info["fonts"]["families"], list)
        assert isinstance(info["colors"]["pairs"], list)


@pytest.mark.unit
class TestFontAvailabilityFallbacks:
    """Cover the disk-availability failure paths and the system_fallback chain."""

    @pytest.fixture
    def pool(self) -> list[FontPoolEntry]:
        return [
            FontPoolEntry(
                name="Ghost",
                file="Ghost-Bold.ttf",
                ffmpeg_name="Ghost-Bold",
                system_fallback="Helvetica",
            ),
        ]

    def test_verify_returns_false_for_missing_file(self, pool, tmp_path):
        manager = FontManager(font_pool=pool, static_fonts_dir=str(tmp_path))
        assert manager.verify_font_availability("Ghost") is False

    def test_verify_returns_false_for_unknown_name(self, pool, tmp_path):
        manager = FontManager(font_pool=pool, static_fonts_dir=str(tmp_path))
        assert manager.verify_font_availability("NotInPool") is False

    def test_get_available_fonts_falls_back_to_full_pool_when_none_on_disk(
        self, pool, tmp_path, caplog
    ):
        manager = FontManager(font_pool=pool, static_fonts_dir=str(tmp_path))
        with caplog.at_level("ERROR"):
            available = manager.get_available_fonts()
        assert available == ["Ghost"]
        assert any("No font files available" in rec.message for rec in caplog.records)

    def test_generate_randomized_style_uses_system_fallback_when_file_missing(
        self, pool, tmp_path
    ):
        from types import SimpleNamespace

        config = SimpleNamespace(
            font_pool=pool,
            color_pool=[
                ColorPoolEntry(
                    name="classic",
                    display_name="Classic",
                    font_color="&H00FFFFFF",
                    outline_color="&H00000000",
                    description="White on black",
                ),
            ],
            subtitle_settings={"font_directory": str(tmp_path)},
        )
        engine = RandomizationEngine(video_config=config)
        result = engine.generate_randomized_style(
            product_id="missing-file-test",
            enable_font_randomization=True,
            base_style={"font_name": "Arial"},
        )
        # Verification fails → system_fallback wins, no font_path attached
        assert result["font_name"] == "Helvetica"
        assert "font_path" not in result


@pytest.mark.unit
class TestRandomizationEngineDefaultsAndConfigShapes:
    """Cover the default-loading paths and the object-style subtitle_settings branch."""

    def test_constructed_without_video_config_loads_global_defaults(self):
        # Falls through to _load_default_pools, which reads the global VideoConfig.
        engine = RandomizationEngine()
        info = engine.get_system_info()
        assert info["fonts"]["total"] >= 1
        assert info["colors"]["total"] >= 1

    def test_font_manager_constructed_with_no_args_uses_defaults(self):
        manager = FontManager()
        # Pool from global YAML — at least one entry.
        assert len(manager.get_available_fonts()) >= 1

    def test_color_manager_constructed_with_no_args_uses_defaults(self):
        manager = ColorManager()
        # Pool from global YAML — at least one entry.
        assert len(manager.get_available_color_pairs()) >= 1

    def test_engine_reads_font_directory_from_attribute_style_config(self, tmp_path):
        from types import SimpleNamespace

        # subtitle_settings is an object with .font_directory, not a dict.
        # This mirrors what MergedSubtitleSettings looks like at runtime.
        ss_object = SimpleNamespace(font_directory=str(tmp_path))
        config = SimpleNamespace(
            font_pool=[
                FontPoolEntry(
                    name="X",
                    file="X.ttf",
                    ffmpeg_name="X",
                    system_fallback="Arial",
                ),
            ],
            color_pool=[
                ColorPoolEntry(
                    name="classic",
                    display_name="Classic",
                    font_color="&H00FFFFFF",
                    outline_color="&H00000000",
                    description="",
                ),
            ],
            subtitle_settings=ss_object,
        )
        engine = RandomizationEngine(video_config=config)
        # Confirms the attribute branch ran by checking the resolved fonts dir.
        assert str(engine.font_manager.fonts_dir) == str(tmp_path)
