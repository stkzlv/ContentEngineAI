"""Tests for font and color randomization system."""

import pytest

from src.video.font_color_manager import (
    ColorManager,
    ColorPair,
    FontFamily,
    FontManager,
    RandomizationEngine,
)


@pytest.mark.unit
class TestFontManager:
    """Test font management and randomization."""

    @pytest.fixture
    def font_manager(self):
        """Create font manager instance."""
        return FontManager()

    def test_get_available_fonts(self, font_manager):
        """Test retrieval of available fonts."""
        fonts = font_manager.get_available_fonts()
        assert isinstance(fonts, list)
        assert len(fonts) > 0
        assert all(isinstance(f, FontFamily) for f in fonts)

    def test_font_families_exist(self, font_manager):
        """Test that expected font families are available."""
        fonts = font_manager.get_available_fonts()
        font_names = [f.value for f in fonts]

        # Check for key fonts from the collection
        expected_fonts = ["Montserrat", "Poppins", "Gabarito"]
        for font in expected_fonts:
            assert any(font in name for name in font_names)

    def test_select_random_font_deterministic(self, font_manager):
        """Test that font selection is deterministic with same seed."""
        seed1 = "test_product_123"
        seed2 = "test_product_123"
        seed3 = "different_product"

        font1 = font_manager.select_random_font(seed1)
        font2 = font_manager.select_random_font(seed2)
        font3 = font_manager.select_random_font(seed3)

        # Same seed should give same font
        assert font1 == font2

        # Different seed might give different font (not guaranteed, but likely)
        # We just verify it returns a valid font
        assert isinstance(font3, FontFamily)

    def test_get_font_info(self, font_manager):
        """Test font info retrieval."""
        font = FontFamily.MONTSERRAT
        info = font_manager.get_font_info(font)

        assert info is not None
        assert isinstance(info.file_path, str)
        assert "Montserrat-Bold" in info.file_path
        assert info.file_path.endswith(".ttf")


@pytest.mark.unit
class TestColorManager:
    """Test color management and randomization."""

    @pytest.fixture
    def color_manager(self):
        """Create color manager instance."""
        return ColorManager()

    def test_get_available_color_pairs(self, color_manager):
        """Test retrieval of available color pairs."""
        pairs = color_manager.get_available_color_pairs()
        assert isinstance(pairs, list)
        assert len(pairs) > 0
        assert all(isinstance(p, ColorPair) for p in pairs)

    def test_color_pairs_exist(self, color_manager):
        """Test that expected color pairs are available."""
        pairs = color_manager.get_available_color_pairs()

        # Should have standard pairs
        assert ColorPair.CLASSIC in pairs
        assert ColorPair.HIGH_CONTRAST in pairs
        assert ColorPair.WARM in pairs

    def test_select_random_color_deterministic(self, color_manager):
        """Test that color selection is deterministic with same seed."""
        seed1 = "test_product_123"
        seed2 = "test_product_123"

        color1 = color_manager.select_random_color_pair(seed1)
        color2 = color_manager.select_random_color_pair(seed2)

        # Same seed should give same color pair
        assert color1 == color2

    def test_get_color_info(self, color_manager):
        """Test color info retrieval."""
        info = color_manager.get_color_info(ColorPair.WARM)

        assert info is not None
        assert info.name == "Warm"
        assert info.font_color == "&H000080FF"  # Orange
        assert info.outline_color == "&H00008000"  # Dark green

    def test_all_color_pairs_have_info(self, color_manager):
        """Test that all color pairs have valid info."""
        pairs = color_manager.get_available_color_pairs()

        for pair in pairs:
            info = color_manager.get_color_info(pair)
            assert info is not None
            assert info.font_color.startswith("&H")
            assert info.outline_color.startswith("&H")
            assert len(info.description) > 0


@pytest.mark.unit
class TestRandomizationEngine:
    """Test randomization engine integration."""

    @pytest.fixture
    def engine(self):
        """Create randomization engine instance."""
        return RandomizationEngine()

    def test_font_randomization_enabled(self, engine):
        """Test font randomization when enabled."""
        base_style = {
            "font_name": "Arial",
            "font_color": "&H00FFFFFF",
            "outline_color": "&H00000000",
        }

        result = engine.generate_randomized_style(
            product_id="test_123",
            enable_font_randomization=True,
            enable_color_randomization=False,
            base_style=base_style,
        )

        # Font should be randomized
        assert "font_name" in result
        assert result["font_name"] != "Arial"  # Should be different from base
        assert "font_path" in result

        # Colors should remain from base
        assert result["font_color"] == "&H00FFFFFF"
        assert result["outline_color"] == "&H00000000"

    def test_color_randomization_enabled(self, engine):
        """Test color randomization when enabled."""
        base_style = {
            "font_name": "Arial",
            "font_color": "&H00FFFFFF",
            "outline_color": "&H00000000",
        }

        result = engine.generate_randomized_style(
            product_id="test_123",
            enable_font_randomization=False,
            enable_color_randomization=True,
            base_style=base_style,
        )

        # Font should remain from base
        assert result["font_name"] == "Arial"

        # Colors should be randomized
        assert "font_color" in result
        assert "outline_color" in result
        # At least one should be different (both might change)
        colors_changed = (
            result["font_color"] != "&H00FFFFFF"
            or result["outline_color"] != "&H00000000"
        )
        assert colors_changed

    def test_both_randomizations_enabled(self, engine):
        """Test both font and color randomization enabled."""
        base_style = {
            "font_name": "Arial",
            "font_color": "&H00FFFFFF",
            "outline_color": "&H00000000",
        }

        result = engine.generate_randomized_style(
            product_id="test_123",
            enable_font_randomization=True,
            enable_color_randomization=True,
            base_style=base_style,
        )

        # Both should be randomized
        assert result["font_name"] != "Arial"
        assert "font_path" in result
        assert "font_color" in result
        assert "outline_color" in result

    def test_no_randomization(self, engine):
        """Test that base style is preserved when randomization disabled."""
        base_style = {
            "font_name": "Arial",
            "font_color": "&H00FFFFFF",
            "outline_color": "&H00000000",
            "bold": True,
        }

        result = engine.generate_randomized_style(
            product_id="test_123",
            enable_font_randomization=False,
            enable_color_randomization=False,
            base_style=base_style,
        )

        # Should match base style
        assert result["font_name"] == "Arial"
        assert result["font_color"] == "&H00FFFFFF"
        assert result["outline_color"] == "&H00000000"
        assert result["bold"] is True

    def test_deterministic_randomization(self, engine):
        """Test that same product_id gives same randomization."""
        base_style = {
            "font_name": "Arial",
            "font_color": "&H00FFFFFF",
            "outline_color": "&H00000000",
        }

        result1 = engine.generate_randomized_style(
            product_id="SAME_ID",
            enable_font_randomization=True,
            enable_color_randomization=True,
            base_style=base_style,
        )

        result2 = engine.generate_randomized_style(
            product_id="SAME_ID",
            enable_font_randomization=True,
            enable_color_randomization=True,
            base_style=base_style,
        )

        # Same product should get same randomization
        assert result1["font_name"] == result2["font_name"]
        assert result1["font_color"] == result2["font_color"]
        assert result1["outline_color"] == result2["outline_color"]

    def test_preserves_other_style_properties(self, engine):
        """Test that randomization preserves non-randomized properties."""
        base_style = {
            "font_name": "Arial",
            "font_color": "&H00FFFFFF",
            "outline_color": "&H00000000",
            "bold": True,
            "outline_thickness": 2,
            "shadow": True,
            "effects": ["karaoke"],
        }

        result = engine.generate_randomized_style(
            product_id="test_123",
            enable_font_randomization=True,
            enable_color_randomization=True,
            base_style=base_style,
        )

        # Non-color, non-font properties should be preserved
        assert result["bold"] is True
        assert result["outline_thickness"] == 2
        assert result["shadow"] is True
        assert result["effects"] == ["karaoke"]
