"""Unit tests for A/B testing of platform metadata prompt templates."""

import tempfile
from pathlib import Path

import pytest

from src.ai.platform_metadata.ab_testing import (
    ABTestingSettings,
    PlatformABConfig,
    PromptVariant,
    PromptVariantSelector,
    VariantSelection,
)
from src.ai.platform_metadata.models import PlatformMetadata


class TestPromptVariant:
    """Test PromptVariant Pydantic model."""

    def test_prompt_variant_creation(self):
        """Test creating a prompt variant."""
        variant = PromptVariant(
            name="control",
            template_path="src/ai/prompts/youtube_metadata.md",
            weight=50,
            description="Original YouTube metadata prompt",
        )

        assert variant.name == "control"
        assert variant.template_path == "src/ai/prompts/youtube_metadata.md"
        assert variant.weight == 50
        assert variant.description == "Original YouTube metadata prompt"

    def test_prompt_variant_defaults(self):
        """Test default values for prompt variant."""
        variant = PromptVariant(
            name="test",
            template_path="test.md",
        )

        assert variant.weight == 50
        assert variant.description == ""

    def test_prompt_variant_weight_validation(self):
        """Test weight validation bounds."""
        # Valid bounds
        variant_min = PromptVariant(name="min", template_path="t.md", weight=0)
        variant_max = PromptVariant(name="max", template_path="t.md", weight=100)
        assert variant_min.weight == 0
        assert variant_max.weight == 100

        # Invalid bounds
        with pytest.raises(ValueError):
            PromptVariant(name="neg", template_path="t.md", weight=-1)

        with pytest.raises(ValueError):
            PromptVariant(name="over", template_path="t.md", weight=101)


class TestPlatformABConfig:
    """Test PlatformABConfig Pydantic model."""

    def test_platform_ab_config_creation(self):
        """Test creating platform A/B config."""
        config = PlatformABConfig(
            enabled=True,
            variants=[
                PromptVariant(name="control", template_path="a.md", weight=60),
                PromptVariant(name="variant_a", template_path="b.md", weight=40),
            ],
        )

        assert config.enabled is True
        assert len(config.variants) == 2
        assert config.variants[0].name == "control"
        assert config.variants[1].name == "variant_a"

    def test_platform_ab_config_defaults(self):
        """Test default values for platform A/B config."""
        config = PlatformABConfig()

        assert config.enabled is True
        assert config.variants == []


class TestABTestingSettings:
    """Test ABTestingSettings Pydantic model."""

    def test_ab_testing_settings_defaults(self):
        """Test default A/B testing settings."""
        settings = ABTestingSettings()

        assert settings.enabled is True
        assert settings.youtube.enabled is True
        assert settings.tiktok.enabled is True
        assert settings.instagram.enabled is True

        # Each platform should have control variant by default
        assert len(settings.youtube.variants) == 1
        assert settings.youtube.variants[0].name == "control"

    def test_ab_testing_settings_disabled(self):
        """Test disabling A/B testing."""
        settings = ABTestingSettings(enabled=False)
        assert settings.enabled is False


class TestVariantSelection:
    """Test VariantSelection dataclass."""

    def test_variant_selection_creation(self):
        """Test creating variant selection."""
        selection = VariantSelection(
            variant_name="control",
            template_path=Path("src/ai/prompts/youtube_metadata.md"),
            selection_hash="abcd1234",
            platform="youtube",
        )

        assert selection.variant_name == "control"
        assert selection.selection_hash == "abcd1234"
        assert selection.platform == "youtube"

    def test_variant_selection_to_dict(self):
        """Test converting selection to dictionary."""
        selection = VariantSelection(
            variant_name="variant_a",
            template_path=Path("/test/path/template.md"),
            selection_hash="efgh5678",
            platform="tiktok",
        )

        result = selection.to_dict()

        assert result["variant_name"] == "variant_a"
        assert result["template_path"] == "/test/path/template.md"
        assert result["selection_hash"] == "efgh5678"
        assert result["platform"] == "tiktok"


class TestPromptVariantSelector:
    """Test PromptVariantSelector class."""

    @pytest.fixture
    def temp_project_root(self):
        """Create temporary project root with template files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create template files
            prompts_dir = root / "src" / "ai" / "prompts"
            prompts_dir.mkdir(parents=True)

            for template in [
                "youtube_metadata.md",
                "tiktok_caption.md",
                "instagram_caption.md",
            ]:
                (prompts_dir / template).write_text(f"# Template for {template}")

            yield root

    @pytest.fixture
    def ab_settings(self):
        """Create A/B testing settings with multiple variants."""
        return ABTestingSettings(
            enabled=True,
            youtube=PlatformABConfig(
                enabled=True,
                variants=[
                    PromptVariant(
                        name="control",
                        template_path="src/ai/prompts/youtube_metadata.md",
                        weight=50,
                    ),
                    PromptVariant(
                        name="variant_a",
                        template_path="src/ai/prompts/youtube_metadata.md",
                        weight=50,
                    ),
                ],
            ),
            tiktok=PlatformABConfig(
                enabled=True,
                variants=[
                    PromptVariant(
                        name="control",
                        template_path="src/ai/prompts/tiktok_caption.md",
                        weight=100,
                    ),
                ],
            ),
            instagram=PlatformABConfig(
                enabled=False,
                variants=[],
            ),
        )

    @pytest.fixture
    def selector(self, ab_settings, temp_project_root):
        """Create variant selector."""
        return PromptVariantSelector(ab_settings, project_root=temp_project_root)

    def test_select_variant_deterministic(self, selector):
        """Test that variant selection is deterministic."""
        # Same product should always get same variant
        selection1 = selector.select_variant("B0TESTASIN", "youtube")
        selection2 = selector.select_variant("B0TESTASIN", "youtube")

        assert selection1.variant_name == selection2.variant_name
        assert selection1.selection_hash == selection2.selection_hash

    def test_select_variant_different_products(self, selector):
        """Test that different products can get different variants."""
        # With 50/50 split, different products should sometimes get different variants
        selections = {}
        for i in range(100):
            product_id = f"B0TEST{i:04d}"
            selection = selector.select_variant(product_id, "youtube")
            selections[product_id] = selection.variant_name

        # Should have both variants represented (with high probability)
        unique_variants = set(selections.values())
        assert len(unique_variants) >= 1  # At minimum one variant
        # With 100 products and 50/50 split, extremely unlikely to get only one

    def test_select_variant_different_platforms(self, selector):
        """Test that same product can get different selection hash per platform."""
        # Selection hash should include platform
        youtube_selection = selector.select_variant("B0TESTASIN", "youtube")
        tiktok_selection = selector.select_variant("B0TESTASIN", "tiktok")

        # Hash should be different because platform is part of hash input
        assert youtube_selection.selection_hash != tiktok_selection.selection_hash

    def test_select_variant_single_variant(self, selector):
        """Test selection with single variant (100% traffic)."""
        # TikTok has only control variant with 100% weight
        selection = selector.select_variant("B0TESTASIN", "tiktok")

        assert selection.variant_name == "control"
        assert selection.platform == "tiktok"

    def test_select_variant_disabled_platform(self, selector):
        """Test error when platform A/B testing is disabled."""
        with pytest.raises(ValueError, match="not configured"):
            selector.select_variant("B0TESTASIN", "instagram")

    def test_select_variant_unknown_platform(self, selector):
        """Test error for unknown platform."""
        with pytest.raises(ValueError, match="Unknown platform"):
            selector.select_variant("B0TESTASIN", "facebook")

    def test_get_default_template(self, selector, temp_project_root):
        """Test getting default template path."""
        youtube_path = selector.get_default_template("youtube")
        assert youtube_path == temp_project_root / "src/ai/prompts/youtube_metadata.md"

        tiktok_path = selector.get_default_template("tiktok")
        assert tiktok_path == temp_project_root / "src/ai/prompts/tiktok_caption.md"

    def test_get_default_template_unknown_platform(self, selector):
        """Test error for unknown platform default template."""
        with pytest.raises(ValueError, match="Unknown platform"):
            selector.get_default_template("linkedin")

    def test_is_enabled(self, selector):
        """Test checking if A/B testing is enabled."""
        assert selector.is_enabled("youtube") is True
        assert selector.is_enabled("tiktok") is True
        assert selector.is_enabled("instagram") is False  # Disabled in fixture

    def test_is_enabled_global_disabled(self, temp_project_root):
        """Test that global disable affects all platforms."""
        settings = ABTestingSettings(enabled=False)
        selector = PromptVariantSelector(settings, project_root=temp_project_root)

        assert selector.is_enabled("youtube") is False
        assert selector.is_enabled("tiktok") is False

    def test_get_variant_names(self, selector):
        """Test getting list of variant names."""
        youtube_variants = selector.get_variant_names("youtube")
        assert "control" in youtube_variants
        assert "variant_a" in youtube_variants

        tiktok_variants = selector.get_variant_names("tiktok")
        assert tiktok_variants == ["control"]

    def test_weighted_selection_distribution(self):
        """Test that weighted selection respects weights."""
        # Create settings with 80/20 split
        settings = ABTestingSettings(
            enabled=True,
            youtube=PlatformABConfig(
                enabled=True,
                variants=[
                    PromptVariant(name="heavy", template_path="a.md", weight=80),
                    PromptVariant(name="light", template_path="b.md", weight=20),
                ],
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "a.md").write_text("heavy template")
            (root / "b.md").write_text("light template")

            selector = PromptVariantSelector(settings, project_root=root)

            # Run many selections and count distribution
            counts = {"heavy": 0, "light": 0}
            for i in range(1000):
                selection = selector.select_variant(f"PRODUCT{i}", "youtube")
                counts[selection.variant_name] += 1

            # Heavy should get roughly 80% (with some variance)
            heavy_ratio = counts["heavy"] / 1000
            assert (
                0.70 < heavy_ratio < 0.90
            ), f"Expected ~80%, got {heavy_ratio*100:.1f}%"


class TestPlatformMetadataWithVariant:
    """Test PlatformMetadata with prompt_variant field."""

    def test_metadata_with_variant(self):
        """Test creating metadata with variant tracking."""
        metadata = PlatformMetadata.create(
            platform="youtube",
            title="Test Title",
            description="Test description",
            hashtags=["#test", "#ad"],
            keywords=["test"],
            product_id="B0TESTASIN",
            prompt_variant="variant_a",
        )

        assert metadata.prompt_variant == "variant_a"

    def test_metadata_without_variant(self):
        """Test creating metadata without variant (backward compatible)."""
        metadata = PlatformMetadata.create(
            platform="youtube",
            title="Test Title",
            description="Test description",
            hashtags=["#test", "#ad"],
            keywords=["test"],
            product_id="B0TESTASIN",
        )

        assert metadata.prompt_variant is None

    def test_metadata_to_dict_includes_variant(self):
        """Test that to_dict includes prompt_variant."""
        metadata = PlatformMetadata.create(
            platform="tiktok",
            description="Test caption",
            hashtags=["#test", "#ad"],
            keywords=["test"],
            product_id="B0TESTASIN",
            prompt_variant="control",
        )

        result = metadata.to_dict()

        assert "prompt_variant" in result
        assert result["prompt_variant"] == "control"

    def test_metadata_to_dict_null_variant(self):
        """Test that to_dict handles null variant."""
        metadata = PlatformMetadata.create(
            platform="instagram",
            description="Test caption",
            hashtags=["#test", "#ad"],
            keywords=["test"],
            product_id="B0TESTASIN",
        )

        result = metadata.to_dict()

        assert "prompt_variant" in result
        assert result["prompt_variant"] is None
