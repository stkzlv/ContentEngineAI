"""Unit tests for platform metadata prompt templates."""

from pathlib import Path

import pytest


def load_prompt_template(path: Path) -> str:
    """Load prompt template from file (test utility)."""
    # Resolve path relative to project root
    project_root = Path(__file__).parent.parent.parent
    full_path = project_root / path
    return full_path.read_text(encoding="utf-8")


class TestPromptTemplateLoading:
    """Test prompt template file loading."""

    def test_load_youtube_template(self):
        """Test loading YouTube metadata template."""
        template_path = Path("src/ai/prompts/youtube_metadata.md")
        template = load_prompt_template(template_path)

        assert template is not None
        assert len(template) > 0
        assert isinstance(template, str)

    def test_load_tiktok_template(self):
        """Test loading TikTok caption template."""
        template_path = Path("src/ai/prompts/tiktok_caption.md")
        template = load_prompt_template(template_path)

        assert template is not None
        assert len(template) > 0
        assert isinstance(template, str)

    def test_load_instagram_template(self):
        """Test loading Instagram caption template."""
        template_path = Path("src/ai/prompts/instagram_caption.md")
        template = load_prompt_template(template_path)

        assert template is not None
        assert len(template) > 0
        assert isinstance(template, str)


class TestYouTubeTemplateStructure:
    """Test YouTube metadata template structure and content."""

    def setup_method(self):
        """Set up test fixtures."""
        template_path = Path("src/ai/prompts/youtube_metadata.md")
        self.template = load_prompt_template(template_path)

    def test_required_placeholders(self):
        """Test that all required placeholders are present."""
        required_placeholders = [
            "{FULL_PRODUCT_NAME}",
            "{PRODUCT_DESCRIPTION}",
            "{PRODUCT_URL}",
        ]

        for placeholder in required_placeholders:
            assert (
                placeholder in self.template
            ), f"Missing required placeholder: {placeholder}"

    def test_contains_instructions(self):
        """Test that template contains clear instructions."""
        # Check for key instruction sections
        assert "Title Requirements:" in self.template
        assert "Description Requirements:" in self.template
        assert "Hashtags:" in self.template
        assert "Keywords:" in self.template

        # Check for critical YouTube-specific requirements
        assert "50-60 characters" in self.template
        assert "#Shorts" in self.template
        assert "#ad" in self.template

    def test_contains_examples(self):
        """Test that template includes examples."""
        assert "Example" in self.template or "EXAMPLE" in self.template
        assert "TITLE:" in self.template
        assert "DESCRIPTION:" in self.template
        assert "HASHTAGS:" in self.template
        assert "KEYWORDS:" in self.template

    def test_output_format_specified(self):
        """Test that template specifies output format."""
        assert "Output Format:" in self.template
        assert "Return your response" in self.template

    def test_template_formatting(self):
        """Test template can be formatted with sample data."""
        sample_data = {
            "FULL_PRODUCT_NAME": "Test Product - Wireless Headphones",
            "PRODUCT_DESCRIPTION": "High-quality wireless headphones with noise cancellation",
            "PRODUCT_URL": "https://example.com/test-product",
        }

        formatted = self.template.format(**sample_data)

        assert "Test Product - Wireless Headphones" in formatted
        assert "High-quality wireless headphones with noise cancellation" in formatted
        assert "https://example.com/test-product" in formatted
        assert "{FULL_PRODUCT_NAME}" not in formatted
        assert "{PRODUCT_DESCRIPTION}" not in formatted
        assert "{PRODUCT_URL}" not in formatted


class TestTikTokTemplateStructure:
    """Test TikTok caption template structure and content."""

    def setup_method(self):
        """Set up test fixtures."""
        template_path = Path("src/ai/prompts/tiktok_caption.md")
        self.template = load_prompt_template(template_path)

    def test_required_placeholders(self):
        """Test that all required placeholders are present."""
        required_placeholders = [
            "{FULL_PRODUCT_NAME}",
            "{PRODUCT_DESCRIPTION}",
            "{PRODUCT_URL}",
        ]

        for placeholder in required_placeholders:
            assert (
                placeholder in self.template
            ), f"Missing required placeholder: {placeholder}"

    def test_contains_instructions(self):
        """Test that template contains clear instructions."""
        # Check for key instruction sections
        assert "Caption Requirements:" in self.template
        assert "Hashtags:" in self.template

        # Check for TikTok-specific requirements
        assert "100-300 characters" in self.template
        assert "#ad" in self.template

        # Check for TikTok SEO guidance
        assert "search" in self.template.lower()

    def test_generic_hashtag_warnings(self):
        """Test that template warns against generic hashtags."""
        # TikTok template should warn against generic viral tags
        generic_tags = ["#fyp", "#foryoupage", "#viral"]

        for tag in generic_tags:
            assert (
                tag in self.template
            ), f"Template should warn against generic tag: {tag}"

    def test_contains_examples(self):
        """Test that template includes examples."""
        assert "Example" in self.template or "EXAMPLE" in self.template
        assert "CAPTION:" in self.template
        assert "HASHTAGS:" in self.template
        assert "KEYWORDS:" in self.template

    def test_output_format_specified(self):
        """Test that template specifies output format."""
        assert "Output Format:" in self.template
        assert "Return your response" in self.template

    def test_template_formatting(self):
        """Test template can be formatted with sample data."""
        sample_data = {
            "FULL_PRODUCT_NAME": "Test Product - Wireless Headphones",
            "PRODUCT_DESCRIPTION": "High-quality wireless headphones with noise cancellation",
            "PRODUCT_URL": "https://example.com/test-product",
        }

        formatted = self.template.format(**sample_data)

        assert "Test Product - Wireless Headphones" in formatted
        assert "High-quality wireless headphones with noise cancellation" in formatted
        assert "https://example.com/test-product" in formatted
        assert "{FULL_PRODUCT_NAME}" not in formatted
        assert "{PRODUCT_DESCRIPTION}" not in formatted
        assert "{PRODUCT_URL}" not in formatted


class TestInstagramTemplateStructure:
    """Test Instagram caption template structure and content."""

    def setup_method(self):
        """Set up test fixtures."""
        template_path = Path("src/ai/prompts/instagram_caption.md")
        self.template = load_prompt_template(template_path)

    def test_required_placeholders(self):
        """Test that all required placeholders are present."""
        required_placeholders = [
            "{FULL_PRODUCT_NAME}",
            "{PRODUCT_DESCRIPTION}",
            "{PRODUCT_URL}",
            "{CAPTION_STYLE}",
            "{EMOJI_ENABLED}",
        ]

        for placeholder in required_placeholders:
            assert (
                placeholder in self.template
            ), f"Missing required placeholder: {placeholder}"

    def test_contains_instructions(self):
        """Test that template contains clear instructions."""
        # Check for key instruction sections
        assert "Caption Requirements" in self.template
        assert "Hashtag Requirements" in self.template

        # Check for Instagram-specific requirements
        assert "15-30 hashtags" in self.template
        assert "#ad" in self.template

        # Check for dual caption style support
        assert "short" in self.template.lower()
        assert "seo" in self.template.lower()

    def test_caption_style_variants(self):
        """Test that template explains both caption styles."""
        # Should have instructions for both short and SEO styles
        assert "3-5 words" in self.template
        assert "100-200 characters" in self.template

    def test_hashtag_requirements(self):
        """Test that template specifies hashtag requirements."""
        # Critical Instagram requirement: hashtags in caption, not comments
        assert "caption" in self.template.lower() and "comment" in self.template.lower()
        assert "15-30" in self.template

    def test_contains_examples(self):
        """Test that template includes examples."""
        assert "Example" in self.template or "EXAMPLE" in self.template
        assert "CAPTION:" in self.template
        assert "HASHTAGS:" in self.template
        assert "KEYWORDS:" in self.template

        # Should have examples for both styles
        assert (
            "SHORT STYLE:" in self.template or "short style:" in self.template.lower()
        )
        assert "SEO STYLE:" in self.template or "seo style:" in self.template.lower()

    def test_output_format_specified(self):
        """Test that template specifies output format."""
        assert "Output Format:" in self.template
        assert "Return your response" in self.template

    def test_template_formatting_basic(self):
        """Test template can be formatted with basic data."""
        sample_data = {
            "FULL_PRODUCT_NAME": "Test Product - Wireless Headphones",
            "PRODUCT_DESCRIPTION": "High-quality wireless headphones with noise cancellation",
            "PRODUCT_URL": "https://example.com/test-product",
            "CAPTION_STYLE": "seo",
            "EMOJI_ENABLED": "Use emojis",
        }

        formatted = self.template.format(**sample_data)

        assert "Test Product - Wireless Headphones" in formatted
        assert "High-quality wireless headphones with noise cancellation" in formatted
        assert "https://example.com/test-product" in formatted
        assert "seo" in formatted
        assert "Use emojis" in formatted
        assert "{FULL_PRODUCT_NAME}" not in formatted
        assert "{PRODUCT_DESCRIPTION}" not in formatted
        assert "{PRODUCT_URL}" not in formatted
        assert "{CAPTION_STYLE}" not in formatted
        assert "{EMOJI_ENABLED}" not in formatted

    def test_template_formatting_short_style(self):
        """Test template formatting with short caption style."""
        sample_data = {
            "FULL_PRODUCT_NAME": "Test Product",
            "PRODUCT_DESCRIPTION": "Test description",
            "PRODUCT_URL": "https://example.com/test",
            "CAPTION_STYLE": "short",
            "EMOJI_ENABLED": "No emojis",
        }

        formatted = self.template.format(**sample_data)

        assert "short" in formatted
        assert "No emojis" in formatted


class TestTemplateConsistency:
    """Test consistency across all platform templates."""

    def setup_method(self):
        """Set up test fixtures."""
        self.youtube_template = load_prompt_template(
            Path("src/ai/prompts/youtube_metadata.md")
        )
        self.tiktok_template = load_prompt_template(
            Path("src/ai/prompts/tiktok_caption.md")
        )
        self.instagram_template = load_prompt_template(
            Path("src/ai/prompts/instagram_caption.md")
        )

    def test_all_templates_have_output_format(self):
        """Test that all templates specify output format."""
        assert "Output Format:" in self.youtube_template
        assert "Output Format:" in self.tiktok_template
        assert "Output Format:" in self.instagram_template

    def test_all_templates_have_examples(self):
        """Test that all templates include examples."""
        assert "Example" in self.youtube_template or "EXAMPLE" in self.youtube_template
        assert "Example" in self.tiktok_template or "EXAMPLE" in self.tiktok_template
        assert (
            "Example" in self.instagram_template or "EXAMPLE" in self.instagram_template
        )

    def test_all_templates_require_ad_disclosure(self):
        """Test that all templates require #ad hashtag."""
        assert "#ad" in self.youtube_template
        assert "#ad" in self.tiktok_template
        assert "#ad" in self.instagram_template

    def test_all_templates_have_product_placeholders(self):
        """Test that all templates have core product placeholders."""
        common_placeholders = ["{FULL_PRODUCT_NAME}", "{PRODUCT_DESCRIPTION}"]

        for placeholder in common_placeholders:
            assert placeholder in self.youtube_template
            assert placeholder in self.tiktok_template
            assert placeholder in self.instagram_template

    def test_all_templates_are_non_empty(self):
        """Test that all templates have substantial content."""
        # Each template should have reasonable length
        assert len(self.youtube_template) > 1000
        assert len(self.tiktok_template) > 1000
        assert len(self.instagram_template) > 1000


class TestTemplateCompleteness:
    """Test that templates provide complete guidance for LLM."""

    def test_youtube_template_completeness(self):
        """Test YouTube template provides complete guidance."""
        template = load_prompt_template(Path("src/ai/prompts/youtube_metadata.md"))

        # Should have all sections
        required_sections = [
            "Instructions:",
            "Product Information:",
            "Output Format:",
            "Examples",
        ]

        for section in required_sections:
            assert section in template, f"Missing section: {section}"

        # Should explain YouTube-specific optimization
        assert "SEO" in template
        assert "first 150" in template.lower()  # First 150 chars of description

    def test_tiktok_template_completeness(self):
        """Test TikTok template provides complete guidance."""
        template = load_prompt_template(Path("src/ai/prompts/tiktok_caption.md"))

        # Should have all sections
        required_sections = [
            "Instructions:",
            "Product Information:",
            "Output Format:",
            "Examples",
        ]

        for section in required_sections:
            assert section in template, f"Missing section: {section}"

        # Should explain TikTok search optimization
        assert "search" in template.lower()
        assert "avoid" in template.lower() or "do not" in template.lower()

    def test_instagram_template_completeness(self):
        """Test Instagram template provides complete guidance."""
        template = load_prompt_template(Path("src/ai/prompts/instagram_caption.md"))

        # Should have all sections
        required_sections = [
            "Instructions:",
            "Caption Requirements",
            "Hashtag Requirements",
            "Product Information:",
            "Output Format:",
            "Examples",
        ]

        for section in required_sections:
            assert section in template, f"Missing section: {section}"

        # Should explain both caption styles
        assert "short" in template.lower() and "seo" in template.lower()

        # Should explain hashtag placement
        assert (
            "caption" in template.lower() and "comment" in template.lower()
        )  # Hashtags in caption, not comments
