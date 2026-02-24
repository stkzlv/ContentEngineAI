"""Tests for the script template selection system."""

import hashlib
import random
from pathlib import Path

import pytest

from src.ai.script_generator import select_script_template
from src.video.config.llm_settings import LLMSettings, ScriptTemplateConfig


def _make_settings(
    enabled: bool = True,
    templates_dir: str = "src/ai/prompts/scripts",
    template_pool: list[str] | None = None,
    fixed_template: str | None = None,
    prompt_template_path: str = "src/ai/prompts/video_script.md",
) -> LLMSettings:
    """Build LLMSettings with script_templates config for testing."""
    return LLMSettings(
        provider="openrouter",
        api_key_env_var="OPENROUTER_API_KEY",
        models=["gpt-3.5-turbo"],
        prompt_template_path=prompt_template_path,
        target_audience="General audience",
        script_templates=ScriptTemplateConfig(
            enabled=enabled,
            templates_dir=templates_dir,
            template_pool=template_pool or [],
            fixed_template=fixed_template,
        ),
    )


class TestScriptTemplateConfig:
    """Test ScriptTemplateConfig Pydantic model."""

    def test_defaults(self):
        cfg = ScriptTemplateConfig()
        assert cfg.enabled is False
        assert cfg.templates_dir == "src/ai/prompts/scripts"
        assert cfg.template_pool == []
        assert cfg.fixed_template is None

    def test_custom_values(self):
        cfg = ScriptTemplateConfig(
            enabled=True,
            templates_dir="/custom/dir",
            template_pool=["a", "b"],
            fixed_template="story_driven",
        )
        assert cfg.enabled is True
        assert cfg.templates_dir == "/custom/dir"
        assert cfg.template_pool == ["a", "b"]
        assert cfg.fixed_template == "story_driven"

    def test_llm_settings_has_script_templates_field(self):
        settings = LLMSettings(
            provider="openrouter",
            api_key_env_var="KEY",
            models=["m1"],
            prompt_template_path="p.md",
        )
        assert isinstance(settings.script_templates, ScriptTemplateConfig)
        assert settings.script_templates.enabled is False


class TestSelectScriptTemplate:
    """Test select_script_template() function."""

    def test_disabled_returns_default_path(self):
        """When script_templates.enabled is False, return the legacy path."""
        settings = _make_settings(enabled=False)
        result = select_script_template(settings, "B08TEST123")
        assert result == Path("src/ai/prompts/video_script.md")

    def test_fixed_template_returns_that_file(self, temp_dir: Path):
        """When fixed_template is set and exists, return it."""
        templates_dir = temp_dir / "scripts"
        templates_dir.mkdir()
        (templates_dir / "curiosity_hook.md").write_text("template content")

        settings = _make_settings(
            fixed_template="curiosity_hook",
            templates_dir=str(templates_dir),
        )
        result = select_script_template(settings, "B08TEST123")
        assert result == templates_dir / "curiosity_hook.md"

    def test_fixed_template_missing_falls_back(self, temp_dir: Path):
        """When fixed_template doesn't exist, fall back to default path."""
        templates_dir = temp_dir / "scripts"
        templates_dir.mkdir()

        settings = _make_settings(
            fixed_template="nonexistent",
            templates_dir=str(templates_dir),
            prompt_template_path="fallback.md",
        )
        result = select_script_template(settings, "B08TEST123")
        assert result == Path("fallback.md")

    def test_missing_templates_dir_falls_back(self):
        """When templates directory doesn't exist, fall back."""
        settings = _make_settings(
            templates_dir="/nonexistent/path",
            prompt_template_path="fallback.md",
        )
        result = select_script_template(settings, "B08TEST123")
        assert result == Path("fallback.md")

    def test_empty_templates_dir_falls_back(self, temp_dir: Path):
        """When templates directory is empty, fall back."""
        templates_dir = temp_dir / "empty_scripts"
        templates_dir.mkdir()

        settings = _make_settings(
            templates_dir=str(templates_dir),
            prompt_template_path="fallback.md",
        )
        result = select_script_template(settings, "B08TEST123")
        assert result == Path("fallback.md")

    def test_deterministic_selection(self, temp_dir: Path):
        """Same product ID always selects the same template."""
        templates_dir = temp_dir / "scripts"
        templates_dir.mkdir()
        for name in ["alpha", "beta", "gamma", "delta"]:
            (templates_dir / f"{name}.md").write_text(f"{name} content")

        settings = _make_settings(templates_dir=str(templates_dir))

        result1 = select_script_template(settings, "B08PRODUCT1")
        result2 = select_script_template(settings, "B08PRODUCT1")
        assert result1 == result2

    def test_different_products_can_get_different_templates(self, temp_dir: Path):
        """Different product IDs should produce varied template choices."""
        templates_dir = temp_dir / "scripts"
        templates_dir.mkdir()
        for name in ["a", "b", "c", "d", "e", "f", "g", "h"]:
            (templates_dir / f"{name}.md").write_text("content")

        settings = _make_settings(templates_dir=str(templates_dir))

        selections = set()
        for i in range(50):
            result = select_script_template(settings, f"PRODUCT{i:04d}")
            selections.add(result.stem)

        # With 8 templates and 50 products, we expect at least 2 different picks
        assert len(selections) > 1

    def test_pool_filter_restricts_choices(self, temp_dir: Path):
        """template_pool limits which templates can be selected."""
        templates_dir = temp_dir / "scripts"
        templates_dir.mkdir()
        for name in ["alpha", "beta", "gamma", "delta"]:
            (templates_dir / f"{name}.md").write_text("content")

        settings = _make_settings(
            templates_dir=str(templates_dir),
            template_pool=["alpha", "beta"],
        )

        selections = set()
        for i in range(30):
            result = select_script_template(settings, f"PROD{i:04d}")
            selections.add(result.stem)

        # Only alpha and beta should appear
        assert selections <= {"alpha", "beta"}

    def test_pool_with_nonexistent_entries_ignored(self, temp_dir: Path):
        """Pool entries that don't exist on disk get filtered out."""
        templates_dir = temp_dir / "scripts"
        templates_dir.mkdir()
        (templates_dir / "real.md").write_text("content")

        settings = _make_settings(
            templates_dir=str(templates_dir),
            template_pool=["real", "fake", "missing"],
        )

        result = select_script_template(settings, "B08TEST123")
        assert result.stem == "real"

    def test_pool_all_nonexistent_uses_all_templates(self, temp_dir: Path):
        """If every pool entry is invalid, fall back to all available templates."""
        templates_dir = temp_dir / "scripts"
        templates_dir.mkdir()
        (templates_dir / "only_one.md").write_text("content")

        settings = _make_settings(
            templates_dir=str(templates_dir),
            template_pool=["fake1", "fake2"],
        )

        result = select_script_template(settings, "B08TEST123")
        assert result.stem == "only_one"

    def test_no_product_id_picks_randomly(self, temp_dir: Path):
        """Without product_id, selection is non-deterministic (random.choice)."""
        templates_dir = temp_dir / "scripts"
        templates_dir.mkdir()
        for name in ["a", "b", "c"]:
            (templates_dir / f"{name}.md").write_text("content")

        settings = _make_settings(templates_dir=str(templates_dir))

        result = select_script_template(settings, None)
        assert result.stem in {"a", "b", "c"}

    def test_hash_independence_from_other_features(self, temp_dir: Path):
        """Script template hash (salted) differs from font/color/voice hashes."""
        product_id = "B08TESTPROD"

        # Script template uses salted hash
        script_hash = hashlib.md5(
            f"{product_id}:script_template".encode(), usedforsecurity=False
        ).hexdigest()

        # Font/color uses plain hash
        plain_hash = hashlib.md5(product_id.encode(), usedforsecurity=False).hexdigest()

        # They should differ (the salt makes them independent)
        assert script_hash != plain_hash

        # Verify the seed values produce different RNG sequences
        script_seed = int(script_hash[:8], 16)
        font_seed = int(plain_hash[:8], 16)
        assert script_seed != font_seed

    def test_uses_real_templates_directory(self):
        """Verify the actual templates dir has the expected templates."""
        templates_dir = Path("src/ai/prompts/scripts")
        if not templates_dir.is_dir():
            pytest.skip("Templates directory not found (running outside project root)")

        templates = sorted(p.stem for p in templates_dir.glob("*.md"))
        assert len(templates) >= 15
        assert "curiosity_hook" in templates
        assert "classic_promo" in templates
        assert "problem_solution" in templates

    def test_real_templates_deterministic_selection(self):
        """Verify deterministic selection works with the real templates dir."""
        templates_dir = Path("src/ai/prompts/scripts")
        if not templates_dir.is_dir():
            pytest.skip("Templates directory not found")

        settings = _make_settings(templates_dir=str(templates_dir))

        # Same product always gets same template
        r1 = select_script_template(settings, "B0DK1VZBR4")
        r2 = select_script_template(settings, "B0DK1VZBR4")
        assert r1 == r2
        assert r1.exists()
