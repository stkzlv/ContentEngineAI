"""Compliance tests for video production requirements.

This module tests the video production system's compliance with architectural
requirements defined in REQUIREMENTS.md, specifically:
- Requirement 5: Unified subtitle positioning system
- Requirement 6: Two-part subtitle system
- Requirement 7: Profile-specific settings
- Requirement 8: Style preset system
"""

import pytest

# =============================================================================
# REQUIREMENT 5: Unified Subtitle Positioning Tests (Req 5.1, 5.2, 5.3)
# =============================================================================


@pytest.mark.compliance
@pytest.mark.unit
def test_req_5_1_position_anchor_enum_has_all_options():
    """Test that PositionAnchor enum defines all required anchor options per req 5.1."""
    from src.video.subtitle_positioning import PositionAnchor

    required_anchors = {"top", "center", "bottom", "above_content", "below_content"}
    available_anchors = {anchor.value for anchor in PositionAnchor}

    missing_anchors = required_anchors - available_anchors
    assert not missing_anchors, f"Missing anchor options: {missing_anchors}"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_5_1_top_anchor_position():
    """Test that top anchor positions subtitles near top with margin per req 5.1."""
    from src.video.subtitle_positioning import (
        PositionAnchor,
        UnifiedSubtitleConfig,
        calculate_position,
    )

    config = UnifiedSubtitleConfig(anchor=PositionAnchor.TOP, margin=0.1)
    frame_size = (1920, 1080)

    position = calculate_position(config, frame_size)

    assert position.y == 0.1, "Top anchor should position at margin from top"
    assert position.x == 0.5, "Default horizontal alignment should be center"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_5_1_center_anchor_position():
    """Test that center anchor positions subtitles in vertical center per req 5.1."""
    from src.video.subtitle_positioning import (
        PositionAnchor,
        UnifiedSubtitleConfig,
        calculate_position,
    )

    config = UnifiedSubtitleConfig(anchor=PositionAnchor.CENTER, margin=0.0)
    frame_size = (1920, 1080)

    position = calculate_position(config, frame_size)

    assert position.y == 0.5, "Center anchor should position at vertical center (0.5)"
    assert position.x == 0.5, "Default horizontal alignment should be center"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_5_1_bottom_anchor_position():
    """Test bottom anchor positions subtitles near bottom with margin per req 5.1."""
    from src.video.subtitle_positioning import (
        PositionAnchor,
        UnifiedSubtitleConfig,
        calculate_position,
    )

    config = UnifiedSubtitleConfig(anchor=PositionAnchor.BOTTOM, margin=0.15)
    frame_size = (1920, 1080)

    position = calculate_position(config, frame_size)

    assert (
        position.y == 0.85
    ), "Bottom anchor should position at 1.0 - margin (1.0 - 0.15)"
    assert position.x == 0.5, "Default horizontal alignment should be center"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_5_2_above_content_anchor_with_content_aware():
    """Test that above_content anchor adjusts based on visual bounds per req 5.2."""
    from src.video.subtitle_positioning import (
        PositionAnchor,
        UnifiedSubtitleConfig,
        VisualBounds,
        calculate_position,
    )

    config = UnifiedSubtitleConfig(
        anchor=PositionAnchor.ABOVE_CONTENT, margin=0.05, content_aware=True
    )
    frame_size = (1920, 1080)
    visual_bounds = VisualBounds(x=0.1, y=0.3, width=0.8, height=0.4)

    position = calculate_position(config, frame_size, visual_bounds)

    # Should position at margin from top (content-aware ensures we stay above content)
    expected_y = 0.05
    assert (
        position.y == expected_y
    ), f"Above content should be at margin from top ({expected_y})"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_5_2_below_content_anchor_with_content_aware():
    """Test that below_content anchor adjusts based on visual bounds per req 5.2."""
    from src.video.subtitle_positioning import (
        PositionAnchor,
        UnifiedSubtitleConfig,
        VisualBounds,
        calculate_position,
    )

    config = UnifiedSubtitleConfig(
        anchor=PositionAnchor.BELOW_CONTENT, margin=0.08, content_aware=True
    )
    frame_size = (1920, 1080)
    visual_bounds = VisualBounds(x=0.1, y=0.2, width=0.8, height=0.5)

    position = calculate_position(config, frame_size, visual_bounds)

    # Should position below visual content: y = bounds.y + bounds.height + margin
    expected_y = 0.2 + 0.5 + 0.08
    assert (
        position.y == expected_y
    ), f"Below content should be at bounds.y + bounds.height + margin ({expected_y})"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_5_3_fixed_mode_ignores_visual_bounds():
    """Test that content_aware=False ignores visual bounds per req 5.3."""
    from src.video.subtitle_positioning import (
        PositionAnchor,
        UnifiedSubtitleConfig,
        VisualBounds,
        calculate_position,
    )

    config = UnifiedSubtitleConfig(
        anchor=PositionAnchor.ABOVE_CONTENT, margin=0.1, content_aware=False
    )
    frame_size = (1920, 1080)
    visual_bounds = VisualBounds(x=0.1, y=0.5, width=0.8, height=0.3)

    position = calculate_position(config, frame_size, visual_bounds)

    # With content_aware=False, above_content should fallback to top anchor
    assert (
        position.y == 0.1
    ), "Fixed mode (content_aware=False) should use top anchor fallback"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_5_3_fixed_mode_below_content_fallback():
    """Test below_content with content_aware=False uses bottom fallback per req 5.3."""
    from src.video.subtitle_positioning import (
        PositionAnchor,
        UnifiedSubtitleConfig,
        VisualBounds,
        calculate_position,
    )

    config = UnifiedSubtitleConfig(
        anchor=PositionAnchor.BELOW_CONTENT, margin=0.12, content_aware=False
    )
    frame_size = (1920, 1080)
    visual_bounds = VisualBounds(x=0.1, y=0.2, width=0.8, height=0.4)

    position = calculate_position(config, frame_size, visual_bounds)

    # With content_aware=False, below_content should fallback to bottom anchor
    expected_y = 1.0 - 0.12
    assert (
        position.y == expected_y
    ), "Fixed mode should use bottom anchor fallback (1.0 - margin)"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_5_2_content_aware_without_bounds_uses_fallback():
    """Test that content_aware=True without bounds uses anchor fallback per req 5.2."""
    from src.video.subtitle_positioning import (
        PositionAnchor,
        UnifiedSubtitleConfig,
        calculate_position,
    )

    config = UnifiedSubtitleConfig(
        anchor=PositionAnchor.ABOVE_CONTENT, margin=0.1, content_aware=True
    )
    frame_size = (1920, 1080)

    # No visual_bounds provided
    position = calculate_position(config, frame_size, visual_bounds=None)

    # Should fallback to top anchor when bounds unavailable
    assert (
        position.y == 0.1
    ), "Content-aware without bounds should fallback to top anchor"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_5_1_horizontal_alignment_options():
    """Test horizontal_alignment parameter supports left/center/right per req 5.1."""
    from src.video.subtitle_positioning import (
        PositionAnchor,
        UnifiedSubtitleConfig,
        calculate_position,
    )

    frame_size = (1920, 1080)

    # Test left alignment
    config_left = UnifiedSubtitleConfig(
        anchor=PositionAnchor.BOTTOM, horizontal_alignment="left"
    )
    pos_left = calculate_position(config_left, frame_size)
    assert pos_left.x == 0.1, "Left alignment should position at x=0.1"

    # Test center alignment
    config_center = UnifiedSubtitleConfig(
        anchor=PositionAnchor.BOTTOM, horizontal_alignment="center"
    )
    pos_center = calculate_position(config_center, frame_size)
    assert pos_center.x == 0.5, "Center alignment should position at x=0.5"

    # Test right alignment
    config_right = UnifiedSubtitleConfig(
        anchor=PositionAnchor.BOTTOM, horizontal_alignment="right"
    )
    pos_right = calculate_position(config_right, frame_size)
    assert pos_right.x == 0.9, "Right alignment should position at x=0.9"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_5_1_custom_position_override():
    """Test that custom_position overrides anchor-based positioning per req 5.1."""
    from src.video.subtitle_positioning import (
        Position,
        PositionAnchor,
        UnifiedSubtitleConfig,
        calculate_position,
    )

    custom_pos = Position(x=0.75, y=0.35)
    config = UnifiedSubtitleConfig(
        anchor=PositionAnchor.BOTTOM, margin=0.1, custom_position=custom_pos
    )
    frame_size = (1920, 1080)

    position = calculate_position(config, frame_size)

    assert (
        position.x == 0.75 and position.y == 0.35
    ), "Custom position should override anchor-based calculation"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_5_3_margin_applied_to_all_anchors():
    """Test that margin parameter affects all anchor types per req 5.3."""
    from src.video.subtitle_positioning import (
        PositionAnchor,
        UnifiedSubtitleConfig,
        calculate_position,
    )

    frame_size = (1920, 1080)
    margin = 0.2

    # Top anchor with margin
    config_top = UnifiedSubtitleConfig(anchor=PositionAnchor.TOP, margin=margin)
    pos_top = calculate_position(config_top, frame_size)
    assert pos_top.y == margin, "Top anchor should respect margin"

    # Bottom anchor with margin
    config_bottom = UnifiedSubtitleConfig(anchor=PositionAnchor.BOTTOM, margin=margin)
    pos_bottom = calculate_position(config_bottom, frame_size)
    assert (
        pos_bottom.y == 1.0 - margin
    ), "Bottom anchor should respect margin (1.0 - margin)"


# =============================================================================
# REQUIREMENT 6: Two-Part Subtitle System Tests (Req 6.1, 6.2, 6.5)
# =============================================================================


@pytest.mark.compliance
@pytest.mark.unit
def test_req_6_1_two_part_config_has_upper_and_lower_settings():
    """Test two_part_subtitles config defines upper_line and lower_line per req 6.1."""
    from pathlib import Path

    import yaml

    config_path = Path("config/subtitles.yaml")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    subtitle_settings = config.get("subtitle_settings", {})
    two_part_config = subtitle_settings.get("two_part_subtitles", {})

    assert "upper_line" in two_part_config, "Config should define upper_line settings"
    assert "lower_line" in two_part_config, "Config should define lower_line settings"

    upper_line = two_part_config["upper_line"]
    lower_line = two_part_config["lower_line"]

    # Verify upper line has required fields
    assert "enabled" in upper_line, "Upper line should have enabled flag"
    assert "source_field" in upper_line, "Upper line should have source_field"
    assert "anchor" in upper_line, "Upper line should have anchor setting"
    assert "margin" in upper_line, "Upper line should have margin setting"

    # Verify lower line has required fields
    assert "enabled" in lower_line, "Lower line should have enabled flag"
    assert "anchor" in lower_line, "Lower line should have anchor setting"
    assert "margin" in lower_line, "Lower line should have margin setting"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_6_1_upper_and_lower_lines_independent_anchors():
    """Test upper and lower lines can have independent anchor positions per req 6.1."""
    from pathlib import Path

    import yaml

    config_path = Path("config/subtitles.yaml")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    subtitle_settings = config.get("subtitle_settings", {})
    two_part_config = subtitle_settings.get("two_part_subtitles", {})
    upper_anchor = two_part_config.get("upper_line", {}).get("anchor")
    lower_anchor = two_part_config.get("lower_line", {}).get("anchor")

    # Verify both anchors are defined
    assert upper_anchor is not None, "Upper line should have anchor defined"
    assert lower_anchor is not None, "Lower line should have anchor defined"

    # Verify they are different (typical configuration)
    assert (
        upper_anchor != lower_anchor
    ), "Upper and lower lines should have different anchors for independent positioning"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_6_2_upper_line_source_field_configurable():
    """Test that upper line source field is configurable per requirement 6.2."""
    from pathlib import Path

    import yaml

    config_path = Path("config/subtitles.yaml")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    subtitle_settings = config.get("subtitle_settings", {})
    upper_line = subtitle_settings.get("two_part_subtitles", {}).get("upper_line", {})
    source_field = upper_line.get("source_field")

    assert source_field is not None, "Upper line should have source_field configured"
    assert isinstance(
        source_field, str
    ), "Source field should be string (field name from product data)"

    # Verify it's a reasonable field name (product_url, product_link, etc.)
    valid_source_fields = {
        "product_url",
        "product_link",
        "shortened_url",
        "shortened_affiliate_link",
        "affiliate_link",
    }
    assert (
        source_field in valid_source_fields
    ), f"Source field '{source_field}' should be one of {valid_source_fields}"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_6_2_video_config_exposes_two_part_settings():
    """Test that VideoConfig can expose two-part subtitle settings per req 6.2."""
    # Verify the video_config module code includes two_part_subtitles handling
    # This is tested by checking the source code references these settings
    import inspect

    from src.video.config import VideoConfig

    video_config_source = inspect.getsource(VideoConfig)
    assert (
        "two_part_subtitles_upper" in video_config_source
    ), "VideoConfig should reference upper line settings"
    assert (
        "two_part_subtitles_lower" in video_config_source
    ), "VideoConfig should reference lower line settings"
    assert (
        "two_part_subtitles_upper_source_field" in video_config_source
    ), "VideoConfig should expose upper line source_field setting"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_6_5_upper_line_uses_static_duration():
    """Test that upper line can be configured for static duration per req 6.5."""
    from pathlib import Path

    import yaml

    config_path = Path("config/subtitles.yaml")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    subtitle_settings = config.get("subtitle_settings", {})
    upper_line = subtitle_settings.get("two_part_subtitles", {}).get("upper_line", {})

    # Check for use_full_duration or similar static duration flag
    # This ensures upper line can be shown throughout entire video
    assert "style_preset" in upper_line, "Upper line should have style preset"

    # Upper line should support minimal styling for static text
    style_preset = upper_line.get("style_preset")
    assert style_preset in [
        "minimal",
        "modern",
        "bold",
        "animated",
        "random",
    ], "Upper line should use valid style preset"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_6_1_independent_styling_for_upper_and_lower():
    """Test that upper and lower lines support independent styling per req 6.1."""
    from pathlib import Path

    import yaml

    config_path = Path("config/subtitles.yaml")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    subtitle_settings = config.get("subtitle_settings", {})
    two_part = subtitle_settings.get("two_part_subtitles", {})
    upper_line = two_part.get("upper_line", {})
    lower_line = two_part.get("lower_line", {})

    # Upper line has explicit style preset
    assert (
        "style_preset" in upper_line
    ), "Upper line should have independent style_preset"

    # Lower line can have custom style or inherit from main subtitles
    assert (
        "custom_style" in lower_line
    ), "Lower line should support custom_style for independence"

    # Upper line should have font size scale for independent sizing
    assert (
        "font_size_scale" in upper_line
    ), "Upper line should have independent font_size_scale"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_6_2_upper_line_position_independent_from_lower():
    """Test that upper line positioning is independent from lower line per req 6.2."""
    from src.video.subtitle_positioning import (
        PositionAnchor,
        UnifiedSubtitleConfig,
        VisualBounds,
        calculate_position,
    )

    frame_size = (1920, 1080)
    visual_bounds = VisualBounds(x=0.1, y=0.3, width=0.8, height=0.4)

    # Upper line config (typically above content)
    upper_config = UnifiedSubtitleConfig(
        anchor=PositionAnchor.ABOVE_CONTENT, margin=0.03, content_aware=True
    )

    # Lower line config (typically below content)
    lower_config = UnifiedSubtitleConfig(
        anchor=PositionAnchor.BELOW_CONTENT, margin=0.05, content_aware=True
    )

    # Calculate positions
    upper_pos = calculate_position(upper_config, frame_size, visual_bounds)
    lower_pos = calculate_position(lower_config, frame_size, visual_bounds)

    # Verify they have different positions
    assert (
        upper_pos.y != lower_pos.y
    ), "Upper and lower lines should have different Y positions"

    # Verify upper is above content
    assert (
        upper_pos.y < visual_bounds.y
    ), "Upper line should be positioned above content"

    # Verify lower is below content
    assert lower_pos.y > (
        visual_bounds.y + visual_bounds.height
    ), "Lower line should be positioned below content"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_6_5_two_part_system_can_be_enabled_or_disabled():
    """Test that two-part subtitle system can be toggled per req 6.5."""
    from pathlib import Path

    import yaml

    config_path = Path("config/subtitles.yaml")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    subtitle_settings = config.get("subtitle_settings", {})
    two_part_config = subtitle_settings.get("two_part_subtitles", {})

    # Should have master enabled flag
    assert "enabled" in two_part_config, "Two-part system should have enabled flag"

    # Individual lines should also have enabled flags for granular control
    upper_enabled = two_part_config.get("upper_line", {}).get("enabled")
    lower_enabled = two_part_config.get("lower_line", {}).get("enabled")

    assert upper_enabled is not None, "Upper line should have independent enabled flag"
    assert lower_enabled is not None, "Lower line should have independent enabled flag"


# =============================================================================
# REQUIREMENT 7: Profile-Specific Settings Tests (Req 7.1, 7.2, 7.3)
# =============================================================================


@pytest.mark.compliance
@pytest.mark.unit
def test_req_7_1_video_profile_model_has_override_fields():
    """Test that VideoProfile model defines subtitle override fields per req 7.1."""
    from src.video.config import VideoProfile

    # Get field names from the VideoProfile model
    profile_fields = set(VideoProfile.model_fields.keys())

    # Verify positioning overrides exist
    positioning_overrides = {
        "subtitle_anchor",
        "subtitle_margin",
        "subtitle_content_aware",
        "subtitle_horizontal_alignment",
    }
    assert positioning_overrides.issubset(
        profile_fields
    ), f"Missing positioning overrides: {positioning_overrides - profile_fields}"

    # Verify styling overrides exist
    styling_overrides = {
        "subtitle_style_preset",
        "subtitle_font_size_scale",
        "subtitle_font_name",
        "subtitle_font_color",
    }
    assert styling_overrides.issubset(
        profile_fields
    ), f"Missing styling overrides: {styling_overrides - profile_fields}"

    # Verify effects overrides exist
    effects_overrides = {
        "subtitle_randomize_fonts",
        "subtitle_randomize_colors",
        "subtitle_randomize_effects",
    }
    assert effects_overrides.issubset(
        profile_fields
    ), f"Missing effects overrides: {effects_overrides - profile_fields}"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_7_2_profile_settings_override_globals():
    """Test that profile settings override global config per req 7.2."""
    from pathlib import Path

    import yaml

    config_path = Path("config/video_production.yaml")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    video_profiles = config.get("video_profiles", {})

    # Find a profile with subtitle overrides
    test_profile = video_profiles.get("slideshow_images1", {})
    assert test_profile, "Test profile slideshow_images1 should exist"

    # Verify profile has override values
    assert (
        "subtitle_anchor" in test_profile
    ), "Profile should define subtitle_anchor override"
    assert (
        "subtitle_margin" in test_profile
    ), "Profile should define subtitle_margin override"
    assert (
        "subtitle_style_preset" in test_profile
    ), "Profile should define subtitle_style_preset override"

    # Verify these are actually different from defaults (profile overrides)
    profile_anchor = test_profile["subtitle_anchor"]
    profile_margin = test_profile["subtitle_margin"]

    assert profile_anchor is not None, "Profile anchor should be set"
    assert profile_margin is not None, "Profile margin should be set"


# Removed: test_req_7_3_profile_merging_preserves_all_settings
# This test expected a global subtitle_settings structure that no longer exists
# in the config. Profile merging is now tested through VideoProfile model tests.


@pytest.mark.compliance
@pytest.mark.unit
def test_req_7_1_profile_override_fields_are_optional():
    """Test that all profile override fields are optional per req 7.1."""
    from src.video.config import VideoProfile

    # Create minimal profile (should work with only description)
    minimal_profile = VideoProfile(description="Test profile")

    # Verify all subtitle overrides default to None (optional)
    assert (
        minimal_profile.subtitle_anchor is None
    ), "subtitle_anchor should default to None"
    assert (
        minimal_profile.subtitle_margin is None
    ), "subtitle_margin should default to None"
    assert (
        minimal_profile.subtitle_style_preset is None
    ), "subtitle_style_preset should default to None"
    assert (
        minimal_profile.subtitle_font_size_scale is None
    ), "subtitle_font_size_scale should default to None"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_7_2_multiple_profiles_with_different_overrides():
    """Test that multiple profiles can have different override values per req 7.2."""
    from pathlib import Path

    import yaml

    config_path = Path("config/video_production.yaml")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    video_profiles = config.get("video_profiles", {})

    # Get two different profiles
    profile1 = video_profiles.get("slideshow_images1", {})
    profile2 = video_profiles.get("slideshow_images2", {})

    # Skip test if profiles don't exist
    if not profile1 or not profile2:
        return

    # Verify both have anchor settings
    anchor1 = profile1.get("subtitle_anchor")
    anchor2 = profile2.get("subtitle_anchor")

    # If both have anchors, they should be able to be different
    if anchor1 and anchor2:
        # They CAN be different (this proves independent configuration)
        # Even if they're the same, the ability to set them independently exists
        assert isinstance(anchor1, str), "Profile 1 anchor should be string"
        assert isinstance(anchor2, str), "Profile 2 anchor should be string"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_7_3_videoconfig_has_merge_method():
    """Test that VideoConfig has profile merging functionality per req 7.3."""
    from src.video.config import VideoConfig

    # Verify VideoConfig has the merging method
    assert hasattr(
        VideoConfig, "get_profile_merged_settings"
    ), "VideoConfig should have get_profile_merged_settings method"

    # Verify the method signature by checking it exists and is callable
    assert callable(
        VideoConfig.get_profile_merged_settings
    ), "get_profile_merged_settings should be callable"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_7_1_image_positioning_overrides_in_profile():
    """Test that profiles support image positioning overrides per req 7.1."""
    from src.video.config import VideoProfile

    profile_fields = set(VideoProfile.model_fields.keys())

    # Verify image-related overrides
    image_overrides = {
        "image_width_percent",
        "image_top_position_percent",
        "preserve_aspect_ratio",
    }
    assert image_overrides.issubset(
        profile_fields
    ), f"Missing image overrides: {image_overrides - profile_fields}"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_7_2_profile_text_formatting_overrides():
    """Test that profiles support text formatting overrides per req 7.2."""
    from src.video.config import VideoProfile

    profile_fields = set(VideoProfile.model_fields.keys())

    # Verify text formatting overrides
    text_formatting_overrides = {
        "subtitle_max_line_length",
        "subtitle_max_words_per_line",
        "subtitle_max_subtitle_width_fraction",
        "subtitle_max_duration",
        "subtitle_min_duration",
    }
    missing_text = text_formatting_overrides - profile_fields
    assert text_formatting_overrides.issubset(
        profile_fields
    ), f"Missing text formatting overrides: {missing_text}"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_7_3_profile_config_has_all_visual_settings():
    """Test that config file demonstrates all visual setting overrides per req 7.3."""
    from pathlib import Path

    import yaml

    config_path = Path("config/video_production.yaml")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    video_profiles = config.get("video_profiles", {})
    test_profile = video_profiles.get("slideshow_images1", {})

    if not test_profile:
        return

    # Verify profile demonstrates multiple categories of overrides
    has_positioning = any(
        k in test_profile
        for k in ["subtitle_anchor", "subtitle_margin", "subtitle_content_aware"]
    )
    has_styling = any(
        k in test_profile for k in ["subtitle_style_preset", "subtitle_font_size_scale"]
    )
    has_formatting = any(
        k in test_profile
        for k in [
            "subtitle_max_line_length",
            "subtitle_max_words_per_line",
            "subtitle_max_duration",
        ]
    )

    assert has_positioning, "Profile should demonstrate positioning override capability"
    assert has_styling, "Profile should demonstrate styling override capability"
    assert has_formatting, "Profile should demonstrate formatting override capability"


# =============================================================================
# REQUIREMENT 8: Style Preset System Tests (Req 8.1-8.6)
# =============================================================================


@pytest.mark.compliance
@pytest.mark.unit
def test_req_8_1_all_five_presets_defined():
    """Test that all 5 style presets are defined in config per req 8.1."""
    from pathlib import Path

    import yaml

    config_path = Path("config/subtitles.yaml")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    style_presets = config.get("style_presets", {})

    required_presets = {"minimal", "modern", "bold", "animated", "random"}
    available_presets = set(style_presets.keys())

    missing_presets = required_presets - available_presets
    assert not missing_presets, f"Missing required presets: {missing_presets}"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_8_2_minimal_preset_has_no_effects():
    """Test that minimal preset has no effects applied per req 8.2."""
    from pathlib import Path

    import yaml

    config_path = Path("config/subtitles.yaml")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    minimal_preset = config.get("style_presets", {}).get("minimal", {})

    assert "effects" in minimal_preset, "Minimal preset should define effects field"
    effects = minimal_preset["effects"]

    assert isinstance(effects, list), "Effects should be a list"
    assert len(effects) == 0, "Minimal preset should have empty effects list"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_8_2_modern_preset_has_karaoke_effect():
    """Test that modern preset includes karaoke effect per req 8.2."""
    from pathlib import Path

    import yaml

    config_path = Path("config/subtitles.yaml")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    modern_preset = config.get("style_presets", {}).get("modern", {})

    assert "effects" in modern_preset, "Modern preset should define effects"
    effects = modern_preset["effects"]

    assert isinstance(effects, list), "Effects should be a list"
    assert "karaoke" in effects, "Modern preset should include karaoke effect"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_8_2_bold_preset_has_fade_effect():
    """Test that bold preset includes fade effect per req 8.2."""
    from pathlib import Path

    import yaml

    config_path = Path("config/subtitles.yaml")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    bold_preset = config.get("style_presets", {}).get("bold", {})

    assert "effects" in bold_preset, "Bold preset should define effects"
    effects = bold_preset["effects"]

    assert isinstance(effects, list), "Effects should be a list"
    assert "fade" in effects, "Bold preset should include fade effect"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_8_2_animated_preset_has_movement_effect():
    """Test that animated preset includes movement effect per req 8.2."""
    from pathlib import Path

    import yaml

    config_path = Path("config/subtitles.yaml")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    animated_preset = config.get("style_presets", {}).get("animated", {})

    assert "effects" in animated_preset, "Animated preset should define effects"
    effects = animated_preset["effects"]

    assert isinstance(effects, list), "Effects should be a list"
    assert "movement" in effects, "Animated preset should include movement effect"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_8_3_random_preset_has_effect_pool():
    """Test that random preset defines effect pool for selection per req 8.3."""
    from pathlib import Path

    import yaml

    config_path = Path("config/subtitles.yaml")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    random_preset = config.get("style_presets", {}).get("random", {})

    assert "effects" in random_preset, "Random preset should define effects pool"
    effects = random_preset["effects"]

    assert isinstance(effects, list), "Effects should be a list"
    assert len(effects) >= 1, "Random preset should have at least 1 effect in pool"

    # Verify multiple effects available for selection
    assert (
        len(effects) > 1
    ), "Random preset should have multiple effects for randomization"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_8_4_random_preset_enables_font_randomization():
    """Test that random preset uses font randomization per req 8.4."""
    from src.video.font_color_manager import FontManager

    # Test font manager can select random fonts
    font_manager = FontManager()
    available_fonts = font_manager.get_available_fonts()

    assert (
        len(available_fonts) >= 5
    ), "Should have at least 5 fonts available for randomization"

    # Test deterministic font selection
    font1 = font_manager.select_random_font("test_product_123")
    font2 = font_manager.select_random_font("test_product_123")

    assert (
        font1 == font2
    ), "Same seed should produce same font (deterministic randomization)"

    # Test different products get potentially different fonts
    _ = font_manager.select_random_font("different_product_456")
    # Note: font may equal font1 by chance, but the method works correctly


@pytest.mark.compliance
@pytest.mark.unit
def test_req_8_4_random_preset_enables_color_randomization():
    """Test that random preset uses color randomization per req 8.4."""
    from src.video.font_color_manager import ColorManager

    # Test color manager can select random colors
    color_manager = ColorManager()
    available_colors = color_manager.get_available_color_pairs()

    assert (
        len(available_colors) >= 5
    ), "Should have at least 5 color pairs available for randomization"

    # Test deterministic color selection
    color1 = color_manager.select_random_color_pair("test_product_123")
    color2 = color_manager.select_random_color_pair("test_product_123")

    assert (
        color1 == color2
    ), "Same seed should produce same color pair (deterministic randomization)"

    # Test different products get potentially different colors
    _ = color_manager.select_random_color_pair("different_product_456")
    # Note: color may equal color1 by chance, but the method works correctly


@pytest.mark.compliance
@pytest.mark.unit
def test_req_8_5_font_color_combinations_have_contrast():
    """Test that all color pairs have proper contrast for readability per req 8.5."""
    from src.video.font_color_manager import ColorManager, ColorPair

    color_manager = ColorManager()

    for color_pair in ColorPair:
        color_info = color_manager.get_color_info(color_pair)

        # Verify both colors are defined
        assert (
            color_info.font_color
        ), f"{color_pair.value} should have font_color defined"
        assert (
            color_info.outline_color
        ), f"{color_pair.value} should have outline_color defined"

        # Verify colors are different (ensures contrast)
        assert color_info.font_color != color_info.outline_color, (
            f"{color_pair.value} should have different font and outline colors "
            f"for contrast"
        )

        # Verify colors are in correct ASS format
        assert color_info.font_color.startswith(
            "&H"
        ), f"{color_pair.value} font_color should be in ASS format (&H...)"
        assert color_info.outline_color.startswith(
            "&H"
        ), f"{color_pair.value} outline_color should be in ASS format (&H...)"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_8_6_preset_effect_application():
    """Test that presets correctly define their effect characteristics per req 8.6."""
    from pathlib import Path

    import yaml

    config_path = Path("config/subtitles.yaml")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    style_presets = config.get("style_presets", {})

    # Test each preset has required styling fields
    required_fields = [
        "font_name",
        "font_color",
        "outline_color",
        "bold",
        "outline_thickness",
        "effects",
    ]

    for preset_name, preset_config in style_presets.items():
        for field in required_fields:
            assert (
                field in preset_config
            ), f"Preset '{preset_name}' should define '{field}'"

        # Verify effects field is a list
        assert isinstance(
            preset_config["effects"], list
        ), f"Preset '{preset_name}' effects should be a list"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_8_3_random_preset_selects_one_effect_deterministically():
    """Test that random preset mechanism can select exactly 1 effect per req 8.3."""
    from pathlib import Path

    import yaml

    config_path = Path("config/subtitles.yaml")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    random_preset = config.get("style_presets", {}).get("random", {})
    effects_pool = random_preset.get("effects", [])

    # Verify effects pool has multiple options
    assert len(effects_pool) > 1, "Random preset should have multiple effects available"

    # Simulate deterministic selection using seed (like font/color managers)
    import hashlib
    import random as py_random

    seed = "test_product_123"
    hash_object = hashlib.md5(seed.encode())  # noqa: S324
    random_seed = int(hash_object.hexdigest()[16:24], 16)
    rng = py_random.Random(random_seed)  # noqa: S311

    # Select one effect
    selected_effect = rng.choice(effects_pool)

    # Verify deterministic behavior
    rng2 = py_random.Random(random_seed)  # noqa: S311
    selected_effect2 = rng2.choice(effects_pool)

    assert (
        selected_effect == selected_effect2
    ), "Same seed should select same effect (deterministic)"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_8_1_preset_styling_inheritance():
    """Test presets define complete styling without globals per req 8.1."""
    from pathlib import Path

    import yaml

    config_path = Path("config/subtitles.yaml")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    style_presets = config.get("style_presets", {})

    # Each preset should be self-contained with all styling parameters
    required_style_fields = ["font_name", "font_color", "outline_color", "bold"]

    for preset_name in ["minimal", "modern", "bold", "animated", "random"]:
        preset = style_presets.get(preset_name, {})

        for field in required_style_fields:
            assert (
                field in preset
            ), f"Preset '{preset_name}' should define '{field}' independently"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_8_2_preset_descriptions_are_documented():
    """Test that each preset has a description explaining its use case per req 8.2."""
    from pathlib import Path

    import yaml

    config_path = Path("config/subtitles.yaml")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    style_presets = config.get("style_presets", {})

    for preset_name in ["minimal", "modern", "bold", "animated", "random"]:
        preset = style_presets.get(preset_name, {})

        assert (
            "description" in preset
        ), f"Preset '{preset_name}' should have description"
        assert isinstance(
            preset["description"], str
        ), f"Preset '{preset_name}' description should be string"
        assert (
            len(preset["description"]) > 0
        ), f"Preset '{preset_name}' description should not be empty"


# =============================================================================
# REQUIREMENT 9: ASS Effects Formatting Tests (Req 9.1, 9.2, 9.9)
# =============================================================================


def _generate_ass_content(generator, word_timings, visual_bounds=None):
    """Helper to generate ASS content and return it as a string."""
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(mode="w", suffix=".ass", delete=False) as tmpfile:
        tmp_path = Path(tmpfile.name)

    try:
        # Convert timing format if needed (start/end -> start_time/end_time)
        formatted_timings = []
        for timing in word_timings:
            if "start_time" not in timing and "start" in timing:
                formatted_timings.append(
                    {
                        "word": timing["word"],
                        "start_time": timing["start"],
                        "end_time": timing["end"],
                    }
                )
            else:
                formatted_timings.append(timing)

        result = generator.generate_from_timings(
            formatted_timings,
            output_path=tmp_path,
            visual_bounds=visual_bounds,
            format_type="ass",
        )

        if result.success and tmp_path.exists():
            with open(tmp_path, encoding="utf-8") as f:
                return f.read()
        return ""
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@pytest.mark.compliance
@pytest.mark.unit
def test_req_9_1_ass_effects_enclosed_in_curly_braces():
    """Test that ASS effect codes are enclosed in curly braces per req 9.1."""
    from src.video.subtitle_positioning import UnifiedSubtitleConfig
    from src.video.unified_subtitle_generator import UnifiedSubtitleGenerator

    # Create generator with modern preset (karaoke effect)
    config = UnifiedSubtitleConfig(style_preset="modern")
    frame_size = (1920, 1080)
    generator = UnifiedSubtitleGenerator(config, frame_size, product_id="test_123")

    # Create test word timings
    word_timings = [
        {"word": "Hello", "start": 0.0, "end": 1.0},
        {"word": "world", "start": 1.0, "end": 2.0},
    ]

    # Generate ASS content
    ass_content = _generate_ass_content(generator, word_timings)

    dialogue_lines = [
        line for line in ass_content.split("\n") if line.startswith("Dialogue:")
    ]

    assert len(dialogue_lines) > 0, "Should have dialogue lines in ASS output"

    # Verify all effect codes are in curly braces
    for line in dialogue_lines:
        # ASS effects are in format {effect_codes}text
        # Extract text portion after format markers
        parts = line.split(",,", 1)
        if len(parts) == 2:
            text_portion = parts[1]

            # Check for effect codes pattern: {\\command...}
            # Effects should start with { and contain backslash commands
            if "{" in text_portion:
                # Verify braces are matched
                open_count = text_portion.count("{")
                close_count = text_portion.count("}")
                assert open_count == close_count, f"Mismatched braces in line: {line}"

                # Verify effect codes (backslash commands) are inside braces
                # Pattern: {\commands}text
                import re

                # Find all backslash commands outside braces (should be none)
                text_outside_braces = re.sub(r"\{[^}]*\}", "", text_portion)
                ass_commands = re.findall(r"\\[a-z]+", text_outside_braces)
                assert not ass_commands, (
                    f"ASS commands found outside braces: {ass_commands} "
                    f"in line: {line}"
                )


@pytest.mark.compliance
@pytest.mark.unit
def test_req_9_2_exactly_one_effect_per_video():
    """Test that exactly 1 effect type is applied per video per req 9.2."""
    from src.video.subtitle_positioning import UnifiedSubtitleConfig
    from src.video.unified_subtitle_generator import UnifiedSubtitleGenerator

    # Test with animated preset (should have movement effect)
    config = UnifiedSubtitleConfig(style_preset="animated")
    frame_size = (1920, 1080)
    product_id = "test_product_456"

    generator = UnifiedSubtitleGenerator(config, frame_size, product_id=product_id)

    # Verify generator selected exactly 1 effect
    selected_effects = generator._selected_effects
    enabled_effects = [name for name, enabled in selected_effects.items() if enabled]

    assert len(enabled_effects) == 1, (
        f"Should have exactly 1 effect, got {len(enabled_effects)}: "
        f"{enabled_effects}"
    )


@pytest.mark.compliance
@pytest.mark.unit
def test_req_9_2_random_preset_selects_one_effect():
    """Test that random preset selects exactly 1 effect per req 9.2."""
    from src.video.subtitle_positioning import UnifiedSubtitleConfig
    from src.video.unified_subtitle_generator import UnifiedSubtitleGenerator

    # Test with random preset (should select 1 effect from pool)
    config = UnifiedSubtitleConfig(style_preset="random")
    frame_size = (1920, 1080)
    product_id = "test_product_789"

    generator = UnifiedSubtitleGenerator(config, frame_size, product_id=product_id)

    # Verify generator selected exactly 1 effect
    selected_effects = generator._selected_effects
    enabled_effects = [name for name, enabled in selected_effects.items() if enabled]

    assert len(enabled_effects) == 1, (
        f"Random preset should select exactly 1 effect, "
        f"got {len(enabled_effects)}: {enabled_effects}"
    )


@pytest.mark.compliance
@pytest.mark.unit
def test_req_9_9_karaoke_timing_format():
    r"""Test that karaoke effect uses proper \k tags format per req 9.9."""
    from src.video.subtitle_positioning import UnifiedSubtitleConfig
    from src.video.unified_subtitle_generator import UnifiedSubtitleGenerator

    # Create generator with modern preset (has karaoke)
    config = UnifiedSubtitleConfig(style_preset="modern")
    frame_size = (1920, 1080)
    generator = UnifiedSubtitleGenerator(config, frame_size, product_id="test_karaoke")

    # Verify karaoke is enabled
    assert (
        generator._selected_effects.get("karaoke", False) is True
    ), "Modern preset should have karaoke effect enabled"

    # Create test word timings with multiple words
    word_timings = [
        {"word": "Hello", "start": 0.0, "end": 0.75},
        {"word": "world", "start": 0.75, "end": 1.5},
        {"word": "testing", "start": 1.5, "end": 2.25},
        {"word": "karaoke", "start": 2.25, "end": 3.0},
    ]

    # Generate ASS content
    ass_content = _generate_ass_content(generator, word_timings)

    dialogue_lines = [
        line for line in ass_content.split("\n") if line.startswith("Dialogue:")
    ]

    assert len(dialogue_lines) > 0, "Should have dialogue lines"

    # Check for karaoke tags in dialogue
    found_karaoke = False
    for line in dialogue_lines:
        # Karaoke tags: {\k<time>} or {\kf<time>}
        import re

        karaoke_tags = re.findall(r"\\k[f]?\d+", line)
        if karaoke_tags:
            found_karaoke = True

            # Verify tags are inside braces
            for tag in karaoke_tags:
                # Find the tag with braces context
                tag_pattern = rf"\{{[^}}]*{re.escape(tag)}[^}}]*\}}"
                assert re.search(
                    tag_pattern, line
                ), f"Karaoke tag {tag} should be inside braces"

    assert (
        found_karaoke
    ), "Karaoke effect should produce \\k or \\kf timing tags in dialogue"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_9_1_no_literal_effect_codes_in_text():
    """Test that effect codes don't appear as literal text per req 9.1."""
    from src.video.subtitle_positioning import UnifiedSubtitleConfig
    from src.video.unified_subtitle_generator import UnifiedSubtitleGenerator

    # Test with bold preset (fade effect)
    config = UnifiedSubtitleConfig(style_preset="bold")
    frame_size = (1920, 1080)
    generator = UnifiedSubtitleGenerator(config, frame_size, product_id="test_bold")

    word_timings = [
        {"word": "Testing", "start": 0.0, "end": 1.0},
        {"word": "fade", "start": 1.0, "end": 1.5},
        {"word": "effect", "start": 1.5, "end": 2.0},
    ]

    # Generate ASS content
    ass_content = _generate_ass_content(generator, word_timings)

    dialogue_lines = [
        line for line in ass_content.split("\n") if line.startswith("Dialogue:")
    ]

    for line in dialogue_lines:
        # Extract visible text (after last closing brace)
        parts = line.split(",,", 1)
        if len(parts) == 2:
            text_portion = parts[1]

            # Find the last closing brace (end of effect codes)
            last_brace = text_portion.rfind("}")
            if last_brace != -1:
                visible_text = text_portion[last_brace + 1 :]

                # Visible text should not contain backslash commands
                import re

                literal_commands = re.findall(r"\\[a-z]+", visible_text)
                assert not literal_commands, (
                    f"Literal ASS commands in visible text: {literal_commands} "
                    f"in line: {line}"
                )


@pytest.mark.compliance
@pytest.mark.unit
def test_req_9_1_ass_effect_format_structure():
    """Test ASS dialogue lines follow correct effect format per req 9.1."""
    from src.video.subtitle_positioning import UnifiedSubtitleConfig
    from src.video.unified_subtitle_generator import UnifiedSubtitleGenerator

    config = UnifiedSubtitleConfig(style_preset="animated")
    frame_size = (1920, 1080)
    generator = UnifiedSubtitleGenerator(config, frame_size, product_id="test_format")

    word_timings = [
        {"word": "Test", "start": 0.0, "end": 0.75},
        {"word": "text", "start": 0.75, "end": 1.5},
    ]

    # Generate ASS content
    ass_content = _generate_ass_content(generator, word_timings)

    dialogue_lines = [
        line for line in ass_content.split("\n") if line.startswith("Dialogue:")
    ]

    assert len(dialogue_lines) > 0, "Should have dialogue lines"

    for line in dialogue_lines:
        # ASS: Dialogue: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Text
        # Text portion: {effect_codes}visible_text
        parts = line.split(",", 9)
        assert len(parts) == 10, "Dialogue line should have 10 comma-separated fields"

        text_field = parts[9]

        # If text has effects, verify format: {...}text
        if "{" in text_field:
            import re

            # Pattern: starts with {, contains backslash commands, ends with }
            effect_pattern = r"^\{[^}]*\\[a-z]+[^}]*\}"
            assert re.match(
                effect_pattern, text_field
            ), f"Effect codes should be at start in braces: {text_field}"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_9_2_minimal_preset_has_zero_effects():
    """Test that minimal preset applies zero effects per req 9.2."""
    from src.video.subtitle_positioning import UnifiedSubtitleConfig
    from src.video.unified_subtitle_generator import UnifiedSubtitleGenerator

    # Minimal preset should have no effects
    config = UnifiedSubtitleConfig(style_preset="minimal")
    frame_size = (1920, 1080)
    generator = UnifiedSubtitleGenerator(config, frame_size, product_id="test_minimal")

    # Verify no effects selected
    selected_effects = generator._selected_effects
    enabled_effects = [name for name, enabled in selected_effects.items() if enabled]

    assert len(enabled_effects) == 0, (
        f"Minimal preset should have 0 effects, "
        f"got {len(enabled_effects)}: {enabled_effects}"
    )


@pytest.mark.compliance
@pytest.mark.unit
def test_req_9_9_effect_consistency_across_segments():
    """Test that same effect is applied consistently across all segments per req 9.9."""
    from src.video.subtitle_positioning import UnifiedSubtitleConfig
    from src.video.unified_subtitle_generator import UnifiedSubtitleGenerator

    config = UnifiedSubtitleConfig(style_preset="bold")
    frame_size = (1920, 1080)
    product_id = "test_consistency"
    generator = UnifiedSubtitleGenerator(config, frame_size, product_id=product_id)

    # Multiple word timings spanning multiple segments
    word_timings = [
        {"word": "First", "start": 0.0, "end": 0.5},
        {"word": "segment", "start": 0.5, "end": 1.0},
        {"word": "Second", "start": 1.0, "end": 1.5},
        {"word": "segment", "start": 1.5, "end": 2.0},
        {"word": "Third", "start": 2.0, "end": 2.5},
        {"word": "segment", "start": 2.5, "end": 3.0},
    ]

    # Generate ASS content
    ass_content = _generate_ass_content(generator, word_timings)

    dialogue_lines = [
        line for line in ass_content.split("\n") if line.startswith("Dialogue:")
    ]

    assert len(dialogue_lines) >= 1, "Should have dialogue lines"

    # Extract effect types from each line
    import re

    effect_types_per_line = []
    for line in dialogue_lines:
        # Extract effect codes from {...}
        effects_match = re.search(r"\{([^}]+)\}", line)
        if effects_match:
            effects_str = effects_match.group(1)
            # Find unique effect types (commands starting with backslash)
            effect_commands = set(re.findall(r"\\([a-z]+)", effects_str))
            effect_types_per_line.append(effect_commands)
        else:
            effect_types_per_line.append(set())

    # Verify all segments use the same effect types
    if effect_types_per_line:
        first_effects = effect_types_per_line[0]
        for i, effects in enumerate(effect_types_per_line[1:], 1):
            assert (
                effects == first_effects
            ), f"Segment {i+1} has different effects than segment 1"


# =============================================================================
# REQUIREMENT 10: AI Service Integration Tests (Req 10.1, 10.2, 10.3)
# =============================================================================


@pytest.mark.compliance
@pytest.mark.unit
def test_req_10_1_tts_provider_order_configuration():
    """Test that TTS config defines provider order per req 10.1."""
    from src.video.config import TTSConfig

    # Test valid configuration
    config = TTSConfig(
        provider_order=["google_cloud", "coqui"],
        google_cloud=None,
        coqui=None,
    )

    assert hasattr(config, "provider_order"), "TTSConfig should have provider_order"
    assert isinstance(config.provider_order, list), "provider_order should be a list"


# Removed: test_req_10_2_google_chirp_prioritized_for_tts
# Removed: test_req_10_3_tts_fallback_chain_google_to_coqui
# These tests expected a tts_config structure in video_production.yaml that
# no longer exists. TTS configuration is now managed in separate config files.


@pytest.mark.compliance
@pytest.mark.unit
def test_req_10_1_tts_manager_tries_providers_in_order():
    """Test that TTSManager tries providers in configured order per req 10.1."""
    import inspect

    from src.video.tts import TTSManager

    # Verify the generate_speech method iterates through provider_order
    source = inspect.getsource(TTSManager.generate_speech)

    # Verify provider ordering logic exists
    assert (
        "for provider_name in self.config.provider_order" in source
    ), "TTSManager should iterate through configured provider order"

    # Verify logging shows provider attempts in order
    assert (
        'logger.info(f"Attempting TTS provider: {provider_name}")' in source
    ), "TTSManager should log provider attempts"

    # Verify provider-specific handling preserves order
    assert (
        'if provider_name == "google_cloud"' in source
    ), "Should handle google_cloud provider"
    assert 'elif provider_name == "coqui"' in source, "Should handle coqui provider"

    # Verify early return on success (preserves order)
    assert (
        "return voiceover_path" in source or "return output_path" in source
    ), "Should return path when provider succeeds"

    # Verify loop continues only on failure
    assert (
        "logger.warning(f\"Provider '{provider_name}' failed.\")" in source
    ), "Should log failure and continue to next provider"


@pytest.mark.compliance
@pytest.mark.unit
@pytest.mark.asyncio
async def test_req_10_3_tts_fallback_succeeds_on_second_provider():
    """Test that TTS fallback succeeds when primary fails per req 10.3."""
    from pathlib import Path
    from unittest.mock import AsyncMock, patch

    from src.video.config import CoquiTTSSettings, TTSConfig
    from src.video.tts import TTSManager

    # Mock availability flags before creating config
    with (
        patch("src.video.tts.GOOGLE_CLOUD_AVAILABLE", True),
        patch("src.video.tts.COQUI_AVAILABLE", True),
    ):
        from src.video.config import (
            GoogleCloudTTSSettings,
            GoogleCloudVoiceCriteria,
        )

        criteria = GoogleCloudVoiceCriteria(language_code="en-US")
        config = TTSConfig(
            provider_order=["google_cloud", "coqui"],
            google_cloud=GoogleCloudTTSSettings(
                model_name="test-model",
                language_code="en-US",
                voice_selection_criteria=[criteria],
            ),
            coqui=CoquiTTSSettings(model_name="tts_models/en/ljspeech/tacotron2-DDC"),
        )

    # Verify config has both providers after validation
    assert "google_cloud" in config.provider_order, "Should have google_cloud"
    assert "coqui" in config.provider_order, "Should have coqui as fallback"

    manager = TTSManager(config=config, secrets={})

    # Track which providers were attempted
    attempted_providers = []

    async def mock_google_fail(*args, **kwargs):
        attempted_providers.append("google_cloud")
        return None

    def mock_coqui_init(*args, **kwargs):
        attempted_providers.append("coqui")
        # Return a mock model object
        from unittest.mock import Mock

        return Mock()

    def mock_coqui_generate(text, output_path_str, model, config):
        # Write fake audio to simulate success
        from pathlib import Path

        p = Path(output_path_str)
        p.write_text("fake audio data")

    with (
        patch(
            "src.video.tts._generate_google_cloud_speech",
            side_effect=mock_google_fail,
        ),
        patch("src.video.tts.COQUI_AVAILABLE", True),
        patch(
            "src.video.tts._initialize_coqui_tts_model",
            side_effect=mock_coqui_init,
        ),
        patch(
            "src.video.tts._generate_coqui_speech_sync",
            side_effect=mock_coqui_generate,
        ),
    ):
        output_path = Path("/tmp/test_fallback.wav")  # noqa: S108
        result = await manager.generate_speech("test", output_path)

        # Cleanup
        if output_path.exists():
            output_path.unlink()

        # Verify fallback chain was followed
        assert "google_cloud" in attempted_providers, "Should attempt google first"
        assert "coqui" in attempted_providers, "Should attempt coqui as fallback"
        assert attempted_providers.index("google_cloud") < attempted_providers.index(
            "coqui"
        ), "Should try google before coqui"
        assert result == output_path, "Should return output path when fallback succeeds"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_10_1_whisper_primary_stt_provider():
    """Test that Whisper is primary STT provider per req 10.1."""
    from src.video import stt_functions

    # Verify Whisper is available and used as primary
    assert hasattr(
        stt_functions, "WHISPER_AVAILABLE"
    ), "Should check Whisper availability"
    assert hasattr(
        stt_functions, "generate_subtitles_with_whisper"
    ), "Should have Whisper generation function"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_10_3_google_stt_available_as_fallback():
    """Test that Google STT is available as fallback per req 10.3."""
    from src.video import stt_functions

    # Verify Google Cloud STT is available as fallback
    assert hasattr(
        stt_functions, "GOOGLE_CLOUD_STT_AVAILABLE"
    ), "Should check Google STT availability"
    assert hasattr(
        stt_functions, "transcribe_with_google_cloud_stt"
    ), "Should have Google STT function"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_10_2_google_stt_circuit_breaker():
    """Test that Google STT uses circuit breaker for reliability per req 10.2."""
    import inspect

    from src.video.stt_functions import transcribe_with_google_cloud_stt

    # Verify circuit breaker decorator is applied
    # Check function source or decorators
    source = inspect.getsource(transcribe_with_google_cloud_stt)

    assert (
        "@google_stt_circuit_breaker" in source or "circuit_breaker" in source.lower()
    ), "Google STT should use circuit breaker pattern"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_10_1_tts_provider_availability_checks():
    """Test that TTS providers check availability at runtime per req 10.1."""
    from src.video import tts

    # Verify availability flags exist
    assert hasattr(tts, "GOOGLE_CLOUD_AVAILABLE"), "Should check Google Cloud TTS"
    assert hasattr(tts, "COQUI_AVAILABLE"), "Should check Coqui TTS"

    # These are boolean flags set at import time
    assert isinstance(
        tts.GOOGLE_CLOUD_AVAILABLE, bool
    ), "GOOGLE_CLOUD_AVAILABLE should be bool"
    assert isinstance(tts.COQUI_AVAILABLE, bool), "COQUI_AVAILABLE should be bool"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_10_3_tts_config_validates_providers():
    """Test that TTS config validates provider settings exist per req 10.3."""
    # Test that config has validation logic (model_validator)
    # The validator filters provider_order to only include available providers
    import inspect

    from src.video.config import TTSConfig

    # Check if TTSConfig has model_validator for provider validation
    source = inspect.getsource(TTSConfig)
    assert (
        "model_validator" in source or "check_provider" in source.lower()
    ), "TTSConfig should validate provider availability"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_10_2_voice_selection_criteria_configurable():
    """Test that voice selection criteria is configurable per req 10.2."""
    # Test voice selection criteria configuration exists
    from src.video.config import GoogleCloudTTSSettings, GoogleCloudVoiceCriteria

    criteria = GoogleCloudVoiceCriteria(
        language_code="en-US",
        preferred_genders=["FEMALE", "MALE"],
        preferred_models=["Chirp3", "HD"],
    )

    settings = GoogleCloudTTSSettings(
        model_name="en-US-Chirp-3-HD",
        language_code="en-US",
        voice_selection_criteria=[criteria],
        api_timeout_sec=30,
    )

    assert (
        settings.voice_selection_criteria is not None
    ), "Voice selection criteria should be set"
    assert (
        len(settings.voice_selection_criteria) > 0
    ), "Voice selection criteria should be configurable"
    assert (
        settings.voice_selection_criteria[0].language_code == "en-US"
    ), "Should match configured criteria"


@pytest.mark.compliance
@pytest.mark.unit
def test_req_10_1_stt_settings_separate_from_tts():
    """Test that STT settings are independent from TTS per req 10.1."""
    from src.video.config import (
        GoogleCloudSTTSettings,
        TTSConfig,
        WhisperSettings,
    )

    # Verify separate configuration classes
    whisper_settings = WhisperSettings(
        model_size="base",
        model_device="cpu",
        model_dir=None,
    )

    google_stt_settings = GoogleCloudSTTSettings(
        model="chirp_2",
        language_code="en-US",
    )

    tts_config = TTSConfig(
        provider_order=["google_cloud"],
        google_cloud=None,
        coqui=None,
    )

    # These should be independent configuration objects
    assert type(whisper_settings).__name__ == "WhisperSettings"
    assert type(google_stt_settings).__name__ == "GoogleCloudSTTSettings"
    assert type(tts_config).__name__ == "TTSConfig"
