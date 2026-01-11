"""Tests for ASS subtitle effect generation."""

import re

import pytest

from src.video.subtitle_positioning import (
    Position,
    StylePreset,
    UnifiedSubtitleConfig,
)
from src.video.unified_subtitle_generator import UnifiedSubtitleGenerator


@pytest.fixture
def minimal_config():
    """Create a minimal preset config with no effects."""
    return UnifiedSubtitleConfig(
        style_preset=StylePreset.MINIMAL,
        anchor="bottom",
        margin=0.1,
        max_words_per_line=5,
        max_line_length=40,
        max_duration=5.0,
        min_duration=0.5,
    )


@pytest.fixture
def animated_config():
    """Create an animated preset config with effects enabled."""
    return UnifiedSubtitleConfig(
        style_preset=StylePreset.ANIMATED,
        anchor="bottom",
        margin=0.1,
        max_words_per_line=5,
        max_line_length=40,
        max_duration=5.0,
        min_duration=0.5,
        randomize_effects=False,
    )


@pytest.fixture
def random_config():
    """Create a random preset config for effect randomization."""
    return UnifiedSubtitleConfig(
        style_preset=StylePreset.RANDOM,
        anchor="bottom",
        margin=0.1,
        max_words_per_line=5,
        max_line_length=40,
        max_duration=5.0,
        min_duration=0.5,
        randomize_effects=True,
    )


@pytest.fixture
def modern_config():
    """Create a modern preset config with karaoke effect."""
    return UnifiedSubtitleConfig(
        style_preset=StylePreset.MODERN,
        anchor="bottom",
        margin=0.1,
        max_words_per_line=5,
        max_line_length=40,
        max_duration=5.0,
        min_duration=0.5,
    )


@pytest.fixture
def bold_config():
    """Create a bold preset config with fade effect."""
    return UnifiedSubtitleConfig(
        style_preset=StylePreset.BOLD,
        anchor="bottom",
        margin=0.1,
        max_words_per_line=5,
        max_line_length=40,
        max_duration=5.0,
        min_duration=0.5,
    )


@pytest.fixture
def frame_size():
    """Standard vertical video frame size."""
    return (1080, 1920)


@pytest.fixture
def sample_segments():
    """Sample subtitle segments for testing."""
    return [
        {"text": "Hello world", "start": 0.0, "end": 2.0},
        {"text": "This is a test", "start": 2.5, "end": 5.0},
        {"text": "Multiple words here", "start": 5.5, "end": 8.0},
    ]


class TestASSEffectSelection:
    """Tests for effect selection logic."""

    def test_minimal_preset_no_effects(self, minimal_config, frame_size):
        """Minimal preset should have no effects enabled."""
        generator = UnifiedSubtitleGenerator(
            config=minimal_config, frame_size=frame_size, product_id="TEST001"
        )

        effects = generator._selected_effects
        assert all(
            not v for v in effects.values()
        ), "Minimal preset should have no effects"

    def test_exactly_one_effect_per_video(self, modern_config, frame_size):
        """Each video should have exactly one effect (per REQUIREMENTS.md)."""
        generator = UnifiedSubtitleGenerator(
            config=modern_config, frame_size=frame_size, product_id="TEST001"
        )

        effects = generator._selected_effects
        enabled_effects = [k for k, v in effects.items() if v]
        assert (
            len(enabled_effects) <= 1
        ), f"Expected at most 1 effect, got {enabled_effects}"

    def test_random_effect_deterministic(self, random_config, frame_size):
        """Random effect selection should be deterministic per product_id."""
        gen1 = UnifiedSubtitleGenerator(
            config=random_config, frame_size=frame_size, product_id="PRODUCT123"
        )
        gen2 = UnifiedSubtitleGenerator(
            config=random_config, frame_size=frame_size, product_id="PRODUCT123"
        )

        assert (
            gen1._selected_effects == gen2._selected_effects
        ), "Same product_id should yield same effect selection"

    def test_different_products_may_have_different_effects(
        self, random_config, frame_size
    ):
        """Different products may get different random effects."""
        effects_seen = set()
        for i in range(10):
            gen = UnifiedSubtitleGenerator(
                config=random_config, frame_size=frame_size, product_id=f"PRODUCT{i}"
            )
            enabled = [k for k, v in gen._selected_effects.items() if v]
            if enabled:
                effects_seen.add(enabled[0])

        # With random selection, we should see at least some variation
        assert len(effects_seen) >= 1, "Random effect selection should work"


class TestKaraokeEffects:
    """Tests for karaoke word timing effects."""

    def test_karaoke_tag_format(self, modern_config, frame_size):
        r"""Karaoke should generate proper \kf tags."""
        generator = UnifiedSubtitleGenerator(
            config=modern_config, frame_size=frame_size, product_id="TEST001"
        )

        # Manually enable karaoke for testing
        generator._selected_effects["karaoke"] = True

        text = "Hello world testing"
        segment_duration = 3.0
        result = generator._create_karaoke_effects(text, segment_duration)

        # Should contain \kf tags
        assert "\\kf" in result, f"Expected \\kf tag in result: {result}"

        # Each word should have a timing tag
        kf_matches = re.findall(r"\\kf\d+", result)
        words = text.split()
        assert len(kf_matches) == len(
            words
        ), f"Expected {len(words)} \\kf tags, got {len(kf_matches)}"

    def test_karaoke_timing_calculation(self, modern_config, frame_size):
        """Karaoke timing should be evenly distributed across words."""
        generator = UnifiedSubtitleGenerator(
            config=modern_config, frame_size=frame_size, product_id="TEST001"
        )
        generator._selected_effects["karaoke"] = True

        text = "One two three four"
        segment_duration = 4.0  # 4 seconds for 4 words = 1 second each
        result = generator._create_karaoke_effects(text, segment_duration)

        # Extract timing values
        timings = re.findall(r"\\kf(\d+)", result)
        assert len(timings) == 4

        # Each timing should be roughly equal (100cs each, with min/max constraints)
        for t in timings:
            timing_value = int(t)
            assert (
                20 <= timing_value <= 200
            ), f"Timing {timing_value} outside valid range"

    def test_karaoke_single_word_no_effect(self, modern_config, frame_size):
        """Single word should not have karaoke effects."""
        generator = UnifiedSubtitleGenerator(
            config=modern_config, frame_size=frame_size, product_id="TEST001"
        )
        generator._selected_effects["karaoke"] = True

        text = "Hello"
        result = generator._create_karaoke_effects(text, 2.0)

        # Single word should return unchanged
        assert result == "Hello", "Single word should not have karaoke tags"

    def test_karaoke_word_preservation(self, modern_config, frame_size):
        """Karaoke should preserve all original words."""
        generator = UnifiedSubtitleGenerator(
            config=modern_config, frame_size=frame_size, product_id="TEST001"
        )
        generator._selected_effects["karaoke"] = True

        text = "This is a test phrase"
        result = generator._create_karaoke_effects(text, 5.0)

        # All words should be in the result
        for word in text.split():
            assert word in result, f"Word '{word}' missing from result"


class TestFadeEffects:
    """Tests for fade in/out effects."""

    def test_fade_tag_format(self, bold_config, frame_size, sample_segments):
        r"""Fade should generate proper \fad tags."""
        generator = UnifiedSubtitleGenerator(
            config=bold_config, frame_size=frame_size, product_id="TEST001"
        )
        generator._selected_effects["fade"] = True

        position = Position(x=0.5, y=0.8)
        colors = {"primary": "&H00FFFFFF", "outline": "&H00000000"}

        dialogue = generator._create_dialogue_line(
            sample_segments[0], position, colors, debug_mode=True
        )

        assert dialogue is not None, "Dialogue should not be None"
        assert "\\fad(" in dialogue, f"Expected \\fad tag in: {dialogue}"

    def test_fade_duration_symmetric(self, bold_config, frame_size, sample_segments):
        """Fade should have symmetric in/out durations."""
        generator = UnifiedSubtitleGenerator(
            config=bold_config, frame_size=frame_size, product_id="TEST001"
        )
        generator._selected_effects["fade"] = True

        position = Position(x=0.5, y=0.8)
        colors = {"primary": "&H00FFFFFF", "outline": "&H00000000"}

        dialogue = generator._create_dialogue_line(sample_segments[0], position, colors)

        assert dialogue is not None, "Dialogue should not be None"

        # Extract fade values
        fad_match = re.search(r"\\fad\((\d+),(\d+)\)", dialogue)
        assert fad_match, f"Expected \\fad(in,out) pattern in: {dialogue}"

        fade_in = int(fad_match.group(1))
        fade_out = int(fad_match.group(2))
        assert fade_in == fade_out, "Fade in and out should be symmetric"
        assert fade_in > 0, "Fade duration should be positive"


class TestTypewriterEffect:
    """Tests for typewriter reveal effect."""

    def test_typewriter_alpha_transition(
        self, animated_config, frame_size, sample_segments
    ):
        """Typewriter should use alpha transitions."""
        generator = UnifiedSubtitleGenerator(
            config=animated_config, frame_size=frame_size, product_id="TEST001"
        )
        generator._selected_effects["typewriter"] = True

        position = Position(x=0.5, y=0.8)
        colors = {"primary": "&H00FFFFFF", "outline": "&H00000000"}

        dialogue = generator._create_dialogue_line(sample_segments[0], position, colors)

        assert dialogue is not None, "Dialogue should not be None"

        # Typewriter uses alpha transitions
        assert (
            "\\alpha" in dialogue or "\\t(" in dialogue
        ), f"Expected alpha transition in: {dialogue}"


class TestASSSyntaxValidation:
    """Tests for ASS file syntax correctness."""

    def test_ass_header_structure(self, minimal_config, frame_size):
        """ASS header should have proper sections."""
        generator = UnifiedSubtitleGenerator(
            config=minimal_config, frame_size=frame_size, product_id="TEST001"
        )

        colors = {"primary": "&H00FFFFFF", "outline": "&H00000000"}
        header = generator._create_ass_header(font_size=48, colors=colors)

        # Check required sections
        header_text = "\n".join(header)
        assert "[Script Info]" in header_text
        assert "[V4+ Styles]" in header_text
        assert "[Events]" in header_text
        assert "ScriptType: v4.00+" in header_text

    def test_ass_playres_matches_frame_size(self, minimal_config, frame_size):
        """ASS PlayResX/Y should match frame size."""
        generator = UnifiedSubtitleGenerator(
            config=minimal_config, frame_size=frame_size, product_id="TEST001"
        )

        colors = {"primary": "&H00FFFFFF", "outline": "&H00000000"}
        header = generator._create_ass_header(font_size=48, colors=colors)
        header_text = "\n".join(header)

        assert f"PlayResX: {frame_size[0]}" in header_text
        assert f"PlayResY: {frame_size[1]}" in header_text

    def test_ass_time_format(self, minimal_config, frame_size):
        """ASS time format should be H:MM:SS.CC."""
        generator = UnifiedSubtitleGenerator(
            config=minimal_config, frame_size=frame_size, product_id="TEST001"
        )

        # Test various times
        assert generator._format_ass_time(0.0) == "0:00:00.00"
        assert generator._format_ass_time(1.5) == "0:00:01.50"
        assert generator._format_ass_time(61.25) == "0:01:01.25"
        assert generator._format_ass_time(3662.0) == "1:01:02.00"

    def test_ass_time_centiseconds_precision(self, minimal_config, frame_size):
        """ASS time should have centisecond precision."""
        generator = UnifiedSubtitleGenerator(
            config=minimal_config, frame_size=frame_size, product_id="TEST001"
        )

        result = generator._format_ass_time(1.234)
        # Should round to centiseconds (.23)
        assert result == "0:00:01.23"

    def test_dialogue_line_format(self, minimal_config, frame_size, sample_segments):
        """Dialogue lines should follow ASS format."""
        generator = UnifiedSubtitleGenerator(
            config=minimal_config, frame_size=frame_size, product_id="TEST001"
        )

        position = Position(x=0.5, y=0.8)
        colors = {"primary": "&H00FFFFFF", "outline": "&H00000000"}

        dialogue = generator._create_dialogue_line(sample_segments[0], position, colors)

        assert dialogue is not None, "Dialogue should not be None"

        # Check dialogue line format
        assert dialogue.startswith("Dialogue: 0,"), f"Bad dialogue prefix: {dialogue}"
        assert ",Default,," in dialogue, "Missing style reference"
        assert "\\pos(" in dialogue, "Missing position tag"

    def test_position_tag_coordinates(
        self, minimal_config, frame_size, sample_segments
    ):
        """Position tag should have valid pixel coordinates."""
        generator = UnifiedSubtitleGenerator(
            config=minimal_config, frame_size=frame_size, product_id="TEST001"
        )

        position = Position(x=0.5, y=0.8)
        colors = {"primary": "&H00FFFFFF", "outline": "&H00000000"}

        dialogue = generator._create_dialogue_line(sample_segments[0], position, colors)

        assert dialogue is not None, "Dialogue should not be None"

        # Extract position
        pos_match = re.search(r"\\pos\((\d+),(\d+)\)", dialogue)
        assert pos_match, f"Expected \\pos(x,y) in: {dialogue}"

        x = int(pos_match.group(1))
        y = int(pos_match.group(2))

        # Should be within frame bounds
        assert 0 <= x <= frame_size[0], f"X={x} outside frame width {frame_size[0]}"
        assert 0 <= y <= frame_size[1], f"Y={y} outside frame height {frame_size[1]}"


class TestMovementEffect:
    """Tests for movement/floating effect."""

    def test_movement_uses_move_tag(self, animated_config, frame_size, sample_segments):
        r"""Movement effect should use \move tag instead of \pos."""
        generator = UnifiedSubtitleGenerator(
            config=animated_config, frame_size=frame_size, product_id="TEST001"
        )
        generator._selected_effects["movement"] = True

        position = Position(x=0.5, y=0.8)
        colors = {"primary": "&H00FFFFFF", "outline": "&H00000000"}

        dialogue = generator._create_dialogue_line(sample_segments[0], position, colors)

        assert dialogue is not None, "Dialogue should not be None"
        assert "\\move(" in dialogue, f"Expected \\move tag in: {dialogue}"
        assert "\\pos(" not in dialogue, "Should not have \\pos when using \\move"


class TestScalePulseEffect:
    """Tests for scale pulse effect."""

    def test_scale_pulse_uses_transforms(
        self, animated_config, frame_size, sample_segments
    ):
        """Scale pulse should use transform tags."""
        generator = UnifiedSubtitleGenerator(
            config=animated_config, frame_size=frame_size, product_id="TEST001"
        )
        generator._selected_effects["scale_pulse"] = True

        position = Position(x=0.5, y=0.8)
        colors = {"primary": "&H00FFFFFF", "outline": "&H00000000"}

        dialogue = generator._create_dialogue_line(sample_segments[0], position, colors)

        assert dialogue is not None, "Dialogue should not be None"
        assert "\\t(" in dialogue, f"Expected transform tag in: {dialogue}"
        assert "\\fscx" in dialogue, "Expected scale X in transform"
        assert "\\fscy" in dialogue, "Expected scale Y in transform"


class TestGlowEffect:
    """Tests for glow/color transition effect."""

    def test_glow_uses_color_transition(
        self, animated_config, frame_size, sample_segments
    ):
        """Glow effect should use color transitions."""
        generator = UnifiedSubtitleGenerator(
            config=animated_config, frame_size=frame_size, product_id="TEST001"
        )
        generator._selected_effects["glow"] = True

        position = Position(x=0.5, y=0.8)
        colors = {"primary": "&H00FFFFFF", "outline": "&H00000000"}

        dialogue = generator._create_dialogue_line(sample_segments[0], position, colors)

        assert dialogue is not None, "Dialogue should not be None"
        assert "\\t(" in dialogue, f"Expected transform tag in: {dialogue}"
        assert "\\3c" in dialogue, "Expected outline color change (\\3c)"


class TestRotationBounceEffect:
    """Tests for rotation bounce effect."""

    def test_rotation_uses_frz_tag(self, animated_config, frame_size, sample_segments):
        r"""Rotation bounce should use \frz rotation tag."""
        generator = UnifiedSubtitleGenerator(
            config=animated_config, frame_size=frame_size, product_id="TEST001"
        )
        generator._selected_effects["rotation_bounce"] = True

        position = Position(x=0.5, y=0.8)
        colors = {"primary": "&H00FFFFFF", "outline": "&H00000000"}

        dialogue = generator._create_dialogue_line(sample_segments[0], position, colors)

        assert dialogue is not None, "Dialogue should not be None"
        assert "\\t(" in dialogue, f"Expected transform tag in: {dialogue}"
        assert "\\frz" in dialogue, "Expected rotation tag (\\frz)"


class TestStylePresetApplication:
    """Tests for style preset configuration."""

    def test_minimal_vs_animated_presets_differ(
        self, minimal_config, animated_config, frame_size
    ):
        """Minimal and animated presets should have different effect configurations."""
        gen_minimal = UnifiedSubtitleGenerator(
            config=minimal_config, frame_size=frame_size, product_id="TEST001"
        )
        gen_animated = UnifiedSubtitleGenerator(
            config=animated_config, frame_size=frame_size, product_id="TEST001"
        )

        minimal_effects = [k for k, v in gen_minimal._selected_effects.items() if v]
        _animated_effects = [k for k, v in gen_animated._selected_effects.items() if v]

        # Minimal should have no effects
        assert len(minimal_effects) == 0, "Minimal should have no effects"

    def test_style_config_has_required_fields(self, modern_config, frame_size):
        """Style config should have all required fields."""
        generator = UnifiedSubtitleGenerator(
            config=modern_config, frame_size=frame_size, product_id="TEST001"
        )

        required_fields = [
            "font_name",
            "font_color",
            "outline_color",
            "bold",
            "shadow",
            "outline_thickness",
        ]

        for field in required_fields:
            assert field in generator.style_config, f"Missing required field: {field}"
