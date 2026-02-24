"""Unit tests for TTS voice profile selection, markup, and metadata."""

import hashlib
import random
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.video.config import VideoConfig
from src.video.config.audio_models import (
    GoogleCloudTTSSettings,
    GoogleCloudVoiceCriteria,
    TextMarkupRule,
    TTSConfig,
    VoiceProfileConfig,
)
from src.video.tts import TTSManager


def _make_tts_config(
    profiles: dict[str, VoiceProfileConfig] | None = None,
    pool: list[str] | None = None,
    enabled: bool = True,
) -> TTSConfig:
    """Build a minimal TTSConfig with voice profiles for testing."""
    return TTSConfig(
        provider_order=["google_cloud"],
        google_cloud=GoogleCloudTTSSettings(
            language_code="en-US",
            voice_selection_criteria=[
                GoogleCloudVoiceCriteria(language_code="en-US", ssml_gender="FEMALE")
            ],
        ),
        voice_profiles_enabled=enabled,
        voice_profiles=profiles or {},
        voice_profile_pool=pool or [],
    )


SAMPLE_PROFILES = {
    "warm": VoiceProfileConfig(
        provider="gemini",
        style_prompt="Speak warmly.",
        speaking_rate=1.05,
        markup_rules=[
            TextMarkupRule(pattern=r"\.\s+", insert_after="[short pause] "),
        ],
    ),
    "calm": VoiceProfileConfig(
        provider="gemini",
        style_prompt="Speak calmly.",
        speaking_rate=0.95,
    ),
    "chirp": VoiceProfileConfig(
        provider="google_cloud",
        voice_criteria=[
            GoogleCloudVoiceCriteria(
                language_code="en-US", name_contains="Chirp3", ssml_gender="FEMALE"
            ),
        ],
    ),
}


class TestVoiceProfileSelection:
    """Tests for _select_voice_profile deterministic selection."""

    def test_disabled_returns_none(self):
        cfg = _make_tts_config(profiles=SAMPLE_PROFILES, enabled=False)
        mgr = TTSManager(cfg, {}, product_id="B0TEST001")
        name, profile = mgr._select_voice_profile()
        assert name is None
        assert profile is None

    def test_empty_profiles_returns_none(self):
        cfg = _make_tts_config(profiles={}, enabled=True)
        mgr = TTSManager(cfg, {})
        name, profile = mgr._select_voice_profile()
        assert name is None
        assert profile is None

    def test_deterministic_by_product_id(self):
        cfg = _make_tts_config(profiles=SAMPLE_PROFILES)
        mgr = TTSManager(cfg, {}, product_id="B09S8WMJY9")
        name1, _ = mgr._select_voice_profile()
        name2, _ = mgr._select_voice_profile()
        assert name1 == name2, "Same product_id should always pick the same profile"

    def test_different_products_can_get_different_profiles(self):
        cfg = _make_tts_config(profiles=SAMPLE_PROFILES)
        results = set()
        for asin in ["B0A", "B0B", "B0C", "B0D", "B0E", "B0F", "B0G", "B0H"]:
            mgr = TTSManager(cfg, {}, product_id=asin)
            name, _ = mgr._select_voice_profile()
            results.add(name)
        assert len(results) > 1, "Different products should produce variety"

    def test_pool_restricts_selection(self):
        cfg = _make_tts_config(profiles=SAMPLE_PROFILES, pool=["chirp"])
        for asin in ["B0A", "B0B", "B0C", "B0D"]:
            mgr = TTSManager(cfg, {}, product_id=asin)
            name, _ = mgr._select_voice_profile()
            assert name == "chirp"

    def test_pool_filters_invalid_names(self):
        cfg = _make_tts_config(profiles=SAMPLE_PROFILES, pool=["nonexistent", "warm"])
        mgr = TTSManager(cfg, {}, product_id="B0TEST001")
        name, profile = mgr._select_voice_profile()
        assert name == "warm"
        assert profile is not None

    def test_no_product_id_uses_random(self):
        cfg = _make_tts_config(profiles=SAMPLE_PROFILES)
        mgr = TTSManager(cfg, {}, product_id=None)
        name, profile = mgr._select_voice_profile()
        assert name in SAMPLE_PROFILES
        assert profile is not None


class TestVoiceProfileOverride:
    """Tests for CLI voice profile override."""

    def test_override_selects_specified_profile(self):
        cfg = _make_tts_config(profiles=SAMPLE_PROFILES)
        mgr = TTSManager(cfg, {}, product_id="B0TEST001", voice_profile_override="calm")
        name, profile = mgr._select_voice_profile()
        assert name == "calm"
        assert profile is not None
        assert profile.style_prompt == "Speak calmly."

    def test_override_ignores_product_hash(self):
        cfg = _make_tts_config(profiles=SAMPLE_PROFILES)
        # Without override, different products get different profiles
        mgr1 = TTSManager(cfg, {}, product_id="B0A", voice_profile_override="warm")
        mgr2 = TTSManager(cfg, {}, product_id="B0B", voice_profile_override="warm")
        name1, _ = mgr1._select_voice_profile()
        name2, _ = mgr2._select_voice_profile()
        assert name1 == name2 == "warm"

    def test_invalid_override_falls_back(self):
        cfg = _make_tts_config(profiles=SAMPLE_PROFILES)
        mgr = TTSManager(
            cfg, {}, product_id="B0TEST001", voice_profile_override="nonexistent"
        )
        name, profile = mgr._select_voice_profile()
        # Should fall back to normal selection, not None
        assert name in SAMPLE_PROFILES
        assert profile is not None


class TestMarkupRules:
    """Tests for _apply_markup_rules and _strip_markup."""

    def test_apply_markup_inserts_after_sentences(self):
        rules = [TextMarkupRule(pattern=r"\.\s+", insert_after="[short pause] ")]
        text = "First sentence. Second sentence. Third."
        result = TTSManager._apply_markup_rules(text, rules)
        assert "[short pause]" in result
        assert result.count("[short pause]") == 2  # after 1st and 2nd periods

    def test_apply_markup_inserts_before(self):
        rules = [TextMarkupRule(pattern=r"!\s+", insert_before="[excited] ")]
        text = "Wow! That's great! Done."
        result = TTSManager._apply_markup_rules(text, rules)
        assert result.count("[excited]") == 2

    def test_apply_markup_empty_rules(self):
        text = "No changes expected."
        result = TTSManager._apply_markup_rules(text, [])
        assert result == text

    def test_apply_markup_multiple_rules(self):
        rules = [
            TextMarkupRule(pattern=r"\.\s+", insert_after="[pause] "),
            TextMarkupRule(pattern=r"!\s+", insert_after="[short pause] "),
        ]
        text = "Sentence one. Wow! Sentence three."
        result = TTSManager._apply_markup_rules(text, rules)
        assert "[pause]" in result
        assert "[short pause]" in result

    def test_strip_markup_removes_pause_tags(self):
        text = "Hello. [short pause] World. [pause] Done."
        result = TTSManager._strip_markup(text)
        assert "[short pause]" not in result
        assert "[pause]" not in result
        assert "Hello" in result
        assert "World" in result

    def test_strip_markup_removes_long_pause(self):
        text = "Wait[long pause]here"
        result = TTSManager._strip_markup(text)
        assert "[long pause]" not in result

    def test_strip_markup_no_tags_unchanged(self):
        text = "Plain text without any tags."
        result = TTSManager._strip_markup(text)
        assert result == text


class TestGenerateSpeechRouting:
    """Tests for generate_speech profile routing and metadata capture."""

    @pytest.mark.asyncio
    async def test_empty_text_returns_none(self):
        cfg = _make_tts_config(profiles=SAMPLE_PROFILES)
        mgr = TTSManager(cfg, {})
        result = await mgr.generate_speech("", Path("/tmp/out.wav"))  # noqa: S108
        assert result is None

    @pytest.mark.asyncio
    async def test_whitespace_text_returns_none(self):
        cfg = _make_tts_config(profiles=SAMPLE_PROFILES)
        mgr = TTSManager(cfg, {})
        result = await mgr.generate_speech("   \n  ", Path("/tmp/out.wav"))  # noqa: S108
        assert result is None

    @pytest.mark.asyncio
    async def test_gemini_profile_tries_gemini_first(self):
        cfg = _make_tts_config(profiles=SAMPLE_PROFILES, pool=["warm"])
        mgr = TTSManager(cfg, {}, product_id="B0TEST")

        fake_path = Path("/tmp/test.wav")  # noqa: S108
        with patch(
            "src.video.tts._generate_gemini_speech", new_callable=AsyncMock
        ) as mock_gemini:
            mock_gemini.return_value = (fake_path, "Kore")
            result = await mgr.generate_speech("Test text.", fake_path)

            mock_gemini.assert_called_once()
            assert result == fake_path
            assert mgr.selected_profile_name == "warm"
            assert mgr.selected_voice_name == "Kore"

    @pytest.mark.asyncio
    async def test_gemini_failure_falls_back_to_google_cloud(self):
        cfg = _make_tts_config(profiles=SAMPLE_PROFILES, pool=["warm"])
        mgr = TTSManager(cfg, {}, product_id="B0TEST")

        fake_path = Path("/tmp/test.wav")  # noqa: S108
        with (
            patch(
                "src.video.tts._generate_gemini_speech", new_callable=AsyncMock
            ) as mock_gemini,
            patch(
                "src.video.tts._generate_google_cloud_speech", new_callable=AsyncMock
            ) as mock_gc,
        ):
            mock_gemini.side_effect = Exception("Gemini unavailable")
            mock_gc.return_value = (fake_path, "en-US-Chirp3-HD-Achird")

            result = await mgr.generate_speech("Test text.", fake_path)

            assert result == fake_path
            assert mgr.selected_voice_name == "en-US-Chirp3-HD-Achird"

    @pytest.mark.asyncio
    async def test_google_cloud_profile_skips_gemini(self):
        cfg = _make_tts_config(profiles=SAMPLE_PROFILES, pool=["chirp"])
        mgr = TTSManager(cfg, {}, product_id="B0TEST")

        fake_path = Path("/tmp/test.wav")  # noqa: S108
        with (
            patch(
                "src.video.tts._generate_gemini_speech", new_callable=AsyncMock
            ) as mock_gemini,
            patch(
                "src.video.tts._generate_google_cloud_speech", new_callable=AsyncMock
            ) as mock_gc,
        ):
            mock_gc.return_value = (fake_path, "en-US-Chirp3-HD-Achird")

            await mgr.generate_speech("Test text.", fake_path)

            mock_gemini.assert_not_called()
            mock_gc.assert_called_once()
            assert mgr.selected_profile_name == "chirp"

    @pytest.mark.asyncio
    async def test_markup_stripped_on_fallback(self):
        """When Gemini fails and fallback to google_cloud, markup should be stripped."""
        cfg = _make_tts_config(profiles=SAMPLE_PROFILES, pool=["warm"])
        mgr = TTSManager(cfg, {}, product_id="B0TEST")

        fake_path = Path("/tmp/test.wav")  # noqa: S108
        with (
            patch(
                "src.video.tts._generate_gemini_speech", new_callable=AsyncMock
            ) as mock_gemini,
            patch(
                "src.video.tts._generate_google_cloud_speech", new_callable=AsyncMock
            ) as mock_gc,
        ):
            mock_gemini.side_effect = Exception("Gemini down")
            mock_gc.return_value = (fake_path, "en-US-Chirp3-HD-Achird")

            await mgr.generate_speech("First sentence. Second sentence.", fake_path)

            # The text passed to google_cloud should not contain markup
            call_args = mock_gc.call_args
            text_arg = call_args[0][0]  # first positional arg is text
            assert "[short pause]" not in text_arg

    @pytest.mark.asyncio
    async def test_no_providers_returns_none(self):
        cfg = _make_tts_config(profiles=SAMPLE_PROFILES, pool=["chirp"])
        cfg.provider_order = []
        mgr = TTSManager(cfg, {}, product_id="B0TEST")

        result = await mgr.generate_speech("Test.", Path("/tmp/test.wav"))  # noqa: S108
        assert result is None


class TestTextMarkupRuleModel:
    """Tests for TextMarkupRule Pydantic model."""

    def test_defaults(self):
        rule = TextMarkupRule(pattern=r"\.\s+")
        assert rule.insert_before == ""
        assert rule.insert_after == ""

    def test_full_config(self):
        rule = TextMarkupRule(
            pattern=r"!\s+",
            insert_before="[excited] ",
            insert_after="[pause] ",
        )
        assert rule.pattern == r"!\s+"
        assert rule.insert_before == "[excited] "
        assert rule.insert_after == "[pause] "


class TestVoiceProfileConfigModel:
    """Tests for VoiceProfileConfig Pydantic model."""

    def test_defaults(self):
        profile = VoiceProfileConfig()
        assert profile.provider == "google_cloud"
        assert profile.style_prompt is None
        assert profile.gemini_model_name == "gemini-2.5-flash-tts"
        assert profile.voice_criteria is None
        assert profile.speaking_rate is None
        assert profile.pitch is None
        assert profile.markup_rules == []

    def test_gemini_profile(self):
        profile = VoiceProfileConfig(
            provider="gemini",
            style_prompt="Speak warmly.",
            gemini_model_name="gemini-2.5-pro-tts",
            speaking_rate=1.1,
        )
        assert profile.provider == "gemini"
        assert profile.gemini_model_name == "gemini-2.5-pro-tts"

    def test_google_cloud_profile_with_criteria(self):
        profile = VoiceProfileConfig(
            provider="google_cloud",
            voice_criteria=[
                GoogleCloudVoiceCriteria(language_code="en-US", name_contains="Chirp3"),
            ],
        )
        assert profile.voice_criteria is not None
        assert len(profile.voice_criteria) == 1
        assert profile.voice_criteria[0].name_contains == "Chirp3"


class TestTTSConfigVoiceProfiles:
    """Tests for TTSConfig voice profile fields."""

    def test_voice_profiles_default_empty(self):
        cfg = _make_tts_config()
        assert cfg.voice_profiles == {}
        assert cfg.voice_profile_pool == []
        assert cfg.voice_profiles_enabled is True

    def test_voice_profiles_loaded(self):
        cfg = _make_tts_config(profiles=SAMPLE_PROFILES)
        assert len(cfg.voice_profiles) == 3
        assert "warm" in cfg.voice_profiles
        assert cfg.voice_profiles["warm"].provider == "gemini"

    def test_voice_profiles_disabled(self):
        cfg = _make_tts_config(profiles=SAMPLE_PROFILES, enabled=False)
        assert cfg.voice_profiles_enabled is False
        # Profiles still loaded, just not used
        assert len(cfg.voice_profiles) == 3
